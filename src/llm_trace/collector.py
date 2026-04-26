"""Collector — loads a model once, runs forward + autoregressive generation per
prompt, and extracts every internal state a renderer might want.

Absorbs what were previously separate scripts/modules for model loading, chat
template application, generation loop, and EOS handling.

Key discipline:

- `attn_implementation="eager"` is forced at load time so `output_attentions`
  returns real values on the first trace pass (SDPA/Flash return None silently).
- `torch.manual_seed` is set from `gen_params["seed"]` so cache keys are honest
  under temperature > 0.
- The generation loop does NOT request attentions/hidden-states on each step —
  only the first trace forward pass does. This keeps per-token cost normal.
- Arrays stored in `TraceData` are trimmed per the `CollectionConfig`
  (attention slicing policy, `hidden_dims_keep`, `top_k`) so cache size is
  bounded. Full tensors never escape this module.

Nothing outside this module imports torch via `llm_trace`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_trace import cache
from llm_trace.trace_data import TraceData  # re-exported below

__all__ = ["CollectionConfig", "LoadedModel", "TraceData", "Collector", "load_model"]


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """What to slice out of a forward pass. Bounds cache size.

    `attention_layers` / `attention_heads` accept either an explicit tuple of
    indices or the string `"all"`. `"all"` resolves to every available
    layer/head at slice time, so a single config works across models of
    different sizes.
    """
    attention_layers: tuple[int, ...] | str = (0,)
    attention_heads: tuple[int, ...] | str = "all"
    hidden_dims_keep: int = 32
    top_k: int = 15
    extra_temps_for_viz: tuple[float, ...] = (0.1, 0.7, 1.5)
    generation_top_k: int = 5   # how many alternatives to record per gen step


# ── Loaded model bundle ────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    model_id: str
    tokenizer: Any
    model: Any
    config: Any
    is_chat: bool
    eos_ids: set[int]
    arch_info: dict[str, Any]   # goes into TraceData.model_meta


def load_model(model_id: str, dtype: Any = None) -> LoadedModel:
    """Load tokenizer + model with eager attention.

    Prints a one-line warning about the ~2× slowdown vs SDPA.
    """
    if dtype is None:
        dtype = torch.float32

    print(f"[collector] loading {model_id} (attn=eager, dtype={dtype})...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000

    cfg = model.config
    is_chat = getattr(tokenizer, "chat_template", None) is not None

    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gc_eos, int):
        eos_ids.add(gc_eos)
    elif isinstance(gc_eos, (list, tuple)):
        eos_ids.update(int(x) for x in gc_eos)

    arch_info = _describe_architecture(model, cfg, tokenizer, eos_ids, load_ms)
    return LoadedModel(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        config=cfg,
        is_chat=is_chat,
        eos_ids=eos_ids,
        arch_info=arch_info,
    )


def _describe_architecture(model, cfg, tokenizer, eos_ids, load_ms) -> dict[str, Any]:
    """Collect architecture metadata for display."""
    has_wpe = hasattr(model, "transformer") and hasattr(getattr(model, "transformer", None), "wpe")
    positional = "learned" if has_wpe else ("rope" if getattr(cfg, "rope_theta", None) else "unknown")

    if hasattr(cfg, "rms_norm_eps"):
        normalization = "rmsnorm"
    elif hasattr(cfg, "layer_norm_epsilon") or hasattr(cfg, "layer_norm_eps"):
        normalization = "layernorm"
    else:
        normalization = "unknown"

    n_head = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    n_kv = getattr(cfg, "num_key_value_heads", n_head)
    n_layer = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    ffn = getattr(cfg, "intermediate_size", None) or (4 * hidden if hidden else None)
    ctx = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_ctx", None)

    return {
        "model_type": getattr(cfg, "model_type", "unknown"),
        "vocab_size": int(cfg.vocab_size),
        "n_layer": int(n_layer) if n_layer else None,
        "n_head": int(n_head) if n_head else None,
        "n_kv_heads": int(n_kv) if n_kv else None,
        "hidden_size": int(hidden) if hidden else None,
        "ffn_intermediate": int(ffn) if ffn else None,
        "ctx_window": int(ctx) if ctx else None,
        "positional_encoding": positional,
        "normalization": normalization,
        "rope_theta": getattr(cfg, "rope_theta", None),
        "eos_ids": sorted(eos_ids),
        "eos_tokens": [tokenizer.decode([i]) for i in sorted(eos_ids)],
        "bos_id": tokenizer.bos_token_id,
        "pad_id": tokenizer.pad_token_id,
        "load_ms": round(load_ms, 1),
    }


# ── Collector ──────────────────────────────────────────────────────────────

class Collector:
    """Orchestrates one trace per (model, prompt) cell. Hits cache first."""

    def __init__(self, cfg: CollectionConfig, cache_dir: Path, use_cache: bool = True):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    # Exposed for tests to spy on (see test_cache_hit).
    _forward_calls: int = 0

    def collect(
        self,
        loaded: LoadedModel,
        prompt: str | list[dict],
        gen_params: dict[str, Any],
        system: str | None = None,
    ) -> TraceData:
        """Main entry. Cache hit → load + return. Miss → run + save + return.

        `system` is an optional system prompt. For chat-template models it's
        prepended as a system role; for base models it's concatenated with a
        one-line warning printed once (since base LMs don't understand system
        prompts but we oblige the user's intent).
        """
        original, templated = _resolve_prompt(
            loaded.tokenizer, prompt, loaded.is_chat, system=system
        )

        # Include system in the cache key (so different systems → different cells).
        effective_gen = {**gen_params, "__system__": system or ""}
        key = cache.cache_key(loaded.model_id, templated, effective_gen)

        if self.use_cache:
            hit = cache.load(key, self.cache_dir)
            if hit is not None:
                arrays, meta = hit
                return TraceData.from_cache_payload(arrays, meta)

        trace = self._run(loaded, original, templated, gen_params, system)

        if self.use_cache:
            arrays, meta = trace.to_cache_payload()
            cache.save(key, arrays, meta, self.cache_dir)

        return trace

    # ── Core trace pipeline (miss path) ──────────────────────────────────

    def _run(
        self,
        loaded: LoadedModel,
        original_prompt: str,
        templated_prompt: str,
        gen_params: dict[str, Any],
        system: str | None,
    ) -> TraceData:
        self._forward_calls += 1
        seed = int(gen_params.get("seed", 42))
        torch.manual_seed(seed)

        tokenizer = loaded.tokenizer
        model = loaded.model
        dims_keep = self.cfg.hidden_dims_keep
        top_k = self.cfg.top_k

        # Tokenize
        token_ids_list = tokenizer.encode(templated_prompt, add_special_tokens=False)
        if not token_ids_list:
            # Fallback: some chat templates rely on add_special_tokens=True
            token_ids_list = tokenizer.encode(templated_prompt)
        tokens = [tokenizer.decode([t]) for t in token_ids_list]

        # First trace forward pass — full hidden + attention output
        t0 = time.perf_counter()
        inp = torch.tensor([token_ids_list])
        with torch.no_grad():
            out = model(inp, output_hidden_states=True, output_attentions=True)
        first_forward_ms = (time.perf_counter() - t0) * 1000
        self._forward_calls += 1

        # ── Embeddings ────────────────────────────────────────────────
        emb_token, emb_pos, emb_combined = _extract_embeddings(
            model, token_ids_list, dims_keep
        )

        # ── Attention slices (policy from CollectionConfig) ───────────
        attentions_sliced = _slice_attentions(
            out.attentions,
            layers=self.cfg.attention_layers,
            heads=self.cfg.attention_heads,
        )

        # ── Hidden state last-token across layers ─────────────────────
        hidden_last = np.stack([
            hs[0, -1].detach().float().numpy()[:dims_keep] for hs in out.hidden_states
        ]).astype(np.float32)
        hidden_last_norms = np.array([
            float(np.linalg.norm(hs[0, -1].detach().float().numpy()))
            for hs in out.hidden_states
        ], dtype=np.float32)

        # Full-width final hidden state (last token, last layer = post-final-LN).
        # This is the exact vector the LM head projects to vocab logits.
        final_hidden_full = (
            out.hidden_states[-1][0, -1].detach().float().numpy().astype(np.float32)
        )

        # ── Per-token residual-stream norms across all layer outputs ──
        # Shape: (n_layers+1, n_tokens). Powers the token × layer heatmap
        # in animated_v2.py — viewer sees one stream per input token.
        hidden_norms = np.stack([
            np.linalg.norm(hs[0].detach().float().numpy(), axis=-1)
            for hs in out.hidden_states
        ]).astype(np.float32)

        # ── Logits / probs top-k at last position ─────────────────────
        logits_full = out.logits[0, -1].detach().float().numpy()
        logits_top_ids, logits_top_values = _top_k(logits_full, top_k)
        logits_top_tokens = [tokenizer.decode([int(i)]) for i in logits_top_ids]
        probs_full = _softmax(logits_full)
        probs_top_ids, probs_top_values = _top_k(probs_full, top_k)
        probs_top_tokens = [tokenizer.decode([int(i)]) for i in probs_top_ids]

        # LM-head weight rows for the top-K predicted ids — these are the
        # exact vectors dot-producted with the final hidden state to produce
        # each logit (logit[i] = <final_hidden_full, W_lm_head[id_i]>). Shipping
        # only the top-K rows keeps payload tiny vs the full [vocab × hidden].
        lm_head = model.get_output_embeddings()
        lm_head_w = lm_head.weight.detach().float().cpu().numpy()
        lm_head_top_rows = np.stack(
            [lm_head_w[int(tid)] for tid in logits_top_ids]
        ).astype(np.float32)
        # Tied-weight detection — used by the LM-head step to explain whether
        # this row is the same as the input embedding (GPT-2) or a different
        # learned matrix (Llama).
        in_emb_w = model.get_input_embeddings().weight
        out_emb_w = lm_head.weight
        tied = bool(in_emb_w.data_ptr() == out_emb_w.data_ptr())
        loaded.arch_info["tied_word_embeddings"] = tied

        # ── Temperature scan ──────────────────────────────────────────
        temp_scan = _temperature_scan(
            logits_full, tokenizer, temps=self.cfg.extra_temps_for_viz, top_k=10
        )

        # ── Autoregressive generation (with per-step capture) ──────────
        generation, per_token_ms, per_step_hidden_norms, per_step_top_ids, per_step_top_probs = _generate(
            model,
            tokenizer,
            token_ids_list,
            gen_params=gen_params,
            eos_ids=loaded.eos_ids,
            top_k_alts=self.cfg.generation_top_k,
            per_step_top_k=self.cfg.top_k,
        )
        self._forward_calls += len(generation)

        # Detokenize the full generated response in one call so we don't lose
        # sentencepiece ▁-marker spaces that per-token decode would drop.
        gen_ids_for_text = [s["id"] for s in generation if not s.get("is_eos")]
        generation_text = tokenizer.decode(gen_ids_for_text, skip_special_tokens=True) \
            if gen_ids_for_text else ""

        timings = {
            "model_load_ms": loaded.arch_info.get("load_ms"),
            "first_forward_ms": round(first_forward_ms, 1),
            "per_token_ms": per_token_ms,
        }

        return TraceData(
            model_id=loaded.model_id,
            prompt=original_prompt,
            system_prompt=system,
            templated_prompt=templated_prompt,
            gen_params=gen_params,
            model_meta=loaded.arch_info,
            tokens=tokens,
            token_ids=list(token_ids_list),
            embeddings_token=emb_token,
            embeddings_pos=emb_pos,
            embeddings_combined=emb_combined,
            attentions=attentions_sliced,
            hidden_last=hidden_last,
            hidden_last_norms=hidden_last_norms,
            final_hidden_full=final_hidden_full,
            hidden_norms=hidden_norms,
            logits_top_ids=logits_top_ids,
            logits_top_values=logits_top_values,
            logits_top_tokens=logits_top_tokens,
            probs_top_ids=probs_top_ids,
            probs_top_values=probs_top_values,
            probs_top_tokens=probs_top_tokens,
            lm_head_top_rows=lm_head_top_rows,
            temp_scan=temp_scan,
            generation=generation,
            generation_text=generation_text,
            per_step_hidden_norms=per_step_hidden_norms,
            per_step_top_ids=per_step_top_ids,
            per_step_top_probs=per_step_top_probs,
            timings=timings,
        )


# ── Helpers ────────────────────────────────────────────────────────────────

_BASE_MODEL_SYSTEM_WARNED = False


def _resolve_prompt(
    tokenizer,
    prompt: str | list[dict],
    is_chat: bool,
    system: str | None = None,
) -> tuple[str, str]:
    """Returns (original_for_display, templated_for_model).

    If `system` is given:
      - chat model: prepended as a {role:system, content} message (unless the
        caller already included one in a message list).
      - base model: concatenated as 'System: <text>\\n<prompt>' with a one-time
        warning, since base LMs don't understand system prompts.
    """
    global _BASE_MODEL_SYSTEM_WARNED

    if isinstance(prompt, list):
        original = json.dumps(prompt)
        messages = list(prompt)  # copy so we can prepend
    elif isinstance(prompt, str):
        original = prompt
        messages = [{"role": "user", "content": prompt}] if is_chat else None
    else:
        raise TypeError(f"prompt must be str or list[dict], got {type(prompt)}")

    if system and is_chat and messages is not None:
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": system}] + messages

    if is_chat and messages is not None:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif system and not is_chat:
        if not _BASE_MODEL_SYSTEM_WARNED:
            print(
                "[collector] warning: model is not chat-tuned; "
                "prepending system text as plain context."
            )
            _BASE_MODEL_SYSTEM_WARNED = True
        templated = f"System: {system}\n\n{original}"
    else:
        templated = original
    return original, templated


def _extract_embeddings(
    model, token_ids: list[int], dims_keep: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Pull token + (optional) position embeddings for each input token.

    Returns (emb_token, emb_pos_or_None, emb_combined) each shaped
    (n_tokens, dims_keep) float32. For RoPE models, emb_pos is None and
    emb_combined equals emb_token.
    """
    emb_layer = model.get_input_embeddings()
    emb_matrix = emb_layer.weight.detach().float().cpu().numpy()

    # Store FULL prompt-token embeddings (not truncated). Cost: ~hidden_size×n_tokens
    # × 4 bytes per cell — trivial. Lets the embed step show complete vectors.
    emb_token = np.stack([emb_matrix[tid] for tid in token_ids]).astype(np.float32)

    wpe = getattr(getattr(model, "transformer", None), "wpe", None)
    if wpe is not None:
        pos_matrix = wpe.weight.detach().float().cpu().numpy()
        emb_pos = np.stack([pos_matrix[i] for i in range(len(token_ids))]).astype(np.float32)
        emb_combined = (emb_token + emb_pos).astype(np.float32)
    else:
        emb_pos = None
        emb_combined = emb_token.copy()

    return emb_token, emb_pos, emb_combined


def _slice_attentions(
    attentions_tuple,
    layers: tuple[int, ...] | str,
    heads: tuple[int, ...] | str,
) -> dict[str, np.ndarray]:
    """Pick configured (layer, head) pairs; store under 'L{layer}H{head}' keys.

    Either argument may be the string "all" to include every available
    layer / head. Numeric indices outside the valid range are silently skipped.
    """
    out: dict[str, np.ndarray] = {}
    n_layers = len(attentions_tuple)
    layer_iter = range(n_layers) if layers == "all" else layers

    for L in layer_iter:
        if L >= n_layers:
            continue
        layer_attn = attentions_tuple[L][0].detach().float().numpy()   # (n_heads, seq, seq)
        n_heads = layer_attn.shape[0]
        head_iter = range(n_heads) if heads == "all" else heads
        for H in head_iter:
            if H >= n_heads:
                continue
            out[f"L{L}H{H}"] = layer_attn[H].astype(np.float32)
    return out


def _top_k(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(values)[-k:][::-1]
    return idx.astype(np.int64), values[idx].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    e = np.exp(shifted)
    return (e / e.sum()).astype(np.float32)


def _temperature_scan(
    logits: np.ndarray, tokenizer, temps: tuple[float, ...], top_k: int
) -> list[dict[str, Any]]:
    scan: list[dict[str, Any]] = []
    for t in temps:
        t_safe = max(float(t), 1e-6)
        scaled = logits / t_safe
        probs = _softmax(scaled)
        ids = np.argsort(probs)[-top_k:][::-1]
        scan.append({
            "temperature": float(t),
            "top": [
                {
                    "token": tokenizer.decode([int(i)]),
                    "id": int(i),
                    "prob": float(probs[i]),
                }
                for i in ids
            ],
        })
    return scan


def _generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    gen_params: dict[str, Any],
    eos_ids: set[int],
    top_k_alts: int,
    per_step_top_k: int = 15,
) -> tuple[list[dict[str, Any]], list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Autoregressive greedy generation with per-step capture.

    Each step pays an extra cost to request output_hidden_states — we use the
    residual-stream norms across layers to visualize how the representation
    evolves at every loop iteration.

    Returns: (steps, per_token_ms,
              per_step_hidden_norms (n_steps, n_layers+1),
              per_step_top_ids      (n_steps, per_step_top_k),
              per_step_top_probs    (n_steps, per_step_top_k))
    """
    max_new = int(gen_params.get("max_new_tokens", 50))
    stop_on_eos = bool(gen_params.get("stop_on_eos", True))

    current = list(prompt_ids)
    steps: list[dict[str, Any]] = []
    per_token_ms: list[float] = []
    hidden_norms_rows: list[np.ndarray] = []
    top_ids_rows: list[np.ndarray] = []
    top_probs_rows: list[np.ndarray] = []

    for i in range(max_new):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(torch.tensor([current]), output_hidden_states=True)
        dt = (time.perf_counter() - t0) * 1000
        per_token_ms.append(round(dt, 1))

        # Residual stream norm at the last token across all layer outputs.
        layer_norms = np.array([
            float(np.linalg.norm(hs[0, -1].detach().float().numpy()))
            for hs in out.hidden_states
        ], dtype=np.float32)
        hidden_norms_rows.append(layer_norms)

        logits = out.logits[0, -1].detach().float().numpy()
        probs = _softmax(logits)

        # Per-step top-K (for the HTML loop table).
        top_k_idx = np.argsort(probs)[-per_step_top_k:][::-1]
        top_ids_rows.append(top_k_idx.astype(np.int64))
        top_probs_rows.append(probs[top_k_idx].astype(np.float32))

        next_id = int(np.argmax(probs))
        is_eos = next_id in eos_ids

        # Smaller top-K kept in-line for the existing top_alts UI.
        top_alt_idx = top_k_idx[:top_k_alts]
        top_alts = [
            {"token": tokenizer.decode([int(t)]), "id": int(t), "prob": float(probs[t])}
            for t in top_alt_idx
        ]

        steps.append({
            "step": i + 1,
            "token": tokenizer.decode([next_id]),
            "id": next_id,
            "prob": float(probs[next_id]),
            "ctx_len": len(current),
            "is_eos": is_eos,
            "top_alts": top_alts,
        })

        current.append(next_id)
        if is_eos and stop_on_eos:
            break

    # Stack per-step arrays. If no steps ran, return zero-row arrays.
    if not steps:
        return steps, per_token_ms, \
            np.zeros((0, 1), dtype=np.float32), \
            np.zeros((0, per_step_top_k), dtype=np.int64), \
            np.zeros((0, per_step_top_k), dtype=np.float32)

    return (
        steps,
        per_token_ms,
        np.stack(hidden_norms_rows).astype(np.float32),
        np.stack(top_ids_rows).astype(np.int64),
        np.stack(top_probs_rows).astype(np.float32),
    )
