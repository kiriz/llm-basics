"""TraceData dataclass — the spine of the data model.

Kept in its own module (no torch, no transformers) so that the `llm-trace
render` subcommand and all renderers can import it without pulling torch.
This preserves the cache-based re-render feature (no model load required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BlockDeepDive:
    """Captured intermediates for ONE chosen (layer, head) of a transformer block.

    Powers the standalone "inside one block" deep-dive renderer. All arrays are
    float32; the chosen head is sliced out of the full QKV (so `q/k/v/context`
    are head_dim wide, not n_heads*head_dim).

    Sequence length matches the prompt that was forwarded; `block_output` for
    layer L equals the residual-stream input to layer L+1.
    """
    layer_index: int
    head_index: int
    n_heads: int
    head_dim: int
    activation: str               # 'gelu' | 'silu' | 'gelu_new'
    has_swiglu_gate: bool         # True for Llama-family (gate_proj * silu(up_proj))

    pre_ln1: np.ndarray           # (seq, hidden) — block input = residual stream
    post_ln1: np.ndarray          # (seq, hidden)
    q: np.ndarray                 # (seq, head_dim)  — chosen head
    k: np.ndarray                 # (seq, head_dim)
    v: np.ndarray                 # (seq, head_dim)
    scores: np.ndarray            # (seq, seq) pre-softmax post-mask
    weights: np.ndarray           # (seq, seq) post-softmax (== HF attentions[L,B,H])
    context: np.ndarray           # (seq, head_dim) weights @ V
    attn_output: np.ndarray       # (seq, hidden) after o_proj (full multi-head merged)
    post_attn_residual: np.ndarray  # (seq, hidden) = pre_ln1 + attn_output
    post_ln2: np.ndarray          # (seq, hidden)
    ffn_pre_act: np.ndarray       # (seq, ffn_dim) — pre-activation up_proj output
    ffn_post_act: np.ndarray      # (seq, ffn_dim) — post-activation
    ffn_gate: np.ndarray | None   # (seq, ffn_dim) for SwiGLU; None for GPT-2-style
    ffn_output: np.ndarray        # (seq, hidden) — post down_proj
    block_output: np.ndarray      # (seq, hidden) = post_attn_residual + ffn_output


@dataclass(frozen=True, slots=True)
class TraceData:
    """One cell of a (model × prompt) matrix. Fully self-describing — a cached
    trace can be re-rendered without re-reading trace.yaml.

    Arrays go to .npz. Everything else (lists, dicts, scalars) goes to .json.
    """
    # Provenance
    model_id: str
    prompt: str
    system_prompt: str | None
    templated_prompt: str
    gen_params: dict[str, Any]

    # Architecture metadata
    model_meta: dict[str, Any]

    # Tokenization
    tokens: list[str]
    token_ids: list[int]

    # Embeddings (per-token, truncated to hidden_dims_keep)
    embeddings_token: np.ndarray                 # (n_tokens, dims_keep) float32
    embeddings_combined: np.ndarray              # (n_tokens, dims_keep) float32
    embeddings_pos: np.ndarray | None            # (n_tokens, dims_keep) or None (RoPE)

    # Attention: keyed "L{layer}H{head}" per slicing policy
    attentions: dict[str, np.ndarray]            # each (seq, seq) float32

    # Hidden states at last token position across all layer outputs
    hidden_last: np.ndarray                      # (n_layers+1, dims_keep) float32
    hidden_last_norms: np.ndarray                # (n_layers+1,) float32

    # Full-width final hidden state at last token (post-final-LN, what the
    # LM head reads). Kept full-width because the LM head step shows the
    # actual vector that gets projected to vocab logits.
    final_hidden_full: np.ndarray                # (hidden_size,) float32

    # Per-token residual-stream norms across all layer outputs.
    # Shape: (n_layers+1, n_tokens) float32. Powers the token×layer heatmap.
    hidden_norms: np.ndarray

    # Top-k logits + probs
    logits_top_ids: np.ndarray                   # (top_k,) int64
    logits_top_values: np.ndarray                # (top_k,) float32
    logits_top_tokens: list[str]
    probs_top_ids: np.ndarray                    # (top_k,) int64
    probs_top_values: np.ndarray                 # (top_k,) float32
    probs_top_tokens: list[str]

    # LM-head weight rows for the top-K logit winners. Shape (top_k, hidden_size).
    # Each row is the column of W projected against the final hidden state to
    # produce that token's logit:  logit[i] = <final_hidden_full, lm_head_top_rows[i]>.
    # For tied-weights models this row equals the input embedding row from
    # Step 3 (transposed); for untied models it's a separately-learned matrix.
    lm_head_top_rows: np.ndarray                 # (top_k, hidden_size) float32

    # Temperature scan
    temp_scan: list[dict[str, Any]]

    # Autoregressive generation trace (one entry per generated token).
    # Each entry is a dict with keys: step, token, id, prob, ctx_len, is_eos,
    # top_alts, hidden_norms (list), top_ids (list), top_probs (list),
    # top_tokens (list). The arrays are stored separately (below) in tensor
    # form for efficient cache storage.
    generation: list[dict[str, Any]]

    # Pre-detokenized full generated string. Done in one call via
    # `tokenizer.decode(gen_ids, skip_special_tokens=True)` so it's correct
    # for sentencepiece tokenizers (where per-token decode drops the ▁ space).
    generation_text: str

    # Per-step arrays (len == len(generation)) — present for v2+ traces.
    # These let renderers plot the residual-stream norm evolution and the
    # probability mass distribution at every loop iteration.
    per_step_hidden_norms: np.ndarray   # (n_steps, n_layers+1) float32
    per_step_top_ids: np.ndarray        # (n_steps, top_k) int64
    per_step_top_probs: np.ndarray      # (n_steps, top_k) float32

    # EOS-step capture (optional). Populated only when the loop actually
    # terminated by emitting EOS. Lets the renderer show the LM-head matmul
    # at the moment the model decided to stop.
    eos_step_hidden_full: np.ndarray | None    # (hidden_size,) float32 | None
    eos_step_top_rows: np.ndarray | None       # (top_k, hidden_size) float32 | None
    eos_step_top_logits: np.ndarray | None     # (top_k,) float32 | None
    eos_step_top_tokens: list[str] | None      # decoded strings for top-K

    # Optional: full inner-state capture for one chosen (layer, head) of one
    # transformer block. Powers the "inside one block" deep-dive renderer.
    # None when block_deepdive collection is disabled in config.
    block_deepdive: BlockDeepDive | None

    # Performance
    timings: dict[str, Any]

    # ── Cache serialization ────────────────────────────────────────────────

    def to_cache_payload(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays: dict[str, np.ndarray] = {
            "embeddings_token": self.embeddings_token,
            "embeddings_combined": self.embeddings_combined,
            "hidden_last": self.hidden_last,
            "hidden_last_norms": self.hidden_last_norms,
            "hidden_norms": self.hidden_norms,
            "final_hidden_full": self.final_hidden_full,
            "logits_top_ids": self.logits_top_ids,
            "logits_top_values": self.logits_top_values,
            "probs_top_ids": self.probs_top_ids,
            "probs_top_values": self.probs_top_values,
            "lm_head_top_rows": self.lm_head_top_rows,
            "per_step_hidden_norms": self.per_step_hidden_norms,
            "per_step_top_ids": self.per_step_top_ids,
            "per_step_top_probs": self.per_step_top_probs,
        }
        if self.embeddings_pos is not None:
            arrays["embeddings_pos"] = self.embeddings_pos
        if self.eos_step_hidden_full is not None:
            arrays["eos_step_hidden_full"] = self.eos_step_hidden_full
            arrays["eos_step_top_rows"] = self.eos_step_top_rows
            arrays["eos_step_top_logits"] = self.eos_step_top_logits
        if self.block_deepdive is not None:
            bd = self.block_deepdive
            for k in (
                "pre_ln1", "post_ln1", "q", "k", "v",
                "scores", "weights", "context",
                "attn_output", "post_attn_residual", "post_ln2",
                "ffn_pre_act", "ffn_post_act", "ffn_output", "block_output",
            ):
                arrays[f"bd__{k}"] = getattr(bd, k)
            if bd.ffn_gate is not None:
                arrays["bd__ffn_gate"] = bd.ffn_gate
        for key, arr in self.attentions.items():
            arrays[f"attn__{key}"] = arr

        meta = {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "templated_prompt": self.templated_prompt,
            "gen_params": self.gen_params,
            "model_meta": self.model_meta,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "logits_top_tokens": self.logits_top_tokens,
            "probs_top_tokens": self.probs_top_tokens,
            "temp_scan": self.temp_scan,
            "generation": self.generation,
            "generation_text": self.generation_text,
            "timings": self.timings,
            "has_embeddings_pos": self.embeddings_pos is not None,
            "has_eos_step": self.eos_step_hidden_full is not None,
            "eos_step_top_tokens": self.eos_step_top_tokens,
            "has_block_deepdive": self.block_deepdive is not None,
        }
        if self.block_deepdive is not None:
            bd = self.block_deepdive
            meta["block_deepdive_meta"] = {
                "layer_index": bd.layer_index,
                "head_index": bd.head_index,
                "n_heads": bd.n_heads,
                "head_dim": bd.head_dim,
                "activation": bd.activation,
                "has_swiglu_gate": bd.has_swiglu_gate,
            }
        return arrays, meta

    @classmethod
    def from_cache_payload(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> TraceData:
        attentions = {
            k.removeprefix("attn__"): v for k, v in arrays.items() if k.startswith("attn__")
        }
        emb_pos = arrays["embeddings_pos"] if meta.get("has_embeddings_pos") else None
        return cls(
            model_id=meta["model_id"],
            prompt=meta["prompt"],
            system_prompt=meta.get("system_prompt"),
            templated_prompt=meta["templated_prompt"],
            gen_params=meta["gen_params"],
            model_meta=meta["model_meta"],
            tokens=meta["tokens"],
            token_ids=meta["token_ids"],
            embeddings_token=arrays["embeddings_token"],
            embeddings_pos=emb_pos,
            embeddings_combined=arrays["embeddings_combined"],
            attentions=attentions,
            hidden_last=arrays["hidden_last"],
            hidden_last_norms=arrays["hidden_last_norms"],
            hidden_norms=arrays["hidden_norms"],
            final_hidden_full=arrays["final_hidden_full"],
            logits_top_ids=arrays["logits_top_ids"],
            logits_top_values=arrays["logits_top_values"],
            logits_top_tokens=meta["logits_top_tokens"],
            probs_top_ids=arrays["probs_top_ids"],
            probs_top_values=arrays["probs_top_values"],
            probs_top_tokens=meta["probs_top_tokens"],
            lm_head_top_rows=arrays["lm_head_top_rows"],
            temp_scan=meta["temp_scan"],
            generation=meta["generation"],
            generation_text=meta.get("generation_text", ""),
            per_step_hidden_norms=arrays["per_step_hidden_norms"],
            per_step_top_ids=arrays["per_step_top_ids"],
            per_step_top_probs=arrays["per_step_top_probs"],
            eos_step_hidden_full=arrays.get("eos_step_hidden_full") if meta.get("has_eos_step") else None,
            eos_step_top_rows=arrays.get("eos_step_top_rows") if meta.get("has_eos_step") else None,
            eos_step_top_logits=arrays.get("eos_step_top_logits") if meta.get("has_eos_step") else None,
            eos_step_top_tokens=meta.get("eos_step_top_tokens"),
            block_deepdive=_load_block_deepdive(arrays, meta),
            timings=meta["timings"],
        )


def _load_block_deepdive(
    arrays: dict[str, np.ndarray], meta: dict[str, Any]
) -> BlockDeepDive | None:
    if not meta.get("has_block_deepdive"):
        return None
    bm = meta["block_deepdive_meta"]
    return BlockDeepDive(
        layer_index=int(bm["layer_index"]),
        head_index=int(bm["head_index"]),
        n_heads=int(bm["n_heads"]),
        head_dim=int(bm["head_dim"]),
        activation=str(bm["activation"]),
        has_swiglu_gate=bool(bm["has_swiglu_gate"]),
        pre_ln1=arrays["bd__pre_ln1"],
        post_ln1=arrays["bd__post_ln1"],
        q=arrays["bd__q"],
        k=arrays["bd__k"],
        v=arrays["bd__v"],
        scores=arrays["bd__scores"],
        weights=arrays["bd__weights"],
        context=arrays["bd__context"],
        attn_output=arrays["bd__attn_output"],
        post_attn_residual=arrays["bd__post_attn_residual"],
        post_ln2=arrays["bd__post_ln2"],
        ffn_pre_act=arrays["bd__ffn_pre_act"],
        ffn_post_act=arrays["bd__ffn_post_act"],
        ffn_gate=arrays.get("bd__ffn_gate"),
        ffn_output=arrays["bd__ffn_output"],
        block_output=arrays["bd__block_output"],
    )
