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
            timings=meta["timings"],
        )
