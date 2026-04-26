"""PNG renderer — matplotlib 4-panel figure.

Ports the 4 plot functions from llm_flow_viz.py. Consumes TraceData only:
    1. Attention heatmap (uses L0H0 or first available slice)
    2. Top-K next-token probabilities (log scale)
    3. Hidden-state trajectory across layers (PCA → 2D via numpy SVD)
    4. Temperature effect on top-3 tokens

Must not import torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project rows to 2D via SVD. Returns (points, variance_ratios)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    points = Xc @ Vt[:2].T
    total_var = (S ** 2).sum()
    ratios = (S[:2] ** 2) / total_var if total_var > 0 else np.zeros(2)
    return points, ratios


def _plot_attention(ax, trace) -> None:
    import matplotlib.pyplot as plt

    if not trace.attentions:
        ax.set_title("Attention (no cached slice)", fontsize=11)
        ax.axis("off")
        return

    key = "L0H0" if "L0H0" in trace.attentions else next(iter(trace.attentions))
    attn = trace.attentions[key]
    mask = np.triu(np.ones_like(attn), k=1) > 0
    masked = np.where(mask, np.nan, attn)

    im = ax.imshow(masked, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    labels = [repr(t) for t in trace.tokens]
    ax.set_xticks(range(len(trace.tokens)))
    ax.set_yticks(range(len(trace.tokens)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Attends to", fontsize=10)
    ax.set_ylabel("From", fontsize=10)
    ax.set_title(f"1. Attention weights  ({key})", fontsize=11, fontweight="bold")

    for i in range(len(trace.tokens)):
        for j in range(i + 1):
            val = float(attn[i, j])
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_top_probs(ax, trace) -> None:
    k = len(trace.probs_top_values)
    labels = [repr(tok) for tok in trace.probs_top_tokens]
    values = trace.probs_top_values.astype(float)

    y = np.arange(k)[::-1]
    bars = ax.barh(y, values, color="steelblue", edgecolor="navy")
    if len(bars) > 0:
        bars[0].set_color("crimson")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Probability (log scale)", fontsize=10)
    ax.set_title(f"2. Top-{k} next-token probabilities", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, p in zip(bars, values, strict=False):
        ax.text(max(p * 1.15, 1e-6), bar.get_y() + bar.get_height() / 2,
                f"{p:.4f}", va="center", fontsize=8)


def _plot_hidden_trajectory(ax, trace) -> None:
    import matplotlib.pyplot as plt

    hidden = trace.hidden_last          # (n_layers+1, dims_keep)
    norms = trace.hidden_last_norms     # (n_layers+1,)
    # Rescale each row by its full norm so the trajectory reflects magnitude growth,
    # not just truncated directional drift.
    rescaled = hidden * (norms[:, None] / np.maximum(
        np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8
    ))

    if rescaled.shape[0] < 2:
        ax.set_title("Hidden-state trajectory (too few layers)", fontsize=11)
        ax.axis("off")
        return

    points, ratios = _pca_2d(rescaled)

    ax.plot(points[:, 0], points[:, 1], "-", color="gray", alpha=0.4,
            linewidth=1, zorder=1)
    colors = plt.cm.plasma(np.linspace(0, 1, len(points)))
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=120,
               edgecolors="black", linewidths=0.5, zorder=3)
    for i, pt in enumerate(points):
        label = "input" if i == 0 else f"L{i}"
        ax.annotate(label, pt, textcoords="offset points", xytext=(8, 5), fontsize=9)

    total = ratios.sum() * 100
    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)", fontsize=10)
    ax.set_title(f"3. Hidden-state trajectory  (last token, {total:.0f}% var)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


def _plot_temperature(ax, trace) -> None:
    if not trace.temp_scan:
        ax.set_title("Temperature scan not available", fontsize=11)
        ax.axis("off")
        return

    # Use the middle temperature's top-3 tokens as the x-axis.
    mid = trace.temp_scan[len(trace.temp_scan) // 2]
    top3_ids = [int(entry["id"]) for entry in mid["top"][:3]]
    labels = [repr(entry["token"]) for entry in mid["top"][:3]]

    x = np.arange(len(top3_ids))
    width = 0.8 / max(len(trace.temp_scan), 1)

    for i, block in enumerate(trace.temp_scan):
        temp = block["temperature"]
        id_to_prob = {int(entry["id"]): float(entry["prob"]) for entry in block["top"]}
        values = [id_to_prob.get(tid, 0.0) for tid in top3_ids]
        offset = (i - (len(trace.temp_scan) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=f"T={temp}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title("4. Temperature effect on top-3", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


def render(trace, cfg: dict[str, Any] | None = None, out_path: Path | str | None = None) -> Path:
    """Write a 4-panel PNG summarising the trace. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_dir = Path((cfg or {}).get("out_dir", "./out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{trace.model_id.replace('/', '_')}__{_slug(trace.prompt)}.png"
    out_path = Path(out_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f'{trace.model_id}  —  prompt: {trace.prompt!r}',
        fontsize=13, fontweight="bold",
    )

    _plot_attention(axes[0, 0], trace)
    _plot_top_probs(axes[0, 1], trace)
    _plot_hidden_trajectory(axes[1, 0], trace)
    _plot_temperature(axes[1, 1], trace)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _slug(s: str, max_len: int = 32) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in s.strip())
    return cleaned[:max_len].strip("_") or "prompt"
