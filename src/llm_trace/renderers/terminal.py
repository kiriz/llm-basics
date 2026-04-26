"""Terminal renderer — rich-based step-by-step trace.

Consumes a TraceData (no torch, no transformers). Combines the 10-step
structure from llm_flow.py with the architecture card from llama_trace.py.

Entry: render(trace, cfg={}).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Small formatting helpers ───────────────────────────────────────────────

def _header(text: str) -> None:
    console.print(f"\n[bold cyan]{'─' * 72}[/bold cyan]")
    console.print(f"[bold white]{text}[/bold white]")
    console.print(f"[bold cyan]{'─' * 72}[/bold cyan]")


def _step(n: int, title: str) -> None:
    console.print(f"\n[bold yellow]Step {n}:[/bold yellow] [bold]{title}[/bold]")


def _dim(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


def _fmt_vec(v: np.ndarray, dims: int = 6) -> str:
    head = ", ".join(f"{x:+.3f}" for x in v[:dims])
    tail = " ..." if v.size > dims else ""
    return f"[{head}{tail}]"


# ── Per-step renderers ────────────────────────────────────────────────────

def _show_header_and_arch(trace) -> None:
    m = trace.model_meta
    _header(f"Llm trace — {trace.model_id}")
    if getattr(trace, "system_prompt", None):
        console.print(f"  System       : [magenta]{trace.system_prompt!r}[/magenta]")
    console.print(f"  Prompt       : [yellow]{trace.prompt!r}[/yellow]")
    if trace.templated_prompt != trace.prompt:
        console.print(f"  Templated    : [yellow]{trace.templated_prompt!r}[/yellow]")
    console.print(f"  Gen params   : {trace.gen_params}")

    arch = Table(show_header=False, box=None, padding=(0, 2))
    arch.add_column("Feature", style="cyan")
    arch.add_column("Value", style="green")
    arch.add_column("Note", style="dim")

    arch.add_row("Model type", str(m.get("model_type")), "")
    arch.add_row("Layers", str(m.get("n_layer")), "transformer blocks stacked")
    arch.add_row("Hidden dim", str(m.get("hidden_size")), "residual stream width")
    arch.add_row("Attention heads", str(m.get("n_head")), "")
    if m.get("n_kv_heads") and m.get("n_head") and m["n_kv_heads"] != m["n_head"]:
        arch.add_row("KV heads", str(m["n_kv_heads"]),
                     f"GQA: {m['n_head'] // m['n_kv_heads']}× sharing")
    arch.add_row("FFN intermediate", str(m.get("ffn_intermediate")), "")
    arch.add_row("Positional encoding", str(m.get("positional_encoding")),
                 f"rope_theta={m['rope_theta']}" if m.get("rope_theta") else "")
    arch.add_row("Normalization", str(m.get("normalization")), "")
    arch.add_row("Vocabulary", f"{m.get('vocab_size'):,}" if m.get("vocab_size") else "?", "")
    arch.add_row("Context window", str(m.get("ctx_window")), "architectural ceiling")
    arch.add_row("EOS token(s)", str(m.get("eos_ids", [])),
                 f"decoded: {m.get('eos_tokens', [])}")
    console.print(arch)


def _show_step_1_tokens(trace) -> None:
    _step(1, "Tokenization  (text → token IDs)")
    _dim("Tokenizer splits text into subword pieces; each maps to an integer id.")

    tbl = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    tbl.add_column("Pos", style="dim")
    tbl.add_column("Token", style="yellow")
    tbl.add_column("ID", style="green")
    for i, (tok, tid) in enumerate(zip(trace.tokens, trace.token_ids, strict=False)):
        tbl.add_row(str(i), repr(tok), str(tid))
    console.print(tbl)
    console.print(f"\n  [bold]{len(trace.tokens)} tokens[/bold] from "
                  f"{len(trace.prompt)} chars of raw prompt")


def _show_step_2_tensor(trace) -> None:
    _step(2, "Input tensor  (IDs → PyTorch tensor)")
    _dim("This is the literal integer tensor the model receives.")
    console.print(f"  shape  : [green][1, {len(trace.token_ids)}][/green]   "
                  f"= (batch_size=1, seq_len={len(trace.token_ids)})")
    console.print(f"  values : [yellow][{trace.token_ids}][/yellow]")


def _show_step_3_embeddings(trace) -> None:
    _step(3, "Embedding lookup  (IDs → dense vectors)")
    _dim(f"Each id indexes into the embedding matrix (vocab × hidden). "
         f"Showing first {trace.embeddings_token.shape[1]} dims.")
    for i, (tok, tid) in enumerate(zip(trace.tokens, trace.token_ids, strict=False)):
        vec = trace.embeddings_token[i]
        console.print(f"  token[{i}] {repr(tok):14s} id={tid:<6}  "
                      f"shape=[green]({vec.shape[0]}+,)[/green]  {_fmt_vec(vec)}")


def _show_step_4_positional(trace) -> None:
    _step(4, "Positional encoding")
    if trace.embeddings_pos is None:
        _dim(f"No learned positional embeddings on this model "
             f"(encoding type: {trace.model_meta.get('positional_encoding')}).")
        _dim("RoPE applies position rotations INSIDE attention, not at embedding time.")
        _dim("So the 'input to layer 0' = token embedding unchanged.")
        return

    _dim("A positional vector is added elementwise to each token embedding.")
    _dim("final_input_to_transformer = token_embed + position_embed")
    for i in range(min(3, len(trace.token_ids))):
        pv = trace.embeddings_pos[i]
        console.print(f"  position[{i}]   {_fmt_vec(pv)}")


def _show_step_5_forward(trace) -> None:
    _step(5, f"Forward pass  ({trace.model_meta.get('n_layer')} transformer layers)")
    _dim("Residual stream grows through layers. Final norm rescales before LM head.")

    n = len(trace.hidden_last_norms)
    probe = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    for i in probe:
        label = "input   " if i == 0 else f"after L{i:2d}"
        norm = trace.hidden_last_norms[i]
        head = _fmt_vec(trace.hidden_last[i], dims=5)
        console.print(f"  {label}   ||h||={norm:8.2f}   first 5 dims: {head}")

    fwd_ms = trace.timings.get("first_forward_ms")
    if fwd_ms:
        console.print(f"  [dim]Wall time for the trace forward pass: {fwd_ms:.1f} ms[/dim]")


def _show_step_6_attention(trace) -> None:
    _step(6, "Attention weights  (Q·Kᵀ / √d → softmax → weighted sum of V)")
    if not trace.attentions:
        _dim("No attention matrix cached for this trace (check slicing policy).")
        return

    # Prefer L0H0 if present, else first available
    key = "L0H0" if "L0H0" in trace.attentions else next(iter(trace.attentions))
    attn = trace.attentions[key]
    _dim(f"Showing {key}. Each row = one token; column = its attention weight to another.")

    seq = attn.shape[0]
    tbl = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    tbl.add_column("From ↓ / To →", style="cyan", min_width=14)
    for t in trace.tokens:
        tbl.add_column(repr(t), justify="right", min_width=7)

    for i, from_tok in enumerate(trace.tokens):
        row: list[str] = []
        for j in range(seq):
            if j > i:
                row.append("[dim]—[/dim]")
                continue
            w = float(attn[i, j])
            if w > 0.3:
                row.append(f"[bold green]{w:.2f}[/bold green]")
            elif w > 0.1:
                row.append(f"[yellow]{w:.2f}[/yellow]")
            else:
                row.append(f"[dim]{w:.2f}[/dim]")
        tbl.add_row(repr(from_tok), *row)
    console.print(tbl)


def _show_step_7_logits(trace) -> None:
    _step(7, "Logits  (raw score per vocabulary token)")
    vocab = trace.model_meta.get("vocab_size")
    _dim(f"After all layers, last hidden state is projected to vocab "
         f"({vocab:,} scores). Only differences matter.")

    tbl = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    tbl.add_column("Rank", style="dim")
    tbl.add_column("Token", style="yellow")
    tbl.add_column("ID", style="dim")
    tbl.add_column("Logit", style="green")
    for r in range(len(trace.logits_top_ids)):
        tid = int(trace.logits_top_ids[r])
        tbl.add_row(str(r + 1), repr(trace.logits_top_tokens[r]), str(tid),
                    f"{float(trace.logits_top_values[r]):.3f}")
    console.print(tbl)


def _show_step_8_softmax(trace) -> None:
    _step(8, "Softmax  (logits → probabilities summing to 1)")

    tbl = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    tbl.add_column("Rank", style="dim")
    tbl.add_column("Token", style="yellow")
    tbl.add_column("Prob bar", style="green")
    tbl.add_column("Logit", style="dim")
    for r in range(len(trace.probs_top_ids)):
        tid = int(trace.probs_top_ids[r])
        prob = float(trace.probs_top_values[r])
        filled = min(20, int(prob * 40))
        bar = "█" * filled + "░" * (20 - filled)
        logit = _logit_for(trace, tid)
        tbl.add_row(str(r + 1), repr(trace.probs_top_tokens[r]),
                    f"{prob:.4f}  {bar}",
                    f"{logit:.2f}" if logit is not None else "—")
    console.print(tbl)


def _show_step_9_temperature(trace) -> None:
    _step(9, "Sampling / Temperature")
    _dim("Temperature scales logits before softmax: low T → peaky, high T → flat.")
    if not trace.temp_scan:
        _dim("No temperature scan in this trace.")
        return
    for block in trace.temp_scan:
        t = block["temperature"]
        top3 = " | ".join(
            f"{entry['token']!r} {entry['prob']:.3f}" for entry in block["top"][:3]
        )
        console.print(f"  T={t:>4.1f}  top-3: [dim]{top3}[/dim]")


def _show_step_10_generation(trace) -> None:
    gp = trace.gen_params
    _step(10, f"Autoregressive generation  (max={gp.get('max_new_tokens')}, "
              f"seed={gp.get('seed')}, stop_on_eos={gp.get('stop_on_eos')})")
    _dim("Each step: forward pass → argmax → append. Stop on EOS (if configured).")
    console.print()

    for step in trace.generation:
        flag = "[bold red]← EOS, break[/bold red]" if step.get("is_eos") else ""
        console.print(
            f"  t+{step['step']:2d}  id={step['id']:6d}  "
            f"p={step['prob']:.3f}  "
            f"ctx_len={step['ctx_len']:3d}  "
            f"{repr(step['token']):20s}  {flag}"
        )

    # Use the pre-detokenized string (correct for sentencepiece / llama-family
    # tokenizers where per-token decode drops the ▁ space marker).
    gen_text = getattr(trace, "generation_text", None) or "".join(
        s["token"] for s in trace.generation if not s.get("is_eos")
    )
    full = trace.prompt + gen_text
    console.print(Panel(
        full.strip(),
        title="[bold green]Complete output[/bold green]",
        expand=False,
    ))


# ── Utility ────────────────────────────────────────────────────────────────

def _logit_for(trace, tid: int) -> float | None:
    for i in range(len(trace.logits_top_ids)):
        if int(trace.logits_top_ids[i]) == tid:
            return float(trace.logits_top_values[i])
    return None


# ── Entry point ────────────────────────────────────────────────────────────

def render(trace, cfg: dict[str, Any] | None = None) -> None:
    """Render a full 10-step trace to stdout."""
    _show_header_and_arch(trace)
    _show_step_1_tokens(trace)
    _show_step_2_tensor(trace)
    _show_step_3_embeddings(trace)
    _show_step_4_positional(trace)
    _show_step_5_forward(trace)
    _show_step_6_attention(trace)
    _show_step_7_logits(trace)
    _show_step_8_softmax(trace)
    _show_step_9_temperature(trace)
    _show_step_10_generation(trace)
