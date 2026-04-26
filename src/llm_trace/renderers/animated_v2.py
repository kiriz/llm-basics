"""Animated renderer V2 — iteration sandbox.

Copy of animated.py for in-progress modifications. The original `animated.py`
stays untouched as a known-good baseline. Switch on per-run via:
    --override 'renderers=["animated_v2"]'

Output filename suffix: __anim_v2.html (so v1 and v2 don't overwrite each other).

Original docstring follows.
================================================================================

Animated renderer — scene-based timeline that plays through the trace.

Self-contained HTML with three acts:
  I   Prologue  — raw text fades in, splits into token chips, each resolves
                  to an integer id.
  II  Deep dive — one full forward pass: embedding lookup → activation wave
                  through the N layers (residual-stream norm meter) → single-
                  head attention heatmap glow → logits bar chart → softmax →
                  argmax token chosen.
  III Loop      — compressed per-step playback of every remaining generation
                  step until EOS or max_new_tokens. Each step: new token chip
                  lands in the cumulative output strip, norm sparkline grows,
                  wallclock ticker updates.

Interactive controls at the bottom: play/pause, scrub bar, speed (0.5× / 1× /
2× / 10×). All data comes from TraceData — no model re-run, no server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Data shaping ───────────────────────────────────────────────────────────

def _short_model_slug(model_id: str) -> str:
    """Pretty short slug for an HF model id, used to compose default filenames.

    Examples:
        'distilgpt2'                                  -> 'distilgpt2'
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0'         -> 'tinyllama'
        'meta-llama/Llama-3.2-1B-Instruct'           -> 'llama'
        'gpt2-medium'                                 -> 'gpt2'

    Heuristic: take the last `/` segment, then everything before the first
    '-' or '.', lowercased. Falls back to slugifying the whole id.
    """
    last = model_id.rsplit("/", 1)[-1]
    head = last.split("-", 1)[0].split(".", 1)[0]
    slug = "".join(c.lower() if c.isalnum() else "_" for c in head).strip("_")
    return slug or "model"


def _build_anim_payload(trace) -> dict[str, Any]:
    m = trace.model_meta or {}

    # Collect every head at layer 0 (cycling target). For heads not in the
    # cache (slicing policy was narrower), keep null so the JS knows.
    n_heads = int(m.get("n_head") or 12)
    attn_heads_l0 = []
    for h in range(n_heads):
        key = f"L0H{h}"
        if key in trace.attentions:
            attn_heads_l0.append(trace.attentions[key].tolist())
        else:
            attn_heads_l0.append(None)
    # Back-compat single matrix (used as fallback if no heads cached at all).
    attn_key = "L0H0" if "L0H0" in trace.attentions else (
        next(iter(trace.attentions)) if trace.attentions else None
    )
    attn_matrix = trace.attentions[attn_key].tolist() if attn_key else None

    # Per-step data for Act III.
    len(trace.generation)
    psn = trace.per_step_hidden_norms
    per_token_ms = trace.timings.get("per_token_ms", [])

    loop_steps = []
    for i, s in enumerate(trace.generation):
        loop_steps.append({
            "step": int(s["step"]),
            "token": s["token"],
            "id": int(s["id"]),
            "prob": float(s["prob"]),
            "ctx_len": int(s["ctx_len"]),
            "is_eos": bool(s.get("is_eos", False)),
            "ms": float(per_token_ms[i]) if i < len(per_token_ms) else 0.0,
            "hidden_norms": psn[i].tolist() if i < psn.shape[0] else [],
            "top_alts": [{"token": a["token"], "prob": float(a["prob"])}
                         for a in s.get("top_alts", [])[:5]],
        })

    # Featured first-step forward-pass artifacts (for Act II).
    hidden_norms_first = trace.hidden_last_norms.tolist()
    probs_top_first = [
        {"token": tok, "id": int(tid), "prob": float(v)}
        for tok, tid, v in zip(
            trace.probs_top_tokens, trace.probs_top_ids, trace.probs_top_values, strict=False
        )
    ][:10]
    logits_top_first = [
        {"token": tok, "id": int(tid), "logit": float(v)}
        for tok, tid, v in zip(
            trace.logits_top_tokens, trace.logits_top_ids, trace.logits_top_values, strict=False
        )
    ][:10]
    # Embedding (token part) of each input token, truncated.
    emb_top_dims = trace.embeddings_token[:, :16].tolist()

    gen_text = getattr(trace, "generation_text", None) or "".join(
        s["token"] for s in trace.generation if not s.get("is_eos")
    )

    return {
        "model_id": trace.model_id,
        "prompt": trace.prompt,
        "system_prompt": getattr(trace, "system_prompt", None),
        "tokens": trace.tokens,
        "token_ids": list(trace.token_ids),
        "model_meta": m,
        "embeddings": emb_top_dims,
        "hidden_norms_first_step": hidden_norms_first,
        "hidden_last": trace.hidden_last.tolist(),   # last-token vector across layers, for LM head viz
        # 2D: (n_layers+1, n_tokens) — token × layer heatmap data
        "hidden_norms_grid": trace.hidden_norms.tolist(),
        "attention_featured_head": attn_key,
        "attention_matrix": attn_matrix,
        "attention_heads_l0": attn_heads_l0,
        "n_heads": n_heads,
        "probs_top_first_step": probs_top_first,
        "logits_top_first_step": logits_top_first,
        "embedding_link": f"embeddings_{_short_model_slug(trace.model_id)}.html",
        "loop_steps": loop_steps,
        "generation_text": gen_text,
        "gen_params": trace.gen_params,
        "timings": trace.timings,
    }


# ── HTML template ──────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm-trace animated — __TITLE__</title>
<style>
:root{--bg:#080b12;--surface:#0e1220;--surface2:#151b2e;--border:#1f2a45;--text:#cdd6f0;--muted:#5a6888;--teal:#00d4aa;--amber:#ffb800;--coral:#ff6b6b;--blue:#4da6ff;--purple:#c084fc;--mono:'SF Mono',Menlo,monospace;--sans:system-ui,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:var(--sans);background:var(--bg);color:var(--text);overflow:hidden}

.stage{
  position:relative;width:100vw;height:100vh;
  display:flex;flex-direction:column;
}

.top-bar{
  display:flex;gap:24px;align-items:baseline;
  padding:14px 28px;border-bottom:1px solid var(--border);
  font-family:var(--mono);font-size:12px;color:var(--muted);
}
.top-bar .model{color:var(--teal);font-weight:700;font-size:13px}
.top-bar .sys  {color:var(--purple)}
.top-bar .prompt{color:var(--amber)}

.scene{
  position:absolute;inset:0 0 240px 0;
  padding:40px;overflow:hidden;
  opacity:0;pointer-events:none;transition:opacity .3s;
}
.scene.active{opacity:1;pointer-events:auto}
.scene h2{font-family:var(--mono);font-size:12px;color:var(--teal);
  letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px}
.scene .hint{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:20px;max-width:680px;line-height:1.55}

/* ── Act I (tokenization) ── */
.act1-text{
  font-family:var(--mono);font-size:28px;color:var(--amber);
  text-align:center;margin-top:40px;letter-spacing:.02em;
  transition:all .6s cubic-bezier(.4,0,.2,1);
}
.act1-chips{
  display:flex;justify-content:center;flex-wrap:wrap;gap:14px;
  margin-top:60px;
}
.act1-chip{
  display:flex;flex-direction:column;align-items:center;gap:4px;
  opacity:0;transform:translateY(30px);
}
.act1-chip.show{opacity:1;transform:translateY(0);
  transition:opacity .35s, transform .35s cubic-bezier(.4,0,.2,1)}
.act1-chip .tok{
  padding:10px 18px;border-radius:8px;
  font-family:var(--mono);font-size:16px;font-weight:600;
  background:var(--surface2);border:2px solid var(--border);
}
.act1-chip.show .tok{border-color:var(--teal);color:var(--teal)}
.act1-chip .id{font-family:var(--mono);font-size:11px;color:var(--muted);opacity:0;transition:opacity .3s}
.act1-chip.id-show .id{opacity:1}

/* ── Act II (deep dive) ── */
.pipeline{
  display:flex;align-items:center;gap:10px;justify-content:center;
  margin-top:10px;
}
.pipe-box{
  padding:10px 14px;border-radius:8px;min-width:92px;
  background:var(--surface2);border:2px solid var(--border);
  font-family:var(--mono);font-size:11px;color:var(--muted);
  text-align:center;transition:all .35s;
}
.pipe-box.active{border-color:var(--teal);color:var(--text);
  box-shadow:0 0 16px rgba(0,212,170,0.35)}
.pipe-box.done{border-color:var(--blue);color:var(--blue)}
.pipe-arrow{color:var(--muted);font-family:var(--mono);font-size:14px}

.act2-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:28px;
}
.act2-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:20px;min-height:220px;
}
.act2-card h3{font-family:var(--mono);font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px}

/* Heatmap: rows = input tokens, columns = layer outputs.
   Each cell shows ‖hidden_state‖ for that (token, layer). */
.heatmap-wrap{overflow-x:auto;padding:4px 0}
.heatmap-svg{display:block;font-family:var(--mono);max-width:100%}
.heatmap-cell{transition:fill .35s ease-out}
.heatmap-row-label{font-family:var(--mono);font-size:11px;fill:var(--text)}
.heatmap-row-label.last{fill:var(--teal);font-weight:700}
.heatmap-col-label{font-family:var(--mono);font-size:10px;fill:var(--muted)}
.heatmap-cell-val{font-family:var(--mono);font-size:9px;text-anchor:middle;dominant-baseline:central;pointer-events:none}
.heatmap-last-row-frame{fill:none;stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3}
.heatmap-hint{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:6px;line-height:1.5}
.heatmap-hint b{color:var(--teal)}

.attn-svg{width:100%;height:220px;display:block}
.attn-hint{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:6px;line-height:1.5}
.attn-hint b{color:var(--teal)}

/* ── LM head card — shows L_N→logits transformation ──────────────────── */
.lm-head-step{display:flex;gap:12px;padding:10px 12px;background:var(--surface2);
  border-radius:6px;border-left:3px solid var(--blue);margin:6px 0;
  opacity:0;transform:translateY(6px);
  transition:opacity .45s,transform .45s,border-color .3s}
.lm-head-step.show{opacity:1;transform:translateY(0)}
.lm-head-step:last-of-type{border-left-color:var(--teal)}
.lm-step-num{width:22px;height:22px;border-radius:50%;background:var(--blue);
  color:#000;font-family:var(--mono);font-size:11px;font-weight:700;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px}
.lm-head-step:last-of-type .lm-step-num{background:var(--teal)}
.lm-step-body{flex:1;min-width:0}
.lm-step-title{font-family:var(--mono);font-size:12.5px;color:var(--text);
  font-weight:600;margin-bottom:3px}
.lm-step-desc{font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5}
.lm-step-desc b{color:var(--text)}
.lm-vec-cells{display:flex;gap:1px;margin-top:8px;flex-wrap:wrap}
.lm-vec-cells .vc{width:10px;height:18px;border-radius:1px}
.lm-vec-cells .more{font-family:var(--mono);font-size:10px;
  color:var(--muted);margin-left:6px;align-self:center}
.lm-head-arrow{text-align:center;padding:2px 0;font-family:var(--mono);
  font-size:10.5px;color:var(--teal);opacity:0;transition:opacity .4s}
.lm-head-arrow.show{opacity:1}
.lm-head-note{margin-top:10px;font-family:var(--mono);font-size:10.5px;
  color:var(--muted);line-height:1.55;padding:8px 12px;background:var(--surface2);
  border-left:2px solid var(--purple);border-radius:4px}
.lm-head-note b{color:var(--purple)}
.shape-strip{display:flex;align-items:center;gap:14px;margin-top:14px;padding:8px 14px;
  background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  font-family:var(--mono);font-size:12px;flex-wrap:wrap}

/* Embedding lookup preview — appears during early Act II as a fade-in overlay
   that briefly explains the lookup before the layer wave starts. Absolute
   positioning so it doesn't jolt the layout when it disappears. */
.embed-lookup-card{position:absolute;top:170px;left:40px;right:40px;
  background:var(--surface);border:1px solid var(--teal);
  box-shadow:0 4px 22px rgba(0,212,170,0.18);border-radius:12px;
  padding:18px 22px;z-index:5;
  opacity:0;pointer-events:none;transition:opacity .55s}
.embed-lookup-card.show{opacity:1;pointer-events:auto}
.embed-lookup-card h3{font-family:var(--mono);font-size:11px;color:var(--teal);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;font-weight:700}
.embed-lookup-card .desc{font-family:var(--mono);font-size:11.5px;color:var(--muted);
  margin-bottom:14px;line-height:1.55;max-width:780px}
.embed-lookup-rows{display:grid;grid-template-columns:auto 1fr;gap:8px 18px;
  align-items:center;font-family:var(--mono);font-size:12px;margin-bottom:12px}
.embed-row-tok{padding:5px 12px;border-radius:5px;background:var(--surface2);
  border:1px solid var(--teal);color:var(--teal);font-weight:600;justify-self:start;font-size:13px}
.embed-row-detail{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.embed-row-detail .arrow{color:var(--muted)}
.embed-row-detail .row-id{color:var(--amber);font-weight:600}
.embed-row-detail .matrix-icon{display:inline-block;width:14px;height:50px;
  background:var(--surface2);border:1px solid var(--border);border-radius:2px;
  position:relative;vertical-align:middle}
.embed-row-detail .matrix-icon .row-mark{position:absolute;left:1px;right:1px;
  height:2px;background:var(--teal);box-shadow:0 0 6px var(--teal)}
.embed-row-detail .v-cells{display:flex;gap:1px}
.embed-row-detail .v-cells .vc{width:9px;height:20px;border-radius:1px}
.embed-row-detail .more-dims{color:var(--muted);font-size:10.5px;margin-left:4px}
.embed-lookup-link{display:inline-block;font-family:var(--mono);font-size:12px;
  color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);
  padding-bottom:1px}
.embed-lookup-link:hover{color:var(--teal);border-bottom-color:var(--teal)}
.shape-strip .label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.1em}
.shape-strip .value{color:var(--teal);font-weight:700;font-size:13px;
  transition:color .3s, transform .3s}
.shape-strip .value.changed{color:var(--amber);transform:scale(1.05)}
.shape-strip .detail{color:var(--text);opacity:.85}

.logits-bars{display:flex;flex-direction:column;gap:4px;margin-top:4px}
.softmax-connector{display:flex;align-items:center;gap:14px;
  padding:10px 16px;margin:14px 0;background:var(--surface2);
  border-left:3px solid var(--teal);border-radius:6px;
  font-family:var(--mono);font-size:11.5px;
  transition:box-shadow .35s,opacity .35s}
.softmax-connector.firing{box-shadow:0 0 18px rgba(0,212,170,0.45);
  background:rgba(0,212,170,0.08)}
.softmax-formula{color:var(--teal);font-weight:700}
.softmax-arrow{color:var(--teal);font-size:18px;font-weight:700;
  transition:transform .35s}
.softmax-connector.firing .softmax-arrow{animation:sm-pulse 0.7s ease-in-out infinite}
@keyframes sm-pulse{0%,100%{transform:translateY(0)}50%{transform:translateY(3px)}}
.softmax-note{color:var(--muted);flex:1}
.act2-substep{font-family:var(--mono);font-size:10.5px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}
.act2-substep b{color:var(--teal)}
.logit-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;opacity:0;transition:opacity .25s}
.logit-row.show{opacity:1}
.logit-row .tk{min-width:80px;text-align:right;color:var(--text);
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.logit-row .bar{flex:1;height:14px;background:var(--surface2);border-radius:3px;overflow:hidden}
.logit-row .fill{height:100%;background:var(--blue);width:0;transition:width .55s cubic-bezier(.4,0,.2,1)}
.logit-row.winner .fill{background:var(--teal)}
.logit-row .pct{min-width:40px;text-align:right;color:var(--muted)}
.logit-row.winner .pct{color:var(--teal);font-weight:700}

/* ── Act III (loop) ── */
.loop-header{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:16px;
}
.loop-stats{font-family:var(--mono);font-size:12px;color:var(--muted);display:flex;gap:20px}
.loop-stats b{color:var(--teal)}
.loop-stats .warn{color:var(--amber)}
.loop-stats .err{color:var(--coral)}

.output-strip{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:16px 20px;
  font-family:var(--mono);font-size:15px;line-height:1.7;
  min-height:120px;max-height:320px;overflow-y:auto;
}
.output-strip .orig{color:var(--muted)}
.output-strip .tok{
  display:inline-block;padding:1px 4px;margin:0 1px;border-radius:3px;
  animation:pop .35s ease-out;
}
.output-strip .tok.eos{background:rgba(255,107,107,.2);color:var(--coral);font-weight:700}
@keyframes pop{
  0%{transform:scale(.5) translateY(-20px);opacity:0}
  60%{transform:scale(1.1) translateY(0);opacity:1}
  100%{transform:scale(1) translateY(0);opacity:1}
}

.loop-sidebar{
  margin-top:20px;display:grid;grid-template-columns:2fr 1fr;gap:20px;
}
.sidebar-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 18px;
}
.sidebar-card h3{font-family:var(--mono);font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.mini-spark{width:100%;height:80px;display:block}
.alt-bars{display:flex;flex-direction:column;gap:5px}
.alt-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;opacity:0;transition:opacity .25s}
.alt-row.show{opacity:1}
.alt-row .alt-tok{min-width:90px;text-align:right;color:var(--text);overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.alt-row .alt-bar{flex:1;height:14px;background:var(--surface2);border-radius:3px;overflow:hidden}
.alt-row .alt-fill{height:100%;border-radius:3px;transition:width .45s cubic-bezier(.4,0,.2,1);display:flex;align-items:center;padding:0 6px;color:#000;font-size:9px;font-weight:700}
.alt-row.winner .alt-fill{background:var(--teal);color:#000}
.alt-row:not(.winner) .alt-fill{background:var(--blue);color:rgba(0,0,0,0.85)}
.alt-row .alt-pct{min-width:46px;text-align:right;color:var(--muted)}
.alt-row.winner .alt-pct{color:var(--teal);font-weight:700}
.stat-line{display:flex;justify-content:space-between;font-family:var(--mono);font-size:12px;margin-bottom:6px}
.stat-line .k{color:var(--muted)}
.stat-line .v{color:var(--teal);font-weight:600}
.stat-line .v.ms-hot{color:var(--coral)}

/* ── End card ── */
.end-card{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:12px;
  pointer-events:none;opacity:0;transition:opacity .4s;
}
.end-card.show{opacity:1}
.end-title{font-size:22px;font-weight:800}
.end-output{
  max-width:820px;padding:20px 28px;
  background:var(--surface);border:2px solid var(--teal);border-radius:12px;
  font-family:var(--mono);font-size:14px;color:var(--text);line-height:1.6;
}
.end-references{
  max-width:820px;margin-top:18px;padding:16px 22px;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  pointer-events:auto;
}
.end-references h4{font-family:var(--mono);font-size:11px;color:var(--teal);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;font-weight:700}
.end-references ul{list-style:none;padding:0;margin:0;
  display:flex;flex-direction:column;gap:6px;font-family:var(--mono);font-size:12px}
.end-references li{display:flex;gap:8px}
.end-references li .tag{display:inline-block;min-width:88px;padding:1px 8px;
  background:var(--surface2);border:1px solid var(--border);border-radius:4px;
  font-size:10px;text-align:center;color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em;flex-shrink:0}
.end-references li a{color:var(--blue);text-decoration:none;
  border-bottom:1px dashed var(--blue);padding-bottom:1px}
.end-references li a:hover{color:var(--teal);border-bottom-color:var(--teal)}
.end-references .key-terms{margin-top:10px;font-family:var(--mono);
  font-size:10.5px;color:var(--muted);line-height:1.7}
.end-references .key-terms b{color:var(--text)}

/* ── Code strip (always visible above controls) ── */
.code-strip{
  position:absolute;bottom:72px;left:0;right:0;height:168px;
  background:rgba(8,11,18,0.96);backdrop-filter:blur(8px);
  border-top:1px solid var(--border);
  display:flex;flex-direction:column;
  padding:10px 28px 8px;
  font-family:var(--mono);
}
.code-header{display:flex;justify-content:space-between;align-items:baseline;
  font-size:10px;letter-spacing:.06em;margin-bottom:6px;}
.code-label{color:var(--teal);font-weight:700;text-transform:uppercase}
.code-label.external{color:var(--amber)}
.code-src{color:var(--muted);font-style:italic}
.code-body{flex:1;overflow:hidden;font-size:12px;line-height:1.55;color:var(--text);
  margin:0;background:transparent;font-family:var(--mono);}
.code-line{display:block;padding:0 8px;border-left:2px solid transparent;}
.code-line.current{background:rgba(0,212,170,0.10);border-left-color:var(--teal);color:#fff}
.code-keyword{color:var(--purple)}
.code-string{color:var(--amber)}
.code-comment{color:var(--muted);font-style:italic}
.code-self{color:var(--blue)}
.code-num{color:var(--coral)}
.code-footer{font-size:10.5px;color:var(--muted);margin-top:4px;
  letter-spacing:.04em;}
.code-footer b{color:var(--teal);font-weight:700}

/* ── Controls ── */
.controls{
  position:absolute;bottom:0;left:0;right:0;height:72px;
  display:flex;align-items:center;gap:14px;padding:0 28px;
  border-top:1px solid var(--border);background:rgba(8,11,18,0.95);
  backdrop-filter:blur(8px);
}
.controls .btn{
  background:var(--teal);color:#000;border:none;cursor:pointer;
  width:44px;height:44px;border-radius:50%;
  font-size:18px;font-weight:700;
  display:flex;align-items:center;justify-content:center;
}
.controls .btn:hover{opacity:.85}
.controls .btn.secondary{background:var(--surface2);color:var(--text);
  border:1px solid var(--border)}
.scrub-wrap{flex:1;position:relative}
.scrub{width:100%;accent-color:var(--teal);height:4px;cursor:pointer}
.speed-select{
  background:var(--surface2);color:var(--text);
  border:1px solid var(--border);border-radius:6px;
  padding:6px 10px;font-family:var(--mono);font-size:12px;cursor:pointer;
}
.time-display{font-family:var(--mono);font-size:12px;color:var(--muted);min-width:80px;text-align:right}
.act-pill{font-family:var(--mono);font-size:10px;color:var(--teal);padding:4px 10px;
  background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);
  border-radius:12px;letter-spacing:.1em;text-transform:uppercase}
</style>
</head>
<body>
<div class="stage">
  <div class="top-bar">
    <span class="model" id="t-model"></span>
    <span class="sys" id="t-sys"></span>
    <span class="prompt" id="t-prompt"></span>
  </div>

  <div id="scene1" class="scene">
    <h2>Act I · Tokenization</h2>
    <div class="hint">The model never sees text directly. The tokenizer splits input into subword pieces, each mapping to a vocabulary integer.</div>
    <div class="act1-text" id="act1-text"></div>
    <div class="act1-chips" id="act1-chips"></div>
  </div>

  <div id="scene2" class="scene">
    <h2>Act II · One forward pass (first generated token)</h2>
    <div class="hint">Embedding lookup → <span id="t-nlayer">N</span> transformer blocks (attention + FFN + residual) → final norm → LM head projection → softmax → argmax.</div>
    <div class="pipeline" id="pipe"></div>
    <div class="shape-strip" id="shape-strip">
      <span class="label">tensor shape</span>
      <span class="value" id="shape-value">[1, seq]</span>
      <span class="detail" id="shape-detail">batch × sequence (raw token ids)</span>
    </div>
    <div class="embed-lookup-card" id="embed-lookup">
      <h3>Embedding lookup — id → row → vector</h3>
      <div class="desc">
        Each token id is a row index into the embedding matrix. The matrix is shaped
        <b>[<span id="el-vocab">?</span> × <span id="el-hidden">?</span>]</b> — one
        row per token in the vocabulary. Pulling out one row gives a
        <span id="el-hidden2">?</span>-number vector. That's the <b>only</b> thing
        the model needs your token ids for; everything after this is just math on
        these vectors.
      </div>
      <div class="embed-lookup-rows" id="embed-lookup-rows"></div>
      <a class="embed-lookup-link" id="embed-link" href="embeddings.html" target="_blank">
        explore all <span id="el-vocab2">?</span> rows in 2D semantic space →
      </a>
    </div>
    <div class="act2-grid">
      <div class="act2-card">
        <h3>Hidden state magnitude — each token × each layer</h3>
        <div class="heatmap-wrap"><svg id="heatmap" class="heatmap-svg"></svg></div>
        <div class="heatmap-hint">
          A <b>LAYER</b> is one transformer block — distilgpt2 stacks 6 of them.
          Each block reads its input and adds a new contribution.
          Each <b>row</b> = one input token's vector (768 numbers).
          Each <b>column</b> = state after one block. Cell = ‖vector‖.
          The <b>last row</b> (teal frame) is what decides the next token.
          <br><br>
          <b>Why do some cells get huge?</b> Each block <i>adds</i> to the
          residual stream without rescaling, so magnitudes accumulate.
          The <b>first token</b> picks up disproportionate activation
          (the "attention sink" effect — every later token can look back
          at it). The final column is labeled <b>L<sub>N</sub>·ln</b> because
          the model's last LayerNorm has been applied — it normalizes each
          token's vector independently, compressing the spike.
        </div>
      </div>
      <div class="act2-card">
        <h3>LM head — turn the final hidden state into vocab scores</h3>
        <div class="lm-head-step" id="lm-step-1">
          <div class="lm-step-num">1</div>
          <div class="lm-step-body">
            <div class="lm-step-title">Take the last row of the heatmap</div>
            <div class="lm-step-desc">
              That's the final state of the last input token —
              <b><span id="lm-hidden-size">768</span> numbers</b>
              after the final LayerNorm. Showing first 20 dims:
            </div>
            <div class="lm-vec-cells" id="lm-vec-cells"></div>
          </div>
        </div>
        <div class="lm-head-arrow" id="lm-arrow-1">↓ multiply by LM-head weight matrix</div>
        <div class="lm-head-step" id="lm-step-2">
          <div class="lm-step-num">2</div>
          <div class="lm-step-body">
            <div class="lm-step-title">Project to vocabulary</div>
            <div class="lm-step-desc">
              Multiply by <b>W ∈ ℝ<sup>[<span id="lm-hidden-size2">768</span>
              × <span id="lm-vocab-size">50,257</span>]</sup></b> — one column
              per token in the vocabulary. Each column is a "pattern"
              the model checks for.
            </div>
          </div>
        </div>
        <div class="lm-head-arrow" id="lm-arrow-2">↓ dot product per vocab token</div>
        <div class="lm-head-step" id="lm-step-3">
          <div class="lm-step-num">3</div>
          <div class="lm-step-body">
            <div class="lm-step-title"><span id="lm-vocab-size2">50,257</span> raw logit scores</div>
            <div class="lm-step-desc">
              One number per vocab token — how strongly the hidden state
              matches each "pattern". These are the <b>logits</b>,
              broken down in the next card below.
            </div>
          </div>
        </div>
        <div class="lm-head-note">
          <b>Tied weights:</b> in GPT-2 this matrix is the SAME as the
          embedding matrix from the start (just transposed). One table is
          used for both <b>id → vector</b> lookup and <b>vector → id scores</b>
          unembedding.
        </div>
      </div>
    </div>
    <div class="act2-card" style="margin-top:16px">
      <div class="act2-substep">Step A · raw <b>logits</b> (LM head output)</div>
      <h3>One score per token in the entire vocabulary — these are arbitrary real numbers (often negative).</h3>
      <div class="logits-bars" id="logit-bars"></div>
      <div class="softmax-connector" id="softmax-connector">
        <span class="softmax-formula">softmax( xᵢ ) = exp(xᵢ) / Σⱼ exp(xⱼ)</span>
        <span class="softmax-arrow">↓</span>
        <span class="softmax-note">turns scores into a probability distribution that sums to 1</span>
      </div>
      <div class="act2-substep">Step B · <b>probabilities</b> (after softmax)</div>
      <h3>Same tokens, now interpretable as %. The argmax (teal) wins.</h3>
      <div class="logits-bars" id="probs-bars"></div>
    </div>
  </div>

  <div id="scene3" class="scene">
    <h2>Act III · Autoregressive loop</h2>
    <div class="hint">Each step is one full forward pass over the growing context. Tokens appear below as they're emitted. Loop ends on EOS or max_new_tokens.</div>
    <div class="loop-header">
      <div class="loop-stats">
        <span>step <b id="cur-step">1</b> / <span id="total-steps"></span></span>
        <span>ctx <b id="cur-ctx">0</b> tok</span>
        <span>ms/pass <b id="cur-ms">0</b></span>
        <span>cum <b id="cum-ms">0</b> ms</span>
      </div>
      <div class="loop-stats">
        <span id="eos-status"></span>
      </div>
    </div>
    <div class="output-strip" id="output-strip"></div>
    <div class="loop-sidebar">
      <div class="sidebar-card">
        <h3>top-5 alternatives — current step (winner in teal)</h3>
        <div class="alt-bars" id="alt-bars"></div>
      </div>
      <div class="sidebar-card">
        <h3>this step</h3>
        <div class="stat-line"><span class="k">token</span><span class="v" id="cur-tok"></span></div>
        <div class="stat-line"><span class="k">p(top)</span><span class="v" id="cur-prob"></span></div>
        <div class="stat-line"><span class="k">id</span><span class="v" id="cur-id"></span></div>
        <div class="stat-line"><span class="k">wallclock</span><span class="v" id="cur-ms-v"></span></div>
      </div>
    </div>
  </div>

  <div class="end-card" id="end-card">
    <div class="act-pill">Final result</div>
    <div class="end-title" id="end-title"></div>
    <div class="end-output" id="end-output"></div>
    <div class="end-references">
      <h4>Further reading · pick what fits your style</h4>
      <ul>
        <li><span class="tag">visual</span>
          <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank">
          Jay Alammar — The Illustrated Transformer</a></li>
        <li><span class="tag">code</span>
          <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY" target="_blank">
          Karpathy — Let's build GPT, from scratch, in code (YouTube, 2 hr)</a></li>
        <li><span class="tag">deep</span>
          <a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank">
          Anthropic — A Mathematical Framework for Transformer Circuits</a></li>
        <li><span class="tag">paper</span>
          <a href="https://arxiv.org/abs/2309.17453" target="_blank">
          Xiao et al. (2023) — Attention Sinks (the first-token-spike phenomenon)</a></li>
        <li><span class="tag">paper</span>
          <a href="https://arxiv.org/abs/1706.03762" target="_blank">
          Vaswani et al. (2017) — "Attention Is All You Need" (the original transformer)</a></li>
        <li><span class="tag">video</span>
          <a href="https://www.youtube.com/watch?v=eMlx5fFNoYc" target="_blank">
          3Blue1Brown — But what is a GPT? (Visualizations of transformers)</a></li>
      </ul>
      <div class="key-terms">
        <b>Key terms to look up:</b>
        token / tokenizer (BPE) · embedding matrix · residual stream · attention head ·
        causal mask · softmax · temperature · greedy vs. sampling · LayerNorm · LM head ·
        EOS token · attention sink · KV cache · autoregressive generation
      </div>
    </div>
  </div>

  <div class="code-strip" id="code-strip">
    <div class="code-header">
      <span class="code-label" id="code-label">inside the model</span>
      <span class="code-src" id="code-src">transformers/models/gpt2/modeling_gpt2.py</span>
    </div>
    <pre class="code-body" id="code-body"></pre>
    <div class="code-footer" id="code-footer"></div>
  </div>

  <div class="controls">
    <button class="btn" id="play-btn">⏵</button>
    <button class="btn secondary" id="restart-btn" title="Restart">↻</button>
    <div class="act-pill" id="act-pill">Act I</div>
    <div class="scrub-wrap">
      <input type="range" class="scrub" id="scrub" min="0" max="1000" value="0" step="1">
    </div>
    <select class="speed-select" id="speed">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
      <option value="5">5×</option>
      <option value="10">10×</option>
    </select>
    <div class="time-display" id="time-display">0.0s / 0.0s</div>
  </div>
</div>

<script>
const D = __DATA__;

// ── Top bar ─────────────────────────────────────────────────────────────
document.getElementById('t-model').textContent = D.model_id;
document.getElementById('t-sys').textContent = D.system_prompt ? `system: ${JSON.stringify(D.system_prompt)}` : '';
document.getElementById('t-prompt').textContent = `prompt: ${JSON.stringify(D.prompt)}`;
document.getElementById('t-nlayer').textContent = D.model_meta.n_layer ?? '?';

// ── Timeline ────────────────────────────────────────────────────────────
const T_ACT1 = 7.0;
const N_LAYERS = (D.hidden_norms_first_step || []).length - 1;
// Layer wave auto-scales with model depth: 1.4 s/layer, clamped to [5, 12] s.
const LAYER_WAVE = Math.min(12, Math.max(5, N_LAYERS * 1.4));
// Act II breakdown (default totals to ~16.5 s for distilgpt2 / 6 layers):
//   pipe Input → Embedding         3 s
//   layer activation wave          LAYER_WAVE s  (~8.4 s for 6 layers)
//     ↳ attention reveal runs IN PARALLEL during first ~2.5 s of this wave
//       (layer 0 is what produces the attention pattern we're showing)
//   "linger" after layer wave      1.5 s   (both cards full, code panel highlights attn math)
//   final-norm + LM head           2 s
//   logits + softmax + winner      2 s
const T_ACT2 = T_ACT1 + 3 + LAYER_WAVE + 1.5 + 2 + 2;
const STEP_DUR = 0.7;     // seconds per generation step at 1× (was 0.35 — too fast)
const LOOP_DUR = Math.max(6, D.loop_steps.length * STEP_DUR);
const T_END = T_ACT2 + LOOP_DUR + 3.0;

const scrub = document.getElementById('scrub');
scrub.max = Math.round(T_END * 100);

const actPill = document.getElementById('act-pill');
function currentAct(t){
  if (t < T_ACT1) return 'Act I · tokenize';
  if (t < T_ACT2) return 'Act II · forward pass';
  if (t < T_ACT2 + LOOP_DUR) return 'Act III · loop';
  return 'Done';
}

// ── Act I: tokenization ─────────────────────────────────────────────────
const act1TextEl = document.getElementById('act1-text');
const act1ChipsEl = document.getElementById('act1-chips');
act1TextEl.textContent = JSON.stringify(D.prompt);
D.tokens.forEach((tok, i) => {
  const el = document.createElement('div');
  el.className = 'act1-chip';
  el.innerHTML = `<div class="tok">${escHtml(tok)}</div><div class="id">id ${D.token_ids[i]}</div>`;
  act1ChipsEl.appendChild(el);
});

function updateAct1(t){
  // 0-1s: text only. 1-2.5s: chips appear staggered. 2.5-4s: ids fade in.
  const textOpacity = t < 2 ? 1 : Math.max(0, 1 - (t - 2) * 2);
  act1TextEl.style.opacity = String(textOpacity);

  const chips = act1ChipsEl.children;
  const chipsStart = 1.0, chipsEnd = 2.8;
  for (let i = 0; i < chips.length; i++){
    const delay = chipsStart + (i / chips.length) * (chipsEnd - chipsStart);
    if (t >= delay) chips[i].classList.add('show');
    else chips[i].classList.remove('show');
    if (t >= delay + 1) chips[i].classList.add('id-show');
    else chips[i].classList.remove('id-show');
  }
}

// ── Act II: deep dive ───────────────────────────────────────────────────
const PIPE_STAGES = [
  'Input IDs','Embedding','Layer 1','...','Layer N','Final norm','LM head','Softmax','Argmax',
];
const pipeEl = document.getElementById('pipe');
PIPE_STAGES.forEach((s,i) => {
  const b = document.createElement('div');
  b.className = 'pipe-box'; b.textContent = s;
  pipeEl.appendChild(b);
  if (i < PIPE_STAGES.length - 1){
    const a = document.createElement('div'); a.className = 'pipe-arrow'; a.textContent = '→';
    pipeEl.appendChild(a);
  }
});

// ── Embedding lookup preview (top of Act II) ──────────────────────────
// Brief panel that fades in for the first ~5 seconds of Act II to show
// "id → row → vector" before the layer wave takes over. Links to the
// standalone embedding-explorer HTML for deeper inspection.
const embedLookupCardEl = document.getElementById('embed-lookup');
(function buildEmbedLookup(){
  const rowsEl = document.getElementById('embed-lookup-rows');
  document.getElementById('el-vocab').textContent  = (D.model_meta.vocab_size || 0).toLocaleString();
  document.getElementById('el-vocab2').textContent = (D.model_meta.vocab_size || 0).toLocaleString();
  document.getElementById('el-hidden').textContent = D.model_meta.hidden_size;
  document.getElementById('el-hidden2').textContent = D.model_meta.hidden_size;
  // Wire the "go deeper" link to a model-specific embedding-explorer file.
  // Convention: out/embeddings_<short-model-slug>.html (e.g. embeddings_distilgpt2.html).
  document.getElementById('embed-link').href = D.embedding_link || 'embeddings.html';

  // Show the first 3 prompt tokens (room for more makes the card too tall).
  const N_SHOW = Math.min(3, D.tokens.length);
  for (let i = 0; i < N_SHOW; i++){
    const tok  = D.tokens[i];
    const tid  = D.token_ids[i];
    const vec  = (D.embeddings && D.embeddings[i]) || [];
    const vocab = D.model_meta.vocab_size || 1;
    const rowFraction = (tid / vocab) * 100;

    const cells = vec.slice(0, 12).map(v => {
      const intensity = Math.min(1, Math.abs(v));
      const col = v >= 0
        ? `rgba(0,${Math.round(140 + intensity*112)},${Math.round(120 + intensity*50)},${0.25 + intensity*0.7})`
        : `rgba(${Math.round(140 + intensity*115)},${Math.round(80 - intensity*60)},${Math.round(80 - intensity*60)},${0.25 + intensity*0.7})`;
      return `<div class="vc" style="background:${col}" title="${v.toFixed(3)}"></div>`;
    }).join('');
    const moreDims = (D.model_meta.hidden_size || 0) - 12;

    rowsEl.innerHTML += `
      <div class="embed-row-tok">${escHtml(tok)}</div>
      <div class="embed-row-detail">
        <span class="arrow">→ id</span>
        <span class="row-id">${tid}</span>
        <span class="arrow">→ row in</span>
        <span class="matrix-icon" title="row ${tid} of ${vocab}">
          <span class="row-mark" style="top:${rowFraction.toFixed(1)}%"></span>
        </span>
        <span class="arrow">→</span>
        <span class="v-cells">${cells}</span>
        <span class="more-dims">+ ${moreDims} more dims</span>
      </div>
    `;
  }
})();

function updateEmbedLookup(dt){
  // Visible during pipe Input + Embedding (0-3s) plus 1s into layer wave so
  // the viewer has time to read it. Then fades out.
  const show = dt > 0.3 && dt < 4.5;
  embedLookupCardEl.classList.toggle('show', show);
}

// Heatmap data: hidden_norms_grid is [n_layers+1][n_tokens]
// We render it as rows=tokens, columns=layers (transposed for display).
const heatmapWrapEl = document.querySelector('.heatmap-wrap');
const HM_GRID = D.hidden_norms_grid || [];
const HM_NLAYERS = HM_GRID.length;            // includes input column
const HM_NTOKENS = (HM_GRID[0] || []).length;
let   HM_MAX = 0;
for (const row of HM_GRID) for (const v of row) { if (v > HM_MAX) HM_MAX = v; }
HM_MAX = Math.max(HM_MAX, 1e-6);

// Used by updateAct2 — kept under the same name as before for compat.
const norms = D.hidden_norms_first_step || [];

function drawHeatmap(progress){
  // progress 0..1 — fraction of layer columns to reveal.
  const colsToShow = Math.ceil(progress * HM_NLAYERS);
  const labelW = 90;
  // Cell width auto-scales so wider models (TinyLlama 22 layers) still fit.
  const targetW = 700;
  const cellW = Math.max(28, Math.floor((targetW - labelW) / Math.max(HM_NLAYERS, 1)));
  const cellH = 28;
  const headerH = 22;
  const W = labelW + cellW * HM_NLAYERS + 6;
  const H = headerH + cellH * HM_NTOKENS + 6;

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="heatmap-svg" xmlns="http://www.w3.org/2000/svg">`;

  // Column headers (input, L1, L2, ..., L_N·ln)
  // The very last column has the model's final LayerNorm already applied,
  // so it's labeled with `·ln` to flag that — it's not just "after layer N".
  for (let l = 0; l < HM_NLAYERS; l++){
    const x = labelW + l * cellW + cellW / 2;
    const isFinal = (l === HM_NLAYERS - 1);
    const lbl = l === 0 ? 'input' : (isFinal ? `L${l}·ln` : `L${l}`);
    const fill = isFinal ? '#00d4aa' : 'var(--muted)';
    const wt   = isFinal ? '700' : '400';
    svg += `<text x="${x}" y="14" text-anchor="middle" class="heatmap-col-label" fill="${fill}" font-weight="${wt}">${lbl}</text>`;
  }

  // Rows
  for (let i = 0; i < HM_NTOKENS; i++){
    const y = headerH + i * cellH;
    const isLast = (i === HM_NTOKENS - 1);
    const labelClass = isLast ? 'heatmap-row-label last' : 'heatmap-row-label';
    const tokTxt = (D.tokens[i] || '').replace(/\s+/g, '·') || '·';
    svg += `<text x="${labelW - 6}" y="${y + cellH/2 + 4}" text-anchor="end" class="${labelClass}">${escHtml(tokTxt)}</text>`;

    for (let l = 0; l < HM_NLAYERS; l++){
      const x = labelW + l * cellW;
      const v = HM_GRID[l][i];
      const visible = l < colsToShow;
      const intensity = visible ? Math.min(1, v / HM_MAX) : 0.0;
      // Teal fill, opacity scales with intensity. Empty cells stay near-bg.
      const fill = visible
        ? `rgba(0,${Math.round(60 + 152 * intensity)},${Math.round(50 + 120 * intensity)},${0.18 + intensity * 0.78})`
        : `rgba(21,27,46,0.7)`;
      svg += `<rect x="${x+1}" y="${y+1}" width="${cellW-2}" height="${cellH-2}" rx="2" fill="${fill}" class="heatmap-cell"/>`;
      // Show numeric value when cell is wide enough.
      if (visible && cellW >= 46){
        const txtColor = intensity > 0.45 ? '#001513' : '#cdd6f0';
        svg += `<text x="${x + cellW/2}" y="${y + cellH/2}" class="heatmap-cell-val" fill="${txtColor}">${v.toFixed(1)}</text>`;
      }
    }

    if (isLast){
      // Dashed teal frame around the last row to flag "this is the predictor".
      svg += `<rect x="${labelW - 2}" y="${y - 1}" width="${HM_NLAYERS * cellW + 4}" height="${cellH}" class="heatmap-last-row-frame"/>`;
    }
  }

  svg += '</svg>';
  heatmapWrapEl.innerHTML = svg;
}

// ── LM head card setup ────────────────────────────────────────────────
// Show the actual final-layer hidden state values (post-ln_f) so the
// viewer connects "heatmap last row" with "what the LM head consumes".
(function setupLmHead(){
  const cellsEl = document.getElementById('lm-vec-cells');
  const lastVec = (D.hidden_last && D.hidden_last[D.hidden_last.length - 1]) || [];
  const showDims = Math.min(20, lastVec.length);
  let html = '';
  for (let i = 0; i < showDims; i++){
    const v = lastVec[i];
    const intensity = Math.min(1, Math.abs(v));
    const col = v >= 0
      ? `rgba(0,${Math.round(140+intensity*112)},${Math.round(120+intensity*50)},${0.25+intensity*0.7})`
      : `rgba(${Math.round(140+intensity*115)},${Math.round(80-intensity*60)},${Math.round(80-intensity*60)},${0.25+intensity*0.7})`;
    html += `<div class="vc" style="background:${col}" title="dim ${i}: ${v.toFixed(3)}"></div>`;
  }
  const remaining = (D.model_meta.hidden_size || 0) - showDims;
  html += `<span class="more">+ ${remaining} more</span>`;
  cellsEl.innerHTML = html;
  document.getElementById('lm-hidden-size').textContent  = D.model_meta.hidden_size;
  document.getElementById('lm-hidden-size2').textContent = D.model_meta.hidden_size;
  document.getElementById('lm-vocab-size').textContent   = (D.model_meta.vocab_size || 0).toLocaleString();
  document.getElementById('lm-vocab-size2').textContent  = (D.model_meta.vocab_size || 0).toLocaleString();
})();

// ── LM head animation timing ───────────────────────────────────────────
// Stages of the right-side card fade in over ~1.5s after the layer wave
// finishes. Boundaries are dt-relative within Act II.
function updateLmHead(dt, T_LAYERS_END){
  const lmDt = dt - T_LAYERS_END - 0.3;     // start ~0.3s after heatmap full
  document.getElementById('lm-step-1').classList.toggle('show', lmDt >= 0);
  document.getElementById('lm-arrow-1').classList.toggle('show', lmDt >= 0.5);
  document.getElementById('lm-step-2').classList.toggle('show', lmDt >= 0.6);
  document.getElementById('lm-arrow-2').classList.toggle('show', lmDt >= 1.1);
  document.getElementById('lm-step-3').classList.toggle('show', lmDt >= 1.2);
}

// ── Logits & probs rows (two separate panels; same row count + order) ────
// Use the SAME tokens for both, in the order softmax preserves the rank.
// (probs are softmax(logits); since softmax is monotonic, ranks match.)
const LOGITS_DATA = (D.logits_top_first_step || []);
const PROBS_DATA  = (D.probs_top_first_step  || []);
const logitsEl = document.getElementById('logit-bars');
const probsEl  = document.getElementById('probs-bars');

LOGITS_DATA.forEach((l, i) => {
  const row = document.createElement('div');
  row.className = 'logit-row' + (i === 0 ? ' winner' : '');
  row.innerHTML = `<div class="tk">${escHtml(l.token)}</div><div class="bar"><div class="fill" id="lfill-${i}"></div></div><div class="pct" id="lpct-${i}">0</div>`;
  logitsEl.appendChild(row);
});
PROBS_DATA.forEach((p, i) => {
  const row = document.createElement('div');
  row.className = 'logit-row' + (i === 0 ? ' winner' : '');
  row.innerHTML = `<div class="tk">${escHtml(p.token)}</div><div class="bar"><div class="fill" id="pfill-${i}"></div></div><div class="pct" id="ppct-${i}">0%</div>`;
  probsEl.appendChild(row);
});

// Logit bars are width-scaled relative to the spread (since logits are
// abstract reals — width has no probability meaning, just relative ranking).
const _logVals = LOGITS_DATA.map(l => l.logit);
const LOG_MIN = Math.min(..._logVals);
const LOG_MAX = Math.max(..._logVals);
const LOG_RANGE = (LOG_MAX - LOG_MIN) || 1;

function setPipeActive(idx){
  const boxes = pipeEl.querySelectorAll('.pipe-box');
  boxes.forEach((b, i) => {
    b.classList.toggle('active', i === idx);
    b.classList.toggle('done', i < idx);
  });
}

// ── Tensor shape strip ────────────────────────────────────────────────
// Updates as data flows through the pipeline so viewer SEES the shape change.
const SEQ_LEN = (D.tokens || []).length || 1;
const HIDDEN  = D.model_meta?.hidden_size ?? '?';
const VOCAB   = D.model_meta?.vocab_size ?? '?';
const shapeValueEl  = document.getElementById('shape-value');
const shapeDetailEl = document.getElementById('shape-detail');
let _lastShapeValue = '';
function setShape(value, detail){
  if (value === _lastShapeValue) return;
  _lastShapeValue = value;
  shapeValueEl.textContent = value;
  shapeDetailEl.textContent = detail;
  // Quick "changed" pulse to draw the eye.
  shapeValueEl.classList.add('changed');
  setTimeout(() => shapeValueEl.classList.remove('changed'), 380);
}
function updateShape(dt, T_PIPE_EMB, T_LAYERS_END, T_ATTN_END, T_FINAL_END){
  if (dt < T_PIPE_EMB){
    setShape(`[1, ${SEQ_LEN}]`, `batch × sequence — raw token ids (integers)`);
  } else if (dt < 3){
    setShape(`[1, ${SEQ_LEN}, ${HIDDEN}]`,
      `each id became a ${HIDDEN}-dim vector via lookup in the ${VOCAB.toLocaleString?.() || VOCAB}×${HIDDEN} embedding matrix`);
  } else if (dt < T_LAYERS_END){
    setShape(`[1, ${SEQ_LEN}, ${HIDDEN}]`,
      `shape stays constant through every transformer block — only the values change`);
  } else if (dt < T_ATTN_END){
    setShape(`[1, ${SEQ_LEN}, ${HIDDEN}]`,
      `still ${HIDDEN}-dim per token after the layer wave; final norm rescales`);
  } else if (dt < T_FINAL_END){
    setShape(`[1, ${SEQ_LEN}, ${VOCAB.toLocaleString?.() || VOCAB}]`,
      `LM head projects each token's vector to vocabulary scores (logits)`);
  } else {
    setShape(`scalar id`,
      `argmax over the last position picks one integer — the next token`);
  }
}

function updateAct2(t){
  const dt = t - T_ACT1;
  if (dt < 0){ setPipeActive(-1); return; }

  // Sub-timeline (boundaries derived from the constants at the top of file):
  //   0           → 1.5s       pipe Input IDs
  //   1.5         → 3s         pipe Embedding
  //   3           → 3+LW       layer activation wave (heatmap columns fill)
  //     3 → 5.5s    in-parallel: attention rows reveal (layer 0 is what produces them)
  //   3+LW        → 3+LW+1.5   linger — both cards full, code panel emphasizes attn math
  //   3+LW+1.5    → 3+LW+3.5   final norm + LM head
  //   3+LW+3.5    → end        logits + softmax + winner highlighted
  const LW = LAYER_WAVE;
  const T_PIPE_EMB    = 1.5;
  const T_LAYERS_END  = 3 + LW;
  const T_ATTN_END    = T_LAYERS_END + 1.5;     // linger window
  const T_FINAL_END   = T_ATTN_END + 2;

  if      (dt < T_PIPE_EMB)    setPipeActive(0);
  else if (dt < 3)             setPipeActive(1);
  else if (dt < T_LAYERS_END)  setPipeActive(2 + Math.min(2, Math.floor((dt-3)/(LW/3))));
  else if (dt < T_ATTN_END-1)  setPipeActive(4);   // last "Layer N" stays lit while attention reveals
  else if (dt < T_ATTN_END)    setPipeActive(5);   // Final norm
  else if (dt < T_FINAL_END)   setPipeActive(6);   // LM head
  else if (dt < T_FINAL_END+1) setPipeActive(7);   // Softmax
  else                         setPipeActive(8);   // Argmax

  // Heatmap: one column per layer (input + N), one row per input token.
  // Reveals columns left-to-right as the layer wave plays.
  const layerProgress = Math.min(1, Math.max(0, (dt - 3) / LW));
  drawHeatmap(layerProgress);

  // LM head card — fades in stage-by-stage AFTER the heatmap is full.
  // Connects "last row of heatmap" to "logits below" by literally showing
  // the matmul as three numbered blocks.
  updateLmHead(dt, T_LAYERS_END);

  // ── Tensor shape strip — updates per active pipe stage ────────────
  updateShape(dt, T_PIPE_EMB, T_LAYERS_END, T_ATTN_END, T_FINAL_END);

  // ── Embedding-lookup preview — fades in/out at start of Act II ────
  updateEmbedLookup(dt);

  // ── Final phase: logits → softmax → probs (split into 2 bar-chart panels)
  // Sub-timeline (2 s total, after T_FINAL_END):
  //   0 → 0.9 s   logit bars stagger in
  //   0.9 → 1.1 s softmax connector "fires" (glows + arrow pulses)
  //   1.0 → 2.0 s prob bars stagger in (overlapping the connector pulse)
  const finalDt = dt - T_FINAL_END;
  const logitProg = Math.max(0, Math.min(1, finalDt / 0.9));
  const probsProg = Math.max(0, Math.min(1, (finalDt - 1.0) / 1.0));
  const softmaxFiring = finalDt >= 0.9 && finalDt < 1.5;
  document.getElementById('softmax-connector').classList.toggle('firing', softmaxFiring);

  // Logit bars: width is RELATIVE to the spread (logits have no inherent
  // [0, 1] meaning — they're arbitrary reals). Show actual logit number.
  LOGITS_DATA.forEach((l, i) => {
    const fill = document.getElementById(`lfill-${i}`);
    const pct  = document.getElementById(`lpct-${i}`);
    const row  = logitsEl.children[i];
    if (logitProg > i / LOGITS_DATA.length){
      const w = Math.max(2, ((l.logit - LOG_MIN) / LOG_RANGE) * 100);
      fill.style.width = w.toFixed(1) + '%';
      pct.textContent = l.logit.toFixed(2);
      row.classList.add('show');
    } else {
      fill.style.width = '0%';
      pct.textContent = '—';
      row.classList.remove('show');
    }
  });

  // Probability bars: ABSOLUTE width = real probability percentage.
  // Min 0.5% so the runners-up aren't swallowed by the border.
  PROBS_DATA.forEach((p, i) => {
    const fill = document.getElementById(`pfill-${i}`);
    const pct  = document.getElementById(`ppct-${i}`);
    const row  = probsEl.children[i];
    if (probsProg > i / PROBS_DATA.length){
      fill.style.width = Math.max(0.5, p.prob * 100) + '%';
      pct.textContent = (p.prob * 100).toFixed(1) + '%';
      row.classList.add('show');
    } else {
      fill.style.width = '0%';
      pct.textContent = '0%';
      row.classList.remove('show');
    }
  });
}

// ── Act III: loop ──────────────────────────────────────────────────────
const outputStrip = document.getElementById('output-strip');
outputStrip.innerHTML = `<span class="orig">${escHtml(D.prompt)}</span>`;
const stepCount = D.loop_steps.length;
document.getElementById('total-steps').textContent = stepCount;

const altBarsEl = document.getElementById('alt-bars');

function drawAltBars(alts){
  if (!alts || !alts.length) { altBarsEl.innerHTML = ''; return; }
  // Render top-5 alternatives. Bar width is scaled to the leader so even
  // very low-prob runners-up are visible. Winner (rank 0) gets teal.
  const max = Math.max(...alts.map(a => a.prob), 0.0001);
  altBarsEl.innerHTML = alts.slice(0, 5).map((a, i) => {
    const w = (a.prob / max * 100).toFixed(1);
    const pctTxt = (a.prob * 100).toFixed(1) + '%';
    return `<div class="alt-row show ${i === 0 ? 'winner' : ''}">
      <div class="alt-tok">${escHtml(a.token)}</div>
      <div class="alt-bar"><div class="alt-fill" style="width:${w}%">${i === 0 ? pctTxt : ''}</div></div>
      <div class="alt-pct">${pctTxt}</div>
    </div>`;
  }).join('');
}

let lastShownStep = -1;
let cumMs = 0;

function updateAct3(t){
  const dt = t - T_ACT2;
  if (dt < 0){
    if (lastShownStep !== -1){
      outputStrip.innerHTML = `<span class="orig">${escHtml(D.prompt)}</span>`;
      lastShownStep = -1;
      cumMs = 0;
      document.getElementById('cur-step').textContent = '0';
      document.getElementById('cur-ctx').textContent = '0';
      document.getElementById('cur-ms').textContent = '0';
      document.getElementById('cum-ms').textContent = '0';
      document.getElementById('cur-tok').textContent = '';
      document.getElementById('cur-prob').textContent = '';
      document.getElementById('cur-id').textContent = '';
      document.getElementById('cur-ms-v').textContent = '';
      document.getElementById('eos-status').textContent = '';
      drawAltBars([]);
    }
    return;
  }

  const stepIdx = Math.min(stepCount - 1, Math.floor(dt / STEP_DUR));
  if (stepIdx === lastShownStep) return;

  // Catch up: if user scrubbed forward, replay missed steps instantly.
  if (stepIdx < lastShownStep){
    // Scrubbed backward: rebuild from scratch.
    outputStrip.innerHTML = `<span class="orig">${escHtml(D.prompt)}</span>`;
    cumMs = 0;
    for (let i = 0; i <= stepIdx; i++) appendStep(i, false);
  } else {
    for (let i = lastShownStep + 1; i <= stepIdx; i++){
      appendStep(i, i === stepIdx);
    }
  }
  lastShownStep = stepIdx;
}

function appendStep(i, animate){
  const s = D.loop_steps[i];
  cumMs += s.ms;

  // Append the generated token as a pill, using generation_text as source of
  // truth when available (correct for sentencepiece) — here we just use s.token
  // for the per-step pill, and show the cleaner full text at the end.
  const span = document.createElement('span');
  span.className = 'tok' + (s.is_eos ? ' eos' : '');
  span.style.background = colorForStep(i) + '22';
  span.style.border = `1px solid ${colorForStep(i)}66`;
  span.style.color = colorForStep(i);
  span.textContent = s.is_eos ? '⏹ EOS' : s.token;
  if (!animate) span.style.animation = 'none';
  outputStrip.appendChild(span);
  outputStrip.scrollTop = outputStrip.scrollHeight;

  document.getElementById('cur-step').textContent = s.step;
  document.getElementById('cur-ctx').textContent = s.ctx_len;
  document.getElementById('cur-ms').textContent = s.ms.toFixed(0);
  document.getElementById('cum-ms').textContent = cumMs.toFixed(0);
  document.getElementById('cur-tok').textContent = JSON.stringify(s.token);
  document.getElementById('cur-prob').textContent = (s.prob*100).toFixed(1) + '%';
  document.getElementById('cur-id').textContent = s.id;

  const msEl = document.getElementById('cur-ms-v');
  msEl.textContent = s.ms.toFixed(1) + ' ms';
  msEl.classList.toggle('ms-hot', s.ms > 80);

  drawAltBars(s.top_alts);

  if (s.is_eos){
    document.getElementById('eos-status').innerHTML = '<span class="err">⏹ EOS emitted — loop ended</span>';
  } else if (i === stepCount - 1){
    document.getElementById('eos-status').innerHTML = '<span class="warn">hit max_new_tokens; no EOS</span>';
  }
}

function colorForStep(i){
  const palette = ['#00d4aa','#ffb800','#4da6ff','#c084fc','#ff6b6b','#34d399','#fb923c','#a78bfa'];
  return palette[i % palette.length];
}

// ── End card ───────────────────────────────────────────────────────────
document.getElementById('end-output').textContent = D.prompt + D.generation_text;
document.getElementById('end-title').textContent = D.loop_steps.some(s=>s.is_eos) ?
  `Loop ended cleanly after ${D.loop_steps.length} steps (EOS emitted)` :
  `Loop hit cap at ${D.loop_steps.length} steps (no EOS)`;

function updateEnd(t){
  const show = t >= T_ACT2 + LOOP_DUR + 0.5;
  document.getElementById('end-card').classList.toggle('show', show);
}

// ── Scene visibility ────────────────────────────────────────────────────
const scenes = [
  document.getElementById('scene1'),
  document.getElementById('scene2'),
  document.getElementById('scene3'),
];
function showScene(t){
  let active = 0;
  if (t >= T_ACT1) active = 1;
  if (t >= T_ACT2) active = 2;
  scenes.forEach((s, i) => s.classList.toggle('active', i === active));
}

// ── Master clock ────────────────────────────────────────────────────────
let t = 0, playing = true, speed = 1, lastFrame = performance.now();

function tick(now){
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (playing){
    t += dt * speed;
    if (t >= T_END){ t = T_END; playing = false; setPlayIcon(); }
  }
  render();
  requestAnimationFrame(tick);
}

// ── Code strip — show what's running INSIDE the model per beat ────────
const CODE_BEATS = (function(){
  // Compute Act II sub-beat boundaries (must match updateAct2's local consts).
  const LW = LAYER_WAVE;
  const T_PIPE_EMB    = 1.5;
  const T_LAYERS_END  = 3 + LW;
  const T_ATTN_END    = T_LAYERS_END + 3;
  const T_FINAL_END   = T_ATTN_END + 2;

  return [
    // Act I — tokenization (pre-model)
    { until: T_ACT1 - 2.5, scope: 'external', header: 'tokenizer.encode',
      src: 'transformers/.../tokenization_gpt2_fast.py',
      lines: [
        'def encode(self, text):',
        '    # Rust BPE: greedy merges on learned rules',
        '    return self._tokenizer.encode(text).ids',
      ], current: 2,
      footer: 'tokenization runs BEFORE the model · just lookup tables' },

    { until: T_ACT1, scope: 'external', header: 'tokenizer · decode',
      src: 'transformers/.../tokenization_utils_base.py',
      lines: [
        'tokens = [tokenizer.decode([t]) for t in token_ids]',
        '# integer ids → string pieces',
      ], current: 0,
      footer: 'each id maps to one entry in the 50,257-row vocab' },

    // Act II beat 1: pipe input + embedding
    { until: T_ACT1 + T_PIPE_EMB, scope: 'inside', header: 'GPT2Model.forward · entry',
      src: 'transformers/models/gpt2/modeling_gpt2.py',
      lines: [
        'def forward(self, input_ids, ...):',
        '    inputs_embeds   = self.wte(input_ids)',
        '    position_embeds = self.wpe(position_ids)',
        '    hidden_states   = inputs_embeds + position_embeds',
        '    for block in self.h:',
        '        hidden_states = block(hidden_states)',
      ], current: 1,
      footer: 'wte = nn.Embedding(50257, 768) · just a lookup table' },

    { until: T_ACT1 + 3, scope: 'inside', header: 'GPT2Model.forward · positions',
      src: 'transformers/models/gpt2/modeling_gpt2.py',
      lines: [
        'inputs_embeds   = self.wte(input_ids)         # token vec',
        'position_embeds = self.wpe(position_ids)      # pos vec',
        'hidden_states   = inputs_embeds + position_embeds',
        '# shape: [1, seq_len, 768]',
      ], current: 2,
      footer: 'wpe is also nn.Embedding — for positions 0..1023' },

    // Act II beat 2: layer wave (one frame, but with dynamic footer)
    { until: T_ACT1 + T_LAYERS_END, scope: 'inside', header: 'GPT2Block.forward',
      src: 'transformers/models/gpt2/modeling_gpt2.py',
      lines: [
        'def forward(self, hidden_states):',
        '    residual = hidden_states',
        '    h = self.ln_1(hidden_states)',
        '    h = self.attn(h)[0]              # multi-head self-attn',
        '    hidden_states = residual + h     # residual add',
        '    residual = hidden_states',
        '    h = self.ln_2(hidden_states)',
        '    h = self.mlp(h)                  # FFN (3072 dim)',
        '    return residual + h              # residual add',
      ], current: 3,
      footer: 'block N of N · ‖h‖ rising layer by layer · norm bars on left' },

    // Act II beat 3: attention reveal
    { until: T_ACT1 + T_ATTN_END, scope: 'inside', header: 'GPT2Attention.forward',
      src: 'transformers/models/gpt2/modeling_gpt2.py',
      lines: [
        'qkv     = self.c_attn(x)             # one Conv1D, 768 → 2304',
        'q, k, v = qkv.split(self.split_size, dim=2)',
        'attn    = (q @ k.transpose(-2,-1)) / sqrt(d_k)',
        'attn    = attn + self.bias            # causal mask',
        'attn    = softmax(attn)',
        'output  = self.c_proj(attn @ v)',
      ], current: 4,
      footer: 'softmax produces the heatmap · each row sums to 1' },

    // Act II beat 4: final norm + LM head
    { until: T_ACT1 + T_FINAL_END, scope: 'inside', header: 'GPT2LMHeadModel.forward',
      src: 'transformers/models/gpt2/modeling_gpt2.py',
      lines: [
        '# transformer returns final hidden_states',
        'hidden_states = self.transformer.ln_f(hidden_states)',
        'lm_logits     = self.lm_head(hidden_states)',
        '# self.lm_head: nn.Linear(768, 50257)',
        '# weights TIED to wte — same parameters!',
      ], current: 2,
      footer: 'one weight matrix doubles as embed and unembed' },

    // Act II beat 5: argmax (external)
    { until: T_ACT2, scope: 'external', header: 'inference engine · argmax',
      src: '(your code · runs after model.forward returns)',
      lines: [
        'logits  = out.logits[:, -1]              # last position only',
        'probs   = torch.softmax(logits, dim=-1)  # 50,257 → distribution',
        'next_id = int(probs.argmax())            # greedy',
        '# the model itself stops at logits.',
        '# softmax + argmax are the orchestrator’s job.',
      ], current: 2,
      footer: 'temperature scaling, top-k, sampling — all live HERE, not inside the model' },

    // Act III: loop
    { until: T_END, scope: 'external', header: 'inference engine · generation loop',
      src: '(your code)',
      lines: [
        'for step in range(max_new_tokens):',
        '    out = model(torch.tensor([current]))   # full forward()',
        '    next_id = int(out.logits[0,-1].argmax())',
        '    if next_id in eos_ids and stop_on_eos:',
        '        break',
        '    current.append(next_id)',
      ], current: 1,
      footer: 'each iteration triggers a complete model.forward() · context grows' },
  ];
})();

const codeBodyEl   = document.getElementById('code-body');
const codeLabelEl  = document.getElementById('code-label');
const codeSrcEl    = document.getElementById('code-src');
const codeFooterEl = document.getElementById('code-footer');

function highlightCode(line){
  // Tiny syntax highlighter — keywords, strings, comments, self.
  return escHtml(line)
    .replace(/(#.*)$/g, '<span class="code-comment">$1</span>')
    .replace(/\b(def|for|if|in|return|break|else|class|with|import|from|and|or|not)\b/g,
             '<span class="code-keyword">$1</span>')
    .replace(/\b(self)\b/g, '<span class="code-self">$1</span>')
    .replace(/\b(\d+)\b/g, '<span class="code-num">$1</span>')
    .replace(/(&#39;[^&]*?&#39;|&quot;[^&]*?&quot;)/g, '<span class="code-string">$1</span>');
}

function pickBeat(t){
  for (const b of CODE_BEATS){ if (t < b.until) return b; }
  return CODE_BEATS[CODE_BEATS.length - 1];
}

function dynamicFooter(beat, t){
  // Specialize footers that need data from the current animation moment.
  if (beat.header === 'GPT2Block.forward'){
    const layerProgress = Math.min(1, Math.max(0, (t - T_ACT1 - 3) / LAYER_WAVE));
    const cur = Math.min(N_LAYERS, Math.max(1, Math.ceil(layerProgress * N_LAYERS)));
    return `block <b>${cur}</b> of <b>${N_LAYERS}</b> · ‖h‖ rising layer by layer · norm bars on left`;
  }
  if (beat.header === 'inference engine · generation loop' && t >= T_ACT2){
    const stepIdx = Math.min(D.loop_steps.length, Math.max(1, Math.ceil((t - T_ACT2) / STEP_DUR)));
    return `step <b>${stepIdx}</b> of <b>${D.loop_steps.length}</b> · each step = one full forward() over growing context`;
  }
  return beat.footer;
}

function updateCodeStrip(t){
  const beat = pickBeat(t);
  codeLabelEl.textContent = (beat.scope === 'inside' ? '⌥ inside the model · ' : '⊘ external · ') + beat.header;
  codeLabelEl.classList.toggle('external', beat.scope === 'external');
  codeSrcEl.textContent = beat.src;
  codeBodyEl.innerHTML = beat.lines.map((line, i) => {
    const klass = (i === beat.current ? 'code-line current' : 'code-line');
    return `<span class="${klass}">${highlightCode(line)}</span>`;
  }).join('');
  codeFooterEl.innerHTML = dynamicFooter(beat, t);
}

function render(){
  showScene(t);
  updateAct1(t);
  updateAct2(t);
  updateAct3(t);
  updateEnd(t);
  updateCodeStrip(t);
  actPill.textContent = currentAct(t);
  scrub.value = String(Math.round(t * 100));
  document.getElementById('time-display').textContent = `${t.toFixed(1)}s / ${T_END.toFixed(1)}s`;
}

// ── Controls ───────────────────────────────────────────────────────────
const playBtn = document.getElementById('play-btn');
function setPlayIcon(){ playBtn.textContent = playing ? '⏸' : '⏵'; }
playBtn.addEventListener('click', ()=>{
  if (t >= T_END){ t = 0; lastShownStep = -1; }
  playing = !playing;
  lastFrame = performance.now();
  setPlayIcon();
});

document.getElementById('restart-btn').addEventListener('click', ()=>{
  t = 0; lastShownStep = -1; playing = true; setPlayIcon();
});

scrub.addEventListener('input', e => {
  t = Number(e.target.value) / 100;
  playing = false;
  setPlayIcon();
});

document.getElementById('speed').addEventListener('change', e => {
  speed = Number(e.target.value);
});

// Space toggles play/pause
document.addEventListener('keydown', e => {
  if (e.code === 'Space'){ e.preventDefault(); playBtn.click(); }
  if (e.code === 'KeyR') document.getElementById('restart-btn').click();
});

// ── Helpers ────────────────────────────────────────────────────────────
function escHtml(s){return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

// Kickoff
lastFrame = performance.now();
setPlayIcon();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def render(trace, cfg: dict[str, Any] | None = None, out_path: Path | str | None = None) -> Path:
    """Write a self-contained animated HTML page visualizing the trace."""
    viz = _build_anim_payload(trace)
    data_json = json.dumps(viz, ensure_ascii=False, default=str).replace("</", "<\\/")

    if out_path is None:
        out_dir = Path((cfg or {}).get("out_dir", "./out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_slug(trace.model_id)}__{_slug(trace.prompt)}__anim_v2.html"
    out_path = Path(out_path)

    title = f"{trace.model_id} — {trace.prompt[:40]}"
    html = (_HTML
            .replace("__DATA__", data_json)
            .replace("__TITLE__", _html_escape(title)))
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace('"', "&quot;"))


def _slug(s: str, max_len: int = 40) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in s.strip())
    return cleaned[:max_len].strip("_") or "x"
