"""Animated renderer V3 — slideshow layout with system-stack sidebar.

Design (post-feedback from v2):
  - Top bar          : model · prompt
  - Pipeline strip   : 9-step horizontal indicator (1=input → 9=done)
  - Body (2 columns) : per-step content card (left) + system-stack sidebar (right)
  - Controls         : prev / play-pause / next / speed / scrubber

The 3-layer sidebar is the anchor — it shows where in the inference stack
each step lives:
   Frontend            (UI capture, output display)
   Inference Runtime   (tokenizer, sampling, generation loop)
   Model               (embedding, attention, FFN, LM head)

When a step is active, its specific component glows inside the right layer,
so the viewer sees the highlight HOP between layers — e.g. softmax happens
in the Runtime layer, NOT inside the Model.

Data-overlay convention: layout is static; the trace's JSON payload is
injected and per-step renderers pull what they need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _short_model_slug(model_id: str) -> str:
    last = model_id.rsplit("/", 1)[-1]
    head = last.split("-", 1)[0].split(".", 1)[0]
    slug = "".join(c.lower() if c.isalnum() else "_" for c in head).strip("_")
    return slug or "model"


# ── Data shaping ───────────────────────────────────────────────────────────

def _build_payload(trace) -> dict[str, Any]:
    m = trace.model_meta or {}

    # Full prompt-token embeddings (post v5: collector keeps the full
    # hidden_size vectors for prompt tokens). Payload is small even for
    # 50k vocab × ~10 tokens × 4 bytes ≈ tens of KB.
    emb_full = trace.embeddings_token.tolist()

    # Per-token-per-layer norms (for Step 4 heatmap).
    hidden_norms_grid = trace.hidden_norms.tolist()

    # Top-K logits & probs (for Step 5/6/7).
    logits_top = [
        {"token": tok, "id": int(tid), "logit": float(v)}
        for tok, tid, v in zip(
            trace.logits_top_tokens, trace.logits_top_ids, trace.logits_top_values, strict=False
        )
    ][:10]
    probs_top = [
        {"token": tok, "id": int(tid), "prob": float(v)}
        for tok, tid, v in zip(
            trace.probs_top_tokens, trace.probs_top_ids, trace.probs_top_values, strict=False
        )
    ][:10]

    # Per-generation-step (for Step 8 loop view).
    per_step = []
    per_token_ms = trace.timings.get("per_token_ms", [])
    for i, s in enumerate(trace.generation):
        per_step.append({
            "step": int(s["step"]),
            "token": s["token"],
            "id": int(s["id"]),
            "prob": float(s["prob"]),
            "is_eos": bool(s.get("is_eos", False)),
            "ctx_len": int(s["ctx_len"]),
            "ms": float(per_token_ms[i]) if i < len(per_token_ms) else 0.0,
            "top_alts": [{"token": a["token"], "prob": float(a["prob"])}
                         for a in s.get("top_alts", [])[:5]],
        })

    gen_text = getattr(trace, "generation_text", None) or "".join(
        s["token"] for s in trace.generation if not s.get("is_eos")
    )

    return {
        "model_id": trace.model_id,
        "prompt": trace.prompt,
        "system_prompt": getattr(trace, "system_prompt", None),
        "model_meta": m,
        "tokens": trace.tokens,
        "token_ids": list(trace.token_ids),
        "embeddings": emb_full,
        "hidden_norms_grid": hidden_norms_grid,
        "hidden_last": trace.hidden_last.tolist(),
        "final_hidden_full": trace.final_hidden_full.tolist(),
        "lm_head_top_rows": trace.lm_head_top_rows.tolist(),
        "logits_top": logits_top,
        "probs_top": probs_top,
        "loop_steps": per_step,
        "generation_text": gen_text,
        "gen_params": trace.gen_params,
        "embedding_link": f"embeddings_{_short_model_slug(trace.model_id)}.html",
    }


# ── HTML template ──────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm trace v3 — __TITLE__</title>
<style>
:root{
  --bg:#080b12;--surface:#0e1220;--surface2:#151b2e;--surface3:#1c2236;
  --border:#1f2a45;--text:#cdd6f0;--muted:#5a6888;
  --teal:#00d4aa;--amber:#ffb800;--coral:#ff6b6b;--blue:#4da6ff;--purple:#c084fc;
  --skip:#3a4259;
  --mono:'SF Mono',Menlo,monospace;--sans:system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.5}

/* ── Stage grid ──────────────────────────────────────────────────── */
.stage{display:grid;grid-template-rows:auto 76px 1fr 64px;height:100vh}

/* ── Top bar (2 rows: identity, then arch chips) ─────────────────── */
.top-bar{display:flex;flex-direction:column;justify-content:center;gap:5px;
  padding:9px 28px;border-bottom:1px solid var(--border);
  font-family:var(--mono);font-size:12px;color:var(--muted);
  min-height:52px}
.top-bar .row{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}
.top-bar .model{color:var(--teal);font-weight:700;font-size:13px}
.top-bar .sys{color:var(--purple)}
.top-bar .prompt{color:var(--amber)}
.top-bar .spacer{flex:1}
.top-bar .stamp{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.06em}

.stat-chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.stat-chip{position:relative;display:inline-block;padding:2px 8px;
  border-radius:4px;background:var(--surface2);border:1px solid var(--border);
  font-family:var(--mono);font-size:10.5px;color:var(--text);cursor:help;
  transition:all .15s}
.stat-chip:hover{border-color:var(--teal);color:var(--teal);
  background:rgba(0,212,170,0.06)}
.stat-chip.arch{border-color:var(--blue);color:var(--blue)}
.stat-chip.arch:hover{border-color:var(--teal);color:var(--teal)}
.stat-chip::after{content:attr(data-tip);position:absolute;
  top:calc(100% + 8px);left:0;
  background:var(--surface3);border:1px solid var(--teal);border-radius:6px;
  padding:8px 12px;font-size:11px;color:var(--text);
  white-space:normal;width:300px;text-align:left;line-height:1.5;
  z-index:100;box-shadow:0 4px 16px rgba(0,0,0,0.6);
  opacity:0;pointer-events:none;transform:translateY(-4px);
  transition:opacity .15s, transform .15s;font-weight:400}
.stat-chip:hover::after{opacity:1;transform:translateY(0)}

/* ── Pipeline indicator (9 steps, horizontal) ─────────────────────── */
.pipeline{display:flex;align-items:center;justify-content:center;gap:0;
  padding:10px 28px;border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(8,11,18,1),rgba(14,18,32,1))}
.pipe-step{display:flex;flex-direction:column;align-items:center;gap:4px;
  cursor:pointer;min-width:80px;position:relative;transition:opacity .25s}
.pipe-step .row{display:flex;align-items:center;gap:0}
.pipe-step .dot{width:18px;height:18px;border-radius:50%;
  background:var(--surface2);border:2px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:9px;font-weight:700;color:var(--muted);
  transition:all .35s;z-index:2;position:relative}
.pipe-step .conn{height:2px;width:34px;background:var(--border);transition:background .35s}
.pipe-step:first-child .conn-before{display:none}
.pipe-step .lbl{font-family:var(--mono);font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.05em;margin-top:4px;text-align:center;
  transition:color .25s}
.pipe-step.done .dot{background:var(--surface3);border-color:var(--blue);color:var(--blue)}
.pipe-step.done .conn{background:var(--blue)}
.pipe-step.active .dot{background:var(--teal);border-color:var(--teal);color:#000;
  box-shadow:0 0 14px rgba(0,212,170,0.6);transform:scale(1.15)}
.pipe-step.active .lbl{color:var(--teal);font-weight:700}
.pipe-step:hover .lbl{color:var(--text)}

/* ── Body (left content / right sidebar) ──────────────────────────── */
.body{display:grid;grid-template-columns:1fr 320px;overflow:hidden}

/* ── Per-step content card (left) ─────────────────────────────────── */
.content{padding:24px 32px;overflow-y:auto;border-right:1px solid var(--border)}
.step-num{font-family:var(--mono);font-size:11px;color:var(--teal);
  letter-spacing:.15em;text-transform:uppercase;margin-bottom:6px;font-weight:700}
.step-title{font-size:24px;font-weight:800;margin-bottom:6px}
.step-desc{font-family:var(--mono);font-size:13px;color:var(--muted);
  line-height:1.6;margin-bottom:18px;max-width:760px}
.step-desc b{color:var(--text)}
.step-visual{margin-top:14px}
.step-code{margin-top:18px;padding:12px 16px;background:var(--surface2);
  border-left:3px solid var(--purple);border-radius:6px;
  font-family:var(--mono);font-size:12px;color:var(--text);
  overflow-x:auto}
.step-code .label{font-family:var(--mono);font-size:9.5px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;display:block}
.step-code .code-body{margin:0;font-family:var(--mono);font-size:12px;
  color:var(--text);white-space:pre;line-height:1.6}
.step-code .code-body .comment{color:var(--muted)}
.step-code .code-link{margin-top:10px;padding-top:8px;
  border-top:1px dashed var(--border);font-family:var(--mono);font-size:10.5px;
  color:var(--muted)}
.step-code .code-link a{color:var(--blue);text-decoration:none;
  border-bottom:1px dashed var(--blue);padding-bottom:1px;margin:0 2px}
.step-code .code-link a:hover{color:var(--teal);border-bottom-color:var(--teal)}
.step-code .code-link .dim{color:var(--muted);font-size:10px;margin:0 4px}

/* big text variations */
.big-prompt{font-family:var(--mono);font-size:32px;color:var(--amber);
  padding:32px 0;text-align:center;letter-spacing:.02em}

.token-chips{display:flex;flex-wrap:wrap;gap:18px;margin-top:18px;
  justify-content:center;align-items:flex-start}
.token-chip{display:flex;flex-direction:column;align-items:center;gap:6px}
.token-chip .t{padding:10px 16px;border-radius:6px;background:var(--surface2);
  border:1px solid var(--border);font-family:var(--mono);font-size:15px;
  color:var(--teal);font-weight:600}
.token-chip .id{font-family:var(--mono);font-size:14px;color:var(--amber);font-weight:700;
  letter-spacing:.02em}

/* ── System-stack sidebar (right) ─────────────────────────────────── */
.stack{padding:18px 18px;overflow-y:auto;
  background:linear-gradient(180deg,#11182a 0%,#161e36 100%);
  border-left:1px solid var(--border);
  box-shadow:inset 1px 0 0 rgba(0,212,170,0.04)}
.stack-title{font-family:var(--mono);font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px;font-weight:700}
.layer{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;margin-bottom:10px;transition:all .35s;position:relative}
.layer.skipped{opacity:.4;border-style:dashed}
.layer.active{border-color:var(--teal);box-shadow:0 0 16px rgba(0,212,170,0.25);
  background:linear-gradient(135deg,var(--surface),rgba(0,212,170,0.05))}
.layer-head{display:flex;align-items:baseline;gap:8px;margin-bottom:6px}
.layer-icon{font-size:14px}
.layer-name{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--text)}
.layer.skipped .layer-name{color:var(--muted)}
.layer.active .layer-name{color:var(--teal)}
.layer-tag{font-family:var(--mono);font-size:9px;color:var(--muted);
  letter-spacing:.06em;margin-left:auto}
.layer.skipped .layer-tag::after{content:'· not used';color:var(--coral);margin-left:4px}
.layer-items{display:flex;flex-direction:column;gap:3px}
.layer-item{font-family:var(--mono);font-size:10.5px;color:var(--muted);
  padding:3px 0;display:flex;align-items:center;gap:6px;
  transition:color .25s}
.layer-item .marker{font-size:8px;color:var(--surface3);transition:color .25s}
/* small step-number prefix — empty for items not mapped to a step */
.layer-item .step-num{font-family:var(--mono);font-size:9px;
  min-width:18px;text-align:center;
  color:var(--muted);background:var(--surface3);
  border:1px solid var(--border);border-radius:3px;
  padding:0 4px;font-weight:700;letter-spacing:0;
  flex-shrink:0;transition:all .25s}
.layer-item .step-num:empty{visibility:hidden}
.layer-item.done{color:var(--text)}
.layer-item.done .marker{color:var(--blue)}
.layer-item.done .step-num{color:var(--blue);
  background:rgba(77,166,255,0.10);border-color:rgba(77,166,255,0.40)}
.layer-item.active{color:var(--teal);font-weight:600}
.layer-item.active .marker{color:var(--teal)}
.layer-item.active .step-num{color:#001513;background:var(--teal);
  border-color:var(--teal)}
.layer-item.active::after{content:'◀ now';font-size:9px;color:var(--teal);
  margin-left:auto;font-weight:700;letter-spacing:.06em}

/* arrow between layers */
.layer-arrow{text-align:center;color:var(--muted);font-size:10px;margin:-2px 0}

/* compact layer (used for "skipped" layers — single-line summary) */
.layer.compact{padding:8px 12px}
.layer.compact .layer-items{display:none}
.layer.compact .layer-head{margin-bottom:0;flex-wrap:wrap;align-items:center}
.layer.compact .layer-summary{font-family:var(--mono);font-size:10px;color:var(--muted);
  flex:1 1 100%;margin-top:4px;line-height:1.5}

/* embed step — single-line per-token rows with inline expand trigger */
.embed-rows{display:flex;flex-direction:column;gap:6px;margin-top:14px}
.embed-row{background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;overflow:hidden;
  transition:border-color .2s, box-shadow .2s}
.embed-row[open]{border-color:var(--teal);box-shadow:0 0 14px rgba(0,212,170,0.18)}
.embed-row summary{display:flex;align-items:center;gap:10px;
  padding:9px 14px;cursor:pointer;list-style:none;user-select:none;
  overflow-x:auto;white-space:nowrap}
.embed-row summary::-webkit-details-marker{display:none}
.embed-row summary:hover{background:rgba(0,212,170,0.04)}
.embed-row[open] summary{border-bottom:1px solid var(--border)}
.embed-row .meta{font-family:var(--mono);font-size:12px;color:var(--text);flex-shrink:0}
.embed-row .meta .tok{color:var(--teal);font-weight:700}
.embed-row .meta .id{color:var(--amber);font-weight:700}
.embed-row .meta .dim{color:var(--muted)}
.embed-row .preview-cells{display:flex;gap:2px;flex-shrink:0}
.embed-row .more-link{font-family:var(--mono);font-size:11px;color:var(--blue);
  border-bottom:1px dashed var(--blue);padding-bottom:1px;
  flex-shrink:0;margin-left:6px;font-weight:600}
.embed-row .more-link:hover{color:var(--teal);border-bottom-color:var(--teal)}
.embed-row:not([open]) .more-link::before{content:'▶ '}
.embed-row[open] .more-link::before{content:'▼ ';color:var(--teal)}
.embed-row[open] .more-link{color:var(--teal);border-bottom-color:var(--teal)}
.embed-row .full-grid{padding:14px 16px;display:flex;flex-wrap:wrap;gap:2px;
  max-height:480px;overflow-y:auto;background:var(--surface)}
.embed-row .vc{display:inline-block;width:46px;height:26px;border-radius:3px;
  font-family:var(--mono);font-size:10px;text-align:center;line-height:26px;
  font-weight:600;flex-shrink:0}

/* ── Step 5: LM head similarity-search ─────────────────────────── */
.lm-frame{font-family:var(--mono);font-size:12.5px;color:var(--text);
  line-height:1.6;margin-bottom:14px;max-width:1100px}
.lm-frame b{color:var(--teal)}
.lm-frame .lm-eq{font-family:var(--mono);background:var(--surface3);
  padding:2px 8px;border-radius:3px;color:var(--teal);
  border:1px solid var(--border);white-space:nowrap}
.lm-frame-sub{font-size:12px;color:var(--muted);line-height:1.6;margin-top:8px;
  padding:8px 12px;background:var(--surface2);border-left:2px solid var(--blue);
  border-radius:4px}
.lm-frame-sub b{color:var(--text)}
.lm-divider{font-family:var(--mono);font-size:11.5px;color:var(--muted);
  text-align:center;margin:14px 0 10px;letter-spacing:.04em}
.lm-divider .lm-divider-arrow{display:inline-block;color:var(--teal);
  font-size:16px;margin-right:8px}
.lm-candidates{display:flex;flex-direction:column;gap:6px;max-width:1100px}
.lm-cand{background:var(--surface);border:1px solid var(--border);
  border-radius:6px;transition:border-color .15s, box-shadow .15s;
  overflow:hidden}
.lm-cand.winner{border-color:rgba(0,212,170,0.6);
  box-shadow:0 0 14px rgba(0,212,170,0.12)}
.lm-cand summary{display:flex;align-items:center;gap:10px;
  padding:8px 12px;cursor:pointer;list-style:none;user-select:none;
  font-family:var(--mono);font-size:11.5px}
.lm-cand summary::-webkit-details-marker{display:none}
.lm-cand summary:hover{background:rgba(0,212,170,0.04)}
.lm-cand-rank{min-width:32px;color:var(--muted);font-weight:700}
.lm-cand.winner .lm-cand-rank{color:var(--teal);font-size:14px}
.lm-cand-tok{min-width:90px;color:var(--text);font-weight:700;
  background:var(--surface2);padding:2px 8px;border-radius:3px;
  border:1px solid var(--border)}
.lm-cand.winner .lm-cand-tok{color:var(--teal);border-color:var(--teal)}
.lm-cand-id{min-width:54px;font-size:10px;color:var(--muted)}
.lm-cand-cells{display:flex;gap:1px;flex-shrink:0}
.lm-cand-cells .vc{width:32px;height:18px;font-size:9px;line-height:18px}
.lm-cand-more{font-size:10px;color:var(--blue);margin-left:6px;
  border-bottom:1px dashed var(--blue);padding-bottom:1px}
.lm-cand[open] .lm-cand-more{color:var(--teal);border-bottom-color:var(--teal)}
.lm-cand-eq{margin-left:auto;font-size:10.5px;color:var(--muted)}
.lm-cand-logit{font-family:var(--mono);font-size:13px;font-weight:700;
  color:var(--text);min-width:70px;text-align:right}
.lm-cand.winner .lm-cand-logit{color:var(--teal)}
.lm-cand-full{padding:12px 14px;background:var(--surface);
  display:flex;flex-wrap:wrap;gap:2px;
  max-height:320px;overflow-y:auto;border-top:1px solid var(--border)}
.lm-cand-full .vc{width:46px;height:26px}
.lm-callouts{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0;
  font-family:var(--mono);font-size:11px;color:var(--muted)}
.lm-callouts .pill{background:var(--surface2);border:1px solid var(--border);
  border-radius:4px;padding:4px 9px;color:var(--text)}
.lm-callouts .pill b{color:var(--teal);margin:0 4px}

/* ── Step 4: residual-stream heatmap ──────────────────────────── */
.layer-intro{font-family:var(--mono);font-size:12.5px;color:var(--text);
  line-height:1.6;margin-bottom:12px;max-width:1100px}
.layer-intro b{color:var(--teal)}
.layer-intro .muted{color:var(--muted)}
.layer-stack{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px;font-family:var(--mono);
  max-width:1100px;overflow-x:auto;max-height:62vh;overflow-y:auto}
.layer-stack .col-axis{display:flex;align-items:center;gap:6px;
  padding-left:108px;margin-bottom:6px;color:var(--muted);font-size:10px;
  letter-spacing:.05em}
.layer-stack .col-axis .axis-label{margin-left:auto;margin-right:8px}
.layer-row{display:flex;align-items:center;gap:6px;padding:1px 0;
  border-radius:3px;transition:background .12s}
.layer-row:hover{background:rgba(0,212,170,0.04)}
.layer-row.input-row .lr-label{color:var(--blue)}
.layer-row.final-row .lr-label{color:var(--teal)}
.lr-label{min-width:102px;font-size:10.5px;color:var(--text);
  text-align:right;padding-right:6px;white-space:nowrap}
.lr-label .lr-idx{color:var(--muted);margin-right:5px}
.lr-cells{display:flex;gap:1px;flex-shrink:0}
.lr-cells .hc{display:inline-block;width:30px;height:18px;border-radius:2px;
  font-family:var(--mono);font-size:9.5px;text-align:center;line-height:18px;
  font-weight:600;flex-shrink:0;cursor:default}
.lr-norm-wrap{display:flex;align-items:center;gap:6px;margin-left:10px;
  min-width:140px;flex-shrink:0}
.lr-norm-bar{flex:1;height:8px;background:var(--surface3);border-radius:2px;
  overflow:hidden;position:relative}
.lr-norm-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));
  border-radius:2px}
.lr-norm-num{font-size:10px;color:var(--muted);min-width:42px;text-align:right;
  font-variant-numeric:tabular-nums}
.layer-callouts{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 0;
  font-family:var(--mono);font-size:11px;color:var(--muted)}
.layer-callouts .pill{background:var(--surface2);border:1px solid var(--border);
  border-radius:4px;padding:4px 9px;color:var(--text)}
.layer-callouts .pill b{color:var(--teal);margin-right:4px}
.layer-schematic-wrap{margin-top:14px;max-width:1100px}
.layer-schematic-wrap > summary{cursor:pointer;list-style:none;
  font-family:var(--mono);font-size:12px;color:var(--blue);
  padding:8px 12px;background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;user-select:none}
.layer-schematic-wrap > summary::-webkit-details-marker{display:none}
.layer-schematic-wrap > summary::before{content:'▶ ';color:var(--teal)}
.layer-schematic-wrap[open] > summary::before{content:'▼ '}
.layer-schematic-wrap > summary:hover{border-color:var(--teal)}
.layer-schematic-wrap[open] > summary{border-color:var(--teal);
  border-bottom-left-radius:0;border-bottom-right-radius:0}
.layer-schematic-body{border:1px solid var(--teal);border-top:none;
  border-bottom-left-radius:8px;border-bottom-right-radius:8px;
  padding:14px 16px;background:rgba(0,212,170,0.02)}

/* ── Step 4 (legacy): transformer block schematic, now collapsible ─ */
.block-intro{font-family:var(--mono);font-size:12.5px;color:var(--text);
  line-height:1.6;margin-bottom:14px;max-width:760px}
.block-intro b{color:var(--teal)}
.block-schematic{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 18px;display:flex;flex-direction:column;
  font-family:var(--mono);max-width:760px}
.block-port{background:var(--surface3);border:1px solid var(--blue);
  border-radius:6px;padding:8px 14px;display:flex;align-items:baseline;gap:12px}
.block-port.output{border-color:var(--teal)}
.block-port .port-label{font-size:10px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.1em;min-width:54px;font-weight:700}
.block-port .port-shape{font-size:12px;color:var(--blue);font-weight:600}
.block-port.output .port-shape{color:var(--teal)}
.block-arrow{color:var(--muted);text-align:center;font-size:16px;padding:4px 0}
.block-op{background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;margin:6px 0;
  transition:border-color .2s, box-shadow .2s}
.block-op summary{display:flex;align-items:center;gap:12px;
  padding:10px 14px;cursor:pointer;list-style:none;user-select:none}
.block-op summary::-webkit-details-marker{display:none}
.block-op summary:hover{background:rgba(0,212,170,0.04)}
.block-op[open]{border-color:var(--teal);box-shadow:0 0 14px rgba(0,212,170,0.18)}
.block-op .op-icon{color:var(--teal);font-size:9px;
  transition:transform .2s;display:inline-block;width:10px}
.block-op[open] .op-icon{transform:rotate(90deg)}
.block-op .op-name{font-size:13px;color:var(--text);font-weight:700;min-width:220px}
.block-op .op-desc{font-size:11px;color:var(--muted);flex:1}
.block-op .op-shape{font-size:11px;color:var(--blue);white-space:nowrap;
  background:var(--surface3);padding:2px 8px;border-radius:3px}
.block-op .op-math{margin:0 14px 12px 14px;padding:12px 14px;
  background:var(--surface3);border-radius:4px;
  font-family:var(--mono);font-size:11px;line-height:1.6;
  color:var(--text);white-space:pre;overflow-x:auto}
.block-op .op-math .comment{color:var(--muted)}
.block-residual-add{background:rgba(192,132,252,0.08);
  border:1px dashed var(--purple);border-radius:4px;
  padding:6px 12px;margin:6px 0;
  font-family:var(--mono);font-size:11px;color:var(--purple);
  font-weight:600;text-align:center;letter-spacing:.04em}
.block-residual-add .small{font-size:10px;color:var(--muted);
  font-weight:400;letter-spacing:0;margin-left:6px}
.block-multiplier{background:linear-gradient(180deg,transparent,rgba(0,212,170,0.04));
  border:1px solid rgba(0,212,170,0.4);border-radius:8px;
  padding:14px 18px;margin-top:18px;max-width:760px;
  font-family:var(--mono);font-size:12px;color:var(--text);line-height:1.6}
.block-multiplier .x{color:var(--teal);font-size:24px;font-weight:700;margin-right:6px}

/* ── Controls ─────────────────────────────────────────────────────── */
.controls{display:flex;align-items:center;gap:14px;padding:0 28px;
  border-top:1px solid var(--border);background:rgba(8,11,18,0.95);
  backdrop-filter:blur(8px)}
.btn{background:var(--surface2);border:1px solid var(--border);
  color:var(--text);width:38px;height:38px;border-radius:50%;cursor:pointer;
  font-size:14px;font-weight:700;display:flex;align-items:center;justify-content:center;
  transition:all .15s}
.btn:hover{border-color:var(--teal);color:var(--teal)}
.btn.primary{background:var(--teal);color:#000;border-color:var(--teal);width:46px;height:46px;font-size:16px}
.btn.primary:hover{opacity:.85;color:#000}
.step-pill{font-family:var(--mono);font-size:11px;color:var(--teal);padding:5px 12px;
  background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);
  border-radius:14px;letter-spacing:.08em;text-transform:uppercase;font-weight:700}
.scrub-wrap{flex:1}
.scrub{width:100%;accent-color:var(--teal);height:4px;cursor:pointer}
.speed{background:var(--surface2);color:var(--text);border:1px solid var(--border);
  border-radius:5px;padding:5px 8px;font-family:var(--mono);font-size:11px;cursor:pointer}
.time-display{font-family:var(--mono);font-size:11px;color:var(--muted);min-width:88px;text-align:right}

/* ── Misc utilities ─────────────────────────────────────────────── */
.placeholder{display:flex;align-items:center;justify-content:center;
  min-height:200px;background:var(--surface2);border:1px dashed var(--border);
  border-radius:10px;color:var(--muted);font-family:var(--mono);font-size:11px}
.note{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:10px;
  padding:8px 12px;background:var(--surface2);border-left:2px solid var(--blue);
  border-radius:4px}
.note b{color:var(--text)}
</style></head>
<body>
<div class="stage">

  <!-- ── Top bar (identity row + arch chip row) ────────────────── -->
  <div class="top-bar">
    <div class="row">
      <span class="model" id="t-model"></span>
      <span class="sys" id="t-sys"></span>
      <span class="prompt" id="t-prompt"></span>
      <span class="spacer"></span>
      <span class="stamp">animated_v3 · slideshow + stack</span>
    </div>
    <div class="row stat-chips" id="t-chips"></div>
  </div>

  <!-- ── Pipeline indicator ─────────────────────────────────────── -->
  <div class="pipeline" id="pipeline"></div>

  <!-- ── Body: content + stack sidebar ──────────────────────────── -->
  <div class="body">
    <div class="content" id="content">
      <!-- per-step content rendered here -->
    </div>
    <aside class="stack">
      <div class="stack-title">where am I in the stack?</div>
      <div id="stack-layers"></div>
    </aside>
  </div>

  <!-- ── Controls ───────────────────────────────────────────────── -->
  <div class="controls">
    <button class="btn" id="btn-prev" title="previous step">⏮</button>
    <button class="btn primary" id="btn-play" title="auto-advance">⏵</button>
    <button class="btn" id="btn-next" title="next step">⏭</button>
    <span class="step-pill" id="step-pill">step 1 / 9</span>
    <div class="scrub-wrap">
      <input type="range" class="scrub" id="scrub" min="0" max="8" value="0" step="1">
    </div>
    <select class="speed" id="speed">
      <option value="2.0">slow (4s/step)</option>
      <option value="1.0" selected>normal (3s/step)</option>
      <option value="0.5">fast (1.5s/step)</option>
    </select>
    <span class="time-display" id="time-display">step 1</span>
  </div>

</div>

<script>
const D = __DATA__;

// ── Top bar fields ─────────────────────────────────────────────────────
document.getElementById('t-model').textContent = D.model_id;
document.getElementById('t-sys').textContent = D.system_prompt
  ? `system: ${JSON.stringify(D.system_prompt)}` : '';
document.getElementById('t-prompt').textContent = `prompt: ${JSON.stringify(D.prompt)}`;

// ── Architecture chips with hover tooltips ───────────────────────────
function buildArchChips() {
  const m = D.model_meta || {};
  const fmt = n => !n ? '?' :
    n >= 1e6 ? `${(n/1e6).toFixed(1)}M` :
    n >= 1e3 ? `${(n/1e3).toFixed(0)}k` : String(n);

  const isGqa = (m.n_head && m.n_kv_heads && m.n_head !== m.n_kv_heads);
  const isRope = (m.positional_encoding === 'rope');
  const isRms  = (m.normalization === 'rmsnorm');

  const chips = [
    {
      cls: 'arch',
      label: m.model_type ? `arch: ${m.model_type}` : 'arch: ?',
      tip: `Model architecture family. distilgpt2 / gpt2 are GPT-2 style — learned positional embeddings, LayerNorm, GELU activation, full multi-head attention. Llama-style models use RoPE, RMSNorm, SwiGLU FFN, and often Grouped-Query Attention.`,
    },
    {
      label: `${m.n_layer ?? '?'} layers`,
      tip: `Number of transformer blocks stacked vertically. Each block runs attention + feed-forward once. The same architecture repeats — just with different learned weights per block.`,
    },
    {
      label: isGqa ? `${m.n_head}/${m.n_kv_heads} heads (GQA)` : `${m.n_head ?? '?'} heads`,
      tip: isGqa
        ? `Multi-head attention with Grouped-Query Attention. ${m.n_head} query heads share ${m.n_kv_heads} key/value heads (${m.n_head/m.n_kv_heads}× sharing). Saves memory and compute during inference vs. full multi-head.`
        : `Multi-head attention — ${m.n_head} parallel attention computations per layer, each learning a different focus pattern. Each head has its own Q, K, V projection.`,
    },
    {
      label: `hidden ${m.hidden_size ?? '?'}`,
      tip: `Width of the residual stream. Each token is represented as a vector of ${m.hidden_size} numbers throughout the model. All transformer blocks preserve this dimensionality.`,
    },
    {
      label: `ffn ${fmt(m.ffn_intermediate)}`,
      tip: `Inner dimensionality of the feed-forward network. Each block expands the ${m.hidden_size}-dim vector to ${m.ffn_intermediate}-dim, applies a nonlinearity, then contracts back. Where most of the model's parameters live.`,
    },
    {
      label: `vocab ${fmt(m.vocab_size)}`,
      tip: `Number of distinct tokens the model knows. Larger vocab → rarer words can be single tokens, but the embedding/LM-head matrices grow proportionally.`,
    },
    {
      label: `ctx ${fmt(m.ctx_window)}`,
      tip: `Maximum sequence length the model was trained to handle. Beyond this, position-encoding accuracy collapses. RoPE models can sometimes extrapolate; learned-position models can't.`,
    },
    {
      label: isRope ? 'RoPE' : 'learned pos',
      tip: isRope
        ? `Rotary Position Embeddings. Position info is encoded by ROTATING query/key vectors INSIDE attention — not added at the input. theta=${m.rope_theta || 10000}.`
        : `Learned positional embeddings. A separate ${m.ctx_window} × ${m.hidden_size} table is added to the token embeddings at the input. Token at position N gets the N-th row added.`,
    },
    {
      label: isRms ? 'RMSNorm' : 'LayerNorm',
      tip: isRms
        ? `Root Mean Square Normalization. Like LayerNorm but skips mean subtraction and the bias parameter. Lighter and standard in modern Llama-style models.`
        : `Layer Normalization. Per-token: subtract the mean, divide by std-dev, scale by learned γ, add learned β. Stabilizes training across layers.`,
    },
  ];

  document.getElementById('t-chips').innerHTML =
    chips.map(c => `<span class="stat-chip ${c.cls||''}" data-tip="${esc(c.tip)}">${esc(c.label)}</span>`).join('');
}
buildArchChips();

// ── Helpers ───────────────────────────────────────────────────────────
function esc(s){return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

// Vector cell — single source of truth for both inline preview AND the
// expanded full-grid view in steps 3 (Embed) and 5 (LM head).
function vecCell(v){
  const i = Math.min(1, Math.abs(v));
  const col = v >= 0
    ? `rgba(0,${(140 + i*112)|0},${(120 + i*50)|0},${0.25 + i*0.7})`
    : `rgba(${(140 + i*115)|0},${(80 - i*60)|0},${(80 - i*60)|0},${0.25 + i*0.7})`;
  const textCol = i > 0.5 ? '#001513' : '#cdd6f0';
  const numText = (v >= 0 ? '+' : '') + v.toFixed(2);
  return `<span class="vc" style="background:${col};color:${textCol}">${numText}</span>`;
}

// Build a `<details class="embed-row">` with consistent single-line summary
// + expanded full-vector grid. Used by step 3 (per token) and step 5 (last
// token's hidden state).
function vectorRow(metaHtml, fullVec){
  const previewN  = Math.min(16, fullVec.length);
  const preview   = fullVec.slice(0, previewN).map(vecCell).join('');
  const fullCells = fullVec.map(vecCell).join('');
  const remaining = fullVec.length - previewN;
  return `
    <details class="embed-row">
      <summary>
        <span class="meta">${metaHtml}</span>
        <span class="preview-cells">${preview}</span>
        <span class="more-link">+ ${remaining} more</span>
      </summary>
      <div class="full-grid">${fullCells}</div>
    </details>
  `;
}

// ── Step manifest ─────────────────────────────────────────────────────
// Each entry maps a step to: title, blurb, which sidebar layer/item is active.
const STEPS = [
  { id:'input',    title:'Input',
    blurb:'You typed text. The model is going to chew on this for a while.',
    layer:'frontend', item:'capture' },
  { id:'tokenize', title:'Tokenize',
    blurb:'Text → integer ids. The model never sees characters; only numbers.',
    layer:'runtime',  item:'tokenizer' },
  { id:'embed',    title:'Embed',
    blurb:'Each id is a row index into a giant lookup table. No math yet — just a fetch.',
    layer:'model',    item:'wte' },
  { id:'layers',   title:'Layers',
    blurb:'The vectors flow through N stacked transformer blocks. Each block adds.',
    layer:'model',    item:'transformer' },
  { id:'lm_head',  title:'Logits',
    blurb:'The last token\'s final hidden state is projected by the LM head into one raw score per vocab token.',
    layer:'model',    item:'lm_head' },
  { id:'softmax',  title:'Softmax',
    blurb:'Logits → probabilities. This is OUTSIDE the model. The runtime does it.',
    layer:'runtime',  item:'sampling' },
  { id:'sample',   title:'Sample',
    blurb:'Pick one token from the distribution. argmax for greedy decoding.',
    layer:'runtime',  item:'sampling' },
  { id:'loop',     title:'Loop',
    blurb:'Append the chosen token, go back to step 3. Stop on EOS or max_new_tokens.',
    layer:'runtime',  item:'generation_loop' },
  { id:'done',     title:'Done',
    blurb:'The full output is returned to the caller. End of the line.',
    layer:'frontend', item:'display' },
];

// ── Sidebar stack manifest ────────────────────────────────────────────
const LAYERS = [
  { id:'frontend', icon:'🌐', name:'Frontend / UI',
    tag:'',
    items:[
      { id:'capture', text:'capture user input' },
      { id:'display', text:'display output' },
      { id:'render',  text:'render UI · React · Vue · CLI' },
    ]},
  { id:'runtime', icon:'🛠', name:'Inference Runtime',
    tag:'HuggingFace transformers',
    items:[
      { id:'tokenizer', text:'tokenizer (BPE / sentencepiece)' },
      { id:'forward',   text:'forward-pass driver' },
      { id:'sampling',  text:'sampling: softmax · argmax · top-k · temp' },
      { id:'generation_loop', text:'autoregressive loop · EOS check' },
      { id:'kvcache',   text:'KV cache' },
    ]},
  { id:'model', icon:'🧠', name:'Model',
    tag:'',
    items:[
      { id:'wte',          text:'embedding matrix (wte) · [vocab × hidden]' },
      { id:'wpe',          text:'position embeddings (wpe) — GPT-2 only' },
      { id:'transformer',  text:'transformer block × N (attention + FFN)' },
      { id:'lm_head',      text:'LM head — linear → vocab scores' },
    ]},
];

// ── Build pipeline indicator ──────────────────────────────────────────
const pipelineEl = document.getElementById('pipeline');
pipelineEl.innerHTML = STEPS.map((s, i) => `
  <div class="pipe-step" data-step="${i}">
    <div class="row">
      ${i > 0 ? '<div class="conn"></div>' : ''}
      <div class="dot">${i + 1}</div>
    </div>
    <div class="lbl">${esc(s.title)}</div>
  </div>
`).join('');
pipelineEl.querySelectorAll('.pipe-step').forEach(el => {
  el.addEventListener('click', () => showStep(parseInt(el.dataset.step)));
});

// ── Build sidebar stack ───────────────────────────────────────────────
// Reverse map: itemId → list of 1-based step numbers that reference it.
// Used to label sidebar items with the steps that actually visit them.
const ITEM_STEP_NUMS = {};
STEPS.forEach((s, i) => {
  if (!ITEM_STEP_NUMS[s.item]) ITEM_STEP_NUMS[s.item] = [];
  ITEM_STEP_NUMS[s.item].push(i + 1);
});

const stackEl = document.getElementById('stack-layers');
stackEl.innerHTML = LAYERS.map((L, i) => {
  const cls = ['layer'];
  if (L.skipped) cls.push('skipped');
  if (L.compact) cls.push('compact');
  const itemsHtml = L.items.map(it => {
    const nums = ITEM_STEP_NUMS[it.id] || [];
    const numText = nums.length ? nums.join(',') : '';
    return `
      <div class="layer-item" data-item="${it.id}">
        <span class="marker">●</span>
        <span class="step-num">${numText}</span>
        <span>${esc(it.text)}</span>
      </div>
    `;
  }).join('');
  const summaryHtml = L.compact && L.summary
    ? `<span class="layer-summary">${esc(L.summary)}</span>` : '';
  return `
    <div class="${cls.join(' ')}" data-layer="${L.id}">
      <div class="layer-head">
        <span class="layer-icon">${L.icon}</span>
        <span class="layer-name">${esc(L.name)}</span>
        <span class="layer-tag">${esc(L.tag)}</span>
        ${summaryHtml}
      </div>
      <div class="layer-items">${itemsHtml}</div>
    </div>
    ${i < LAYERS.length - 1 ? '<div class="layer-arrow">↓</div>' : ''}
  `;
}).join('');

// ── Per-step content renderers ────────────────────────────────────────
// Each function returns an HTML string for the .content area. They use
// real D data; layout-first review can scan placeholders where it's
// not yet richly visualized.

function step_input() {
  return `
    <div class="big-prompt">"${esc(D.prompt)}"</div>
    <div class="step-desc" style="text-align:center">
      That's the entire input — a single string of characters.
      Nothing happens to it yet.
    </div>
  `;
}

function step_tokenize() {
  const chips = D.tokens.map((t, i) => `
    <div class="token-chip">
      <div class="t">${esc(t)}</div>
      <div class="id">id ${D.token_ids[i]}</div>
    </div>
  `).join('');
  const code = [
    'ids = tokenizer.encode(prompt)',
    `# BPE merges the text into ${D.tokens.length} pieces, each mapping`,
    `# to one of the ${(D.model_meta.vocab_size||0).toLocaleString()} ids in the vocabulary`,
  ].join('\n');
  return `
    <div class="step-visual"><div class="token-chips">${chips}</div></div>
    <div class="step-code">
      <span class="label">⌥ inside the runtime</span>
      <pre class="code-body">${esc(code)}</pre>
    </div>
  `;
}

function step_embed() {
  const vocab  = (D.model_meta.vocab_size || 0).toLocaleString();
  const hsize  = D.model_meta.hidden_size;
  const tableShape = `[${vocab} × ${hsize}]`;

  const rowsHtml = D.tokens.map((tok, i) => {
    const tid     = D.token_ids[i];
    const fullVec = (D.embeddings && D.embeddings[i]) || [];
    const meta = `<span class="tok">"${esc(tok)}"</span><span class="dim"> · id </span><span class="id">${tid}</span><span class="dim"> · row ${tid} of ${tableShape}</span>`;
    return vectorRow(meta, fullVec);
  }).join('');

  return `
    <div class="step-visual">
      <div style="font-family:var(--mono);font-size:13px;color:var(--text);line-height:1.7;max-width:760px">
        Each token is a <b>row index</b> into the embedding table —
        a <span style="color:var(--blue)">${tableShape}</span> grid.
        Pulling out one row gives a <b>${hsize}-number vector</b>.
        Same token always returns the same row. Think of it like fetching a
        book by its library call number — no math yet, just a lookup.
        Click <b>show all ${hsize} numbers</b> on any row to see the full vector.
      </div>
      <div class="embed-rows">${rowsHtml}</div>
    </div>
    <div class="step-code">
      <span class="label">⌥ inside the model</span>
      <pre class="code-body">${esc(`inputs_embeds = self.wte(input_ids)
# self.wte = nn.Embedding(${vocab}, ${hsize})`)}</pre>
    </div>
    <a href="${esc(D.embedding_link)}" target="_blank"
       style="display:inline-block;margin-top:14px;font-family:var(--mono);font-size:12px;
              color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);padding-bottom:1px">
      explore all ${vocab} rows in 2D semantic space →
    </a>
  `;
}

// Compact heatmap cell — same +/- color logic as vecCell, smaller footprint,
// number rendered only when |v| is large enough to be readable.
function heatCell(v){
  const i = Math.min(1, Math.abs(v));
  const col = v >= 0
    ? `rgba(0,${(140 + i*112)|0},${(120 + i*50)|0},${0.22 + i*0.7})`
    : `rgba(${(140 + i*115)|0},${(80 - i*60)|0},${(80 - i*60)|0},${0.22 + i*0.7})`;
  const textCol = i > 0.55 ? '#001513' : '#cdd6f0';
  const numText = i >= 0.18 ? ((v >= 0 ? '+' : '') + v.toFixed(1)) : '';
  const tip = (v >= 0 ? '+' : '') + v.toFixed(3);
  return `<span class="hc" style="background:${col};color:${textCol}" title="${tip}">${numText}</span>`;
}

function step_layers() {
  const m = D.model_meta || {};
  const hsize  = m.hidden_size || 768;
  const ffn    = m.ffn_intermediate || (4 * hsize);
  const nHead  = m.n_head || 12;
  const headDim = Math.max(1, Math.floor(hsize / nHead));
  const nLayers = m.n_layer || 6;

  // Residual-stream evolution: hidden_last is [n_layer+1 × dims_keep].
  // Row 0 is the input embedding (post pos-add for GPT-2). Rows 1..N are
  // the state after each transformer block. Last row is post-final-LN
  // (this is what the LM head actually reads).
  const hl = D.hidden_last || [];
  const dimsKept = hl.length ? hl[0].length : 0;
  const totalRows = hl.length;     // n_layer + 1, possibly +1 if final LN added

  // Norm of the last token at each layer — comes from the last column of
  // hidden_norms_grid (same indexing: row 0 = input, then per layer).
  const grid = D.hidden_norms_grid || [];
  const lastTokIdx = (D.tokens || []).length - 1;
  const norms = grid.map(row => row[lastTokIdx] || 0);
  const normMax = Math.max(...norms, 1e-9);
  const normMin = Math.min(...norms);
  const normPeakLayer = norms.indexOf(Math.max(...norms));

  // Row label: special-case input (row 0) and final (last row).
  function rowLabel(idx){
    if (idx === 0) return `<span class="lr-idx">in</span>embedding`;
    if (idx === totalRows - 1 && totalRows > nLayers + 1) {
      return `<span class="lr-idx">→</span>final LN`;
    }
    return `<span class="lr-idx">L${idx}</span>after block`;
  }

  function rowClass(idx){
    if (idx === 0) return 'input-row';
    if (idx === totalRows - 1) return 'final-row';
    return '';
  }

  const rowsHtml = hl.map((vec, idx) => {
    const cells = vec.map(heatCell).join('');
    const normPct = (norms[idx] / normMax) * 100;
    return `
      <div class="layer-row ${rowClass(idx)}">
        <div class="lr-label">${rowLabel(idx)}</div>
        <div class="lr-cells">${cells}</div>
        <div class="lr-norm-wrap">
          <div class="lr-norm-bar"><div class="lr-norm-fill" style="width:${normPct.toFixed(1)}%"></div></div>
          <div class="lr-norm-num">${norms[idx].toFixed(2)}</div>
        </div>
      </div>
    `;
  }).join('');

  // Math text per operation. Kept as plain template strings; whitespace
  // matters because they render inside <pre>.
  const lnMath =
`# normalize PER TOKEN (mean & variance over the hidden dim)
mean = x.mean(dim=-1, keepdim=True)
var  = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / sqrt(var + eps)

# learned scale + shift, per channel
out = γ * x_norm + β       # γ, β each [${hsize}]`;

  const attnMath =
`# split input into Q, K, V (one Conv1D produces all three)
qkv = self.c_attn(x)                   # [seq × ${hsize*3}]
Q, K, V = qkv.split(${hsize}, dim=-1)

# reshape into ${nHead} parallel "heads"
Q = Q.view(seq, ${nHead}, ${headDim}).transpose(0,1)   # [${nHead} × seq × ${headDim}]
K = K.view(seq, ${nHead}, ${headDim}).transpose(0,1)
V = V.view(seq, ${nHead}, ${headDim}).transpose(0,1)

# for each head, compute attention:
scores  = Q @ K.transpose(-2,-1) / sqrt(${headDim})   # [seq × seq] per head
scores += causal_mask                                  # upper triangle = -inf
weights = softmax(scores, dim=-1)                      # rows sum to 1
head_o  = weights @ V                                  # [seq × ${headDim}]

# concatenate ${nHead} head outputs and project back to hidden
out = self.c_proj(concat(head_0, ..., head_${nHead-1}))  # [seq × ${hsize}]`;

  const ffnMath =
`# Per-token nonlinear transform. NO cross-token interaction here —
# every token is processed independently.
h1 = self.c_fc(x)        # linear: [seq × ${hsize}] → [seq × ${ffn}]
h2 = gelu(h1)            # GELU activation (smooth ReLU-like)
out = self.c_proj(h2)    # linear: [seq × ${ffn}] → [seq × ${hsize}]`;

  function blockOp(id, name, desc, shape, math){
    return `
      <details class="block-op" data-op="${id}">
        <summary>
          <span class="op-icon">▶</span>
          <span class="op-name">${esc(name)}</span>
          <span class="op-desc">${esc(desc)}</span>
          <span class="op-shape">${esc(shape)}</span>
        </summary>
        <pre class="op-math">${esc(math)}</pre>
      </details>
    `;
  }

  const lastTokDisplay = esc((D.tokens || [])[lastTokIdx] || '?');
  const peakLabel = normPeakLayer === 0
    ? 'input'
    : (normPeakLayer === totalRows - 1 ? 'final LN' : `block ${normPeakLayer}`);

  // Header row: dim labels (0, 4, 8, ..., 28). 32 cells × 30px each.
  const dimTicks = [];
  for (let d = 0; d < dimsKept; d += 4) {
    dimTicks.push(`<span style="display:inline-block;width:${4*30+3*1}px">dim ${d}</span>`);
  }
  const colAxis = `${dimTicks.join('')}<span class="axis-label">‖ residual norm</span>`;

  const callouts = [
    `<span class="pill"><b>${totalRows - 1}</b> transformations stacked (input + ${nLayers} blocks${totalRows > nLayers + 1 ? ' + final LN' : ''})</span>`,
    `<span class="pill"><b>norm</b> ${norms[0].toFixed(2)} → peak ${normMax.toFixed(2)} @ ${peakLabel} → ${norms[totalRows-1].toFixed(2)} (final)</span>`,
    `<span class="pill"><b>showing</b> first ${dimsKept} of ${hsize} dims</span>`,
  ].join('');

  return `
    <div class="step-visual">
      <div class="layer-intro">
        The <b>same vector</b> for the last token “<b>${lastTokDisplay}</b>” passes through
        <b>${nLayers}</b> stacked transformer blocks. Each row below is the same
        ${dimsKept}-dim slice of that vector, <span class="muted">after one more block has touched it</span>.
        Watch how cells change row-to-row — <b>that's a layer "doing something"</b>. The bar on the
        right is the full ${hsize}-dim vector's L2 norm at that depth.
      </div>

      <div class="layer-stack">
        <div class="col-axis">${colAxis}</div>
        ${rowsHtml}
      </div>

      <div class="layer-callouts">${callouts}</div>

      <details class="layer-schematic-wrap">
        <summary>Inside one block — what the model actually does at each row</summary>
        <div class="layer-schematic-body">
          <div class="block-intro">
            Each transformer block runs this fixed pipeline. The block is invoked
            <b>${nLayers}</b> times with different learned weights — same shape going in,
            same shape coming out. Click any operation for the math.
          </div>
          <div class="block-schematic">
            <div class="block-port input">
              <span class="port-label">input</span>
              <span class="port-shape">[seq × ${hsize}]</span>
            </div>
            <div class="block-arrow">↓</div>
            ${blockOp('ln1', 'LayerNorm 1', 'normalize each token’s vector',
                      `[seq × ${hsize}]`, lnMath)}
            <div class="block-arrow">↓</div>
            ${blockOp('attn', 'Multi-head Self-Attention', 'tokens look at each other',
                      `${nHead} heads · head_dim ${headDim}`, attnMath)}
            <div class="block-residual-add">⊕ residual add
              <span class="small">— the original input bypasses attention and is added back</span>
            </div>
            ${blockOp('ln2', 'LayerNorm 2', 'normalize again before FFN',
                      `[seq × ${hsize}]`, lnMath)}
            <div class="block-arrow">↓</div>
            ${blockOp('ffn', 'Feed-Forward Network', 'per-token nonlinear transform',
                      `${hsize} → ${ffn} → ${hsize}`, ffnMath)}
            <div class="block-residual-add">⊕ residual add
              <span class="small">— bypass + add back, again</span>
            </div>
            <div class="block-port output">
              <span class="port-label">output</span>
              <span class="port-shape">[seq × ${hsize}]</span>
            </div>
          </div>
        </div>
      </details>
    </div>
  `;
}

function step_lm_head() {
  const hidden = D.final_hidden_full || [];
  const rows   = D.lm_head_top_rows || [];
  const logits = D.logits_top || [];
  const vocab  = (D.model_meta.vocab_size || 0).toLocaleString();
  const hsize  = D.model_meta.hidden_size;
  const tied   = !!D.model_meta.tied_word_embeddings;

  // Show top-3 candidates side-by-side with hidden state. Each: token,
  // 16-dim preview of W-row, dot-product = logit.
  const N_CANDIDATES = 3;
  const previewN = Math.min(16, hsize);
  const candidates = rows.slice(0, N_CANDIDATES).map((row, i) => {
    const tok = (logits[i] && logits[i].token) || '?';
    const id  = (logits[i] && logits[i].id) || 0;
    const lg  = (logits[i] && logits[i].logit) || 0;
    const cells = row.slice(0, previewN).map(vecCell).join('');
    const fullCells = row.map(vecCell).join('');
    const remain = row.length - previewN;
    const isWinner = i === 0;
    return `
      <details class="lm-cand ${isWinner ? 'winner' : ''}">
        <summary>
          <span class="lm-cand-rank">#${i+1}</span>
          <span class="lm-cand-tok">${esc(tok)}</span>
          <span class="lm-cand-id">id ${id}</span>
          <span class="lm-cand-cells">${cells}</span>
          <span class="lm-cand-more">+ ${remain} more</span>
          <span class="lm-cand-eq">⟨h, w⟩ =</span>
          <span class="lm-cand-logit">${lg >= 0 ? '+' : ''}${lg.toFixed(2)}</span>
        </summary>
        <div class="lm-cand-full">${fullCells}</div>
      </details>
    `;
  }).join('');

  const hiddenMeta = `<span class="dim">hidden state · </span><span class="id">${hsize}</span><span class="dim"> dims · the "query"</span>`;

  const tiedSubFrame = tied
    ? `<b>In this model (tied):</b> it's literally the same matrix as Step 3,
       doing double duty — once as a dictionary on the way in, once as a similarity
       database on the way out.`
    : `<b>In this model (untied):</b> a separately-learned matrix of the same
       shape as Step 3. The model decided that "input meaning" and "output prediction
       key" benefit from different weights, and trained them independently.`;

  return `
    <div class="step-visual">
      <div class="lm-frame">
        The LM head is <b>structurally identical to the embedding table from Step 3</b> —
        same <span class="lm-eq">[${vocab} × ${hsize}]</span> shape, one row per vocab token.
        But it's used for the <b>inverse operation</b>: instead of <i>id → vector</i>
        (lookup), it's <i>vector → which id is closest</i> (similarity search via
        matmul against every row in parallel).
        <div class="lm-frame-sub">${tiedSubFrame}</div>
        <div class="lm-frame-sub">
          <span class="lm-eq">logit[token] = ⟨ hidden_state , W_row[token] ⟩</span>
          — read as <i>"how similar is the hidden state to this token's W-row?"</i>
        </div>
      </div>

      ${vectorRow(hiddenMeta, hidden)}

      <div class="lm-divider">
        <span class="lm-divider-arrow">↓</span>
        ${vocab} dot products in one matmul · top-${N_CANDIDATES} shown below
      </div>

      <div class="lm-candidates">${candidates}</div>

      <div class="lm-callouts">
        <span class="pill">winner: <b>${esc(logits[0]?.token || '?')}</b> · logit ${(logits[0]?.logit || 0).toFixed(2)}</span>
        <span class="pill">${vocab} dot products run in parallel (one matmul)</span>
        <span class="pill">cells where hidden &amp; W-row <b>agree in sign</b> add to the score</span>
      </div>

      <div class="step-code">
        <span class="label">⌥ inside the model · transformers v5.6.2</span>
        <pre class="code-body">${esc(`# 1. Define the LM head — a single linear projection (no bias).
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# 2. Apply it to the final hidden state of every position; we
#    only care about the last position to predict the next token.
logits = self.lm_head(hidden_states)   # → [seq × ${vocab}]
next_token_logits = logits[:, -1, :]   # → [${vocab}]`)}</pre>
        ${(() => {
          const mt = (D.model_meta && D.model_meta.model_type) || '';
          const v = 'v5.6.2';
          let path = null, defL = null, callL = null;
          if (mt === 'gpt2') { path = 'gpt2/modeling_gpt2.py'; defL = 651; callL = 706; }
          else if (mt === 'llama') { path = 'llama/modeling_llama.py'; defL = 438; callL = 487; }
          if (!path) return '';
          const base = `https://github.com/huggingface/transformers/blob/${v}/src/transformers/models/${path}`;
          return `
            <div class="code-link">source ·
              <a href="${base}#L${defL}" target="_blank" rel="noopener">${path}#L${defL}</a>
              <span class="dim">(definition)</span>
              ·
              <a href="${base}#L${callL}" target="_blank" rel="noopener">L${callL}</a>
              <span class="dim">(call site)</span>
            </div>
          `;
        })()}
      </div>
    </div>
  `;
}

function step_softmax() {
  const log = D.logits_top || [];
  const pr  = D.probs_top || [];
  const minL = Math.min(...log.map(l => l.logit));
  const maxL = Math.max(...log.map(l => l.logit));
  const rangeL = (maxL - minL) || 1;
  const logBars = log.map((l, i) => {
    const w = ((l.logit - minL) / rangeL * 100).toFixed(1);
    const c = i === 0 ? '#00d4aa' : '#4da6ff';
    return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;font-family:var(--mono);font-size:11px">
      <span style="min-width:80px;text-align:right;color:var(--text)">${esc(l.token)}</span>
      <span style="flex:1;height:12px;background:var(--surface2);border-radius:2px;overflow:hidden">
        <span style="display:block;width:${w}%;height:100%;background:${c}"></span>
      </span>
      <span style="min-width:50px;color:var(--muted)">${l.logit.toFixed(2)}</span>
    </div>`;
  }).join('');
  const probBars = pr.map((p, i) => {
    const w = Math.max(0.5, p.prob * 100).toFixed(1);
    const c = i === 0 ? '#00d4aa' : (p.prob > 0.15 ? '#ffb800' : '#4da6ff');
    return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;font-family:var(--mono);font-size:11px">
      <span style="min-width:80px;text-align:right;color:${i===0?'#00d4aa':'var(--text)'};font-weight:${i===0?'700':'400'}">${esc(p.token)}</span>
      <span style="flex:1;height:12px;background:var(--surface2);border-radius:2px;overflow:hidden">
        <span style="display:block;width:${w}%;height:100%;background:${c}"></span>
      </span>
      <span style="min-width:50px;color:${i===0?'#00d4aa':'var(--muted)'};font-weight:${i===0?'700':'400'}">${(p.prob*100).toFixed(1)}%</span>
    </div>`;
  }).join('');
  return `
    <div class="step-visual">
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px">A · raw logits (real numbers)</div>
      ${logBars}
      <div style="margin:14px 0;padding:10px 14px;background:var(--surface2);border-left:3px solid var(--teal);border-radius:5px;font-family:var(--mono);font-size:11px">
        <span style="color:var(--teal);font-weight:700">softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)</span>
        <span style="color:var(--muted);margin-left:8px">— turns scores into probabilities summing to 1</span>
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px">B · probabilities (after softmax)</div>
      ${probBars}
    </div>
    <div class="note">
      Softmax happens <b>outside</b> the model — the runtime calls it on the
      logits the model returned. Same with temperature, top-k, top-p sampling.
    </div>
  `;
}

function step_sample() {
  const winner = (D.probs_top || [])[0] || {};
  const other = (D.probs_top || []).slice(1, 6);
  const otherList = other.map(p =>
    `<span style="font-family:var(--mono);font-size:11px;margin-right:14px">
      <span style="color:var(--text)">${esc(p.token)}</span>
      <span style="color:var(--muted)">${(p.prob*100).toFixed(1)}%</span>
    </span>`
  ).join('');
  return `
    <div class="step-visual">
      <div style="text-align:center;padding:24px 0;font-family:var(--mono)">
        <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">argmax picks the biggest</div>
        <div style="font-size:38px;color:var(--teal);font-weight:700;letter-spacing:.02em">${esc(winner.token || '?')}</div>
        <div style="font-size:13px;color:var(--amber);margin-top:6px">${(winner.prob*100||0).toFixed(1)}%  ·  id ${winner.id||0}</div>
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:12px">
        runners-up (didn't win):
      </div>
      <div style="margin-top:6px">${otherList}</div>
    </div>
    <div class="step-code">
      <span class="label">⌥ inside the runtime · greedy decoding</span>
      <pre class="code-body">${esc(`next_id = int(probs.argmax())
# for sampling: next_id = int(torch.multinomial(probs, 1))`)}</pre>
    </div>
  `;
}

function step_loop() {
  const steps = D.loop_steps || [];
  const colors = ['#00d4aa','#ffb800','#4da6ff','#c084fc','#ff6b6b','#34d399','#fb923c','#a78bfa'];
  const chips = steps.map((s, i) => {
    if (s.is_eos) return `<span style="display:inline-block;padding:3px 8px;margin:2px;border-radius:4px;background:rgba(255,107,107,0.2);color:#ff6b6b;font-family:var(--mono);font-size:12px;font-weight:700">⏹ EOS</span>`;
    const c = colors[i % colors.length];
    return `<span style="display:inline-block;padding:3px 8px;margin:2px;border-radius:4px;background:${c}22;border:1px solid ${c}66;color:${c};font-family:var(--mono);font-size:12px">${esc(s.token)}</span>`;
  }).join('');
  const totalMs = steps.reduce((a, s) => a + s.ms, 0);
  return `
    <div class="step-visual">
      <div style="font-family:var(--mono);font-size:12px;color:var(--muted);margin-bottom:8px">
        <span style="color:var(--amber)">prompt</span> + ${steps.length} generated tokens:
      </div>
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 18px;font-family:var(--mono);font-size:13px;line-height:1.9">
        <span style="color:var(--muted)">${esc(D.prompt)}</span>${chips}
      </div>
      <div style="display:flex;gap:24px;margin-top:14px;font-family:var(--mono);font-size:11px;color:var(--muted)">
        <span>steps: <b style="color:var(--teal)">${steps.length}</b></span>
        <span>total: <b style="color:var(--teal)">${totalMs.toFixed(0)} ms</b></span>
        <span>avg: <b style="color:var(--teal)">${(totalMs/Math.max(steps.length,1)).toFixed(1)} ms/step</b></span>
        <span>${steps.some(s => s.is_eos) ? '<span style="color:#ff6b6b;font-weight:700">EOS emitted ✓</span>' : '<span style="color:var(--amber)">hit max_new_tokens (no EOS)</span>'}</span>
      </div>
    </div>
    <div class="step-code">
      <span class="label">⌥ inside the runtime · the autoregressive loop</span>
      <pre class="code-body">${esc(`for step in range(max_new_tokens):
    out = model(torch.tensor([current]))
    next_id = int(out.logits[0, -1].argmax())
    if next_id in eos_ids: break
    current.append(next_id)`)}</pre>
    </div>
  `;
}

function step_done() {
  const text = D.prompt + (D.generation_text || '');
  return `
    <div class="step-visual">
      <div style="font-family:var(--mono);font-size:11px;color:var(--teal);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">final output</div>
      <div style="background:var(--surface);border:2px solid var(--teal);border-radius:10px;padding:18px 24px;font-family:var(--mono);font-size:14px;color:var(--text);line-height:1.7;white-space:pre-wrap">${esc(text)}</div>
    </div>
    <div style="margin-top:24px;padding:16px 22px;background:var(--surface);border:1px solid var(--border);border-radius:10px">
      <div style="font-family:var(--mono);font-size:11px;color:var(--teal);text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;font-weight:700">further reading</div>
      <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px;font-family:var(--mono);font-size:12px">
        <li><a target="_blank" style="color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);padding-bottom:1px"
               href="https://jalammar.github.io/illustrated-transformer/">Jay Alammar — The Illustrated Transformer</a></li>
        <li><a target="_blank" style="color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);padding-bottom:1px"
               href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Karpathy — Let's build GPT, from scratch, in code</a></li>
        <li><a target="_blank" style="color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);padding-bottom:1px"
               href="https://transformer-circuits.pub/2021/framework/index.html">Anthropic — Mathematical Framework for Transformer Circuits</a></li>
        <li><a target="_blank" style="color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);padding-bottom:1px"
               href="https://arxiv.org/abs/2309.17453">Xiao et al. (2023) — Attention Sinks (the first-token spike)</a></li>
        <li><a target="_blank" style="color:var(--blue);text-decoration:none;border-bottom:1px dashed var(--blue);padding-bottom:1px"
               href="https://www.youtube.com/watch?v=eMlx5fFNoYc">3Blue1Brown — But what is a GPT?</a></li>
      </ul>
    </div>
  `;
}

const RENDERERS = {
  input: step_input, tokenize: step_tokenize, embed: step_embed,
  layers: step_layers, lm_head: step_lm_head, softmax: step_softmax,
  sample: step_sample, loop: step_loop, done: step_done,
};

// ── State + show step ─────────────────────────────────────────────────
let currentStep = 0;
let playing = false;
let speed = 1.0;
let autoplayTimer = null;

function showStep(n){
  n = Math.max(0, Math.min(STEPS.length - 1, n));
  currentStep = n;
  const step = STEPS[n];

  // Pipeline highlight
  document.querySelectorAll('.pipe-step').forEach((el, i) => {
    el.classList.remove('done', 'active', 'future');
    if (i < n) el.classList.add('done');
    else if (i === n) el.classList.add('active');
    else el.classList.add('future');
  });

  // Sidebar highlight: items mapped to a previous step → done; current → active.
  // Items can be referenced by multiple steps (e.g. sampling: softmax + sample);
  // active wins over done so the "now" label always points at the current step.
  const doneItems = new Set(STEPS.slice(0, n).map(s => s.item));
  const currentItem = step.item;
  document.querySelectorAll('.layer').forEach(el => {
    el.classList.toggle('active', el.dataset.layer === step.layer);
  });
  document.querySelectorAll('.layer-item').forEach(el => {
    const id = el.dataset.item;
    el.classList.remove('active', 'done');
    if (id === currentItem) el.classList.add('active');
    else if (doneItems.has(id)) el.classList.add('done');
  });

  // Content
  const renderer = RENDERERS[step.id] || (() => '<div class="placeholder">[ content not yet implemented ]</div>');
  document.getElementById('content').innerHTML = `
    <div class="step-num">step ${n + 1} of ${STEPS.length}</div>
    <div class="step-title">${esc(step.title)}</div>
    <div class="step-desc">${esc(step.blurb)}</div>
    ${renderer()}
  `;

  // Controls + scrubber
  document.getElementById('step-pill').textContent = `step ${n + 1} / ${STEPS.length}`;
  document.getElementById('time-display').textContent = `${esc(step.title)}`;
  document.getElementById('scrub').value = String(n);
}

// ── Controls ──────────────────────────────────────────────────────────
document.getElementById('btn-prev').addEventListener('click', () => showStep(currentStep - 1));
document.getElementById('btn-next').addEventListener('click', () => showStep(currentStep + 1));

const playBtn = document.getElementById('btn-play');
function setPlayIcon(){ playBtn.textContent = playing ? '⏸' : '⏵'; }
function tickAutoplay(){
  if (!playing) return;
  if (currentStep >= STEPS.length - 1){ playing = false; setPlayIcon(); return; }
  showStep(currentStep + 1);
  autoplayTimer = setTimeout(tickAutoplay, 3000 / speed);
}
playBtn.addEventListener('click', () => {
  playing = !playing;
  setPlayIcon();
  if (playing){
    if (currentStep >= STEPS.length - 1) showStep(0);
    autoplayTimer = setTimeout(tickAutoplay, 3000 / speed);
  } else if (autoplayTimer){ clearTimeout(autoplayTimer); }
});

document.getElementById('scrub').addEventListener('input', e => {
  if (autoplayTimer){ clearTimeout(autoplayTimer); }
  playing = false; setPlayIcon();
  showStep(parseInt(e.target.value));
});

document.getElementById('speed').addEventListener('change', e => {
  speed = parseFloat(e.target.value);
});

document.addEventListener('keydown', e => {
  if (e.code === 'ArrowRight') showStep(currentStep + 1);
  if (e.code === 'ArrowLeft')  showStep(currentStep - 1);
  if (e.code === 'Space'){ e.preventDefault(); playBtn.click(); }
});

// ── Init ──────────────────────────────────────────────────────────────
showStep(0);
</script>
</body></html>
"""


def render(trace, cfg: dict[str, Any] | None = None, out_path: Path | str | None = None) -> Path:
    """Write the v3 slideshow HTML."""
    payload = _build_payload(trace)
    data_json = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")

    if out_path is None:
        out_dir = Path((cfg or {}).get("out_dir", "./out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_slug(trace.model_id)}__{_slug(trace.prompt)}__anim_v3.html"
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
