"""Inside one transformer block — standalone deep-dive renderer.

A 6-substep slideshow walking through one chosen (layer, head):

  1. LN1 + Q/K/V projection
  2. Scores: Q · K^T (cell-by-cell stagger fill)
  3. Softmax morph (raw → exp → normalized)
  4. Attention output: weights @ V → o_proj → ⊕ residual
  5. LN2 + FFN expand + activation
  6. FFN contract + ⊕ residual = block output

Pure TraceData → HTML. Must not import torch / transformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm_trace.renderers._util import html_escape as _esc
from llm_trace.renderers._util import slug as _slug


def _build_payload(trace) -> dict[str, Any] | None:
    bd = trace.block_deepdive
    if bd is None:
        return None
    spot = len(trace.tokens) - 1  # spotlight = last prompt token

    # Mask sentinel: replace -inf (above-diagonal) with a JS-friendly value.
    # JS code knows that anything <= -900 means "masked" and renders greyed.
    scores_serial = bd.scores.copy()
    scores_serial[np.isinf(scores_serial)] = -999.0

    return {
        "model_id": trace.model_id,
        "prompt": trace.prompt,
        "system_prompt": getattr(trace, "system_prompt", None),
        "model_meta": trace.model_meta,
        "tokens": trace.tokens,
        "token_ids": list(trace.token_ids),
        "block": {
            "layer_index": int(bd.layer_index),
            "head_index": int(bd.head_index),
            "n_heads": int(bd.n_heads),
            "head_dim": int(bd.head_dim),
            "activation": bd.activation,
            "has_swiglu_gate": bool(bd.has_swiglu_gate),
            "spotlight_idx": int(spot),
            "seq_len": int(bd.scores.shape[0]),
            "hidden_size": int(bd.pre_ln1.shape[1]),
            "ffn_dim": int(bd.ffn_pre_act.shape[1]),
            "pre_ln1": bd.pre_ln1[spot].tolist(),
            "post_ln1": bd.post_ln1[spot].tolist(),
            "q": bd.q.tolist(),
            "k": bd.k.tolist(),
            "v": bd.v.tolist(),
            "scores": scores_serial.tolist(),
            "weights": bd.weights.tolist(),
            "context": bd.context[spot].tolist(),
            "attn_output": bd.attn_output[spot].tolist(),
            "post_attn_residual": bd.post_attn_residual[spot].tolist(),
            "post_ln2": bd.post_ln2[spot].tolist(),
            "ffn_pre_act": bd.ffn_pre_act[spot].tolist(),
            "ffn_post_act": bd.ffn_post_act[spot].tolist(),
            "ffn_gate": (
                bd.ffn_gate[spot].tolist() if bd.ffn_gate is not None else None
            ),
            "ffn_output": bd.ffn_output[spot].tolist(),
            "block_output": bd.block_output[spot].tolist(),
        },
    }


def render(
    trace,
    cfg: dict[str, Any] | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Write the inside-one-block HTML for this trace.

    Either pass `out_path` directly, or pass `cfg={"out_dir": ...}` and
    the filename will be auto-derived from model_id + prompt.
    """
    if out_path is None:
        out_dir = Path((cfg or {}).get("out_dir", "./out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"{_slug(trace.model_id)}__{_slug(trace.prompt)}__inside_block.html"
        )
    out_path = Path(out_path)

    payload = _build_payload(trace)
    if payload is None:
        out_path.write_text(_no_data_html(trace), encoding="utf-8")
        return out_path

    # Compact JSON, escape `</` so it doesn't close the <script> early.
    data_json = json.dumps(payload, separators=(",", ":"), default=str)
    data_safe = data_json.replace("</", r"<\/")

    blk = payload["block"]
    title = (
        f"L{blk['layer_index']} · H{blk['head_index']} · "
        f"{trace.model_id.split('/')[-1]} · {trace.prompt[:40]}"
    )
    html = _HTML.replace("__TITLE__", _esc(title)).replace("__DATA__", data_safe)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _no_data_html(trace) -> str:
    return (
        f"<!DOCTYPE html><html><head><title>no block_deepdive</title></head>"
        f"<body style=\"font-family:monospace;padding:40px;color:#888;background:#080b12\">"
        f"<h1 style=\"color:#ff6b6b\">No block deep-dive data</h1>"
        f"<p>This trace was collected without <code>block_deepdive</code>, or "
        f"on an architecture not yet supported (works on GPT-2 + Llama families).</p>"
        f"<p>Model: <code>{_esc(trace.model_id)}</code></p></body></html>"
    )


# ── HTML template ──────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root{
  --bg:#080b12;--surface:#0e1220;--surface2:#151b2e;--surface3:#1c2236;
  --border:#1f2a45;--text:#cdd6f0;--muted:#5a6888;
  --teal:#00d4aa;--amber:#ffb800;--coral:#ff6b6b;--blue:#4da6ff;--purple:#c084fc;
  --pink:#f472b6;
  --mono:'SF Mono',Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,'Segoe UI',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.5}

.stage{display:grid;grid-template-rows:auto 64px 1fr 60px;height:100vh}

/* ── Top bar ─────────────────────────────────────────── */
.top-bar{padding:10px 28px;border-bottom:1px solid var(--border);
  font-family:var(--mono);font-size:12px;color:var(--muted);
  display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.top-bar .stamp{color:var(--purple);font-weight:700;letter-spacing:.05em}
.top-bar .model{color:var(--teal);font-weight:700}
.top-bar .focus{color:var(--amber)}
.top-bar .prompt{color:var(--text)}
.top-bar .arch{margin-left:auto;color:var(--blue)}

/* ── Sub-pipeline indicator ──────────────────────────── */
.pipeline{display:flex;align-items:center;justify-content:center;gap:0;
  padding:14px 28px;border-bottom:1px solid var(--border)}
.pipe-step{display:flex;flex-direction:column;align-items:center;gap:5px;
  cursor:pointer;user-select:none}
.pipe-step .row{display:flex;align-items:center;gap:0}
.pipe-step .conn{height:2px;width:48px;background:var(--border);transition:background .35s}
.pipe-step:first-child .conn{display:none}
.pipe-step .dot{width:22px;height:22px;border-radius:50%;
  background:var(--surface3);border:2px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:10px;color:var(--muted);
  transition:all .35s}
.pipe-step .lbl{font-family:var(--mono);font-size:10px;color:var(--muted);
  letter-spacing:.05em;transition:color .35s;white-space:nowrap}
.pipe-step.done .dot{background:var(--surface3);border-color:var(--blue);color:var(--blue)}
.pipe-step.done .conn{background:var(--blue)}
.pipe-step.active .dot{background:var(--teal);border-color:var(--teal);color:#000;
  font-weight:700;animation:pulse 1.4s ease-in-out infinite}
.pipe-step.active .lbl{color:var(--teal);font-weight:700}
.pipe-step:hover .lbl{color:var(--text)}

@keyframes pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(0,212,170,0.6)}
  50%{box-shadow:0 0 0 6px rgba(0,212,170,0)}
}

/* ── Body ─────────────────────────────────────────────── */
.body{padding:24px 32px;overflow-y:auto;overflow-x:hidden}
.panel{max-width:1200px;margin:0 auto;animation:slideIn .35s ease-out}

@keyframes slideIn{
  from{opacity:0;transform:translateY(8px)}
  to{opacity:1;transform:none}
}

h2.step-title{font-size:24px;font-weight:800;margin-bottom:6px;
  color:var(--text);letter-spacing:-0.01em}
.step-blurb{font-family:var(--mono);font-size:13px;color:var(--muted);
  line-height:1.6;margin-bottom:22px;max-width:900px}
.step-blurb b{color:var(--teal)}

/* ── Vector cell (full-width vector display) ─────────── */
.vec-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.vec-row .meta{font-family:var(--mono);font-size:11px;color:var(--muted);
  min-width:200px;line-height:1.4}
.vec-row .meta b{color:var(--text)}
.vec-row .meta .id{color:var(--amber);font-weight:700}
.vec-cells{display:flex;gap:1px;flex-wrap:wrap}
.vc{display:inline-block;min-width:42px;height:24px;padding:0 4px;
  border-radius:3px;font-family:var(--mono);font-size:10px;
  text-align:center;line-height:24px;font-weight:600;flex-shrink:0}

/* ── Q/K/V triplet ───────────────────────────────────── */
.qkv-block{display:grid;grid-template-columns:60px 1fr;gap:10px 12px;
  margin-top:14px;padding:14px 18px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px}
.qkv-label{font-family:var(--mono);font-size:14px;font-weight:700;
  align-self:center;text-align:right}
.qkv-label.q{color:var(--teal)}
.qkv-label.k{color:var(--blue)}
.qkv-label.v{color:var(--amber)}
.qkv-cells{display:flex;gap:1px;flex-wrap:nowrap;overflow-x:auto}

/* ── Scores grid (substep 2) ─────────────────────────── */
.scores-wrap{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
.scores-grid{display:grid;gap:2px;background:var(--surface);
  padding:14px;border-radius:10px;border:1px solid var(--border)}
.sc-row{display:flex;gap:2px}
.sc-cell{width:var(--sc-w,42px);height:var(--sc-h,30px);border-radius:3px;
  font-family:var(--mono);font-size:var(--sc-fz,9.5px);text-align:center;
  line-height:var(--sc-h,30px);
  font-weight:600;flex-shrink:0;
  opacity:0;animation:cellFillIn 240ms forwards}
.scores-grid.dense{--sc-w:24px;--sc-h:20px;--sc-fz:8px}
.sc-cell.masked{background:var(--surface3)!important;color:var(--muted)!important;
  opacity:0.35;animation:none}
.sc-axis{font-family:var(--mono);font-size:9px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}

@keyframes cellFillIn{
  0%{transform:scale(0.5);opacity:0}
  60%{transform:scale(1.1);opacity:1}
  100%{transform:scale(1);opacity:1}
}

/* ── Softmax bars (substep 3) ────────────────────────── */
.softmax-stage-label{font-family:var(--mono);font-size:11px;color:var(--teal);
  text-transform:uppercase;letter-spacing:.1em;margin:14px 0 6px;font-weight:700}
.softmax-bars{display:flex;align-items:flex-end;gap:6px;
  background:var(--surface);padding:18px;border-radius:10px;
  border:1px solid var(--border);min-height:240px;
  position:relative}
.softmax-bars .baseline{position:absolute;left:18px;right:18px;height:1px;
  background:var(--border);bottom:90px}
.softmax-bar{flex:1;display:flex;flex-direction:column;align-items:center;
  position:relative;height:200px}
.softmax-bar .b{width:80%;border-radius:3px 3px 0 0;
  transition:height 700ms cubic-bezier(.2,.8,.2,1),
             background-color 700ms,bottom 700ms;
  position:absolute;bottom:90px}
.softmax-bar .label{font-family:var(--mono);font-size:9.5px;color:var(--muted);
  position:absolute;bottom:0;text-align:center;width:100%}
.softmax-bar .val{font-family:var(--mono);font-size:9.5px;color:var(--text);
  font-weight:600;position:absolute;
  transition:bottom 700ms cubic-bezier(.2,.8,.2,1),color 400ms}
.softmax-stage-controls{display:flex;gap:8px;margin-top:10px;
  font-family:var(--mono);font-size:11px}
.sm-btn{background:var(--surface2);border:1px solid var(--border);
  color:var(--text);padding:6px 14px;border-radius:5px;cursor:pointer}
.sm-btn:hover{border-color:var(--teal);color:var(--teal)}
.sm-btn.active{background:var(--teal);color:#000;border-color:var(--teal)}

/* ── Attention output (substep 4) ────────────────────── */
.attn-out-grid{display:grid;grid-template-columns:140px 1fr;gap:12px 16px;
  margin-top:12px;padding:14px 18px;background:var(--surface);
  border-radius:10px;border:1px solid var(--border)}
.attn-out-label{font-family:var(--mono);font-size:11px;color:var(--muted);
  align-self:center;text-align:right;line-height:1.4}
.attn-out-label b{color:var(--text);display:block}
.weight-row{display:flex;gap:2px;align-items:center;flex-wrap:wrap}
.weight-bar{display:inline-block;height:22px;background:var(--teal);
  border-radius:2px;min-width:2px;
  font-family:var(--mono);font-size:9.5px;color:#001513;
  text-align:center;line-height:22px;padding:0 4px;font-weight:600;
  transition:width 400ms ease-out,background-color 400ms}
.v-heat{display:flex;flex-direction:column;gap:1px;max-height:150px;overflow-y:auto}
.v-heat .v-head-row{display:flex;gap:1px;align-items:center}
.v-heat .v-tok{min-width:60px;font-family:var(--mono);font-size:10px;
  color:var(--muted);padding-right:8px}
.v-heat .v-cells{display:flex;gap:1px;flex-wrap:nowrap;overflow-x:auto}

/* ── FFN strip (substep 5/6) ─────────────────────────── */
.ffn-strip-wrap{margin-top:12px;padding:14px 18px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;overflow-x:auto}
.ffn-strip-label{font-family:var(--mono);font-size:11px;color:var(--muted);
  margin-bottom:6px}
.ffn-strip-label b{color:var(--text)}
.ffn-strip{display:flex;gap:1px;align-items:center;height:38px;
  min-width:100%}
.ffn-cell{height:34px;flex-shrink:0;border-radius:1px;
  width:0;transition:width 600ms cubic-bezier(.2,.8,.2,1),
                  background-color 400ms}
.ffn-strip.expanded .ffn-cell{width:var(--cw,3px)}
.ffn-strip.contracted .ffn-cell{width:var(--cw-narrow,2px)}

/* ── Activation curve (SVG inline) ───────────────────── */
.act-curve{margin-top:12px;padding:14px 18px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px}
.act-curve svg{width:100%;height:140px}
.act-curve .axis{stroke:var(--border);stroke-width:1}
.act-curve .curve{stroke:var(--teal);stroke-width:2;fill:none}
.act-curve .label{font-family:var(--mono);font-size:11px;fill:var(--muted)}

/* ── Residual add visual ─────────────────────────────── */
.residual-add{margin-top:14px;padding:14px 22px;
  background:linear-gradient(180deg,transparent,rgba(192,132,252,0.08));
  border:1px dashed var(--purple);border-radius:8px;
  font-family:var(--mono);font-size:13px;color:var(--purple);
  text-align:center;font-weight:600;letter-spacing:.04em}
.residual-add b{color:var(--text)}

/* ── Output row (final block output) ─────────────────── */
.block-output-row{margin-top:18px;padding:16px 20px;background:var(--surface);
  border:2px solid var(--teal);border-radius:10px;
  box-shadow:0 0 16px rgba(0,212,170,0.18)}

/* ── Equation chip ───────────────────────────────────── */
.eq-chip{display:inline-block;padding:3px 10px;
  background:var(--surface3);color:var(--teal);
  border:1px solid var(--border);border-radius:4px;
  font-family:var(--mono);font-size:11.5px;margin:0 4px}

/* ── Note callout ────────────────────────────────────── */
.note{font-family:var(--mono);font-size:11.5px;color:var(--text);
  margin-top:14px;padding:10px 14px;background:var(--surface2);
  border-left:2px solid var(--blue);border-radius:4px;line-height:1.6}
.note b{color:var(--text)}
.note.warn{border-left-color:var(--amber);background:rgba(255,184,0,0.04)}

/* ── Controls bar ────────────────────────────────────── */
.controls{display:flex;align-items:center;gap:12px;padding:0 28px;
  border-top:1px solid var(--border);background:rgba(8,11,18,0.95)}
.btn{background:var(--surface2);border:1px solid var(--border);
  color:var(--text);width:34px;height:34px;border-radius:50%;cursor:pointer;
  font-size:13px;font-weight:700;
  display:flex;align-items:center;justify-content:center;transition:all .15s}
.btn:hover{border-color:var(--teal);color:var(--teal)}
.btn.primary{background:var(--teal);color:#000;border-color:var(--teal);
  width:42px;height:42px;font-size:15px}
.btn.primary:hover{opacity:.85;color:#000}
.step-pill{font-family:var(--mono);font-size:11px;color:var(--teal);
  padding:5px 12px;background:rgba(0,212,170,.1);
  border:1px solid rgba(0,212,170,.3);border-radius:14px;
  letter-spacing:.08em;text-transform:uppercase;font-weight:700}
.scrub-wrap{flex:1}
.scrub{width:100%;accent-color:var(--teal);height:4px;cursor:pointer}
.speed{background:var(--surface2);color:var(--text);
  border:1px solid var(--border);border-radius:5px;padding:5px 8px;
  font-family:var(--mono);font-size:11px;cursor:pointer}

/* ── Misc ────────────────────────────────────────────── */
.muted{color:var(--muted)}
.token-pill{display:inline-block;padding:2px 6px;background:var(--surface3);
  border-radius:3px;border:1px solid var(--border);
  font-family:var(--mono);font-size:11px;color:var(--text)}
.spotlight{color:var(--amber);font-weight:700}
</style>
</head>
<body>
<div class="stage">

  <!-- ── Top bar ─────────────────────────────────────── -->
  <div class="top-bar">
    <span class="stamp">INSIDE ONE BLOCK</span>
    <span class="model" id="t-model"></span>
    <span class="focus" id="t-focus"></span>
    <span class="prompt" id="t-prompt"></span>
    <span class="arch" id="t-arch"></span>
  </div>

  <!-- ── Sub-pipeline ─────────────────────────────────── -->
  <div class="pipeline" id="pipeline"></div>

  <!-- ── Body (per-substep panel) ─────────────────────── -->
  <div class="body">
    <div class="panel" id="panel"></div>
  </div>

  <!-- ── Controls ─────────────────────────────────────── -->
  <div class="controls">
    <button class="btn" id="btn-prev" title="prev (←)">‹</button>
    <button class="btn primary" id="btn-play" title="play (space)">▶</button>
    <button class="btn" id="btn-next" title="next (→)">›</button>
    <span class="step-pill" id="step-pill">step 1 / 6</span>
    <div class="scrub-wrap"><input type="range" id="scrub" class="scrub" min="0" max="5" value="0" step="1"></div>
    <select class="speed" id="speed">
      <option value="0.5">slow (8s/step)</option>
      <option value="1" selected>normal (4s/step)</option>
      <option value="2">fast (2s/step)</option>
    </select>
  </div>

</div>

<script>
const D = __DATA__;

// ── Helpers ───────────────────────────────────────────────────────────
function esc(s){return String(s).replace(/[&<>"']/g, c=>(
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

function vecCell(v){
  const i = Math.min(1, Math.abs(v));
  const col = v >= 0
    ? `rgba(0,${(140 + i*112)|0},${(120 + i*50)|0},${0.25 + i*0.7})`
    : `rgba(${(140 + i*115)|0},${(80 - i*60)|0},${(80 - i*60)|0},${0.25 + i*0.7})`;
  const txt = i > 0.5 ? '#001513' : '#cdd6f0';
  const num = (v >= 0 ? '+' : '') + v.toFixed(2);
  return `<span class="vc" style="background:${col};color:${txt}">${num}</span>`;
}

function vecRow(label, fullVec, opts){
  opts = opts || {};
  const previewN = Math.min(opts.preview || 24, fullVec.length);
  const cells = fullVec.slice(0, previewN).map(vecCell).join('');
  const remain = fullVec.length - previewN;
  return `
    <div class="vec-row">
      <span class="meta">${label}</span>
      <span class="vec-cells">${cells}${remain > 0 ? `<span style="font-family:var(--mono);font-size:10px;color:var(--blue);align-self:center;margin-left:6px">+${remain} more</span>` : ''}</span>
    </div>
  `;
}

function smallCell(v, mode){
  const i = Math.min(1, Math.abs(v));
  if (mode === 'masked') {
    return `<span class="sc-cell masked">·</span>`;
  }
  const col = v >= 0
    ? `rgba(0,${(140 + i*112)|0},${(120 + i*50)|0},${0.25 + i*0.75})`
    : `rgba(${(140 + i*115)|0},${(80 - i*60)|0},${(80 - i*60)|0},${0.25 + i*0.75})`;
  const txt = i > 0.55 ? '#001513' : '#cdd6f0';
  const num = isFinite(v) ? v.toFixed(2) : '·';
  return `<span class="sc-cell" style="background:${col};color:${txt}">${num}</span>`;
}

// ── STEP definitions ─────────────────────────────────────────────────
const STEPS = [
  { id:'qkv',     title:'Step 1 · LayerNorm + Q / K / V projection' },
  { id:'scores',  title:'Step 2 · Scores: Q · K^T' },
  { id:'softmax', title:'Step 3 · Softmax → attention weights' },
  { id:'attn',    title:'Step 4 · Weights @ V → output projection → ⊕ residual' },
  { id:'ffn',     title:'Step 5 · LayerNorm 2 + FFN expand + activation' },
  { id:'out',     title:'Step 6 · FFN contract + ⊕ residual = block output' },
];

// ── Top bar fill-in ──────────────────────────────────────────────────
{
  const m = D.model_meta || {};
  const blk = D.block;
  document.getElementById('t-model').textContent = D.model_id;
  document.getElementById('t-focus').textContent =
    `layer ${blk.layer_index} / head ${blk.head_index}` +
    `  (of ${m.n_layer} × ${m.n_head}, head_dim ${blk.head_dim})`;
  document.getElementById('t-prompt').textContent =
    `prompt: "${(D.prompt || '').slice(0, 60)}"`;
  document.getElementById('t-arch').textContent =
    `${m.model_type} · hidden ${blk.hidden_size} · ffn ${blk.ffn_dim} · ` +
    (blk.has_swiglu_gate ? 'SwiGLU' : (blk.activation || ''));
}

// ── Sub-pipeline indicator ───────────────────────────────────────────
{
  const el = document.getElementById('pipeline');
  el.innerHTML = STEPS.map((s, i) => `
    <div class="pipe-step" data-step="${i}">
      <div class="row">
        ${i > 0 ? '<div class="conn"></div>' : ''}
        <div class="dot">${i + 1}</div>
      </div>
      <div class="lbl">${esc(s.id)}</div>
    </div>
  `).join('');
  el.querySelectorAll('.pipe-step').forEach(p => {
    p.addEventListener('click', () => showStep(parseInt(p.dataset.step)));
  });
}

// ── Per-step renderers ───────────────────────────────────────────────

function step_qkv() {
  const b = D.block;
  const spotTok = D.tokens[b.spotlight_idx] || '?';
  return `
    <h2 class="step-title">${esc(STEPS[0].title)}</h2>
    <div class="step-blurb">
      Inside layer <b>${b.layer_index}</b>, head <b>${b.head_index}</b>.
      We're spotlighting the <span class="spotlight">last prompt token</span>
      <span class="token-pill">${esc(spotTok)}</span> (id ${D.token_ids[b.spotlight_idx]}).
      <br>The block input vector hits LayerNorm 1, then is linearly projected into
      <b>three</b> separate vectors — Q (the <i>query</i>), K (the <i>key</i>),
      V (the <i>value</i>) — each of width <b>head_dim = ${b.head_dim}</b>.
    </div>

    ${vecRow(
      `<b>pre-LN1 (residual stream)</b><br>
       <span class="muted">block input · ${b.hidden_size} dims · first 24 shown</span>`,
      b.pre_ln1, {preview: 24}
    )}

    <div style="text-align:center;color:var(--teal);margin:6px 0;font-family:var(--mono);font-size:13px">↓ LayerNorm</div>

    ${vecRow(
      `<b>post-LN1</b><br>
       <span class="muted">normalized · same ${b.hidden_size} dims</span>`,
      b.post_ln1, {preview: 24}
    )}

    <div style="text-align:center;color:var(--teal);margin:6px 0;font-family:var(--mono);font-size:13px">
      ↓ split via three linear projections (q_proj / k_proj / v_proj)
    </div>

    <div class="qkv-block">
      <div class="qkv-label q">Q</div>
      <div class="qkv-cells">${b.q[b.spotlight_idx].map(vecCell).join('')}</div>
      <div class="qkv-label k">K</div>
      <div class="qkv-cells">${b.k[b.spotlight_idx].map(vecCell).join('')}</div>
      <div class="qkv-label v">V</div>
      <div class="qkv-cells">${b.v[b.spotlight_idx].map(vecCell).join('')}</div>
    </div>

    <div class="note">
      <b>Q</b> = "what am I looking for?"   <b>K</b> = "what do I represent?"
      <b>V</b> = "what should I contribute if attended to?"
      Same input, three different learned projections.
      Showing only the spotlighted token's row of each;
      the full Q/K/V cover all <b>${b.seq_len}</b> tokens.
    </div>

    ${b.has_swiglu_gate ? `
      <div class="note warn">
        Llama-family models apply <b>RoPE rotation</b> to Q and K
        <i>before</i> the dot product. We show the post-projection,
        pre-rotation vectors here (the rotation reshapes them slightly
        based on token position).
      </div>
    ` : ''}
  `;
}

function step_scores() {
  const b = D.block;
  const seq = b.seq_len;
  // Find max abs for color scaling (excluding masked).
  let maxAbs = 0;
  for (let r = 0; r < seq; r++) for (let c = 0; c < seq; c++) {
    const v = b.scores[r][c];
    if (v > -900 && Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  }
  const scale = maxAbs || 1;
  // Render with stagger — each cell gets animation-delay
  let cells = '';
  for (let r = 0; r < seq; r++) {
    let row = '';
    for (let c = 0; c < seq; c++) {
      const v = b.scores[r][c];
      const masked = v <= -900;
      const norm = masked ? 0 : v / scale;
      const delay = (r * seq + c) * 28;
      const cell = smallCell(masked ? -1 : norm, masked ? 'masked' : null);
      row += cell.replace('class="sc-cell', `class="sc-cell" style="animation-delay:${delay}ms`).replace('></span>', `></span>`);
      // Re-write to inject delay properly:
    }
    cells += `<div class="sc-row">${row}</div>`;
  }
  // We need to inject animation-delay. Cleaner: rebuild here.
  cells = '';
  // Cap total fill time to ~1.6s regardless of seq size; tiny seq still gets
  // a perceptible stagger.
  const totalCells = seq * seq;
  const perCellMs = Math.max(8, Math.min(28, Math.floor(1600 / totalCells)));
  for (let r = 0; r < seq; r++) {
    let row = '';
    for (let c = 0; c < seq; c++) {
      const v = b.scores[r][c];
      const masked = v <= -900;
      const delay = (r * seq + c) * perCellMs;
      if (masked) {
        row += `<span class="sc-cell masked">·</span>`;
      } else {
        const i = Math.min(1, Math.abs(v / scale));
        const col = v >= 0
          ? `rgba(0,${(140 + i*112)|0},${(120 + i*50)|0},${0.25 + i*0.75})`
          : `rgba(${(140 + i*115)|0},${(80 - i*60)|0},${(80 - i*60)|0},${0.25 + i*0.75})`;
        const txt = i > 0.55 ? '#001513' : '#cdd6f0';
        row += `<span class="sc-cell" style="background:${col};color:${txt};animation-delay:${delay}ms">${v.toFixed(1)}</span>`;
      }
    }
    cells += `<div class="sc-row">${row}</div>`;
  }
  return `
    <h2 class="step-title">${esc(STEPS[1].title)}</h2>
    <div class="step-blurb">
      Compute <span class="eq-chip">scores[i,j] = Q[i] · K[j] / √${b.head_dim}</span>
      for every (query, key) pair. Each cell fills as the dot product completes;
      <b>upper-triangle cells are masked</b> (a token can't attend to a future
      token in causal attention). Color: green = high score, red = low.
    </div>

    <div class="scores-wrap">
      <div>
        <div class="sc-axis">key index →</div>
        <div class="scores-grid${seq > 12 ? ' dense' : ''}">${cells}</div>
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);max-width:280px;line-height:1.6">
        <b style="color:var(--text)">Reading the matrix:</b><br>
        Row <i>i</i> = "what does token <i>i</i>'s query attend to?"<br>
        Col <i>j</i> = "how relevant is token <i>j</i>'s key?"<br><br>
        For ${seq} tokens, that's ${seq * seq} dot products
        — but only ${(seq * (seq + 1)) / 2} unmasked.<br><br>
        ${b.has_swiglu_gate
          ? '<i>Note:</i> for Llama, scores are derived from the post-RoPE attention pattern (so softmax exactly reproduces the model\'s actual weights).'
          : ''}
      </div>
    </div>
  `;
}

function step_softmax() {
  const b = D.block;
  const lastRow = b.seq_len - 1;
  // Row of scores (excluding masked) for last query token
  const rawRow = [];
  for (let j = 0; j <= lastRow; j++) {
    rawRow.push(b.scores[lastRow][j]);
  }
  const wRow = b.weights[lastRow].slice(0, lastRow + 1);
  // exp(scores)
  const expRow = rawRow.map(v => Math.exp(v));
  const sumExp = expRow.reduce((a, x) => a + x, 0);

  // For visualization we render as bars — three stages (raw / exp / norm).
  // The bar heights are computed at JS-runtime when stage changes.
  return `
    <h2 class="step-title">${esc(STEPS[2].title)}</h2>
    <div class="step-blurb">
      Softmax converts the scores row for the
      <span class="spotlight">last query token</span> into a probability
      distribution over keys. Three stages: <b>raw</b> →
      <b>exp(score)</b> → <b>÷ Σ</b> = a row that sums to 1. Click the
      buttons or watch the auto-walk.
    </div>

    <div class="softmax-stage-controls">
      <button class="sm-btn active" data-stage="raw">raw scores</button>
      <button class="sm-btn" data-stage="exp">e<sup>x</sup></button>
      <button class="sm-btn" data-stage="norm">÷ Σ = probs</button>
    </div>

    <div class="softmax-stage-label" id="sm-stage-label">stage: raw scores (some negative)</div>

    <div class="softmax-bars" id="sm-bars">
      <div class="baseline"></div>
      ${rawRow.map((v, i) => `
        <div class="softmax-bar" data-raw="${v}" data-exp="${expRow[i]}" data-norm="${wRow[i]}">
          <div class="b" style="background:${v >= 0 ? 'var(--teal)' : 'var(--coral)'}"></div>
          <div class="val"></div>
          <div class="label">${esc((D.tokens[i] || '').replace(/\\s/g, '·').slice(0,8))}</div>
        </div>
      `).join('')}
    </div>

    <div class="note">
      Σ probs = ${wRow.reduce((a,x)=>a+x,0).toFixed(4)} (should be 1.0).
      Highest weight: <b>${(Math.max(...wRow) * 100).toFixed(1)}%</b> on token
      <span class="token-pill">${esc((D.tokens[wRow.indexOf(Math.max(...wRow))] || '').replace(/\\s/g, '·'))}</span>.
    </div>
  `;
}

// Apply a softmax-bars stage. Called after step_softmax DOM is built.
function applySoftmaxStage(stage) {
  const wrap = document.getElementById('sm-bars');
  if (!wrap) return;
  const bars = Array.from(wrap.querySelectorAll('.softmax-bar'));
  let max = 0, mode = stage;
  bars.forEach(bar => {
    const v = parseFloat(bar.dataset[mode]);
    if (Math.abs(v) > max) max = Math.abs(v);
  });
  max = max || 1;
  const baselineY = mode === 'raw' ? 90 : 0;   // px from bottom (matches CSS bottom:90px for raw)
  bars.forEach(bar => {
    const v = parseFloat(bar.dataset[mode]);
    const b = bar.querySelector('.b');
    const val = bar.querySelector('.val');
    let h, bot, color, txt;
    if (mode === 'raw') {
      // bidirectional bars about baseline at 90px
      h = Math.abs(v) / max * 90;
      bot = v >= 0 ? 90 : 90 - h;
      color = v >= 0 ? 'var(--teal)' : 'var(--coral)';
      txt = (v >= 0 ? '+' : '') + v.toFixed(2);
    } else if (mode === 'exp') {
      h = v / max * 180;
      bot = 0;
      color = 'var(--amber)';
      txt = v.toFixed(2);
    } else {
      h = v * 180;
      bot = 0;
      color = 'var(--purple)';
      txt = (v * 100).toFixed(1) + '%';
    }
    b.style.height = h + 'px';
    b.style.bottom = bot + 'px';
    b.style.background = color;
    val.textContent = txt;
    val.style.bottom = (bot + h + 4) + 'px';
    val.style.color = mode === 'norm' ? 'var(--purple)' : 'var(--text)';
  });
  // Update active button
  document.querySelectorAll('.sm-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === mode);
  });
  const lbl = document.getElementById('sm-stage-label');
  if (lbl) {
    lbl.textContent =
      mode === 'raw'  ? 'stage 1 / 3 · raw scores (some negative)' :
      mode === 'exp'  ? 'stage 2 / 3 · exp(score) — all positive now' :
                        'stage 3 / 3 · ÷ Σ — sums to 1';
  }
}

function step_attn_output() {
  const b = D.block;
  const spotI = b.spotlight_idx;
  const wRow = b.weights[spotI];
  const seq = b.seq_len;

  // Weight bars (top row). Width proportional to weight; min 18px so labels show.
  const maxW = Math.max(...wRow);
  const bars = wRow.map((w, j) => {
    const widthPct = (w / maxW * 100);
    const tok = (D.tokens[j] || '').replace(/\s/g, '·').slice(0, 10);
    return `<span class="weight-bar" style="width:${Math.max(widthPct, 2)}%"
                  title="${esc(D.tokens[j] || '')} = ${(w*100).toFixed(2)}%">${(w*100).toFixed(0)}%
              <span style="color:var(--muted);margin-left:4px">${esc(tok)}</span></span>`;
  }).join('');

  // V matrix as a heatmap (per-token rows).
  const vRows = b.v.map((row, i) => {
    const cells = row.slice(0, 24).map(vecCell).join('');
    return `<div class="v-head-row">
      <span class="v-tok">${esc((D.tokens[i] || '').replace(/\s/g, '·').slice(0, 10))}</span>
      <span class="v-cells">${cells}</span>
    </div>`;
  }).join('');

  return `
    <h2 class="step-title">${esc(STEPS[3].title)}</h2>
    <div class="step-blurb">
      The weights from the spotlighted query token (last) get multiplied by each
      token's V vector, then summed → <b>context</b> vector.
      Then a final linear <b>output projection</b> brings it back to ${b.hidden_size} dims,
      and the <b>residual</b> is added to keep the original signal.
    </div>

    <div class="attn-out-grid">
      <div class="attn-out-label"><b>weights</b><span class="muted">(spotlight row, sum=1)</span></div>
      <div class="weight-row">${bars}</div>

      <div class="attn-out-label"><b>V matrix</b><span class="muted">(${seq} × ${b.head_dim} per head)</span></div>
      <div class="v-heat">${vRows}</div>

      <div class="attn-out-label"><b>= context</b><span class="muted">(weighted sum, head_dim)</span></div>
      <div class="vec-cells">${b.context.map(vecCell).join('')}</div>
    </div>

    <div style="text-align:center;color:var(--teal);margin:14px 0;font-family:var(--mono);font-size:13px">
      ↓ heads concatenated · <b>o_proj</b> linear: head_dim × n_heads → ${b.hidden_size}
    </div>

    ${vecRow(
      `<b>attn_output</b><br><span class="muted">post-projection · ${b.hidden_size} dims</span>`,
      b.attn_output, {preview: 24}
    )}

    <div class="residual-add">⊕ residual add — pre_ln1 + attn_output</div>

    ${vecRow(
      `<b>post-attn residual</b><br><span class="muted">= pre_ln1 + attn_output</span>`,
      b.post_attn_residual, {preview: 24}
    )}
  `;
}

function step_ffn_expand() {
  const b = D.block;
  // Cell width: aim for ~600 px total strip for ffn_dim cells.
  const targetWidth = 900;
  const cw = Math.max(1, Math.floor(targetWidth / b.ffn_dim));

  // Build strips. We render the FULL ffn_dim cells (typically 3072 / 5632)
  // because the visual point IS the width.
  function strip(values, label, expandedClass) {
    let max = 0;
    for (const v of values) if (Math.abs(v) > max) max = Math.abs(v);
    max = max || 1;
    const cells = values.map(v => {
      const i = Math.min(1, Math.abs(v) / max);
      const col = v >= 0
        ? `rgba(0,${(140 + i*112)|0},${(120 + i*50)|0},${0.20 + i*0.7})`
        : `rgba(${(140 + i*115)|0},${(80 - i*60)|0},${(80 - i*60)|0},${0.20 + i*0.7})`;
      return `<span class="ffn-cell" style="background:${col}"></span>`;
    }).join('');
    return `
      <div class="ffn-strip-wrap">
        <div class="ffn-strip-label">${label}</div>
        <div class="ffn-strip ${expandedClass}" style="--cw:${cw}px">${cells}</div>
      </div>
    `;
  }

  const actName = b.activation === 'silu' ? 'SiLU' : (b.activation === 'gelu_new' ? 'GELU (NewGELU)' : 'GELU');
  const actCurve = activationCurveSvg(b.activation);

  return `
    <h2 class="step-title">${esc(STEPS[4].title)}</h2>
    <div class="step-blurb">
      The post-attn residual goes through <b>LayerNorm 2</b>, then a linear
      projection expands it from <b>${b.hidden_size}</b> dims to
      <b>${b.ffn_dim}</b> dims (${(b.ffn_dim / b.hidden_size).toFixed(1)}× wider).
      Then the <b>${actName}</b> activation introduces nonlinearity — without it,
      stacking layers would collapse into one big linear map.
    </div>

    ${vecRow(
      `<b>post-LN2</b><br><span class="muted">normalized · ${b.hidden_size} dims</span>`,
      b.post_ln2, {preview: 24}
    )}

    <div style="text-align:center;color:var(--teal);margin:10px 0;font-family:var(--mono);font-size:13px">
      ↓ up_proj: ${b.hidden_size} → ${b.ffn_dim}
    </div>

    ${strip(b.ffn_pre_act, `<b>pre-activation</b> · ${b.ffn_dim} cells (each is one channel)`, 'expanded')}

    <div class="act-curve">${actCurve}</div>

    ${strip(b.ffn_post_act, `<b>post-${actName}</b> · same width, values pushed through the curve`, 'expanded')}

    ${b.has_swiglu_gate ? `
      <div class="note">
        <b>SwiGLU gate (Llama):</b> there's a parallel <code>gate_proj</code>
        path. The actual operation is
        <span class="eq-chip">silu(gate_proj(x)) ⊙ up_proj(x)</span>
        — a Hadamard product. The "post-activation" strip above already
        reflects this combined output.
      </div>
    ` : ''}
  `;
}

function step_block_output() {
  const b = D.block;
  const targetWidth = 900;
  const cw = Math.max(1, Math.floor(targetWidth / b.ffn_dim));

  return `
    <h2 class="step-title">${esc(STEPS[5].title)}</h2>
    <div class="step-blurb">
      The wide intermediate gets contracted back to <b>${b.hidden_size}</b> dims
      via <code>down_proj</code>, then added to the post-attn residual. That's
      the block's output — the <b>residual stream input for layer ${b.layer_index + 1}</b>.
    </div>

    <div style="text-align:center;color:var(--teal);margin:10px 0;font-family:var(--mono);font-size:13px">
      ${b.ffn_dim} dims  →  down_proj  →  ${b.hidden_size} dims
    </div>

    ${vecRow(
      `<b>ffn_output</b><br><span class="muted">post down_proj · ${b.hidden_size} dims</span>`,
      b.ffn_output, {preview: 24}
    )}

    <div class="residual-add">⊕ residual add — post_attn_residual + ffn_output</div>

    <div class="block-output-row">
      ${vecRow(
        `<b style="color:var(--teal)">block_output</b><br><span class="muted">= post_attn_residual + ffn_output · the new residual stream</span>`,
        b.block_output, {preview: 24}
      )}
    </div>

    <div class="note">
      <b>This vector becomes the input to layer ${b.layer_index + 1}.</b>
      The same pipeline (LN → Q/K/V → attention → ⊕ → LN → FFN → ⊕)
      runs again, with different learned weights, ${(D.model_meta.n_layer || 0) - b.layer_index - 1}
      more times. Then the LM head turns the final residual into vocab logits.
    </div>
  `;
}

function activationCurveSvg(act) {
  // Inline SVG showing the activation curve. Domain x in [-4, 4].
  const W = 600, H = 120;
  const xs = [];
  for (let i = 0; i <= 80; i++) xs.push(-4 + i * 0.1);
  function y(x) {
    if (act === 'silu') return x / (1 + Math.exp(-x));
    if (act === 'gelu_new') {
      const t = Math.tanh(Math.sqrt(2/Math.PI) * (x + 0.044715*x*x*x));
      return 0.5 * x * (1 + t);
    }
    // gelu (default)
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2/Math.PI)*(x + 0.044715*x*x*x)));
  }
  // Map (x,y) to svg coords. y range [-2, 4].
  const xmap = x => 30 + (x + 4) / 8 * (W - 50);
  const ymap = yy => H - 20 - (yy + 2) / 6 * (H - 30);
  const pts = xs.map(x => `${xmap(x).toFixed(1)},${ymap(y(x)).toFixed(1)}`).join(' ');
  const xZero = xmap(0), yZero = ymap(0);
  const yMin = ymap(-2), yMax = ymap(4);
  return `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      <line class="axis" x1="${xZero}" y1="${yMax}" x2="${xZero}" y2="${yMin}"></line>
      <line class="axis" x1="30" y1="${yZero}" x2="${W - 20}" y2="${yZero}"></line>
      <polyline class="curve" points="${pts}"></polyline>
      <text class="label" x="${W - 60}" y="${yZero - 6}">x</text>
      <text class="label" x="${xZero + 6}" y="20">${esc(act)}(x)</text>
    </svg>
  `;
}

const RENDERERS = {
  qkv: step_qkv,
  scores: step_scores,
  softmax: step_softmax,
  attn: step_attn_output,
  ffn: step_ffn_expand,
  out: step_block_output,
};

// ── Slideshow infrastructure ─────────────────────────────────────────
let currentStep = 0;
let playing = false;
let autoplayTimer = null;

function showStep(n) {
  n = Math.max(0, Math.min(STEPS.length - 1, n));
  currentStep = n;
  const step = STEPS[n];

  document.querySelectorAll('.pipe-step').forEach((el, i) => {
    el.classList.remove('done', 'active');
    if (i < n) el.classList.add('done');
    else if (i === n) el.classList.add('active');
  });

  document.getElementById('panel').innerHTML = RENDERERS[step.id]();

  // Wire up step-specific interactions
  if (step.id === 'softmax') {
    // Run an automated walk: raw → exp → norm at 1.5s intervals
    setTimeout(() => applySoftmaxStage('raw'), 80);
    setTimeout(() => applySoftmaxStage('exp'), 1500);
    setTimeout(() => applySoftmaxStage('norm'), 3000);
    document.querySelectorAll('.sm-btn').forEach(btn => {
      btn.addEventListener('click', () => applySoftmaxStage(btn.dataset.stage));
    });
  }

  document.getElementById('step-pill').textContent = `step ${n + 1} / ${STEPS.length}`;
  document.getElementById('scrub').value = String(n);
}

document.getElementById('btn-prev').addEventListener('click', () => showStep(currentStep - 1));
document.getElementById('btn-next').addEventListener('click', () => showStep(currentStep + 1));
document.getElementById('btn-play').addEventListener('click', togglePlay);
document.getElementById('scrub').addEventListener('input', e => showStep(parseInt(e.target.value)));

document.addEventListener('keydown', e => {
  if (e.code === 'ArrowLeft') showStep(currentStep - 1);
  if (e.code === 'ArrowRight') showStep(currentStep + 1);
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
});

function setPlayIcon() {
  document.getElementById('btn-play').textContent = playing ? '❚❚' : '▶';
}

function togglePlay() {
  playing = !playing;
  setPlayIcon();
  if (playing) {
    if (currentStep >= STEPS.length - 1) showStep(0);
    autoplayTimer = setInterval(tickAutoplay, 4000 / parseFloat(
      document.getElementById('speed').value));
  } else {
    if (autoplayTimer) clearInterval(autoplayTimer);
  }
}

function tickAutoplay() {
  if (currentStep >= STEPS.length - 1) {
    playing = false;
    setPlayIcon();
    if (autoplayTimer) clearInterval(autoplayTimer);
    return;
  }
  showStep(currentStep + 1);
}

document.getElementById('speed').addEventListener('change', () => {
  if (playing) {
    if (autoplayTimer) clearInterval(autoplayTimer);
    autoplayTimer = setInterval(tickAutoplay, 4000 / parseFloat(
      document.getElementById('speed').value));
  }
});

// Init
showStep(0);
</script>
</body></html>"""
