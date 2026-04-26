"""Animated renderer — scene-based timeline that plays through the trace.

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

def _build_anim_payload(trace) -> dict[str, Any]:
    m = trace.model_meta or {}
    # Flatten the attention dict into a single featured head (prefer L0H0).
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
        "attention_featured_head": attn_key,
        "attention_matrix": attn_matrix,
        "probs_top_first_step": probs_top_first,
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
  position:absolute;inset:0 0 100px 0;
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

.norm-layers{display:flex;flex-direction:column;gap:6px}
.norm-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px}
.norm-row .lbl{min-width:60px;color:var(--muted);text-align:right}
.norm-row .bar{flex:1;height:12px;background:var(--surface2);border-radius:3px;overflow:hidden}
.norm-row .fill{height:100%;background:linear-gradient(90deg,#4da6ff,#00d4aa);
  width:0;transition:width .35s cubic-bezier(.4,0,.2,1)}
.norm-row .val{min-width:50px;color:var(--teal);text-align:right}

.attn-svg{width:100%;height:220px;display:block}

.logits-bars{display:flex;flex-direction:column;gap:4px;margin-top:4px}
.logit-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;opacity:0;transition:opacity .25s}
.logit-row.show{opacity:1}
.logit-row .tk{min-width:80px;text-align:right;color:var(--text);
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.logit-row .bar{flex:1;height:14px;background:var(--surface2);border-radius:3px;overflow:hidden}
.logit-row .fill{height:100%;background:var(--blue);width:0;transition:width .35s}
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
.scrub-labels{display:flex;justify-content:space-between;
  font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:4px}
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
    <div class="act2-grid">
      <div class="act2-card">
        <h3>Residual-stream norm across layers</h3>
        <div class="norm-layers" id="norm-layers"></div>
      </div>
      <div class="act2-card">
        <h3>Attention heatmap (<span id="attn-key"></span>)</h3>
        <svg class="attn-svg" id="attn-svg"></svg>
      </div>
    </div>
    <div class="act2-card" style="margin-top:16px">
      <h3>Logits → softmax → top-10 probabilities (argmax wins)</h3>
      <div class="logits-bars" id="logit-bars"></div>
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
        <h3>‖h‖ by layer — current step</h3>
        <svg class="mini-spark" id="norm-spark"></svg>
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
  </div>

  <div class="controls">
    <button class="btn" id="play-btn">⏵</button>
    <button class="btn secondary" id="restart-btn" title="Restart">↻</button>
    <div class="act-pill" id="act-pill">Act I</div>
    <div class="scrub-wrap">
      <input type="range" class="scrub" id="scrub" min="0" max="1000" value="0" step="1">
      <div class="scrub-labels">
        <span>0s</span>
        <span id="lbl-a1">Act I</span>
        <span id="lbl-a2">Act II</span>
        <span id="lbl-a3">Act III</span>
        <span id="lbl-end">end</span>
      </div>
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
const T_ACT1 = 4.5;
const T_ACT2 = T_ACT1 + 12;
const STEP_DUR = 0.35;   // seconds per generation step at 1× speed
const LOOP_DUR = Math.max(6, D.loop_steps.length * STEP_DUR);
const T_END = T_ACT2 + LOOP_DUR + 2.5;

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

// Norm bars
const normEl = document.getElementById('norm-layers');
const norms = D.hidden_norms_first_step || [];
const maxNorm = Math.max(...norms, 1);
norms.forEach((n, i) => {
  const row = document.createElement('div');
  row.className = 'norm-row';
  const label = i === 0 ? 'input' : `L${i}`;
  row.innerHTML = `<div class="lbl">${label}</div><div class="bar"><div class="fill" id="nfill-${i}"></div></div><div class="val" id="nval-${i}">0.0</div>`;
  normEl.appendChild(row);
});

// Attention SVG
const attnSvg = document.getElementById('attn-svg');
document.getElementById('attn-key').textContent = D.attention_featured_head || 'none';

function drawAttention(progress){
  const mat = D.attention_matrix;
  if (!mat) { attnSvg.innerHTML = '<text x="10" y="20" fill="#5a6888" font-size="11">no attention slice cached</text>'; return; }
  const n = mat.length;
  const size = Math.min(38, Math.floor(220 / n));
  const pad = 30;
  const W = pad + n*size + 6, H = pad + n*size + 6;
  attnSvg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  attnSvg.setAttribute('width', W);
  attnSvg.setAttribute('height', H);
  let g = '';
  const maxRow = Math.ceil(n * progress);
  for (let i = 0; i < n; i++){
    for (let j = 0; j < n; j++){
      const v = mat[i][j];
      const x = pad + j*size, y = pad + i*size;
      if (v === null || j > i){
        g += `<rect x="${x}" y="${y}" width="${size-2}" height="${size-2}" rx="2" fill="#0a0c14" opacity="0.4"/>`;
      } else {
        const opacity = i < maxRow ? 0.15 + v * 0.85 : 0.03;
        g += `<rect x="${x}" y="${y}" width="${size-2}" height="${size-2}" rx="2" fill="rgba(0,${Math.round(212*v)},${Math.round(170*v)},${opacity})"/>`;
      }
    }
    g += `<text x="${pad-3}" y="${pad + i*size + size/2 + 3}" text-anchor="end" font-size="9" fill="#5a6888" font-family="monospace">${escHtml((D.tokens[i]||'').trim()||'·')}</text>`;
  }
  attnSvg.innerHTML = g;
}

// Logits rows
const logitsEl = document.getElementById('logit-bars');
D.probs_top_first_step.forEach((p, i) => {
  const row = document.createElement('div');
  row.className = 'logit-row' + (i === 0 ? ' winner' : '');
  row.innerHTML = `<div class="tk">${escHtml(p.token)}</div><div class="bar"><div class="fill" id="lfill-${i}"></div></div><div class="pct" id="lpct-${i}">0%</div>`;
  logitsEl.appendChild(row);
});

function setPipeActive(idx){
  const boxes = pipeEl.querySelectorAll('.pipe-box');
  boxes.forEach((b, i) => {
    b.classList.toggle('active', i === idx);
    b.classList.toggle('done', i < idx);
  });
}

function updateAct2(t){
  const dt = t - T_ACT1;
  if (dt < 0){ setPipeActive(-1); return; }

  // 0-1s: pipeline highlights input/embedding
  // 1-6s: layers activate one by one, norm bars fill
  // 6-8s: attention glows row by row
  // 8-10s: logits bars grow, winner highlighted
  // 10-12s: argmax emphasized, pipe "Argmax" active

  if      (dt < 0.8) setPipeActive(0);
  else if (dt < 1.6) setPipeActive(1);
  else if (dt < 6)   setPipeActive(2 + Math.min(2, Math.floor((dt-1.6)/1.5)));
  else if (dt < 7.5) setPipeActive(5);
  else if (dt < 8.5) setPipeActive(6);
  else if (dt < 9.5) setPipeActive(7);
  else               setPipeActive(8);

  // Norm fills
  const layerProgress = Math.min(1, Math.max(0, (dt - 1.6) / 4.4));
  const layersToShow = Math.floor(layerProgress * norms.length);
  for (let i = 0; i < norms.length; i++){
    const fill = document.getElementById(`nfill-${i}`);
    const val = document.getElementById(`nval-${i}`);
    if (i <= layersToShow){
      fill.style.width = Math.min(100, (norms[i] / maxNorm) * 100) + '%';
      val.textContent = norms[i].toFixed(1);
    } else {
      fill.style.width = '0%';
      val.textContent = '0.0';
    }
  }

  // Attention reveal
  const attnProgress = Math.max(0, Math.min(1, (dt - 5.5) / 2.5));
  drawAttention(attnProgress);

  // Logits fill
  const logitProgress = Math.max(0, Math.min(1, (dt - 8) / 1.5));
  D.probs_top_first_step.forEach((p, i) => {
    const fill = document.getElementById(`lfill-${i}`);
    const pct = document.getElementById(`lpct-${i}`);
    const row = logitsEl.children[i];
    if (logitProgress > i / D.probs_top_first_step.length){
      fill.style.width = Math.max(3, p.prob * 100 / (D.probs_top_first_step[0].prob || 1) * 100) + '%';
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

const normSparkEl = document.getElementById('norm-spark');

function drawSparkline(norms){
  if (!norms || norms.length < 2) { normSparkEl.innerHTML = ''; return; }
  const W = 280, H = 80;
  const maxN = Math.max(...norms, 1);
  const xs = i => (i/(norms.length-1)) * (W-10) + 5;
  const ys = v => H - 8 - (v/maxN) * (H-16);
  const path = norms.map((v,i)=>`${i===0?'M':'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');
  const lastX = xs(norms.length-1), lastY = ys(norms[norms.length-1]);
  normSparkEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  normSparkEl.innerHTML = `
    <path d="${path}" stroke="#4da6ff" stroke-width="2" fill="none"/>
    <circle cx="${lastX}" cy="${lastY}" r="4" fill="#00d4aa"/>
    <text x="${lastX-6}" y="${lastY-8}" font-size="10" text-anchor="end" fill="#00d4aa" font-family="monospace">${norms[norms.length-1].toFixed(1)}</text>
  `;
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
      drawSparkline([]);
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

  drawSparkline(s.hidden_norms);

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

function render(){
  showScene(t);
  updateAct1(t);
  updateAct2(t);
  updateAct3(t);
  updateEnd(t);
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
        out_path = out_dir / f"{_slug(trace.model_id)}__{_slug(trace.prompt)}__anim.html"
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
