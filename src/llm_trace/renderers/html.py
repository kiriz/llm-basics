"""HTML renderer — self-contained interactive visualization.

Single-page design (simpler than the original llm_flow_visual.py's 10-page
router but covers the same 10 sections). The HTML/CSS/JS template is a
Python constant in this file — no external template directory.

Fixed bugs from the original:
    1. Broken attention head switcher (renderAttnHead undefined).
    2. Unused D3 CDN import.
    3. Missing `</script>` escape in injected JSON.

Must not import torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_trace.renderers._util import html_escape as _html_escape
from llm_trace.renderers._util import slug as _slug

# ── Data shaping ──────────────────────────────────────────────────────────

def _build_viz_data(trace) -> dict[str, Any]:
    """Transform TraceData into the shape the HTML template consumes."""
    m = trace.model_meta

    # Prefer the collector's one-shot detokenization (correct for sentencepiece);
    # fall back to string-joining individual token strings for legacy traces.
    gen_text = getattr(trace, "generation_text", None) or "".join(
        s["token"] for s in trace.generation if not s.get("is_eos")
    )
    full_output = trace.prompt + gen_text

    embeddings = []
    has_pos = trace.embeddings_pos is not None
    for i, (tok, tid) in enumerate(zip(trace.tokens, trace.token_ids, strict=False)):
        entry = {
            "token": tok,
            "id": int(tid),
            "token_vec": trace.embeddings_token[i].tolist(),
            "combined": trace.embeddings_combined[i].tolist(),
            "pos_vec": trace.embeddings_pos[i].tolist() if has_pos else None,
        }
        embeddings.append(entry)

    # Attention: build a (n_heads_total, seq, seq) structure with None for
    # heads not in the cache. The JS "head switcher" disables tabs for None.
    seq = len(trace.tokens)
    n_heads_total = int(m.get("n_head") or 1)
    cached_layer0 = {
        int(k[3:]): trace.attentions[k]
        for k in trace.attentions
        if k.startswith("L0H")
    }
    attn_heads = []
    for h in range(n_heads_total):
        if h in cached_layer0:
            mat = cached_layer0[h]
            rows = [
                [float(mat[i, j]) if j <= i else None for j in range(seq)]
                for i in range(seq)
            ]
            attn_heads.append(rows)
        else:
            attn_heads.append(None)

    hidden_layers = []
    for i in range(trace.hidden_last.shape[0]):
        hidden_layers.append({
            "layer": i,
            "label": "input" if i == 0 else f"L{i}",
            "vec": trace.hidden_last[i][:16].tolist(),
            "norm": float(trace.hidden_last_norms[i]),
        })

    logits_top = [
        {"token": tok, "id": int(tid), "logit": float(v)}
        for tok, tid, v in zip(
            trace.logits_top_tokens, trace.logits_top_ids, trace.logits_top_values, strict=False
        )
    ]
    probs_top = [
        {"token": tok, "id": int(tid), "prob": float(v)}
        for tok, tid, v in zip(
            trace.probs_top_tokens, trace.probs_top_ids, trace.probs_top_values, strict=False
        )
    ]

    temp_data = {}
    for block in trace.temp_scan:
        temp_data[f"{block['temperature']:g}"] = [
            {"token": e["token"], "prob": float(e["prob"])} for e in block["top"]
        ]

    # Per-step generation data: combine the dict-based `generation` with the
    # array-based per_step_* fields into one list-of-dicts for the HTML table.
    per_step = []
    len(trace.generation)
    psn = trace.per_step_hidden_norms
    psi = trace.per_step_top_ids
    psp = trace.per_step_top_probs
    per_token_ms = trace.timings.get("per_token_ms", [])

    for idx, s in enumerate(trace.generation):
        norms_row = psn[idx].tolist() if idx < psn.shape[0] else []
        top_ids_row = psi[idx].tolist() if idx < psi.shape[0] else []
        top_probs_row = psp[idx].tolist() if idx < psp.shape[0] else []
        # Decode top-K tokens. We don't carry a tokenizer here; best-effort
        # uses top_alts (always decoded) and known prompt tokens.
        lookup = {int(a["id"]): a["token"] for a in s.get("top_alts", [])}
        top_tokens_row = [lookup.get(int(tid), f"#{tid}") for tid in top_ids_row]
        ms = float(per_token_ms[idx]) if idx < len(per_token_ms) else 0.0
        per_step.append({
            "step": int(s["step"]),
            "token": s["token"],
            "id": int(s["id"]),
            "prob": float(s["prob"]),
            "ctx_len": int(s["ctx_len"]),
            "is_eos": bool(s.get("is_eos", False)),
            "ms": ms,
            "top_alts": [{"token": a["token"], "prob": float(a["prob"])}
                         for a in s.get("top_alts", [])[:5]],
            "hidden_norms": norms_row,
            "top_ids": top_ids_row,
            "top_probs": top_probs_row,
            "top_tokens": top_tokens_row,
        })

    return {
        "prompt": trace.prompt,
        "system_prompt": getattr(trace, "system_prompt", None),
        "templated_prompt": trace.templated_prompt,
        "full_output": full_output,
        "model_id": trace.model_id,
        "model_meta": m,
        "fwd_ms": trace.timings.get("first_forward_ms", 0),
        "per_token_ms": trace.timings.get("per_token_ms", []),
        "has_pos_embed": has_pos,
        "tokens": trace.tokens,
        "token_ids": list(trace.token_ids),
        "embeddings": embeddings,
        "attn_heads": attn_heads,
        "hidden_layers": hidden_layers,
        "logits_top": logits_top,
        "probs_top": probs_top,
        "temp_data": temp_data,
        "generated": per_step,
        "gen_params": trace.gen_params,
    }


# ── HTML template ─────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM trace — __TITLE__</title>
<style>
:root{--bg:#080b12;--surface:#0e1220;--surface2:#151b2e;--border:#1f2a45;--text:#cdd6f0;--muted:#5a6888;--teal:#00d4aa;--amber:#ffb800;--coral:#ff6b6b;--blue:#4da6ff;--purple:#c084fc;--mono:'SF Mono',Menlo,monospace;--sans:system-ui,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.45}
.wrap{max-width:1100px;margin:0 auto;padding:40px 32px 80px}
h1{font-size:28px;font-weight:800;margin-bottom:4px}
h1 .m{color:var(--teal)}
.sub{font-family:var(--mono);color:var(--muted);font-size:13px;margin-bottom:28px}
.section{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:16px}
.step-num{font-family:var(--mono);font-size:10px;color:var(--teal);letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px}
.step-title{font-size:20px;font-weight:700;margin-bottom:8px}
.step-desc{color:var(--muted);font-size:13px;margin-bottom:16px;max-width:720px;font-family:var(--mono)}
.stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px}
.stat .k{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.stat .v{font-family:var(--mono);font-size:18px;font-weight:600;color:var(--teal);margin-top:2px}
.stat .s{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:2px}
.insight{margin-top:12px;padding:12px 16px;border-left:3px solid var(--teal);background:rgba(0,212,170,0.05);font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.55}
.insight strong{color:var(--text)}
.chip{display:inline-flex;flex-direction:column;gap:2px;margin:4px 6px;align-items:center}
.chip .t{padding:6px 12px;border-radius:6px;font-family:var(--mono);font-size:13px;background:var(--surface2);border:1px solid var(--border)}
.chip .i{font-family:var(--mono);font-size:10px;color:var(--muted)}
table.plain{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12px}
table.plain th{color:var(--muted);text-transform:uppercase;letter-spacing:.1em;font-size:10px;text-align:left;padding:6px 10px;border-bottom:1px solid var(--border)}
table.plain td{padding:6px 10px;border-bottom:1px solid var(--border)}
table.plain td.tok{color:var(--amber)}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.bar-label{font-family:var(--mono);font-size:12px;min-width:120px;text-align:right;color:var(--text)}
.bar-track{flex:1;height:20px;background:var(--surface2);border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;padding:0 8px;display:flex;align-items:center;font-family:var(--mono);font-size:11px;color:#000;font-weight:600;transition:width .3s}
.heat{display:inline-block;vertical-align:top}
.heat rect{stroke:var(--bg);stroke-width:.5}
.heat-lbl{font-family:var(--mono);font-size:10px;fill:var(--muted)}
.attn-tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px}
.attn-tab{padding:3px 10px;border-radius:14px;font-family:var(--mono);font-size:10px;background:var(--surface2);border:1px solid var(--border);cursor:pointer;color:var(--muted)}
.attn-tab.active{background:var(--teal);color:#000;border-color:var(--teal);font-weight:600}
.attn-tab:disabled{opacity:.3;cursor:not-allowed}
.slider-row{display:flex;align-items:center;gap:14px;padding:12px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;margin-bottom:12px}
.slider-row label{font-family:var(--mono);font-size:12px;min-width:100px}
.slider-row input[type=range]{flex:1;accent-color:var(--teal)}
.slider-row .val{font-family:var(--mono);font-size:16px;color:var(--teal);font-weight:600;min-width:40px;text-align:right}
.gen-output{font-family:var(--mono);font-size:18px;line-height:1.7;padding:20px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;margin-bottom:16px}
.gen-output .orig{color:var(--muted)}
.gtok{display:inline-block;padding:2px 5px;margin:0 1px;border-radius:3px;position:relative;cursor:default}
.gtok .tip{position:absolute;bottom:100%;left:50%;transform:translateX(-50%);background:var(--surface2);border:1px solid var(--border);padding:6px 10px;border-radius:4px;font-size:10px;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity .15s;margin-bottom:4px;z-index:10;color:var(--text)}
.gtok:hover .tip{opacity:1}
svg{display:block;max-width:100%}
</style>
</head>
<body>
<div class="wrap">
<h1>LLM trace — <span class="m">__MODEL_ID__</span></h1>
<div class="sub" id="sub-line"></div>
<div id="root"></div>
</div>
<script>
const D = __DATA__;

// ── Helpers ─────────────────────────────────────────────────────────────
function esc(s){return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function probColor(p){if(p>.4)return '#00d4aa';if(p>.15)return '#ffb800';if(p>.05)return '#4da6ff';return '#5a6888'}
function fmt(x,d){return x.toFixed(d)}
function heatColor(v,vmin,vmax){const t=(v-vmin)/(vmax-vmin+1e-9);if(t<.5){const s=t*2;return `rgb(${26+s*179|0},${157+s*55|0},${223+s*-53|0})`}else{const s=(t-.5)*2;return `rgb(${205+s*50|0},${212+s*-28|0},${170+s*-170|0})`}}

// ── Title line ──────────────────────────────────────────────────────────
document.getElementById('sub-line').textContent =
  (D.system_prompt ? `system: ${JSON.stringify(D.system_prompt)}  ·  ` : '')
  + `prompt: ${JSON.stringify(D.prompt)}  ·  gen: ${JSON.stringify(D.gen_params)}  ·  first fwd: ${fmt(D.fwd_ms,0)}ms`;

// ── Section builders ────────────────────────────────────────────────────
function section(num, title, desc, body){
  return `<div class="section">
    <div class="step-num">${num}</div>
    <div class="step-title">${title}</div>
    ${desc?`<div class="step-desc">${desc}</div>`:''}
    ${body}
  </div>`;
}

function archCard(){
  const m = D.model_meta || {};
  const cells = [
    ['Model type', m.model_type, ''],
    ['Layers', m.n_layer, 'transformer blocks'],
    ['Hidden dim', m.hidden_size, 'residual stream'],
    ['Attn heads', m.n_head, ''],
    ['KV heads', m.n_kv_heads, m.n_head&&m.n_kv_heads&&m.n_head!==m.n_kv_heads ? `GQA ${m.n_head/m.n_kv_heads}×` : ''],
    ['FFN inter.', m.ffn_intermediate, ''],
    ['Positional', m.positional_encoding, m.rope_theta?`θ=${m.rope_theta}`:''],
    ['Normalization', m.normalization, ''],
    ['Vocabulary', (m.vocab_size||0).toLocaleString(), ''],
    ['Ctx window', m.ctx_window, 'hard ceiling'],
    ['EOS ids', JSON.stringify(m.eos_ids||[]), (m.eos_tokens||[]).map(s=>JSON.stringify(s)).join(' ')],
  ];
  const rows = cells.map(([k,v,s])=>`<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v===null||v===undefined?'?':v)}</div><div class="s">${esc(s)}</div></div>`).join('');
  return `<div class="stat-row">${rows}</div>`;
}

function step1(){
  const chips = D.tokens.map((t,i)=>`<span class="chip"><span class="t">${esc(t)}</span><span class="i">pos ${i} · id ${D.token_ids[i]}</span></span>`).join('');
  return section('Step 01', 'Tokenization',
    `Text splits into subword pieces; each maps to an integer id. Deterministic — same text → same ids.`,
    `<div style="padding:4px 0">${chips}</div>
     <div style="margin-top:12px;font-family:var(--mono);font-size:12px;color:var(--muted)">
       ${D.tokens.length} tokens from ${D.prompt.length} chars
       (ratio ${(D.prompt.length/Math.max(D.tokens.length,1)).toFixed(1)} chars/token)
     </div>`);
}

function step2(){
  return section('Step 02', 'Input tensor',
    `Integer tensor fed to model.forward(). Batch size 1, sequence length ${D.tokens.length}.`,
    `<div style="font-family:var(--mono);color:var(--teal);font-size:14px">
       shape: [1, ${D.token_ids.length}]
     </div>
     <div style="font-family:var(--mono);color:var(--amber);font-size:14px;margin-top:6px;word-break:break-all">
       values: [${D.token_ids.join(', ')}]
     </div>`);
}

function step3(){
  // Embedding heatmap: rows = tokens, cols = first N dims of token_vec
  if (!D.embeddings.length) return '';
  const ndims = D.embeddings[0].token_vec.length;
  const cellW = Math.min(20, Math.floor(720/ndims));
  const cellH = 28;
  const labelW = 90;
  const W = labelW + ndims*cellW + 20;
  const H = D.embeddings.length*cellH + 36;
  const allVals = D.embeddings.flatMap(e=>e.token_vec);
  const vmin = Math.min(...allVals), vmax = Math.max(...allVals);
  let cells = '';
  D.embeddings.forEach((e,i)=>{
    e.token_vec.forEach((v,j)=>{
      cells += `<rect x="${labelW+j*cellW}" y="${24+i*cellH}" width="${cellW-1}" height="${cellH-4}" rx="1" fill="${heatColor(v,vmin,vmax)}"><title>${esc(e.token)} d${j}: ${v.toFixed(3)}</title></rect>`;
    });
    cells += `<text x="${labelW-6}" y="${24+i*cellH+cellH/2+3}" text-anchor="end" class="heat-lbl">${esc(e.token.trim()||'·')}</text>`;
  });
  return section('Step 03', 'Embedding lookup',
    `Each id indexes into the embedding matrix (${(D.model_meta.vocab_size||0).toLocaleString()} × ${D.model_meta.hidden_size||'?'}). Shown: first ${ndims} dims of each token's embedding vector.`,
    `<div style="overflow-x:auto"><svg class="heat" width="${W}" height="${H}">${cells}</svg></div>
     <div style="display:flex;gap:10px;align-items:center;margin-top:8px;font-family:var(--mono);font-size:10px;color:var(--muted)">
       <div style="height:8px;width:120px;background:linear-gradient(to right,#1a9ddf,#cdd4ea,#ffb800);border-radius:4px"></div>
       <span>negative ← 0 → positive</span>
     </div>`);
}

function step4(){
  if (!D.has_pos_embed){
    return section('Step 04', 'Positional encoding',
      `This model uses <b>${esc(D.model_meta.positional_encoding||'unknown')}</b>, not a learned position-embedding table.`,
      `<div style="font-family:var(--mono);font-size:13px;color:var(--muted)">
         Position info enters INSIDE attention (via rotations on Q/K), so the input
         to layer 0 = token embeddings unchanged.
       </div>`);
  }
  const e = D.embeddings[0];
  const makeBar = (vec, label, color) => {
    const max = Math.max(...vec.map(Math.abs)) + 1e-9;
    const bars = vec.map((v,i)=>{
      const h = Math.abs(v)/max*30;
      const y = v>=0 ? 32-h : 32;
      return `<rect x="${2+i*8}" y="${y}" width="6" height="${h}" fill="${v>=0?color:'#ff6b6b'}" rx="1"/>`;
    }).join('');
    return `<div style="margin-bottom:14px"><div style="font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:4px">${label}</div>
      <svg width="${vec.length*8+4}" height="64" style="background:var(--surface2);border-radius:6px">${bars}</svg></div>`;
  };
  return section('Step 04', 'Positional encoding (learned)',
    `A positional vector is added elementwise to each token embedding. Shown: token[0] breakdown.`,
    `${makeBar(e.token_vec, `token embedding (id=${e.id})`, '#00d4aa')}
     ${makeBar(e.pos_vec, 'position 0 embedding', '#ffb800')}
     ${makeBar(e.combined, 'combined (token + pos)', '#4da6ff')}`);
}

function step5(){
  const norms = D.hidden_layers.map(h=>h.norm);
  const maxN = Math.max(...norms);
  const W = 720, H = 180, padL = 40, padB = 32, padT = 10;
  const innerW = W-padL-20, innerH = H-padT-padB;
  const xs = i => padL + (i/(D.hidden_layers.length-1))*innerW;
  const ys = v => padT + innerH - (v/maxN)*innerH;
  const path = D.hidden_layers.map((h,i)=>`${i===0?'M':'L'}${xs(i)},${ys(h.norm)}`).join(' ');
  const dots = D.hidden_layers.map((h,i)=>
    `<circle cx="${xs(i)}" cy="${ys(h.norm)}" r="4" fill="${i===0?'#5a6888':i===D.hidden_layers.length-1?'#00d4aa':'#4da6ff'}"><title>${h.label}: ‖h‖=${h.norm.toFixed(2)}</title></circle>`).join('');
  const labels = D.hidden_layers.filter((_,i)=>i%Math.max(1,Math.floor(D.hidden_layers.length/7))===0 || i===D.hidden_layers.length-1)
    .map(h=>`<text x="${xs(h.layer)}" y="${H-10}" text-anchor="middle" class="heat-lbl">${h.label}</text>`).join('');
  return section('Step 05', `Forward pass — ${D.hidden_layers.length-1} layers`,
    `Hidden-state magnitude (‖h‖) at the last token position across layer outputs. The residual stream grows as each layer adds to it.`,
    `<svg width="${W}" height="${H}" style="width:100%">
       <path d="${path}" fill="none" stroke="#00d4aa" stroke-width="2.5"/>
       ${dots}${labels}
       <text x="${padL-4}" y="${padT+5}" text-anchor="end" class="heat-lbl">${maxN.toFixed(0)}</text>
       <text x="${padL-4}" y="${padT+innerH}" text-anchor="end" class="heat-lbl">0</text>
     </svg>`);
}

function step6(){
  const tabs = D.attn_heads.map((h,i)=>{
    const cached = h !== null;
    return `<button class="attn-tab${i===0 && cached?' active':''}" data-head="${i}" ${cached?'':'disabled title="not cached — widen attention_heads in config"'}>H${i}</button>`;
  }).join('');
  return section('Step 06', 'Attention weights (layer 0)',
    `Each row = one token. Column = attention weight onto another. Upper-right is blank because of the causal mask. Only heads in the cache are enabled; the rest require a re-run with a wider slicing policy.`,
    `<div class="attn-tabs" id="attn-tabs">${tabs}</div>
     <div id="attn-heatmap"></div>`);
}

function renderAttnHead(h){
  const mat = D.attn_heads[h];
  const container = document.getElementById('attn-heatmap');
  if (!mat){ container.innerHTML = `<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:12px">Head ${h} not cached.</div>`; return; }
  const seq = D.tokens.length;
  const cellSize = Math.min(64, Math.floor(520/Math.max(seq,1)));
  const labelW = 80, topPad = 50;
  const W = labelW + seq*cellSize + 20;
  const H = topPad + seq*cellSize + 10;
  let cells = '';
  let vmax = 0;
  mat.forEach(row=>row.forEach(v=>{if(v!==null)vmax=Math.max(vmax,v)}));
  for (let i=0;i<seq;i++){
    for (let j=0;j<seq;j++){
      const v = mat[i][j];
      const x = labelW+j*cellSize, y = topPad+i*cellSize;
      if (v===null){
        cells += `<rect x="${x}" y="${y}" width="${cellSize-2}" height="${cellSize-2}" rx="3" fill="#0a0c14" opacity=".5"/>`;
      } else {
        const t = v/(vmax+1e-9);
        cells += `<rect x="${x}" y="${y}" width="${cellSize-2}" height="${cellSize-2}" rx="3" fill="rgba(0,${212*t|0},${170*t|0},${.15+t*.85})"><title>${esc(D.tokens[i])}→${esc(D.tokens[j])}: ${v.toFixed(3)}</title></rect>`;
        if (cellSize>28) cells += `<text x="${x+cellSize/2-1}" y="${y+cellSize/2+4}" text-anchor="middle" font-family="var(--mono)" font-size="${cellSize>42?11:9}" fill="${t>.5?'#000':'#cdd6f0'}">${v.toFixed(2)}</text>`;
      }
    }
    cells += `<text x="${labelW-6}" y="${topPad+i*cellSize+cellSize/2+4}" text-anchor="end" class="heat-lbl">${esc(D.tokens[i].trim()||'·')}</text>`;
    cells += `<text x="${labelW+i*cellSize+cellSize/2}" y="${topPad-8}" text-anchor="middle" class="heat-lbl">${esc(D.tokens[i].trim()||'·')}</text>`;
  }
  container.innerHTML = `<svg width="${W}" height="${H}" style="width:100%;max-width:${W}px">${cells}</svg>`;
}

function wireAttnTabs(){
  document.querySelectorAll('#attn-tabs .attn-tab').forEach(btn=>{
    if (btn.disabled) return;
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('#attn-tabs .attn-tab').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      renderAttnHead(parseInt(btn.dataset.head));
    });
  });
  const firstOk = D.attn_heads.findIndex(h=>h!==null);
  if (firstOk>=0) renderAttnHead(firstOk);
}

function step7(){
  const max = Math.max(...D.logits_top.map(l=>l.logit));
  const min = Math.min(...D.logits_top.map(l=>l.logit));
  const bars = D.logits_top.map((l,i)=>{
    const w = ((l.logit-min)/(max-min+1e-9)*100).toFixed(1);
    const color = i===0?'#00d4aa':i<3?'#ffb800':'#4da6ff';
    return `<div class="bar-row"><div class="bar-label">${esc(l.token)}</div><div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}">${l.logit.toFixed(2)}</div></div></div>`;
  }).join('');
  return section('Step 07', 'Logits', `Raw scores — one per vocabulary token (${(D.model_meta.vocab_size||0).toLocaleString()} total). Range for top ${D.logits_top.length}: ${min.toFixed(1)} to ${max.toFixed(1)}.`, bars);
}

function step8(){
  const total = D.probs_top.reduce((s,p)=>s+p.prob,0);
  const bars = D.probs_top.map((p,i)=>{
    const w = (p.prob/Math.max(D.probs_top[0].prob,1e-9)*100).toFixed(1);
    const color = i===0?'#00d4aa':probColor(p.prob);
    return `<div class="bar-row"><div class="bar-label">${esc(p.token)}</div><div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}">${(p.prob*100).toFixed(1)}%</div></div></div>`;
  }).join('');
  return section('Step 08', 'Probabilities (softmax)', `All ${(D.model_meta.vocab_size||0).toLocaleString()} tokens get a probability; top ${D.probs_top.length} cover ${(total*100).toFixed(1)}%.`, bars);
}

function step9(){
  const keys = Object.keys(D.temp_data);
  if (!keys.length) return '';
  const idx0 = Math.floor(keys.length/2);
  const initKey = keys[idx0];
  return section('Step 09', 'Sampling / temperature',
    `Temperature scales logits before softmax: low T → peaked, high T → flat. Drag the slider to see the top-10 distribution reshape.`,
    `<div class="slider-row">
       <label>Temperature</label>
       <input type="range" min="0" max="${keys.length-1}" value="${idx0}" id="temp-slider">
       <div class="val" id="temp-val">${initKey}</div>
     </div>
     <div id="temp-bars"></div>`);
}

function renderTempBars(key){
  const data = D.temp_data[key];
  const max = Math.max(...data.map(p=>p.prob), 1e-9);
  const html = data.map(p=>{
    const w = (p.prob/max*100).toFixed(1);
    return `<div class="bar-row"><div class="bar-label">${esc(p.token)}</div><div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${probColor(p.prob)}">${(p.prob*100).toFixed(1)}%</div></div></div>`;
  }).join('');
  document.getElementById('temp-bars').innerHTML = html;
}

function wireTempSlider(){
  const slider = document.getElementById('temp-slider');
  if (!slider) return;
  const keys = Object.keys(D.temp_data);
  const update = ()=>{
    const k = keys[parseInt(slider.value)];
    document.getElementById('temp-val').textContent = k;
    renderTempBars(k);
  };
  slider.addEventListener('input', update);
  update();
}

function step10(){
  const colors = ['#00d4aa','#ffb800','#4da6ff','#c084fc','#ff6b6b','#34d399','#fb923c','#a78bfa'];
  const toks = D.generated.map((g,i)=>{
    if (g.is_eos) return `<span class="gtok" style="background:#ff6b6b33;border:1px solid #ff6b6b;color:#ff6b6b">⏹ EOS</span>`;
    const c = colors[i%colors.length];
    const alts = (g.top_alts || []).slice(0,5);
    const tip = `${(g.prob*100).toFixed(1)}% — top: ${alts.map(t=>`${esc(t.token)} ${(t.prob*100).toFixed(0)}%`).join(' · ')}`;
    return `<span class="gtok" style="background:${c}22;border:1px solid ${c}66;color:${c}">${esc(g.token)}<span class="tip">${tip}</span></span>`;
  }).join('');
  const rows = D.generated.map((g,i)=>{
    const c = colors[i%colors.length];
    const eosFlag = g.is_eos ? '<span style="color:#ff6b6b;font-family:var(--mono);font-size:11px">EOS</span>' : '';
    return `<tr><td style="color:var(--muted)">t+${g.step}</td><td class="tok" style="color:${c}">${esc(g.token)}</td><td>${g.id}</td><td style="color:var(--teal)">${(g.prob*100).toFixed(1)}%</td><td>ctx=${g.ctx_len}</td><td>${eosFlag}</td></tr>`;
  }).join('');
  return section('Step 10', 'Autoregressive generation',
    `Append the chosen token, feed the whole sequence back in, repeat. Each step is one full forward pass. Hover any token for top-5 alternatives.`,
    `<div class="gen-output"><span class="orig">${esc(D.prompt)}</span>${toks}</div>
     <table class="plain">
       <thead><tr><th>step</th><th>token</th><th>id</th><th>prob</th><th>ctx</th><th>flags</th></tr></thead>
       <tbody>${rows}</tbody>
     </table>`);
}

// ── Forward-pass schematic (what happens inside each loop iteration) ───
function schematic(){
  const boxes = [
    ['Input IDs', 'shape=[1, seq_len]', '#5a6888'],
    ['Embedding', `lookup (${(D.model_meta.vocab_size||0).toLocaleString()} × ${D.model_meta.hidden_size||'?'})`, '#4da6ff'],
    ['Positional', D.has_pos_embed ? 'learned, added' : 'RoPE, inside attention', '#4da6ff'],
    [`${D.model_meta.n_layer||'?'} × TransformerBlock`, `attn (Q·Kᵀ/√d · V) → ${D.model_meta.normalization||'norm'} → FFN (${D.model_meta.ffn_intermediate||'?'})`, '#c084fc'],
    ['Final norm', D.model_meta.normalization||'norm', '#ffb800'],
    ['LM head', `linear → (${(D.model_meta.vocab_size||0).toLocaleString()} logits)`, '#00d4aa'],
    ['Sample', `argmax (T=${D.gen_params.temperature||0}) / EOS?`, '#ff6b6b'],
  ];
  const html = boxes.map((b,i)=>`
    <div style="display:flex;align-items:center;gap:6px">
      <div style="padding:10px 12px;border-radius:6px;background:${b[2]}22;border:1px solid ${b[2]}66;min-width:120px">
        <div style="font-family:var(--mono);font-size:11px;color:${b[2]};font-weight:700">${esc(b[0])}</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:2px">${esc(b[1])}</div>
      </div>
      ${i<boxes.length-1?'<div style="color:var(--muted);font-family:var(--mono)">→</div>':''}
    </div>`).join('');
  return section('Forward pass', 'What the model computes on every loop iteration',
    `This runs <b>once per generated token</b>. With ${D.generated.length} steps in this trace, it ran ${D.generated.length} times — each one a fresh pass over the entire context so far.`,
    `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px">${html}</div>
     <div style="margin-top:14px;font-family:var(--mono);font-size:11px;color:var(--muted)">
       Complexity per pass: O(seq² × hidden × layers) for attention, O(seq × hidden × ffn × layers) for FFN.
       Context grows each step, so later iterations are progressively slower.
     </div>`);
}

// ── Generation-loop table (every step the model took) ──────────────────
function genLoopTable(){
  if (!D.generated.length) return '';
  const n = D.generated.length;
  const maxCtx = Math.max(...D.generated.map(g=>g.ctx_len));
  const emitted = D.generated.filter(g=>g.is_eos).length > 0;

  const maxNormAll = Math.max(
    ...D.generated.flatMap(g => g.hidden_norms || [0]), 1e-9
  );

  const rows = D.generated.map(g=>{
    const eosFlag = g.is_eos ? '<span style="color:#ff6b6b;font-weight:700">EOS</span>' : '';
    const norms = g.hidden_norms || [];
    const spark = norms.length ? sparkline(norms, maxNormAll) : '';
    const top5bars = (g.top_alts||[]).slice(0,5).map(a=>{
      const w = Math.round(a.prob*100);
      const c = probColor(a.prob);
      return `<div style="display:flex;align-items:center;gap:4px;margin-bottom:1px"><div style="font-family:var(--mono);font-size:10px;min-width:60px;color:var(--text);text-align:right">${esc(a.token)}</div><div style="flex:1;height:8px;background:var(--surface2);border-radius:2px;overflow:hidden"><div style="width:${w}%;height:100%;background:${c}"></div></div><div style="font-family:var(--mono);font-size:9px;color:var(--muted);min-width:34px">${(a.prob*100).toFixed(0)}%</div></div>`;
    }).join('');
    // Color the ms cell: green < 30, amber < 80, red otherwise. Helps spot
    // slow steps as the context grows.
    const ms = g.ms || 0;
    const msColor = ms < 30 ? '#00d4aa' : ms < 80 ? '#ffb800' : '#ff6b6b';
    const msCell = `<td style="font-family:var(--mono);color:${msColor};text-align:right">${ms.toFixed(1)}</td>`;
    return `<tr>
      <td style="color:var(--muted);font-family:var(--mono)">t+${g.step}</td>
      <td style="font-family:var(--mono);color:var(--amber)">${esc(g.token)}</td>
      <td style="color:var(--teal);font-family:var(--mono)">${(g.prob*100).toFixed(1)}%</td>
      ${msCell}
      <td style="font-family:var(--mono);color:var(--muted)">${g.ctx_len}</td>
      <td>${spark}</td>
      <td style="min-width:230px">${top5bars}</td>
      <td>${eosFlag}</td>
    </tr>`;
  }).join('');

  const totalMs = D.generated.reduce((a,g)=>a+(g.ms||0),0);
  const avgMs = D.generated.length ? totalMs/D.generated.length : 0;
  const minMs = Math.min(...D.generated.map(g=>g.ms||0));
  const maxMs = Math.max(...D.generated.map(g=>g.ms||0));

  return section('Generation loop', `${n} step${n===1?'':'s'}, ${n} forward pass${n===1?'':'es'}`,
    `Every row is one full trip through the forward-pass schematic above. "‖h‖ by layer" is the residual-stream norm at the newly-generated token position; the sparkline shows how it grows as the representation passes through each layer. "ms" is wallclock time for that single forward pass (color: green < 30 ms, amber < 80, red beyond). ${emitted?'Loop ended when the model emitted EOS (red badge).':'Loop hit the max_new_tokens cap; model did not emit EOS on its own.'}`,
    `<div style="overflow-x:auto">
       <table class="plain" style="width:100%">
         <thead><tr>
           <th>step</th><th>token</th><th>prob</th><th style="text-align:right">ms</th>
           <th>ctx</th>
           <th>‖h‖ by layer</th><th>top-5 alternatives</th><th>flags</th>
         </tr></thead>
         <tbody>${rows}</tbody>
       </table>
     </div>
     <div style="margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--muted)">
       Max context: <b>${maxCtx}</b> tokens ·
       Wallclock per pass — min <b style="color:var(--teal)">${minMs.toFixed(1)} ms</b>,
       avg <b style="color:var(--teal)">${avgMs.toFixed(1)} ms</b>,
       max <b style="color:var(--amber)">${maxMs.toFixed(1)} ms</b>,
       total <b>${totalMs.toFixed(0)} ms</b>.
     </div>`);
}

function sparkline(vals, maxAll){
  const w = 120, h = 22;
  const n = vals.length;
  if (n < 2) return '';
  const xs = i => (i/(n-1))*(w-4)+2;
  const ys = v => h - 2 - (v/maxAll)*(h-4);
  const path = vals.map((v,i)=>`${i===0?'M':'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');
  const endDot = `<circle cx="${xs(n-1)}" cy="${ys(vals[n-1])}" r="2" fill="#00d4aa"/>`;
  return `<svg width="${w}" height="${h}" style="display:block"><path d="${path}" stroke="#4da6ff" stroke-width="1.3" fill="none"/>${endDot}<title>‖h‖ across ${n} layer outputs: ${vals.map(v=>v.toFixed(1)).join(' → ')}</title></svg>`;
}

// ── Mount ───────────────────────────────────────────────────────────────
document.getElementById('root').innerHTML = [
  section('Architecture', D.model_id, '', archCard()),
  step1(), step2(), step3(), step4(), step5(), step6(),
  step7(), step8(), step9(),
  schematic(),
  genLoopTable(),
  step10(),
].join('');

wireAttnTabs();
wireTempSlider();
</script>
</body>
</html>"""


_COMPARISON_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>LLM trace comparison — __TITLE__</title>
<style>
:root{--bg:#080b12;--surface:#0e1220;--surface2:#151b2e;--border:#1f2a45;--text:#cdd6f0;--muted:#5a6888;--teal:#00d4aa;--amber:#ffb800;--coral:#ff6b6b;--blue:#4da6ff;--mono:'SF Mono',Menlo,monospace;--sans:system-ui,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.45}
.wrap{max-width:1400px;margin:0 auto;padding:32px 24px}
h1{font-size:24px;font-weight:800;margin-bottom:4px}
.sub{font-family:var(--mono);color:var(--muted);font-size:12px;margin-bottom:24px}
.grid{display:grid;gap:14px}
.cell{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px;min-height:180px}
.cell h3{font-family:var(--mono);font-size:11px;color:var(--teal);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px}
.cell .mp{font-family:var(--mono);font-size:12px;color:var(--muted);margin-bottom:10px}
.cell .out{font-family:var(--mono);font-size:14px;color:var(--text);background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px;margin-bottom:10px;white-space:pre-wrap;word-break:break-word}
.cell .bars .bar-row{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.cell .bars .bl{font-family:var(--mono);font-size:11px;min-width:100px;text-align:right;color:var(--text)}
.cell .bars .bt{flex:1;height:14px;background:var(--surface2);border-radius:3px;overflow:hidden}
.cell .bars .bf{height:100%;padding:0 6px;display:flex;align-items:center;font-family:var(--mono);font-size:9px;color:#000;font-weight:600}
.cell .meta{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:10px}
.cell .eos{color:#ff6b6b;font-weight:600}
.empty{opacity:.4}
</style>
</head>
<body>
<div class="wrap">
<h1>Trace comparison — __TITLE__</h1>
<div class="sub">Rows: models. Columns: prompts. Hover a probability bar for raw value.</div>
<div class="grid" id="grid" style="grid-template-columns: repeat(__NCOLS__, 1fr)">__CELLS__</div>
</div>
</body>
</html>"""


# ── Public entry points ────────────────────────────────────────────────────

def render(trace, cfg: dict[str, Any] | None = None, out_path: Path | str | None = None) -> Path:
    """Write a self-contained HTML page visualizing the trace."""
    viz = _build_viz_data(trace)
    data_json = json.dumps(viz, ensure_ascii=False, default=str)
    data_json = data_json.replace("</", "<\\/")   # XSS guard for `</script>` in prompt

    if out_path is None:
        out_dir = Path((cfg or {}).get("out_dir", "./out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_slug(trace.model_id)}__{_slug(trace.prompt)}.html"
    out_path = Path(out_path)

    html = (_HTML_TEMPLATE
            .replace("__DATA__", data_json)
            .replace("__MODEL_ID__", _html_escape(trace.model_id))
            .replace("__TITLE__", _html_escape(f"{trace.model_id} — {trace.prompt[:40]}")))

    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_comparison(
    traces: list,
    cfg: dict[str, Any] | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Side-by-side grid: rows = distinct models, columns = distinct prompts."""
    if not traces:
        raise ValueError("render_comparison requires at least one trace")

    models = []
    prompts = []
    for t in traces:
        if t.model_id not in models:
            models.append(t.model_id)
        if t.prompt not in prompts:
            prompts.append(t.prompt)

    trace_by_cell: dict[tuple[str, str], Any] = {
        (t.model_id, t.prompt): t for t in traces
    }

    cells_html = []
    for model_id in models:
        for prompt in prompts:
            t = trace_by_cell.get((model_id, prompt))
            cells_html.append(_comparison_cell(model_id, prompt, t))

    title = f"{len(models)} models × {len(prompts)} prompts"
    html = (_COMPARISON_TEMPLATE
            .replace("__CELLS__", "\n".join(cells_html))
            .replace("__NCOLS__", str(len(prompts)))
            .replace("__TITLE__", _html_escape(title)))

    if out_path is None:
        out_dir = Path((cfg or {}).get("out_dir", "./out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "compare.html"
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _comparison_cell(model_id: str, prompt: str, trace) -> str:
    """Single cell of the comparison grid — shows top-5 probs + generated output."""
    header = (
        f'<h3>{_html_escape(model_id)}</h3>'
        f'<div class="mp">prompt: {_html_escape(repr(prompt))}</div>'
    )
    if trace is None:
        return f'<div class="cell empty">{header}<div class="out">not run</div></div>'

    output = trace.prompt + "".join(
        s["token"] for s in trace.generation if not s.get("is_eos")
    )
    had_eos = any(s.get("is_eos") for s in trace.generation)

    bars = []
    for tok, p in list(zip(trace.probs_top_tokens, trace.probs_top_values, strict=False))[:5]:
        p = float(p)
        color = "#00d4aa" if p > 0.4 else "#ffb800" if p > 0.15 else "#4da6ff"
        w = min(100, p * 100)
        bars.append(
            f'<div class="bar-row"><div class="bl">{_html_escape(repr(tok))}</div>'
            f'<div class="bt"><div class="bf" style="width:{w:.1f}%;background:{color}">'
            f'{p*100:.1f}%</div></div></div>'
        )
    bars_html = f'<div class="bars">{"".join(bars)}</div>'

    meta_bits = [
        f'{len(trace.tokens)} in-tokens',
        f'{len(trace.generation)} gen',
        f'fwd {trace.timings.get("first_forward_ms", 0):.0f}ms',
    ]
    if had_eos:
        meta_bits.append('<span class="eos">EOS</span>')

    return (
        f'<div class="cell">{header}'
        f'<div class="out">{_html_escape(output)}</div>'
        f'{bars_html}'
        f'<div class="meta">{" · ".join(meta_bits)}</div>'
        f'</div>'
    )


