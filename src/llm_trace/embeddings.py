"""Embedding matrix explorer — dump the model's wte and visualize it.

The metaphor: the embedding matrix is a library; each token id is a call number;
each row is the "book" (a 768-number vector). Tokens that mean similar things
end up with similar vectors → they sit on neighboring shelves.

Two-section HTML output:
  1. The lookup — a literal "id N → row N → 32-dim vector preview" demo
                   for each prompt token.
  2. The neighborhood — 2D PCA scatter of ~500 tokens (prompt + their nearest
                        neighbors + curated landmark groups) showing semantic
                        clustering.

CLI entry: `llm-trace embeddings --prompt "..."`
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── Curated landmark tokens (semantic anchor groups) ──────────────────────
# Each entry is a string with the leading-space variant — that's how GPT-2's
# BPE tokenizer represents most words mid-sentence, and they'll be single
# tokens (not split). Multi-token strings are skipped at lookup time.
CURATED_GROUPS: dict[str, list[str]] = {
    "months":   [" January", " February", " March", " April", " May", " June",
                 " July", " August", " September", " October", " November", " December"],
    "days":     [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday",
                 " Saturday", " Sunday"],
    "numbers":  [" one", " two", " three", " four", " five", " six", " seven",
                 " eight", " nine", " ten", " hundred", " thousand"],
    "animals":  [" cat", " dog", " bird", " fish", " lion", " tiger", " bear",
                 " horse", " cow", " pig", " sheep", " elephant"],
    "colors":   [" red", " blue", " green", " yellow", " black", " white",
                 " orange", " purple", " pink", " brown"],
    "verbs":    [" run", " walk", " eat", " drink", " sleep", " work", " play",
                 " read", " write", " think", " talk", " jump"],
    "adjectives": [" big", " small", " hot", " cold", " fast", " slow", " good",
                   " bad", " old", " new", " happy", " sad"],
    "punctuation": [".", ",", "!", "?", ";", ":"],
}


@dataclass
class EmbeddingExplorerOutput:
    out_path: Path
    n_tokens_plotted: int
    pca_variance_explained: float


# ── Public entry ──────────────────────────────────────────────────────────

def explore(
    model_id: str,
    prompt: str,
    out_path: Path | str,
    n_neighbors: int = 8,
    n_random_background: int = 200,
    cap_total: int = 500,
) -> EmbeddingExplorerOutput:
    """Build a self-contained HTML page exploring the model's embedding matrix.

    Loads the model just to read `wte.weight`, then never calls `forward` again.
    """
    print(f"[embeddings] loading {model_id} (we just need the embedding matrix)...")
    tokenizer, emb_matrix = _load_embeddings(model_id)
    print(f"[embeddings] embedding matrix shape: {emb_matrix.shape} "
          f"({emb_matrix.nbytes / 1e6:.1f} MB at fp32)")

    # Tokenize the prompt — same as collector would.
    prompt_ids = tokenizer.encode(prompt)
    prompt_tokens = [tokenizer.decode([t]) for t in prompt_ids]

    # ── Build the set of token ids to plot ────────────────────────────
    selected: dict[int, str] = {}    # id → category

    # Always include the prompt tokens.
    for tid in prompt_ids:
        selected[int(tid)] = "prompt"

    # Per-prompt-token nearest neighbors by cosine similarity.
    for tid in set(prompt_ids):
        for nbr in _nearest_neighbors(emb_matrix, int(tid), k=n_neighbors):
            selected.setdefault(int(nbr), "neighbor")

    # Curated landmark groups — for visual context.
    curated_lookup = _resolve_curated(tokenizer)
    for cat, ids_with_text in curated_lookup.items():
        for tid, _txt in ids_with_text:
            selected.setdefault(int(tid), cat)

    # Random background sample for "density" texture.
    rng = np.random.RandomState(42)
    vocab_size = emb_matrix.shape[0]
    bg = rng.choice(vocab_size, size=n_random_background, replace=False).tolist()
    for tid in bg:
        selected.setdefault(int(tid), "background")

    # Cap. Drop background first if over cap.
    if len(selected) > cap_total:
        keep_priorities = ["prompt", "neighbor"] + list(curated_lookup.keys()) + ["background"]
        priority = {cat: i for i, cat in enumerate(keep_priorities)}
        ordered = sorted(selected.items(), key=lambda kv: priority.get(kv[1], 99))
        selected = dict(ordered[:cap_total])

    selected_ids = list(selected.keys())
    n_plotted = len(selected_ids)
    print(f"[embeddings] selected {n_plotted} tokens for the 2D scatter")

    # ── PCA on the full vocab; project the selected subset ────────────
    print(f"[embeddings] running PCA on full vocab ({vocab_size:,} × {emb_matrix.shape[1]})...")
    points_2d, var_ratio = _pca_2d_full(emb_matrix)
    selected_points = points_2d[selected_ids]

    # ── Per-token data dicts ──────────────────────────────────────────
    DIMS_PREVIEW = 32
    norms_full = np.linalg.norm(emb_matrix, axis=1)

    token_records = []
    for tid in selected_ids:
        text = tokenizer.decode([int(tid)])
        idx = selected_ids.index(tid)
        token_records.append({
            "id": int(tid),
            "text": text,
            "x": float(selected_points[idx, 0]),
            "y": float(selected_points[idx, 1]),
            "norm": float(norms_full[tid]),
            "category": selected[tid],
            "preview": emb_matrix[tid][:DIMS_PREVIEW].tolist(),
        })

    # ── Build payload and write HTML ──────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": model_id,
        "prompt": prompt,
        "prompt_ids": [int(t) for t in prompt_ids],
        "prompt_tokens": prompt_tokens,
        "matrix_shape": list(emb_matrix.shape),
        "matrix_size_mb": round(emb_matrix.nbytes / 1e6, 1),
        "pca_var_pc1": float(var_ratio[0]),
        "pca_var_pc2": float(var_ratio[1]),
        "tokens": token_records,
        "categories": _category_meta(),
        "dims_preview": DIMS_PREVIEW,
    }

    html = _build_html(payload)
    out_path.write_text(html, encoding="utf-8")
    print(f"[embeddings] wrote {out_path}  ({len(html) / 1024:.1f} KB)")

    return EmbeddingExplorerOutput(
        out_path=out_path,
        n_tokens_plotted=n_plotted,
        pca_variance_explained=float(var_ratio[0] + var_ratio[1]),
    )


# ── Internal helpers ──────────────────────────────────────────────────────

def _load_embeddings(model_id: str):
    """Load tokenizer + the wte (token-embedding) weight matrix as numpy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    emb_layer = model.get_input_embeddings()      # works for any HF causal LM
    emb_matrix = emb_layer.weight.detach().float().cpu().numpy()
    # We don't need the model anymore — let GC reclaim it.
    del model
    return tokenizer, emb_matrix


def _nearest_neighbors(matrix: np.ndarray, query_id: int, k: int = 8) -> list[int]:
    """Cosine-similarity top-k neighbors of `query_id`. Excludes the query itself."""
    q = matrix[query_id]
    q_norm = np.linalg.norm(q) + 1e-12
    norms = np.linalg.norm(matrix, axis=1) + 1e-12
    sims = (matrix @ q) / (norms * q_norm)
    sims[query_id] = -np.inf   # don't return self
    top = np.argpartition(-sims, k)[:k]
    # Sort descending by sim.
    top = top[np.argsort(-sims[top])]
    return [int(i) for i in top]


def _resolve_curated(tokenizer) -> dict[str, list[tuple[int, str]]]:
    """For each curated string, find its single-token id (skip if multi-token)."""
    out: dict[str, list[tuple[int, str]]] = {}
    for cat, words in CURATED_GROUPS.items():
        keep: list[tuple[int, str]] = []
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                keep.append((int(ids[0]), w))
        if keep:
            out[cat] = keep
    return out


def _pca_2d_full(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA project rows of X to 2D via SVD on the centered matrix.

    Returns (points (n_rows, 2), variance_ratio (2,)).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # Memory-friendly: use SVD on the smaller (768×768) covariance via numpy's
    # randomized SVD-equivalent. Just full SVD is fine at this scale.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    points = Xc @ Vt[:2].T
    total_var = (S ** 2).sum()
    ratios = (S[:2] ** 2) / total_var if total_var > 0 else np.zeros(2)
    return points, ratios


def _category_meta() -> dict[str, dict[str, str]]:
    """Per-category visual styling — color, marker size, label visibility."""
    return {
        "prompt":      {"color": "#00d4aa", "size": "10", "label": "prompt token"},
        "neighbor":    {"color": "#4da6ff", "size": "7",  "label": "nearest neighbor"},
        "months":      {"color": "#ffb800", "size": "6",  "label": "month"},
        "days":        {"color": "#fb923c", "size": "6",  "label": "day"},
        "numbers":     {"color": "#a78bfa", "size": "6",  "label": "number"},
        "animals":     {"color": "#34d399", "size": "6",  "label": "animal"},
        "colors":      {"color": "#f472b6", "size": "6",  "label": "color word"},
        "verbs":       {"color": "#22d3ee", "size": "6",  "label": "verb"},
        "adjectives":  {"color": "#c084fc", "size": "6",  "label": "adjective"},
        "punctuation": {"color": "#94a3b8", "size": "6",  "label": "punctuation"},
        "background":  {"color": "#3a4564", "size": "3",  "label": "random vocab"},
    }


# ── HTML template ─────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Embedding matrix explorer — __MODEL__</title>
<style>
:root{--bg:#080b12;--surface:#0e1220;--surface2:#151b2e;--border:#1f2a45;--text:#cdd6f0;--muted:#5a6888;--teal:#00d4aa;--amber:#ffb800;--blue:#4da6ff;--coral:#ff6b6b;--mono:'SF Mono',Menlo,monospace;--sans:system-ui,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.5}
.wrap{max-width:1200px;margin:0 auto;padding:32px 28px}
h1{font-size:24px;font-weight:800;margin-bottom:4px}
h1 .m{color:var(--teal)}
.sub{font-family:var(--mono);color:var(--muted);font-size:12px;margin-bottom:24px;line-height:1.6}
h2{font-size:16px;font-weight:700;color:var(--teal);text-transform:uppercase;letter-spacing:.08em;margin-top:32px;margin-bottom:8px;font-family:var(--mono)}
.section{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-bottom:18px}
.intro{font-family:var(--mono);font-size:12.5px;color:var(--text);line-height:1.65;max-width:880px;margin-bottom:14px}
.intro b{color:var(--teal)}
/* ── Section 1: lookup demo ── */
.lookup{display:grid;grid-template-columns:auto 1fr;gap:8px 18px;align-items:center;font-family:var(--mono);font-size:13px;margin-top:14px}
.lookup .tok{padding:5px 12px;border-radius:6px;background:var(--surface2);border:1px solid var(--teal);color:var(--teal);font-weight:600;font-size:14px;justify-self:start}
.lookup .arrow{color:var(--muted)}
.lookup .id{color:var(--amber);font-weight:600}
.lookup .row-label{color:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase}
.lookup .vec{display:flex;gap:2px;flex-wrap:wrap;align-items:center}
.lookup .v-cell{width:14px;height:24px;border-radius:2px}
.lookup .v-more{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-left:8px}
.matrix-bar{display:flex;align-items:center;gap:10px;margin-top:18px;padding:12px 16px;background:var(--surface2);border-radius:8px;font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.matrix-bar b{color:var(--teal)}
/* ── Section 2: scatter ── */
.scatter-wrap{position:relative;background:var(--surface2);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-top:14px}
svg.scatter{display:block;width:100%;height:560px}
.dot{cursor:pointer;transition:r .15s}
.dot:hover{stroke:#fff;stroke-width:1}
.dot-label{font-family:var(--mono);font-size:10px;fill:var(--text);pointer-events:none;text-shadow:0 0 2px var(--bg),0 0 2px var(--bg)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:14px;padding:12px 16px;background:var(--surface2);border-radius:8px;font-family:var(--mono);font-size:11px}
.legend .item{display:flex;align-items:center;gap:6px;color:var(--text)}
.legend .swatch{width:10px;height:10px;border-radius:50%;flex:none}
.tooltip{position:fixed;pointer-events:none;background:rgba(8,11,18,0.97);border:1px solid var(--teal);border-radius:6px;padding:10px 14px;font-family:var(--mono);font-size:11px;color:var(--text);max-width:340px;box-shadow:0 4px 16px rgba(0,0,0,0.5);opacity:0;transition:opacity .15s;z-index:1000}
.tooltip.show{opacity:1}
.tooltip .tk{color:var(--amber);font-weight:700;font-size:13px}
.tooltip .id{color:var(--muted)}
.tooltip .preview{display:flex;gap:1px;margin-top:6px}
.tooltip .preview .vc{width:7px;height:18px;border-radius:1px}
.tooltip .meta{color:var(--muted);margin-top:6px;font-size:10px}
</style></head><body>
<div class="wrap">
  <h1>Embedding matrix explorer — <span class="m">__MODEL__</span></h1>
  <div class="sub" id="sub-line"></div>

  <div class="section">
    <h2>Section 1 · The lookup</h2>
    <div class="intro">
      The model's <b>embedding matrix</b> is a giant table — one row per token in its vocabulary,
      and each row is a vector of <b id="i-hidden">?</b> numbers. When the tokenizer turns your
      prompt into integer ids, the model uses each id as a <b>row index</b> into this table to
      pull out one row. That's literally it — no math yet. Just a lookup.
    </div>
    <div id="lookup-grid" class="lookup"></div>
    <div class="matrix-bar">
      Embedding matrix shape: <b id="i-shape">?</b> ·
      <b id="i-mb">?</b> at fp32 ·
      <b id="i-nrows">?</b> rows total · we showed <b id="i-prompt-len">?</b> rows above
    </div>
  </div>

  <div class="section">
    <h2>Section 2 · The neighborhood (PCA · 2D)</h2>
    <div class="intro">
      Plotting <b id="s-count">?</b> tokens after squashing the <b id="s-hidden">?</b>-dim
      embedding space down to 2 dimensions via PCA. The two axes are the directions of greatest
      spread across the full vocabulary (PC1 = <b id="s-pc1">?</b>%, PC2 = <b id="s-pc2">?</b>%
      of total variance). Tokens with similar meanings often end up near each other —
      this is <i>learned</i> structure, not designed.
    </div>
    <div class="scatter-wrap"><svg class="scatter" id="scatter"></svg></div>
    <div class="legend" id="legend"></div>
  </div>

</div>

<div class="tooltip" id="tooltip"></div>

<script>
const D = __DATA__;

// ── Header / intro fields ──────────────────────────────────────────────
document.getElementById('sub-line').innerHTML =
  `prompt: <span style="color:var(--amber)">${esc(JSON.stringify(D.prompt))}</span><br>` +
  `embedding matrix: <b style="color:var(--teal)">${D.matrix_shape[0].toLocaleString()} × ${D.matrix_shape[1]}</b> ` +
  `(${D.matrix_size_mb} MB at fp32, never re-loaded after this page builds)`;
document.getElementById('i-hidden').textContent = D.matrix_shape[1];
document.getElementById('i-shape').textContent = `[${D.matrix_shape[0].toLocaleString()}, ${D.matrix_shape[1]}]`;
document.getElementById('i-mb').textContent = `${D.matrix_size_mb} MB`;
document.getElementById('i-nrows').textContent = D.matrix_shape[0].toLocaleString();
document.getElementById('i-prompt-len').textContent = D.prompt_ids.length;
document.getElementById('s-count').textContent = D.tokens.length.toLocaleString();
document.getElementById('s-hidden').textContent = D.matrix_shape[1];
document.getElementById('s-pc1').textContent = (D.pca_var_pc1 * 100).toFixed(1);
document.getElementById('s-pc2').textContent = (D.pca_var_pc2 * 100).toFixed(1);

// ── Section 1: per-prompt-token lookup mini-row ────────────────────────
const lookupEl = document.getElementById('lookup-grid');
const promptDataById = {};
D.tokens.forEach(t => { if (t.category === 'prompt') promptDataById[t.id] = t; });

D.prompt_ids.forEach((tid, i) => {
  const t = promptDataById[tid] || D.tokens.find(x => x.id === tid);
  const tokText = D.prompt_tokens[i] || '?';
  const cellsHtml = (t ? t.preview.slice(0, 32) : []).map(v => {
    const intensity = Math.min(1, Math.abs(v));
    const col = v >= 0
      ? `rgba(0,${Math.round(140+intensity*112)},${Math.round(120+intensity*50)},${0.25+intensity*0.7})`
      : `rgba(${Math.round(140+intensity*115)},${Math.round(80-intensity*60)},${Math.round(80-intensity*60)},${0.25+intensity*0.7})`;
    return `<div class="v-cell" style="background:${col}" title="${v.toFixed(3)}"></div>`;
  }).join('');
  const remaining = D.matrix_shape[1] - 32;
  lookupEl.innerHTML += `
    <div class="tok">${esc(tokText)}</div>
    <div><span class="arrow">→ id</span> <span class="id">${tid}</span> <span class="arrow">→ row ${tid} of [${D.matrix_shape[0].toLocaleString()} × ${D.matrix_shape[1]}]</span></div>
    <div class="row-label">vector</div>
    <div class="vec">${cellsHtml}<span class="v-more">+ ${remaining} more dims</span></div>
  `;
});

// ── Section 2: 2D scatter ──────────────────────────────────────────────
const svg = document.getElementById('scatter');
const SVG_W = 1100, SVG_H = 560, PAD = 36;
svg.setAttribute('viewBox', `0 0 ${SVG_W} ${SVG_H}`);

// Bounds
let minX = +Infinity, maxX = -Infinity, minY = +Infinity, maxY = -Infinity;
D.tokens.forEach(t => {
  if (t.x < minX) minX = t.x; if (t.x > maxX) maxX = t.x;
  if (t.y < minY) minY = t.y; if (t.y > maxY) maxY = t.y;
});
const xToSvg = x => PAD + (x - minX) / (maxX - minX) * (SVG_W - 2*PAD);
const yToSvg = y => SVG_H - PAD - (y - minY) / (maxY - minY) * (SVG_H - 2*PAD);

// Axes (faint)
let axes = `<line x1="${PAD}" y1="${SVG_H-PAD}" x2="${SVG_W-PAD}" y2="${SVG_H-PAD}" stroke="#1f2a45" stroke-width="1"/>`;
axes += `<line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${SVG_H-PAD}" stroke="#1f2a45" stroke-width="1"/>`;
axes += `<text x="${SVG_W-PAD-50}" y="${SVG_H-PAD-6}" font-family="monospace" font-size="10" fill="#5a6888">PC1 →</text>`;
axes += `<text x="${PAD+8}" y="${PAD+12}" font-family="monospace" font-size="10" fill="#5a6888">↑ PC2</text>`;

// Sort: background first (under), then by category, prompt last (on top)
const order = ['background','adjectives','verbs','colors','animals','numbers','days','months','punctuation','neighbor','prompt'];
const sorted = [...D.tokens].sort((a,b) => order.indexOf(a.category) - order.indexOf(b.category));

let dots = '';
let labels = '';
sorted.forEach(t => {
  const meta = D.categories[t.category] || D.categories.background;
  const cx = xToSvg(t.x).toFixed(1), cy = yToSvg(t.y).toFixed(1);
  const r = meta.size;
  dots += `<circle class="dot" cx="${cx}" cy="${cy}" r="${r}" fill="${meta.color}" data-id="${t.id}"/>`;
  // Label visibility: prompt + neighbor + meaningful curated → labeled. Background unlabeled.
  if (t.category !== 'background') {
    const fontSize = t.category === 'prompt' ? 12 : 10;
    const fontWeight = t.category === 'prompt' ? '700' : '400';
    labels += `<text class="dot-label" x="${parseFloat(cx)+r+3}" y="${parseFloat(cy)+4}" font-size="${fontSize}" font-weight="${fontWeight}" fill="${t.category==='prompt'?'#fff':'#cdd6f0'}">${esc(t.text)}</text>`;
  }
});

svg.innerHTML = axes + dots + labels;

// ── Tooltip ────────────────────────────────────────────────────────────
const tooltip = document.getElementById('tooltip');
const tokById = {}; D.tokens.forEach(t => { tokById[t.id] = t; });

svg.addEventListener('mouseover', e => {
  if (!e.target.classList.contains('dot')) return;
  const id = parseInt(e.target.getAttribute('data-id'));
  const t = tokById[id];
  if (!t) return;
  const cells = t.preview.map(v => {
    const intensity = Math.min(1, Math.abs(v));
    const col = v >= 0
      ? `rgba(0,${Math.round(140+intensity*112)},${Math.round(120+intensity*50)},${0.3+intensity*0.65})`
      : `rgba(${Math.round(140+intensity*115)},${Math.round(80-intensity*60)},${Math.round(80-intensity*60)},${0.3+intensity*0.65})`;
    return `<div class="vc" style="background:${col}"></div>`;
  }).join('');
  tooltip.innerHTML = `
    <div><span class="tk">${esc(t.text)}</span> <span class="id">id ${t.id}</span></div>
    <div class="meta">category: ${t.category} · ‖vector‖ = ${t.norm.toFixed(2)}</div>
    <div class="preview">${cells}</div>
    <div class="meta">first ${t.preview.length} of ${D.matrix_shape[1]} dims · green = positive, red = negative</div>
  `;
  tooltip.classList.add('show');
});
svg.addEventListener('mouseout', e => {
  if (e.target.classList.contains('dot')) tooltip.classList.remove('show');
});
svg.addEventListener('mousemove', e => {
  if (!tooltip.classList.contains('show')) return;
  const x = e.clientX + 14, y = e.clientY + 14;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
});

// ── Legend ─────────────────────────────────────────────────────────────
const legendEl = document.getElementById('legend');
order.forEach(cat => {
  const meta = D.categories[cat];
  if (!meta) return;
  const present = D.tokens.some(t => t.category === cat);
  if (!present) return;
  legendEl.innerHTML += `<span class="item"><span class="swatch" style="background:${meta.color}"></span>${meta.label}</span>`;
});

function esc(s){return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
</script>
</body></html>
"""


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    return (_HTML
            .replace("__DATA__", data_json)
            .replace("__MODEL__", _html_escape(payload["model_id"])))


def _html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace('"', "&quot;"))
