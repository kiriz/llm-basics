# LLM Runtimes & Tooling — Reference Notes

Scratchpad notes captured while building this tool. Some sections reference
early standalone scripts (`llm_flow.py`, `llm_flow_viz.py`, `llama_trace.py`)
that were the v0 predecessors of `src/llm_trace/` and are not in this repo —
the design rationale they justify still stands.

---

## 1. What an LLM actually is on disk

An LLM is a **bundle of files**, not a single file:

```
model_name/
├── config.json              # architecture blueprint (layers, hidden dim, heads)
├── model.safetensors        # the weights — millions/billions of floats
├── tokenizer.json           # vocab + BPE rules (how text → token IDs)
└── tokenizer_config.json    # chat template, special token IDs (EOS, BOS, etc.)
```

Weights file sizes:
- GPT-2 (124M params)  → ~500 MB
- TinyLlama (1.1B)     → ~2.2 GB
- Llama-3-8B           → ~15 GB
- Llama-3-70B          → ~140 GB

**Different runtimes use different file formats** for the same underlying model:
- `.safetensors` / `.bin` → PyTorch (HuggingFace transformers)
- `.gguf`                 → llama.cpp / Ollama (often 4-bit quantized for CPU)
- `.mlx`                  → Apple MLX

Ollama typically ships a 4-bit quantized GGUF — smaller, faster on CPU, but lossy.
HuggingFace usually ships fp16/bf16 safetensors — full precision, larger.

---

## 2. What "running" a model means

Every runtime does the same abstract loop:

```
user text → tokenizer → [ID, ID, ID] → forward pass → logits → sample → token → repeat
```

The differences are **how** the forward pass runs and **what internals you can see**.

---

## 3. Runtime comparison

| Runtime | Language | Speed | Expose attention / hidden states? | Good for |
|---|---|---|---|---|
| **HuggingFace transformers** | Python + PyTorch | slow (5–10× baseline) | **YES** (via `output_attentions=True`) | research, teaching, fine-tuning |
| **Ollama / llama.cpp** | C++ | fastest on CPU | **NO** — internals are private | chatbots, local apps |
| **vLLM / TGI** | Python + CUDA kernels | fastest on GPU, batched | limited | production API serving |
| **MLX** | Swift/Python (Apple Silicon) | fast on M-series Macs | manual (via hooks) | Mac-native apps |
| **Raw PyTorch (write it yourself)** | Python | slow | YES | deep understanding |

---

## 4. Why we used HuggingFace (not Ollama) in these scripts

Our goal: **trace every intermediate tensor** — embeddings, per-layer hidden states,
attention matrices, logits, probabilities.

These two flags are the whole reason HF was the only option:

```python
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    output_attentions=True,      # return attention matrices from every layer
    output_hidden_states=True,   # return hidden states between every layer
)
```

Ollama's C++ runtime computes those same matrices — they MUST exist in RAM during
inference — but it doesn't expose them. After the forward pass completes, those
buffers are reused or freed. You get text in, text out. That's it.

**Rule of thumb:**
- Understanding / research → HuggingFace
- Production chatbot     → Ollama / vLLM
- Port from HF to Ollama when you're done experimenting

Car analogy: Ollama is Uber (fast, don't see the engine). Our scripts put the car
on a hydraulic lift with sensors on every piston (slow, see everything).

---

## 5. About the "HuggingFace" dependency

When the script says `from_pretrained("gpt2")` it:

1. Checks the local cache (`~/.cache/huggingface/hub/`) for the files
2. If missing, downloads them once from HF Hub (~500 MB for GPT-2)
3. Loads the weights into PyTorch tensors in RAM

**At inference time there are zero network calls.** HF Hub is just a CDN for weight
files. The actual forward pass is 100% local PyTorch on your CPU/GPU.

**To go fully offline** (skip even the cache-freshness ping):
```bash
HF_HUB_OFFLINE=1 python llm_flow.py --prompt "..."
```
or in-script: `os.environ.setdefault("HF_HUB_OFFLINE", "1")` before imports.

---

## 6. What the scripts in this project are

- **`llm_flow.py`** — GPT-2 (2019 arch), 10-step terminal trace with `rich` tables.
  Base model (not chat-tuned). No EOS stopping — just N-token greedy generation.
- **`llm_flow_viz.py`** — matplotlib PNG with 4 panels (attention heatmap,
  top-10 probs, hidden-state PCA trajectory, temperature effect).
- **`llm_flow_visual.py`** — self-contained interactive HTML viz (10 pages,
  arrow-key nav, temperature slider, attention head switcher).
- **`llama_trace.py`** — TinyLlama-1.1B-Chat (modern llama arch with RoPE, RMSNorm,
  SwiGLU, GQA). Adds chat template, EOS stopping, and the "what if we ignore EOS?"
  experiment. This is the one that answers *how generation stops*.

---

## 7. Package install — use `uv`, not pip

`pyproject.toml` uses `hatchling` as the build backend, which works with both
pip and uv natively. Prefer uv — it's 10-20× faster and makes the
`PYTHONPATH=src python -m …` hack unnecessary.

```bash
uv pip install -e ".[trace]"   # 2 seconds; package editable, CLI on PATH
llm-trace run                  # vs. PYTHONPATH=src python -m llm_trace.cli run
```

For full reproducibility, use `uv sync --extra trace` which produces a
`uv.lock` file to commit. `uv run <cmd>` then runs any command against the
synced env without needing activation.

## 8. Three mechanisms that stop generation

(From `llama_trace.py` summary — important to remember:)

1. **EOS token** — model learned during training to emit a specific token
   (e.g. `</s>`, `<|eot_id|>`) at the end of a response. The generation for-loop
   checks each step and breaks when it sees one.
2. **`max_new_tokens`** — hard cap from the caller (safety net).
3. **Context window** — architectural ceiling (GPT-2: 1024 tokens, TinyLlama: 2048,
   Llama-3: 8192). RoPE frequencies only trained up to this length.

**Critical insight:** Nothing *inside* the transformer stops generation. The for-loop
lives in Python. The model just outputs a probability distribution every step —
the *caller* decides when to stop. Without the EOS check the model happily
hallucinates past its own stop signal (often faking a new user turn).
