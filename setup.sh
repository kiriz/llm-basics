#!/usr/bin/env bash
# =============================================================
# setup.sh — one-time install for the llm-trace tool.
#
# Installs the package + trace-specific deps (transformers, torch, pyyaml)
# via `uv` into the current Python environment, then verifies the
# `llm-trace` CLI is on PATH.
#
# Usage:
#   ./setup.sh
#   ./setup.sh --skip-model-warmup      # skip the distilgpt2 prefetch
#   ./setup.sh --venv                   # create a fresh .venv first
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { printf "%b→%b %s\n" "$BLUE" "$NC" "$*"; }
success() { printf "%b✓%b %s\n" "$GREEN" "$NC" "$*"; }
warn()    { printf "%b!%b %s\n" "$YELLOW" "$NC" "$*"; }
error()   { printf "%b✗%b %s\n" "$RED" "$NC" "$*"; exit 1; }

# ── Flags ────────────────────────────────────────────────────
SKIP_WARMUP=0
MODE="active"   # active | sync | both
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-model-warmup) SKIP_WARMUP=1; shift ;;
        --mode)              MODE="${2:-}"; shift 2 ;;
        --mode=*)            MODE="${1#*=}"; shift ;;
        -h|--help)
            cat <<EOF
Usage: ./setup.sh [--skip-model-warmup] [--mode MODE]

  --skip-model-warmup   Don't prefetch distilgpt2 weights (~350 MB one-time).
  --mode MODE           How to install:
                          'active' (default) — 'uv pip install -e .' into
                                    the current Python environment (conda,
                                    system, etc.). llm-trace goes on PATH.
                          'sync'   — 'uv sync' creates a project-local .venv
                                    with a uv.lock file. Use 'uv run llm-trace ...'
                                    afterward (no activation needed).
                          'both'   — do both. Useful when collaborators may
                                    use either workflow.
EOF
            exit 0
            ;;
        *) error "unknown flag: $1. Try ./setup.sh --help" ;;
    esac
done

case "$MODE" in
    active|sync|both) ;;
    *) error "--mode must be one of: active | sync | both (got: $MODE)" ;;
esac

# ── Check uv is available ────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    warn "uv is not installed."
    echo "  Install it with one of:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "    brew install uv"
    echo "    pipx install uv"
    error "uv is required."
fi
success "uv found: $(uv --version)"

# ── Install according to chosen mode ─────────────────────────
DO_ACTIVE=0; DO_SYNC=0
case "$MODE" in
    active) DO_ACTIVE=1 ;;
    sync)   DO_SYNC=1 ;;
    both)   DO_ACTIVE=1; DO_SYNC=1 ;;
esac

if [[ $DO_ACTIVE -eq 1 ]]; then
    info "Installing into active environment (uv pip install -e .) ..."
    uv pip install -e .
    success "Active env install complete."
fi

if [[ $DO_SYNC -eq 1 ]]; then
    info "Syncing project .venv with uv.lock (uv sync) ..."
    uv sync
    success "Project .venv ready. Use 'uv run llm-trace ...' (no activation needed)."
fi

# ── Verify llm-trace is reachable ────────────────────────────
if [[ $DO_ACTIVE -eq 1 ]] && command -v llm-trace >/dev/null 2>&1; then
    LLM_TRACE_PATH="$(command -v llm-trace)"
    success "llm-trace on PATH: $LLM_TRACE_PATH"
    PY_CMD="python"
elif [[ $DO_SYNC -eq 1 ]]; then
    LLM_TRACE_PATH="uv run llm-trace"
    success "llm-trace accessible via: $LLM_TRACE_PATH"
    PY_CMD="uv run python"
else
    error "llm-trace not reachable after install. Check mode and environment."
fi

# ── Optional: warm the HF cache for the default model ──────
if [[ $SKIP_WARMUP -eq 0 ]]; then
    info "Prefetching distilgpt2 (~350 MB, one-time) so ./try-prompt.sh is fast ..."
    $PY_CMD - <<'PY'
import os, sys
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    AutoTokenizer.from_pretrained("distilgpt2")
    AutoModelForCausalLM.from_pretrained("distilgpt2")
    print("  cached in ~/.cache/huggingface")
except Exception as e:
    print(f"  warm-up skipped: {e}", file=sys.stderr)
    sys.exit(0)
PY
    success "Model warm-up complete."
else
    warn "Skipped model warm-up (--skip-model-warmup given)."
fi

# ── Summary ──────────────────────────────────────────────────
echo
success "Setup done. Next steps:"
echo "   ./try-prompt.sh \"January, February, March,\""
echo "   ./try-prompt.sh \"Hello\" --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max 60"
if [[ $DO_ACTIVE -eq 1 ]]; then
    echo "   llm-trace --help            # direct CLI (active env)"
fi
if [[ $DO_SYNC -eq 1 ]]; then
    echo "   uv run llm-trace --help     # via uv-synced .venv"
fi
echo "   pytest tests/llm_trace/     # run tests"
