#!/usr/bin/env bash
# =============================================================
# try-prompt.sh — quick experiment: send one prompt through a
# model and render terminal + animated_v3 HTML.
#
# Usage:
#   ./try-prompt.sh "opposite of good is"
#   ./try-prompt.sh "January, February, March," --max 50
#   ./try-prompt.sh "Hello" --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --system "You are a Physicist"
#   ./try-prompt.sh "What is gravity?" --renderers '["html","animated_v3"]'
#
# Resolves the llm-trace CLI in this order:
#   1. `llm-trace` on PATH (active env install — see ./setup.sh)
#   2. `uv run llm-trace` (project .venv via uv)
#   3. error
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()    { printf "%b→%b %s\n" "$BLUE" "$NC" "$*"; }
success() { printf "%b✓%b %s\n" "$GREEN" "$NC" "$*"; }
warn()    { printf "%b!%b %s\n" "$YELLOW" "$NC" "$*"; }
error()   { printf "%b✗%b %s\n" "$RED" "$NC" "$*"; exit 1; }

# ── Defaults ─────────────────────────────────────────────────
PROMPT=""
MODEL="distilgpt2"
MAX=20
SYSTEM=""
RENDERERS='["terminal","animated_v3"]'
STOP_ON_EOS="true"

# ── Argparse ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";       shift 2 ;;
        --max)           MAX="$2";         shift 2 ;;
        --system)        SYSTEM="$2";      shift 2 ;;
        --renderers)     RENDERERS="$2";   shift 2 ;;
        --no-eos-stop)   STOP_ON_EOS="false"; shift ;;
        -h|--help)
            sed -n 's/^# \?//p' "$0" | head -20
            exit 0 ;;
        --*)
            error "unknown flag: $1. Try ./try-prompt.sh --help" ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"; shift
            else
                error "extra positional argument: $1 (prompt already set)"
            fi ;;
    esac
done

[[ -z "$PROMPT" ]] && error "no prompt given. Usage: ./try-prompt.sh \"your prompt\""

# ── Resolve llm-trace ────────────────────────────────────────
if command -v llm-trace >/dev/null 2>&1; then
    LLM_TRACE=(llm-trace)
elif command -v uv >/dev/null 2>&1 && [[ -d "$SCRIPT_DIR/.venv" ]]; then
    LLM_TRACE=(uv run llm-trace)
else
    error "llm-trace not on PATH and no uv .venv found. Run ./setup.sh first."
fi

# ── Run ──────────────────────────────────────────────────────
info "model:     $MODEL"
info "prompt:    \"$PROMPT\""
[[ -n "$SYSTEM" ]] && info "system:    \"$SYSTEM\""
info "max_new:   $MAX  (stop_on_eos=$STOP_ON_EOS)"
info "renderers: $RENDERERS"
echo

ARGS=(--prompt "$PROMPT"
      --override "models=[{\"id\":\"$MODEL\"}]"
      --override "renderers=$RENDERERS"
      --override "generation.max_new_tokens=$MAX"
      --override "generation.stop_on_eos=$STOP_ON_EOS")

[[ -n "$SYSTEM" ]] && ARGS+=(--system "$SYSTEM")

"${LLM_TRACE[@]}" run "${ARGS[@]}"

echo
success "Done. HTMLs under ./out/  —  open with:  open out/*$(echo "$PROMPT" | tr -c '[:alnum:]' '_' | cut -c1-30)*"
