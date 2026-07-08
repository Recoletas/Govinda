#!/usr/bin/env bash
# AI-generated, awaiting verification by recoletas on 2026-07-09.
#
# Run inside the SCNet container to switch the shared vLLM source tree to a
# GitLab branch/commit. This helper deliberately does not start or stop vLLM.

set -euo pipefail

Q="${Q:-/public/home/xdzs2026_c087}"
VLLM_DIR="${VLLM_DIR:-$Q/vllm_cscc}"
REMOTE="${REMOTE:-origin}"
REMOTE_URL="${REMOTE_URL:-}"
REF="${1:-main}"

if [[ ! -d "$VLLM_DIR/.git" ]]; then
    echo "ERROR: $VLLM_DIR is not a git checkout" >&2
    exit 1
fi

cd "$VLLM_DIR"

if [[ -n "$REMOTE_URL" ]]; then
    REMOTE="${REMOTE_NAME:-codex-gitlab}"
    if git remote get-url "$REMOTE" >/dev/null 2>&1; then
        git remote set-url "$REMOTE" "$REMOTE_URL"
    else
        git remote add "$REMOTE" "$REMOTE_URL"
    fi
fi

echo "=== before ==="
git status --short --branch
git log --oneline -3

echo "=== fetch $REMOTE $REF ==="
git fetch "$REMOTE" "$REF"

echo "=== switch tree ==="
git checkout --detach FETCH_HEAD

echo "=== after ==="
git status --short --branch
git log --oneline -5

echo "=== active experiment env knobs ==="
echo "VLLM_TRITON_PREFILL_TILE64_POLICY=${VLLM_TRITON_PREFILL_TILE64_POLICY:-mid}"
echo "VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=${VLLM_GDN_CAUSAL_CONV1D_BLOCK_M:-unset}"
echo "VLLM_GDN_CHUNK_SIZE=${VLLM_GDN_CHUNK_SIZE:-unset}"
echo "FLA_TRIL_PRECISION=${FLA_TRIL_PRECISION:-unset}"
echo "VLLM_GFX936_FUSED_GATE_UP_SILU=${VLLM_GFX936_FUSED_GATE_UP_SILU:-unset}"
