#!/usr/bin/env bash
# AI-generated, awaiting verification by recoletas on 2026-07-09.
#
# Container-side orchestration helper for one P0 smoke attempt.
# Defaults are non-destructive: it does not kill processes and does not start
# vLLM unless START_VLLM=1 is set.

set -euo pipefail

Q="${Q:-/public/home/xdzs2026_c087}"
GOVINDA_DIR="${GOVINDA_DIR:-$Q/Govinda}"
REF="${REF:-main}"
RANGE="${RANGE:-4-8K}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
START_VLLM="${START_VLLM:-0}"
WAIT_HEALTH_TIMEOUT_S="${WAIT_HEALTH_TIMEOUT_S:-3600}"
PORT="${PORT:-8000}"

echo "=== P0 smoke once $(date +%F_%T) ==="
echo "REF=$REF RANGE=$RANGE NUM_PROMPTS=$NUM_PROMPTS START_VLLM=$START_VLLM"

bash "$GOVINDA_DIR/tools/codex_checkout_vllm_branch.sh" "$REF"

if [[ "$START_VLLM" == "1" ]]; then
    echo "=== starting vLLM in background ==="
    nohup bash "$GOVINDA_DIR/tools/codex_start_vllm_p0.sh" \
        > /root/codex_start_vllm_p0.wrapper.log 2>&1 &
    echo "vllm_start_pid=$!"
fi

echo "=== waiting for health on port $PORT ==="
deadline=$((SECONDS + WAIT_HEALTH_TIMEOUT_S))
while ! curl -fsS --noproxy 127.0.0.1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "ERROR: /health did not become ready within ${WAIT_HEALTH_TIMEOUT_S}s" >&2
        exit 1
    fi
    sleep 10
done

echo "=== running smoke ==="
RANGE="$RANGE" NUM_PROMPTS="$NUM_PROMPTS" PORT="$PORT" \
    bash "$GOVINDA_DIR/tools/codex_smoke_p0_gdn.sh"
