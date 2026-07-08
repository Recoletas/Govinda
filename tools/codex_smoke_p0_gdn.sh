#!/usr/bin/env bash
# AI-generated, awaiting verification by recoletas on 2026-07-09.
#
# Run inside the SCNet container after deploying the selected vLLM tree.
# This is a smoke helper only; it does not change scheduler, sampling, weights,
# inputs, or the OpenAI-compatible service path.

set -euo pipefail

Q="${Q:-/public/home/xdzs2026_c087}"
MODEL="${MODEL:-$Q/Qwen3.5-27B}"
DATA="${DATA:-$Q/testdata}"
OUT="${OUT:-$DATA/test/codex_p0_gdn_smoke}"
PORT="${PORT:-8000}"
RANGE="${RANGE:-4-8K}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

echo "=== smoke $(date +%F_%T) ==="
echo "range=$RANGE prompts=$NUM_PROMPTS"

VLLM_PID="$(
    pgrep -f "vllm.entrypoints|vllm serve|VLLM::EngineCore" | head -1 || true
)"
if [[ -n "$VLLM_PID" && -r "/proc/$VLLM_PID/environ" ]]; then
    echo "vllm_pid=$VLLM_PID"
    tr '\0' '\n' < "/proc/$VLLM_PID/environ" | grep -E \
        '^(VLLM_TRITON_PREFILL_TILE64_POLICY|VLLM_GDN_CAUSAL_CONV1D_BLOCK_M|VLLM_GDN_CHUNK_SIZE|FLA_GDN_FIX_BT|FLA_TRIL_PRECISION|VLLM_GFX936_FUSED_GATE_UP_SILU)=' \
        || true
else
    echo "vllm_pid=not-found"
fi
echo "bench_env_tile64_policy=${VLLM_TRITON_PREFILL_TILE64_POLICY:-unset}"
echo "bench_env_gdn_conv_block_m=${VLLM_GDN_CAUSAL_CONV1D_BLOCK_M:-unset}"
echo "bench_env_gdn_chunk_size=${VLLM_GDN_CHUNK_SIZE:-unset}"
echo "bench_env_fla_gdn_fix_bt=${FLA_GDN_FIX_BT:-unset}"
echo "bench_env_fla_tril_precision=${FLA_TRIL_PRECISION:-unset}"
echo "bench_env_gfx936_fused_gate_up_silu=${VLLM_GFX936_FUSED_GATE_UP_SILU:-unset}"

curl -fsS --noproxy 127.0.0.1 "http://127.0.0.1:${PORT}/health" >/dev/null

mkdir -p "$OUT/${RANGE}_n${NUM_PROMPTS}"
cd "$DATA"
python3.10 -m vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 --port "$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name custom \
    --dataset-path "$DATA/${RANGE}_throughput.jsonl" \
    --num-prompts "$NUM_PROMPTS" \
    --no-oversample \
    --max-concurrency 1 \
    --request-rate 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$OUT/${RANGE}_n${NUM_PROMPTS}" \
    --result-filename result.json

python3.10 - "$OUT/${RANGE}_n${NUM_PROMPTS}/result.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

print("=== result ===")
for key in (
    "completed",
    "failed",
    "output_throughput",
    "ttft_p99",
    "tpot_p99",
    "itl_p99",
    "e2el_p99",
):
    print(f"{key}: {d.get(key)}")
PY
