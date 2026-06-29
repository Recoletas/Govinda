#!/bin/bash
# 容器 web shell 一键: vllm wheel 装 + 启 + 4-8K smoke.
# 不走 build_ext --inplace (要 libgalaxyhip, srun host 上找不到).
# 改用 $Q/Govinda/dist/vllm-0.18.1+das.dtk2604-cp310-*.whl 预编译 wheel.
#
# 用法 (web shell 跑): bash /public/home/xdzs2026_c087/Govinda/scripts/run_in_container.sh
#
# 跑 ~15-20 min: 装 wheel (1min) + vllm 启 (5-10min 模型加载) + smoke (~1min)

set -u
set -o pipefail

export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export VLLM_WHEEL=$GOVINDA_DIR/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_LOG=/root/vllm.log

echo "=== [$(date +%H:%M:%S)] 0. 装 vllm wheel (--no-deps, 预编译 _C 进来) ==="
/usr/bin/pip install --no-deps "$VLLM_WHEEL" 2>&1 | tail -3

echo "=== [$(date +%H:%M:%S)] 0.5 验证 vllm._C ==="
/usr/bin/python3.10 -c "import vllm; print('vllm', vllm.__version__); import vllm._C; print('vllm._C OK, silu_and_mul:', vllm._C.silu_and_mul)" 2>&1 | head -3

echo "=== [$(date +%H:%M:%S)] 1. 杀旧 vllm (如有) ==="
pkill -f "vllm.entrypoints" 2>/dev/null
sleep 3
pkill -9 -f "vllm.entrypoints" 2>/dev/null
sleep 2

echo "=== [$(date +%H:%M:%S)] 2. 启 vllm serve (用之前能起来的命令) ==="
cd "$Q"
nohup /usr/bin/python3.10 -m vllm.entrypoints.cli.main serve \
    "$Q/Qwen3.5-27B" \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --served-model-name Qwen3.5-27B \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 128 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "  PID=$VLLM_PID, log=$VLLM_LOG"

echo "=== [$(date +%H:%M:%S)] 3. 等 ready (curl 200) ==="
SECONDS=0
MAX_WAIT=900
while ! curl -s --noproxy 127.0.0.1 http://127.0.0.1:8001/health >/dev/null 2>&1; do
  sleep 15
  ELAPSED=$SECONDS
  if [[ $ELAPSED -gt $MAX_WAIT ]]; then
    echo "  ✗ 超时 ${MAX_WAIT}s, 看 $VLLM_LOG"
    tail -30 "$VLLM_LOG"
    exit 1
  fi
  if (( ELAPSED % 60 < 15 )); then
    echo "  [${ELAPSED}s] 还在加载..."
  fi
done
echo "  ✓ vllm ready (${SECONDS}s)"

echo "=== [$(date +%H:%M:%S)] 4. 跑 4-8K smoke (5 prompts) ==="
cd /public/home/xdzs2026_c087/testdata
MODEL_DIR=/public/home/xdzs2026_c087/Qwen3.5-27B \
  /usr/bin/python3.10 -m vllm.entrypoints.cli.main bench serve \
    --backend openai-chat \
    --host 127.0.0.1 --port 8001 \
    --endpoint /v1/chat/completions \
    --model Qwen3.5-27B \
    --tokenizer /public/home/xdzs2026_c087/Qwen3.5-27B \
    --dataset-name custom \
    --dataset-path ./4-8K_throughput.jsonl \
    --num-prompts 5 \
    --no-oversample \
    --max-concurrency 1 \
    --request-rate 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir ./test/4-8K_throughput \
    --result-filename result.json 2>&1 | tail -20

echo "=== [$(date +%H:%M:%S)] 5. 关键数字 ==="
/usr/bin/python3.10 - <<'PYEOF'
import json
d = json.load(open('/public/home/xdzs2026_c087/testdata/test/4-8K_throughput/result.json'))
print(f"  output_throughput:  {d['output_throughput']:.2f} tok/s")
print(f"  TTFT  P50/P99:      {d['ttft_p50']:.0f} / {d['ttft_p99']:.0f} ms")
print(f"  TPOT  P50/P99:      {d['tpot_p50']:.1f} / {d['tpot_p99']:.1f} ms")
PYEOF

echo ""
echo "=== [$(date +%H:%M:%S)] ✓ 全部完成. vllm 还在跑 (PID $VLLM_PID), 你可以继续跑 8-16K / 16-32K ==="