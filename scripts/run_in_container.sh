#!/bin/bash
# 一站式: build vllm._C 扩展 + 启 vllm serve + 4-8K smoke benchmark.
# 容器 web shell 里跑: bash /public/home/xdzs2026_c087/Govinda/scripts/run_in_container.sh
# 跑完输出会落到 /root/vllm.log + /public/home/xdzs2026_c087/testdata/test/4-8K_throughput/result.json

set -u
set -o pipefail

export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export VLLM_SRC_DIR=$Q/vllm_cscc
export VLLM_WHEEL_DIR=$GOVINDA_DIR/dist

# ===== 1. build vllm._C C 扩展 (editable install 必须, ~5-10min 首次) =====
cd "$VLLM_SRC_DIR" || { echo "FAIL: $VLLM_SRC_DIR 不存在"; exit 1; }
if ! /usr/bin/python3.10 -c "import vllm._C; print(vllm._C.silu_and_mul)" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] 编 vllm._C C 扩展 (~5-10min)..."
  /usr/bin/python3.10 setup.py build_ext --inplace 2>&1 | tail -5
else
  echo "[$(date +%H:%M:%S)] vllm._C 已编译, 跳过"
fi

# ===== 2. 启 vllm serve (后台, 首次 ~10min 加载模型) =====
MODEL_DIR=$Q/Qwen3.5-27B
LOG=/root/vllm.log

# 杀掉之前的 (如有)
pkill -f "vllm.entrypoints" 2>/dev/null
sleep 3

# 启 (跟之前能起来的官方 start_vllm.sh 命令一致, 只换 CLI 调用)
nohup /usr/bin/python3.10 -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
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
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "[$(date +%H:%M:%S)] vllm 启动 PID=$VLLM_PID, log: $LOG"

# ===== 3. 等 ready =====
echo "[$(date +%H:%M:%S)] 等 vllm ready (curl http://127.0.0.1:8001/health 返回 200)..."
SECONDS=0
MAX_WAIT=900  # 15 min
while ! curl -s --noproxy 127.0.0.1 http://127.0.0.1:8001/health >/dev/null 2>&1; do
  sleep 15
  ELAPSED=$SECONDS
  if [[ $ELAPSED -gt $MAX_WAIT ]]; then
    echo "[$(date +%H:%M:%S)] ✗ 启动超时 (${MAX_WAIT}s), 看 $LOG"
    tail -30 "$LOG"
    exit 1
  fi
  # 偶尔看一眼进度
  if (( ELAPSED % 60 < 15 )); then
    echo "[$(date +%H:%M:%S)] 还在等 (${ELAPSED}s)..."
  fi
done
echo "[$(date +%H:%M:%S)] ✓ vllm ready (${SECONDS}s)"

# ===== 4. Smoke: 4-8K 5 prompts =====
echo "[$(date +%H:%M:%S)] 跑 4-8K smoke (5 prompts)..."
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
    --result-filename result.json 2>&1 | tail -40

# ===== 5. 关键数字 (跟 baseline 对比) =====
echo ""
echo "[$(date +%H:%M:%S)] 关键数字:"
/usr/bin/python3.10 - <<'PYEOF'
import json
d = json.load(open('/public/home/xdzs2026_c087/testdata/test/4-8K_throughput/result.json'))
print(f"  output_throughput:  {d['output_throughput']:.2f} tok/s")
print(f"  TTFT  P50/P99:      {d['ttft_p50']:.0f} / {d['ttft_p99']:.0f} ms")
print(f"  TPOT  P50/P99:      {d['tpot_p50']:.1f} / {d['tpot_p99']:.1f} ms")
PYEOF
echo ""
echo "[$(date +%H:%M:%S)] ✓ 全部完成"