#!/bin/bash
# 容器里一键: 重装 vllm + 起服务 + 跑 4-8K baseline.
# 用法 (web shell 或 3 跳 ssh 进来跑):  bash /public/home/xdzs2026_c087/Govinda/scripts/run_baseline.sh
#
# 必须 inline 设 LD_LIBRARY_PATH (容器 ssh session 没自动 source DTK env).

set -u
set -o pipefail

export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export WHEEL=$Q/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_LOG=/root/vllm.log

# DTK runtime libs (必须, 否则 torch import 报 libgalaxyhip.so.5 缺失)
DTK_LIBS=$(find /opt/dtk-26.04-DCC2602-0317 -maxdepth 4 -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${DTK_LIBS}${LD_LIBRARY_PATH:-}"
export PATH=/usr/local/bin:/opt/dtk-26.04-DCC2602-0317/bin:$PATH

echo "=== [$(date +%H:%M:%S)] 0. 重装 vllm wheel (~30s) ==="
pip install --no-deps "$WHEEL" 2>&1 | tail -3

echo "=== [$(date +%H:%M:%S)] 1. 杀旧 vllm ==="
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "start_vllm" 2>/dev/null
sleep 3

echo "=== [$(date +%H:%M:%S)] 2. 启 vllm (官方 start_vllm.sh, 后台) ==="
cd "$Q/testdata"
setsid nohup env MODEL_DIR="$Q/Qwen3.5-27B" \
  ./start_vllm.sh > "$VLLM_LOG" 2>&1 < /dev/null &
disown
sleep 5
echo "  pid: $(pgrep -f 'vllm serve' | head -1)"
echo "  log: $VLLM_LOG"

echo "=== [$(date +%H:%M:%S)] 3. 等 ready (curl /health, 最长 15min) ==="
SECONDS=0
MAX_WAIT=900
while ! curl -s --noproxy 127.0.0.1 http://127.0.0.1:8001/health >/dev/null 2>&1; do
  sleep 20
  ELAPSED=$SECONDS
  if [[ $ELAPSED -gt $MAX_WAIT ]]; then
    echo "  ✗ 超时 ${MAX_WAIT}s, 查 log:"
    tail -40 "$VLLM_LOG"
    exit 1
  fi
  if (( ELAPSED % 60 < 20 )); then
    LOAD=$(tail -1 "$VLLM_LOG" 2>/dev/null | grep -oE '[0-9]+% Completed \|[0-9]+/11' | head -1)
    echo "  [${ELAPSED}s] /health 未通, load=$LOAD"
  fi
done
echo "  ✓ vllm ready (${SECONDS}s)"

echo "=== [$(date +%H:%M:%S)] 4. 跑 4-8K baseline (10 prompts) ==="
cd "$Q/testdata"
# vllm bench serve 用 trust_env=True 的 aiohttp, 容器 SCNet proxy (10.13.17.166:3128)
# 会劫持 127.0.0.1:8001 -> 503. NO_PROXY 让 aiohttp 跳过 loopback 走直连.
# (不动 vllm flags / 权重 / KV 量化, 这是纯 client 端网络修复)
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
MODEL_DIR="$Q/Qwen3.5-27B" \
  vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 --port 8001 \
    --endpoint /v1/chat/completions \
    --model Qwen3.5-27B \
    --tokenizer "$Q/Qwen3.5-27B" \
    --dataset-name custom \
    --dataset-path ./4-8K_throughput.jsonl \
    --num-prompts 10 \
    --no-oversample \
    --max-concurrency 1 \
    --request-rate 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir ./test/4-8K_throughput \
    --result-filename result.json 2>&1 | tail -25

echo "=== [$(date +%H:%M:%S)] 5. 关键数字 ==="
RESULT="$Q/testdata/test/4-8K_throughput/result.json"
if [[ -f "$RESULT" ]]; then
  python3.10 - <<PYEOF
import json
d = json.load(open("$RESULT"))
print(f"  output_throughput:  {d.get('output_throughput', 0):.2f} tok/s")
print(f"  total_tokens:       {d.get('total_tokens_output', 0)}")
print(f"  TTFT  P50/P99:      {d.get('ttft_p50', 0):.0f} / {d.get('ttft_p99', 0):.0f} ms")
print(f"  TPOT  P50/P99:      {d.get('tpot_p50', 0):.1f} / {d.get('tpot_p99', 0):.1f} ms")
PYEOF
else
  echo "  ✗ $RESULT 不在"
fi

echo ""
echo "=== [$(date +%H:%M:%S)] ✓ baseline 完成. vllm 还在跑 (pid $(pgrep -f 'vllm serve' | head -1)) ==="