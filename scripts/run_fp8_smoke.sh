#!/bin/bash
# Stream B quick path: vllm fp8 KV cache smoke + bench, 完全用 vllm 现成 fp8 路径,
# 不写新 kernel. 用 start_vllm_bench.sh 加速 (持久 cache 二次启动 < 1 min).
#
# 用法 (容器里跑):
#   bash /public/home/xdzs2026_c087/Govinda/scripts/run_fp8_smoke.sh <block_size>
#
# 假设:
#   - vllm wheel 已装
#   - 容器里 vllm import 走 site-packages (默认)
#   - /public/home/xdzs2026_c087/Qwen3.5-27B/ 模型在
#   - /public/home/xdzs2026_c087/testdata/4-8K_throughput.jsonl 在

set -u
set -o pipefail

BS=${1:-16}   # block_size, 默认 16

export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export VLLM_LOG=/root/vllm_fp8.log

# DTK runtime (容器 ssh 进来不会自动 source)
DTK_LIBS=$(find /opt/dtk-26.04-DCC2602-0317 -maxdepth 4 -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${DTK_LIBS}${LD_LIBRARY_PATH:-}"
export PATH=/usr/local/bin:/opt/dtk-26.04-DCC2602-0317/bin:/opt/dtk-26.04-DCC2602-0317/llvm/bin:$PATH

# /opt/rocm symlink (Triton 必需). 容器重启后可能丢, 重建.
ln -sf /opt/dtk-26.04-DCC2602-0317 /opt/rocm 2>/dev/null

# 强杀上次 vllm + 等 DCU
pkill -9 -f "vllm" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "EngineCoreProc" 2>/dev/null
pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
sleep 8
echo "DCU free: $(python3.10 -c 'import torch; f,_=torch.cuda.mem_get_info(); print(int(f/1024**3))' 2>/dev/null) GB"

echo ""
echo "=== [$(date +%H:%M:%S)] 启 vllm (fp8 KV cache, block_size=$BS) ==="
cd "$Q/testdata"
MODEL_DIR="$Q/Qwen3.5-27B" \
nohup vllm serve "$Q/Qwen3.5-27B" \
    --served-model-name Qwen3.5-27B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --load-format runai_streamer \
    --compilation-config '{"cudagraph_capture_sizes":[1]}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype fp8_e4m3fnuz \
    --block-size "$BS" \
    > "$VLLM_LOG" 2>&1 &
disown
sleep 10

# 等 ready (Model loading took + Cache the graph + /health 200)
SECONDS=0
MAX_WAIT=1500
while true; do
    LOAD_OK=$(grep -c "Model loading took" "$VLLM_LOG" 2>/dev/null; true)
    COMPILE_OK=$(grep -c "Capturing CUDA graph\|Cache the graph\|compile range" "$VLLM_LOG" 2>/dev/null; true)
    HEALTH=$(curl -s --noproxy 127.0.0.1 -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8001/health 2>&1)
    if [[ "$LOAD_OK" =~ ^[0-9]+$ ]] && [[ "$LOAD_OK" -gt 0 ]] \
       && [[ "$COMPILE_OK" =~ ^[0-9]+$ ]] && [[ "$COMPILE_OK" -gt 0 ]] \
       && [[ "$HEALTH" == "200" ]]; then
        echo "  ✓ ready (${SECONDS}s)"
        break
    fi
    sleep 20
    ELAPSED=$SECONDS
    if [[ $ELAPSED -gt $MAX_WAIT ]]; then
        echo "  ✗ 超时 ${MAX_WAIT}s"
        tail -40 "$VLLM_LOG"
        exit 1
    fi
    if (( ELAPSED % 60 < 20 )); then
        LOAD=$(tail -1 "$VLLM_LOG" 2>/dev/null | grep -oE 'Completed \| [0-9]+/11' | head -1)
        echo "  [${ELAPSED}s] load=$LOAD compile=$COMPILE_OK health=$HEALTH"
    fi
done

# Smoke (5 prompts)
echo ""
echo "=== [$(date +%H:%M:%S)] smoke 5 prompts ==="
cd "$Q/testdata"
RESULT_DIR="$Q/testdata/test/fp8_block${BS}_smoke"
mkdir -p "$RESULT_DIR"
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
    --num-prompts 5 \
    --no-oversample \
    --max-concurrency 1 \
    --request-rate 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename result.json 2>&1 | tail -3

echo ""
echo "=== [$(date +%H:%M:%S)] smoke 结果 ==="
RESULT="$RESULT_DIR/result.json"
if [[ -f "$RESULT" ]]; then
    python3.10 - <<PYEOF
import json
d = json.load(open("$RESULT"))
ttft_p99 = d.get("ttft_p99", d.get("p99_ttft_ms", 0))
tpot_p99 = d.get("tpot_p99", d.get("p99_tpot_ms", 0))
throughput = d.get("output_throughput", 0)
print(f"  output_throughput: {throughput:.2f} tok/s")
print(f"  TTFT  P99:         {ttft_p99:.0f} ms  (SLA ≤ 6836)")
print(f"  TPOT  P99:         {tpot_p99:.1f} ms  (SLA ≤ 104.69)")
go = "GO" if ttft_p99 <= 6836 and tpot_p99 <= 104.69 else "NO-GO"
print(f"  SLA check: {go}")
print(f"  baseline 4-8K: 12.95 tok/s, fp8 gain so far: {throughput/12.95*100-100:.1f}%")
PYEOF
fi

echo ""
echo "=== [$(date +%H:%M:%S)] ✓ smoke 完, vllm 还在跑, 检查 vllm.log ==="
tail -5 "$VLLM_LOG"