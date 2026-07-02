#!/bin/bash
# Stream A: block size sweep (16 / 32 / 64), 4-8K 子集.
# 每个 block_size 启一次 vllm, 跑 5 prompts bench, 比较 throughput.
# 关键: 每次启 vllm 前必须确认 DCU 显存释放 (前次 vllm::EngineCore 可能残留).

set -u
set -o pipefail

export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export VLLM_LOG=/root/vllm_sweep.log

# DTK runtime libs
DTK_LIBS=$(find /opt/dtk-26.04-DCC2602-0317 -maxdepth 4 -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${DTK_LIBS}${LD_LIBRARY_PATH:-}"
export PATH=/usr/local/bin:/opt/dtk-26.04-DCC2602-0317/bin:/opt/dtk-26.04-DCC2602-0317/llvm/bin:$PATH

# 强杀所有 vllm 关联进程 (含 EngineCore / multiprocessing.resource_tracker)
kill_all_vllm() {
  pkill -9 -f "vllm" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -f "EngineCoreProc" 2>/dev/null
  pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
  sleep 8
}

# 等 DCU 显存释放到 > 50GB free
wait_dcu_free() {
  for i in $(seq 1 30); do
    FREE=$(python3.10 -c "import torch; f,_=torch.cuda.mem_get_info(); print(int(f/1024**3))" 2>/dev/null)
    if [[ "$FREE" -gt 50 ]]; then
      echo "  DCU free: ${FREE} GB (ok)"
      return 0
    fi
    echo "  DCU free: ${FREE} GB, 等..."
    sleep 5
  done
  echo "  ✗ DCU 30s 内没释放, 强杀再来"
  return 1
}

start_vllm_with_block_size() {
  local bs=$1
  cd "$Q/testdata"
  MODEL_DIR="$Q/Qwen3.5-27B" vllm serve "$Q/Qwen3.5-27B" \
    --served-model-name Qwen3.5-27B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --block-size "$bs"
}

run_bench() {
  local label=$1
  cd "$Q/testdata"
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
      --result-dir "./test/block_size_${label}" \
      --result-filename result.json 2>&1 | tail -3
}

wait_ready() {
  SECONDS=0
  MAX_WAIT=1500  # 25 min — 27B 模型冷启动 + triton compile (cudagraph capture_sizes 1-256) 约 14-18 min
  # vLLM /health 在模型加载完之前就返 200, 必须等模型 loaded + EngineCore proc 稳定
  # 信号: EngineCore log 出现 "Model loading took ... GiB" + triton "Cache the graph"
  while true; do
    LOAD_OK=$(grep -c "Model loading took" "$VLLM_LOG" 2>/dev/null; true)
    COMPILE_OK=$(grep -c "Capturing CUDA graph\|Cache the graph\|compile range" "$VLLM_LOG" 2>/dev/null; true)
    LOAD_OK=${LOAD_OK:-0}
    COMPILE_OK=${COMPILE_OK:-0}
    HEALTH=$(curl -s --noproxy 127.0.0.1 -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8001/health 2>&1)
    if [[ "$LOAD_OK" =~ ^[0-9]+$ ]] && [[ "$LOAD_OK" -gt 0 ]] \
       && [[ "$COMPILE_OK" =~ ^[0-9]+$ ]] && [[ "$COMPILE_OK" -gt 0 ]] \
       && [[ "$HEALTH" == "200" ]]; then
      echo "  ✓ ready (${SECONDS}s, model loaded + compile OK + /health 200)"
      return 0
    fi
    sleep 20
    ELAPSED=$SECONDS
    if [[ $ELAPSED -gt $MAX_WAIT ]]; then
      echo "  ✗ 超时 ${MAX_WAIT}s, 最后状态: load=$LOAD_OK compile=$COMPILE_OK health=$HEALTH"
      tail -40 "$VLLM_LOG"
      return 1
    fi
    if (( ELAPSED % 60 < 20 )); then
      LOAD=$(tail -1 "$VLLM_LOG" 2>/dev/null | grep -oE 'Completed \| [0-9]+/11' | head -1)
      echo "  [${ELAPSED}s] load=$LOAD compile=$COMPILE_OK health=$HEALTH"
    fi
  done
}

for BS in 16 32 64; do
  echo ""
  echo "============================================================"
  echo "  block_size=$BS"
  echo "============================================================"

  echo "=== [$(date +%H:%M:%S)] 杀旧 + 等 DCU 释放 ==="
  kill_all_vllm
  wait_dcu_free || { echo "DCU 不释放, 跳过 $BS"; continue; }

  echo "=== [$(date +%H:%M:%S)] 启 vllm --block-size $BS ==="
  setsid nohup bash -c "$(declare -f start_vllm_with_block_size); start_vllm_with_block_size $BS" \
    > "$VLLM_LOG" 2>&1 < /dev/null &
  disown
  sleep 10

  if ! wait_ready; then
    echo "  ✗ block_size=$BS 启失败, 跳过"
    continue
  fi

  echo "=== [$(date +%H:%M:%S)] 跑 bench (5 prompts) ==="
  if ! run_bench "$BS"; then
    echo "  ✗ bench 失败, 看 vllm.log"
    tail -20 "$VLLM_LOG"
    continue
  fi

  echo "=== [$(date +%H:%M:%S)] 结果 ==="
  RESULT="$Q/testdata/test/block_size_${BS}/result.json"
  if [[ -f "$RESULT" ]]; then
    python3.10 - <<PYEOF
import json
d = json.load(open("$RESULT"))
print(f"  output_throughput: {d.get('output_throughput', 0):.2f} tok/s")
print(f"  TTFT  P99:         {d.get('ttft_p99', d.get('p99_ttft_ms', 0)):.0f} ms")
print(f"  TPOT  P99:         {d.get('tpot_p99', d.get('p99_tpot_ms', 0)):.1f} ms")
PYEOF
  else
    echo "  ✗ $RESULT 不在"
  fi
done

echo ""
echo "=== [$(date +%H:%M:%S)] sweep 完 ==="