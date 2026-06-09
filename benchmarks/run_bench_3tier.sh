#!/usr/bin/env bash
# benchmarks/run_bench_3tier.sh
# 3 档 baseline 跑分入口 (DCU-only; CPU 端 vllm 跑不动 27B)
# 用法: bash benchmarks/run_bench_3tier.sh [--tier all|4k-8k|8k-16k|16k-32k] [--output benchmarks/baseline]
# 每档 50 prompts × 3 iter 取稳态, 配比跟 spec §5 一致 (4k-8k 20% / 8k-16k 50% / 16k-32k 30%)

set -euo pipefail

# 默认档位 + 输出目录
TIER="${1:-all}"
OUTPUT="${2:-benchmarks/baseline}"

# 准备输出目录
mkdir -p "$OUTPUT"

# 每档 prompt 数 (DCU 稳态跑 3 次, 取最后一次)
declare -A PROMPTS=(
    ["4k-8k"]=50
    ["8k-16k"]=50
    ["16k-32k"]=50
)

# 顺序跑 3 档, 单档内跑 3 iter
for t in "4k-8k" "8k-16k" "16k-32k"; do
    # 过滤: --tier all 跑全部, 否则只跑指定档
    if [ "$TIER" != "all" ] && [ "$TIER" != "$t" ]; then continue; fi
    echo "[$(date +%H:%M:%S)] Running tier: $t (${PROMPTS[$t]} prompts × 3 iters)"
    for i in 1 2 3; do
        # 调 run_bench.py 跑单档单 iter
        python3 -m benchmarks.run_bench \
            --tier "$t" \
            --num-prompts "${PROMPTS[$t]}" \
            --output "$OUTPUT" \
            --extra-args "--result-filename $t-iter${i}"
    done
done

echo "[$(date +%H:%M:%S)] Done. Results in $OUTPUT"
