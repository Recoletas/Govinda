#!/bin/bash
# 干净的 vllm serve 命令, 严格对齐比赛规则 §9(7)(8).
#
# LOCKED 参数 (赛方不允许在 bench 阶段调):
#   - model / tokenizer / tokenizer-mode / chat template
#   - temperature (锁 0)
#   - max_tokens
#   - max-num-seqs
#   - max-num-batched-tokens
#   - 任何 batch scheduler 相关参数
#
# 只允许加的 flag: --max-model-len 32768
# 其它都是 vllm 默认值.
#
# 用法 (容器内):
#   bash scripts/start_vllm_bench.sh
#
# 不传 --max-num-seqs 128, --max-num-batched-tokens 4096,
# --default-chat-template-kwargs 等. 这些都是 dev-only.

set -euo pipefail

vllm serve /public/home/xdzs2026_c087/Qwen3.5-27B \
  --port 8001 \
  --max-model-len 32768