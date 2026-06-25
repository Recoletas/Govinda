#!/bin/bash
# DEV-only vllm serve 命令 — 含比赛规则 §9 锁定的参数, 仅内部用.
#
# LOCKED 参数 (赛方不允许在 bench 阶段调): max-num-seqs / max-num-batched-tokens /
# chat template kwarg. 这里用了是为了 dev 加速 (--max-num-seqs 128 让 batch scheduler
# 跑得更激进, --default-chat-template-kwargs 关 thinking mode 让输出 token 少).
#
# **绝对不能**用于 P5 提交, 提交用 scripts/start_vllm_bench.sh.
#
# 详细解读见 docs/decisions/0013-competition-rules-interpretation.md.

set -euo pipefail

vllm serve /public/home/xdzs2026_c087/Qwen3.5-27B \
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
  --default-chat-template-kwargs '{"enable_thinking": false}'