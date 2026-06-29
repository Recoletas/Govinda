#!/usr/bin/env bash
# 严格按规则 §9(7)(8) 的 vllm serve — 仅作 P5 提交 / 官方评测用.
#
# 锁定参数 (§9(8) 不可调):
#   - chat template / tokenizer-mode / model weights
#   - max_tokens / temperature (= 0) / max-num-seqs
#   - max-num-batched-tokens 及其它 batch scheduler 相关
#   - --served-model-name / OpenAI API 路径 / host:port (平台固定)
#
# 唯一允许加的 flag: --max-model-len 32768 (§9(7))
# 其它默认.
#
# 优化 (cache / loader format) 走 article 推荐的 — 不违反 LOCKED, 仍合规.
# 详细解读: docs/decisions/0014-dcu-startup-optimization.md

set -u
set -o pipefail

export Q=/public/home/xdzs2026_c087
mkdir -p "$Q/vllm_cache" "$Q/triton_cache" "$Q/miopen_cache"

# ===== DCU / HIP 特有 (env 变量, 不算 "flag") =====
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_FORCE_DEV_KERNARG=1
export SAFETENSORS_FAST_GPU=1

# ===== 缓存持久化 (env, 允许) =====
export VLLM_CACHE_ROOT=$Q/vllm_cache
export TRITON_CACHE_DIR=$Q/triton_cache
export MIOPEN_USER_DB_PATH=$Q/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$Q/miopen_cache

# ===== vLLM 行为 (env, 允许) =====
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=0

vllm serve "$Q/Qwen3.5-27B" \
    --max-model-len 32768 \
    --load-format runai_streamer \
    --compilation-config '{"cudagraph_capture_sizes":[1]}'