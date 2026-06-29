#!/usr/bin/env bash
# DCU vllm serve (DEV ONLY — 内部用, 不作提交).
#
# 原则: **最小化启动 → 验证 baseline → 一个一个加优化 → 看哪个炸**.
# 之前堆 10+ env 互相打架 (HSA_OVERRIDE_GFX_VERSION / SAFETENSORS_FAST_GPU /
# VLLM_USE_TRITON_FLASH_ATTN / --load-format runai_streamer ...). 重置到最简,
# 用官方 start_vllm.sh 那个用户已经验证能起来的命令 + 改 python -m CLI 调起.
#
# LOCKED 参数 (赛方不允许 bench 阶段加): max-num-seqs / max-num-batched-tokens /
# chat template kwarg. 这里默认就不传这些.
#
# **绝对不能**用于 P5 提交, 提交用 scripts/start_vllm_bench.sh.

set -u
set -o pipefail

# 持久盘 + 路径常量
export Q=/public/home/xdzs2026_c087
export VLLM_SRC_DIR=$Q/vllm_cscc                  # vllm 源码 + editable install

# ===== vllm 调用方式 =====
# image 装 vllm 是 editable install, `vllm` CLI 不在 PATH. 用 python -m 调起 + cd 到源码目录.
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_RUNNER="$PYTHON_BIN -m vllm.entrypoints.cli.main"

cd "$VLLM_SRC_DIR" || { echo "FAIL: $VLLM_SRC_DIR 不存在"; exit 1; }

# ===== 启动 vllm (用官方已经验证的命令, 改 python -m 调用) =====
# 这是官方 testdata/start_vllm.sh 的命令, 用户实测能起来:
#   vllm serve $MODEL_DIR --port 8001 --trust-remote-code --dtype bfloat16
#     --served-model-name Qwen3.5-27B --gpu-memory-utilization 0.95
#     --max-num-batched-tokens 4096 --max-num-seqs 128 ...
#
# vllm CLI 不可用 → 改成 python -m vllm.entrypoints.cli.main serve
exec $VLLM_RUNNER serve "$Q/Qwen3.5-27B" \
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

# ===== 下一步 (这个脚本跑通后再加, 一个一个试) =====
# 1. HSA_OVERRIDE_GFX_VERSION=9.0.0  (实测 gfx90a 不一定对, 启动失败就撤)
# 2. VLLM_USE_TRITON_FLASH_ATTN=1    (默认就是 TRITON_ATTN, 不一定需要显式设)
# 3. VLLM_ROCM_USE_AITER=0            (默认应该就是 0)
# 4. PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# 5. --load-format runai_streamer (需先 pip install runai-model-streamer)
# 6. VLLM_CACHE_ROOT / TRITON_CACHE_DIR / MIOPEN_USER_DB_PATH (持久化)
# 7. --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'