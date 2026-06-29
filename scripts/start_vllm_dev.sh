#!/usr/bin/env bash
# DCU 优化版 vllm serve (DEV ONLY — 内部用, 不作提交).
#
# 参考: "海光 DCU 上 vLLM 部署 Qwen 大模型: 冷启动 15min → 6min" 实战 + LOCKED-param 约束.
# 完整解读: docs/decisions/0014-dcu-startup-optimization.md
#
# LOCKED 参数 (赛方不允许 bench 阶段加): max-num-seqs / max-num-batched-tokens /
# chat template kwarg. 这里默认就不传这些.
#
# **绝对不能**用于 P5 提交, 提交用 scripts/start_vllm_bench.sh.

set -u
set -o pipefail

# 持久盘 (重开不丢) — 用 /public/home 共享 NFS.
export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export VLLM_WHEEL_DIR=$GOVINDA_DIR/dist          # 自编译 vllm wheel 落地点 (bdist_wheel 输出)
export VLLM_SRC_DIR=$Q/vllm_cscc                  # vllm 源码 + editable install

# ===== Python + vllm 调用方式 =====
# image 装 vllm 是 editable install (pip show Location=$VLLM_SRC_DIR), `vllm` CLI 不在 PATH.
# 用 `python -m vllm.entrypoints.cli.main <cmd>` 调起, 配合 cwd=$VLLM_SRC_DIR 让
# editable install 能 import. 已验证 `cd vllm_cscc && python -c "import vllm"` OK.
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_RUNNER="$PYTHON_BIN -m vllm.entrypoints.cli.main"

# ===== Pre-flight: 按需装 vllm + runai_streamer =====
VLLM_EXPECTED="0.18.1"
if ! "$PYTHON_BIN" -c "import vllm; assert vllm.__version__ == '$VLLM_EXPECTED'" 2>/dev/null; then
  echo "[start_vllm_dev] vllm $VLLM_EXPECTED 不在 / 版本不对, 按需装..."
  if ls "$VLLM_WHEEL_DIR"/vllm-${VLLM_EXPECTED}*.whl 1> /dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --no-deps -q "$VLLM_WHEEL_DIR"/vllm-${VLLM_EXPECTED}*.whl
    echo "  -> 从 $VLLM_WHEEL_DIR 装好"
  else
    echo "  WARN: 找不到 $VLLM_WHEEL_DIR/vllm-${VLLM_EXPECTED}*.whl"
    echo "  第一次跑需先编译: cd $VLLM_SRC_DIR && $PYTHON_BIN setup.py bdist_wheel"
    exit 1
  fi
else
  echo "[start_vllm_dev] vllm $VLLM_EXPECTED OK, 跳过安装"
fi

# runai_streamer 必装 (article 推荐的快速权重加载器)
"$PYTHON_BIN" -c "import runai_streamer" 2>/dev/null || "$PYTHON_BIN" -m pip install -q runai-model-streamer

# ===== DCU / HIP 特有 (海光 ROCm 体系) =====
export HIP_VISIBLE_DEVICES=0                            # 指定可见 DCU
# HSA_OVERRIDE_GFX_VERSION: 按 rocminfo | grep gfx 查实际架构填; gfx90a = 9.0.0.
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # 缓解长跑显存碎片
export HIP_FORCE_DEV_KERNARG=1                          # HIP kernel 参数直传, 降 launch 开销
export SAFETENSORS_FAST_GPU=1                            # safetensors 更快搬到显存

# ===== 三类缓存全指持久盘, 重开跳过 torch.compile / Triton / MIOPEN =====
mkdir -p "$Q/vllm_cache" "$Q/triton_cache" "$Q/miopen_cache"
export VLLM_CACHE_ROOT=$Q/vllm_cache
export TRITON_CACHE_DIR=$Q/triton_cache
export MIOPEN_USER_DB_PATH=$Q/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$Q/miopen_cache

# ===== vLLM 行为调优 =====
export VLLM_USE_TRITON_FLASH_ATTN=1                     # DCU 上 attention 走 Triton (默认)
export VLLM_ROCM_USE_AITER=0                            # aiter 部分 gfx9 反而慢, 先关

MODEL_DIR=$Q/Qwen3.5-27B

# ===== 启动 vllm =====
# 必须 cd 到 editable install 目录, python 才能 import vllm
cd "$VLLM_SRC_DIR" || { echo "FAIL: $VLLM_SRC_DIR 不存在"; exit 1; }

# cudagraph_capture_sizes=[1] 对应评测 concurrency=1. dev 调试时扩到
# [1,2,4,8,16,32] 可以看 multi-batch, 但提交时只截 [1].
exec $VLLM_RUNNER serve "$MODEL_DIR" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --load-format runai_streamer \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'