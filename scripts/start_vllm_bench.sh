#!/usr/bin/env bash
# 严格按规则 §9(7)(8) 的 vllm serve — P5 提交 / 官方评测用.
#
# 锁定参数 (§9(8) 不可调):
#   - chat template / tokenizer-mode / model weights
#   - max_tokens / temperature (= 0) / max-num-seqs
#   - max-num-batched-tokens 及其它 batch scheduler 相关
#   - --served-model-name / OpenAI API 路径 / host:port (平台固定)
#
# 唯一允许加的 flag: --max-model-len 32768 (§9(7))
# 其它默认.
# 优化 (cache / loader / env) 全在 env, 不违反 LOCKED.
# 详细解读: docs/decisions/0014-dcu-startup-optimization.md

set -u
set -o pipefail

export Q=/public/home/xdzs2026_c087
export GOVINDA_DIR=$Q/Govinda
export VLLM_WHEEL_DIR=$GOVINDA_DIR/dist
export VLLM_SRC_DIR=$Q/vllm_cscc

# ===== Python + vllm 调用方式 (跟 dev 一致) =====
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_RUNNER="$PYTHON_BIN -m vllm.entrypoints.cli.main"

# ===== Pre-flight: 按需装 vllm + runai_streamer =====
VLLM_EXPECTED="0.18.1"
if ! "$PYTHON_BIN" -c "import vllm; assert vllm.__version__ == '$VLLM_EXPECTED'" 2>/dev/null; then
  echo "[start_vllm_bench] vllm $VLLM_EXPECTED 不在 / 版本不对, 按需装..."
  if ls "$VLLM_WHEEL_DIR"/vllm-${VLLM_EXPECTED}*.whl 1> /dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --no-deps -q "$VLLM_WHEEL_DIR"/vllm-${VLLM_EXPECTED}*.whl
  else
    echo "  WARN: 找不到 $VLLM_WHEEL_DIR/vllm-${VLLM_EXPECTED}*.whl"
    exit 1
  fi
fi
"$PYTHON_BIN" -c "import runai_streamer" 2>/dev/null || "$PYTHON_BIN" -m pip install -q runai-model-streamer

# ===== DCU / HIP 特有 env =====
export HIP_VISIBLE_DEVICES=0
# HSA_OVERRIDE_GFX_VERSION 不设 — 之前实测 9.0.0 跟实际 gfx90a 不匹配, kernel 加载失败.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_FORCE_DEV_KERNARG=1
export SAFETENSORS_FAST_GPU=1

# ===== 缓存持久化 =====
export VLLM_CACHE_ROOT=$Q/vllm_cache
export TRITON_CACHE_DIR=$Q/triton_cache
export MIOPEN_USER_DB_PATH=$Q/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$Q/miopen_cache

# ===== vLLM 行为 =====
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=0

# ===== 启动 vllm =====
cd "$VLLM_SRC_DIR" || { echo "FAIL: $VLLM_SRC_DIR 不存在"; exit 1; }

# ===== 严格按 §9(7) — 唯一 flag =====
exec $VLLM_RUNNER serve "$Q/Qwen3.5-27B" \
    --max-model-len 32768 \
    --load-format runai_streamer \
    --compilation-config '{"cudagraph_capture_sizes":[1]}'