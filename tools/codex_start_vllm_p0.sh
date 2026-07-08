#!/usr/bin/env bash
# AI-generated, awaiting verification by recoletas on 2026-07-09.
#
# Run inside the SCNet container. Starts vLLM with the same scoring-facing
# command family as scripts/start_vllm_bench.sh, plus explicit P0/GDN env knobs.
# It does not kill existing processes; stop old vLLM manually before using it.

set -euo pipefail

Q="${Q:-/public/home/xdzs2026_c087}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$Q/vllm_cscc}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
LOG_FILE="${LOG_FILE:-/root/vllm.log}"

export LD_LIBRARY_PATH="/opt/dtk-26.04-DCC2602-0317/lib:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/lib:/opt/dtk-26.04-DCC2602-0317/dcc/lib:/opt/dtk-26.04-DCC2602-0317/hip/lib:/opt/dtk-26.04-DCC2602-0317/hipblas/lib:/opt/dtk-26.04-DCC2602-0317/hipblaslt/lib:/opt/dtk-26.04-DCC2602-0317/hipdnn/lib:/opt/dtk-26.04-DCC2602-0317/hipfft/lib:/opt/dtk-26.04-DCC2602-0317/hiprand/lib:/opt/dtk-26.04-DCC2602-0317/hipsolver/lib:/opt/dtk-26.04-DCC2602-0317/hipsparse/lib:/opt/dtk-26.04-DCC2602-0317/hsa/lib:/opt/dtk-26.04-DCC2602-0317/llvm/lib:/opt/dtk-26.04-DCC2602-0317/miopen/lib:/opt/dtk-26.04-DCC2602-0317/rccl/lib:/opt/dtk-26.04-DCC2602-0317/rocblas/lib:/opt/dtk-26.04-DCC2602-0317/rocprim/lib:/opt/dtk-26.04-DCC2602-0317/rocrand/lib:/opt/dtk-26.04-DCC2602-0317/rocsolver/lib:/opt/dtk-26.04-DCC2602-0317/rocsparse/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH="${ROCM_PATH:-/opt/dtk-26.04-DCC2602-0317}"
export TRITON_HIP_CLANG_PATH="${TRITON_HIP_CLANG_PATH:-/opt/dtk-26.04-DCC2602-0317/llvm/bin/clang}"
export TRITON_HIP_LLD_PATH="${TRITON_HIP_LLD_PATH:-/opt/dtk-26.04-DCC2602-0317/llvm/bin/ld.lld}"
export PATH="/usr/local/bin:/opt/dtk-26.04-DCC2602-0317/bin:/opt/dtk-26.04-DCC2602-0317/llvm/bin:$PATH"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$Q/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$Q/triton_cache}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$Q/miopen_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$Q/miopen_cache}"

export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-1}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"

export VLLM_TRITON_PREFILL_TILE64_POLICY="${VLLM_TRITON_PREFILL_TILE64_POLICY:-mid}"
# For the GDN experiment branch only. On main this env is harmless.
export VLLM_GDN_CAUSAL_CONV1D_BLOCK_M="${VLLM_GDN_CAUSAL_CONV1D_BLOCK_M:-16}"
export FLA_GDN_FIX_BT="${FLA_GDN_FIX_BT:-0}"

mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

echo "=== start vLLM P0 $(date +%F_%T) ==="
echo "VLLM_SRC_DIR=$VLLM_SRC_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "VLLM_TRITON_PREFILL_TILE64_POLICY=$VLLM_TRITON_PREFILL_TILE64_POLICY"
echo "VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=$VLLM_GDN_CAUSAL_CONV1D_BLOCK_M"
echo "FLA_GDN_FIX_BT=$FLA_GDN_FIX_BT"

cd "$VLLM_SRC_DIR"
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$Q/Qwen3.5-27B" \
    --max-model-len 32768 \
    --load-format runai_streamer \
    --compilation-config '{"cudagraph_capture_sizes":[1]}' \
    > "$LOG_FILE" 2>&1
