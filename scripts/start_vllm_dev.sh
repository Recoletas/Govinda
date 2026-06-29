#!/usr/bin/env bash
# DCU 优化版 vllm serve (DEV ONLY — 内部用, 不作提交).
#
# 参考: "海光 DCU 上 vLLM 部署 Qwen 大模型: 冷启动 15min → 6min" 实战 + 我们的 LOCKED-param 约束.
# 完整解读见 docs/decisions/0014-dcu-startup-optimization.md
#
# LOCKED 参数 (赛方不允许 bench 阶段加): max-num-seqs / max-num-batched-tokens /
# chat template kwarg. 这里默认就不传这些.
#
# **绝对不能**用于 P5 提交, 提交用 scripts/start_vllm_bench.sh.

set -u
set -o pipefail

# 持久盘 (重开不丢) — 这里用 /public/home/xdzs2026_c087/ 共享 NFS,
# 你自己也可以换成 /data/qwen 之类的本地数据盘.
export Q=/public/home/xdzs2026_c087
export VLLM_WHEEL_DIR=$Q/Govinda/dist          # 自编译 vllm wheel 落地点 (bdist_wheel 输出)

# ===== DCU / HIP 特有 (海光 ROCm 体系) =====
export HIP_VISIBLE_DEVICES=0                            # 指定可见 DCU
# HSA_OVERRIDE_GFX_VERSION: 按 rocminfo | grep gfx 查实际架构填; gfx90a = 9.0.0.
# 填错会启动失败或算子崩. 不确定就先用 9.0.0 (CDNA2 fallback 安全值).
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
# TORCH_BLAS_PREFER_HIPBLASLT=1 / 0 — 小 batch decode 用 rocBLAS 可能更好, A/B 实测决定

MODEL_DIR=$Q/Qwen3.5-27B

# cudagraph_capture_sizes=[1] 对应评测 concurrency=1. dev 调试时扩到
# [1,2,4,8,16,32] 可以看 multi-batch, 但提交时只截 [1].
vllm serve "$MODEL_DIR" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --load-format runai_streamer \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'

# vLLM 会自动选 runai_streamer (上面指定), 不需要额外参数