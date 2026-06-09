# ADR 0010: Attention backend 选型 (vLLM 0.18.1, DCU)

**日期**: 2026-06-09
**状态**: Proposed
**Owner**: 队员 B (vLLM backend 调研, 6 h/周 × 1 周)
**关联**:
- spec §5.1 决策表 (Decode attention 提速 / Custom backend 路径)
- spec §5.2 backend 注册机制, §5.3 FlashAttention ROCm FP8, §5.4 FP8 SKU 限制
- ADR 0006 (vLLM 0.18.1 源码阅读笔记), 0006a (Docker 镜像), 0001 (DCU SKU)
- Task 0.4 (P0 backend 路径 smoke), Task 0.5 (Triton DCU FP8 验证), Task 3B 系列 (KV 量化)

## Context

vLLM 0.18.1 `AttentionBackendEnum` 有 24 个值 (`vllm/v1/attention/backends/registry.py:34-87`)。DCU 实际可用 3 个: `TRITON_ATTN` (默认) / `ROCM_AITER_FA` (开 aiter) / `ROCM_AITER_UNIFIED_ATTN` (开 aiter)。选错 backend 的沉没成本高: prefill 长上下文 + decode 高吞吐对 backend 选择敏感, 且 cudagraph capture 状态与 backend 绑定, 中途切 backend 需重 capture。9 周预算 + 4 人 5-12 h/周, 不允许反复试错。

## Decision

**默认走 `TRITON_ATTN` (ROCm 平台默认), P3 集成日再视 aiter 实测结果决定是否切。**

阶段化:
- **P0 末 (Task 0.4 backend smoke)**: 在 DCU 上跑通 `vllm bench serve` 1 个 prompt 验证 `TRITON_ATTN` 可注册可跑, 同时 dry-run 切 `--attention-backend ROCM_AITER_FA` 验证 aiter import 不挂
- **P2 末 (3 档 baseline)**: 全程 `TRITON_ATTN`, 走 spec §5.1 决策
- **P3 集成日**: 若 P0/P2 数据 + 队员 B 在 P3 期间在 DCU 上跑 aiter 三组对比 (TRITON vs AITER-FA vs AITER-Unified, 3 档 × 50 prompts, ADR 0010 决策准则同 ADR 0008/0013), 切到赢家并把 `_get_backend_priorities()` 源码改了 (改法见 learning.md 新 section 末段)。**仅当 SLA (TTFT/TPOT P99 ≤ Baseline × 1.5) 不破 + Δ ≤ 3% 才切**。
- **P3 stretch**: 若 aiter 都不如自写 decode kernel (PagedAttention on DCU FP8), 走 `AttentionBackendEnum.register_backend(CUSTOM, "my.path.MyBackend")` 继承 `TritonAttentionBackend` 改 1 行, 走 spec §5.1 决策表 "Custom backend 路径"

## Consequence

- **P0-P2**: TRITON_ATTN 即 baseline, 不必额外实验; P2 末 baseline 数字天然带 "TRITON_ATTN 默认" 标签
- **P3 集成日**: 选 1 个 backend 锁死进 Dockerfile + serve 启动脚本, 1 行 `--attention-backend` flag 或改源码 priority 2 选 1
- **风险**: TRITON_ATTN 在 16k-32k 长上下文 prefill 阶段可能不如 aiter (FP8 加速 aiter 汇编更优); 缓解: P2 末 profile 一次 (Task 2.3) 拿到 prefill/decode 时间占比
- **不切到 aiter 的兜底**: 若 `ROCM_AITER_FA` 在 CDNA2 (gfx90a, 排除) 不可用, 只剩 TRITON + ROCM_AITER_UNIFIED_ATTN; CDNA3 (gfx942) 三家全可用
- **不再单独起 ADR** 跟踪 aiter 三方对比实验; 实验数据进 P3 末 "集成日" 决定, 此 ADR 状态届时改 "已决策" 并附最终选择

## 接入方式 (1 行命令 vs 改源码)

| 方式 | 适用场景 | 风险 |
|------|---------|------|
| `--attention-backend TRITON_ATTN` | P0-P2 (vLLM 0.18.1 已加 flag, 源码 `vllm/engine/arg_utils.py:782`) | 低, 一行 flag, 不动 vLLM 镜像 |
| 改 `_get_backend_priorities()` 源码 (镜像外) | P3 集成日锁定配置 | 中, 改坏要 restore; 镜像更新会覆盖 |
| `register_backend(AttentionBackendEnum.CUSTOM, ...)` | P3 stretch 自定义 kernel | 高, 需 import 顺序 + `get_class()` 校验 |

## 待办

- [ ] 队员 B: P0 末跑通 `--attention-backend` 3 候选 (TRITON/AITER_FA/AITER_UNIFIED) smoke (Task 0.4)
- [ ] 队员 B: 验证 `VLLM_ATTENTION_BACKEND` env var **不存在** (v0.18.1 `envs.py` grep 0 命中) — 已在 learning.md 标
- [ ] 队员 B: 把 `learning.md` 新 section "Attention backend 选型" 中 "未验" 项在 P0 0.4 / 0.5 跑通后回填
- [ ] P3 集成日: 根据 aiter 三方对比实验数据 + SLA/精度双门槛, 锁定最终 backend 并更新本 ADR 状态
