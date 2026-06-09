# ADR 0011: P0 末调研输出 (vLLM backend / DCU 性能 / KV 量化)

**日期**: 2026-06-09
**状态**: Proposed — 等 P0 0.4/0.5 跑通后回填/改 Accepted
**Owner**: 队长 (合) + 队员 B (vLLM 块) + 队员 A (Kernel 块) + 队员 C (DCU 块) 联合产出
**关联**:
- spec §12 "剩余硬卡门" 第 3 条 (P0 末调研)
- ADR 0001 (DCU SKU), 0006 (vLLM 源码阅读), 0006a (Docker 镜像)
- ADR 0009 (KV 量化策略), 0010 (attention backend 选型)
- Task 0.4 (P0 backend smoke), Task 0.5 (Triton DCU FP8)

## Context

spec §12 三条 "剩余硬卡门" 中第 3 条要求 P0 末交三块输出: vLLM 0.18.1 backend 接入点定位、DCU 性能特征、KV 量化方案对比。本 ADR 收口这三块调研结论, 跟 ADR 0006 (源码阅读) / 0009 (KV 量化) / 0010 (backend 选型) 互补: ADR 0006/0010 已涵盖 backend 选型与源码笔记, 本 ADR 补 backend 接入的 3 个层次对比 + DCU 瓶颈维度 + 6 个量化方案横评。决策落地在 P3 集成日。

## Decision

### A. vLLM 0.18.1 backend 接入点 (3 层对比)

| 层次 | 触发位置 (实读源码) | 改 1 行示例 | DCU 预期行为 | 风险 |
|------|---------------------|-------------|---------------|------|
| L1 CLI flag | `vllm/engine/arg_utils.py:782` 加 `--attention-backend` flag; 字段在同文件 `:580` (`attention_backend: AttentionBackendEnum | None = AttentionConfig.backend`) | `vllm serve ... --attention-backend TRITON_ATTN` | ROCm 平台走 `TRITON_ATTN` 默认, 不改源码 | 低, 1 行 flag, 不动 vLLM 镜像; cudagraph 重新 capture 一次 |
| L2 源码 priority | `vllm/platforms/rocm.py:309-352` `_get_backend_priorities()` 函数, 默认返回 `[TRITON_ATTN]` 末尾; `:411` 在 `RocmPlatform.get_valid_backends()` 被调用 | 在 `:351` 前 `backends.insert(0, AttentionBackendEnum.ROCM_AITER_FA)` | DCU 启动时按 priority 顺序选第一个 `validate_configuration()` 过的 backend | 中, 改镜像内文件, 镜像更新会覆盖; 改坏需 restore |
| L3 register API | `vllm/v1/attention/backends/registry.py:210-258` `register_backend(backend, class_path, is_mamba=False)`; 走 `_ATTN_OVERRIDES[backend] = class_path` 路径 | `register_backend(AttentionBackendEnum.CUSTOM, "my.mod.MyBackend")` (`:242-245` 示例) | 启动时 override enum 对应 class; 需 import 顺序 + `get_class()` 校验 (`:417`) | 高, 需自写 backend 类 (PagedAttention on DCU FP8), 调试面积大 |

推荐顺序 **L1 > L2 > L3** (跟 ADR 0010 一致): P0-P2 走 L1 flag, P3 集成日视 aiter 实测结果决定是否改 L2 priority 锁死, stretch 走 L3 自写 decode kernel。

### B. DCU 性能特征 (CDNA2 vs CDNA3, 27B 视角)

| 维度 | CDNA2 (gfx90a, MI250X) | CDNA3 (gfx942, MI300X) | 27B 推理瓶颈影响 |
|------|------------------------|------------------------|------------------|
| HBM 带宽 (per GPU) | 1.6 TB/s (HBM2e) | 3.2 TB/s (HBM3) | 长上下文 decode 带宽 bound, CDNA3 翻倍 |
| FP8 FNUZ 算力 (dense) | 无原生 FP8 | ~2 PFLOPS | prefill 算力 bound 时 CDNA3 优势巨大 |
| HBM 容量 (per GPU) | 128 GB (MI250X) | 192 GB (MI300X) | 27B FP16 权重 ~54 GB 单卡装不下, FP8 ~27 GB 可装 |
| KV cache (32k ctx) | bf16 ≈ 28 GB | bf16 ≈ 28 GB | 占 HBM 14-22%, 量化收益直接转吞吐 |
| L2 cache | 8 MB (per GCD) | 4 MB (per XCD) | 27B attention L2 miss 高, CDNA3 miss 次数更敏感 |

瓶颈判别 (基于 roofline 估算): **prefill 算力 bound, decode 带宽 bound**。P2 末 profile (Task 2.3) 应验证此假设 — 若 decode 阶段 HBM 带宽利用率 < 60% 即未达带宽 bound, 需重审 KV layout/调度而非量化。优化方向: 量化降带宽 (ADR 0009), compile 降 launch (ADR 0013), block-size 降碎片 (ADR 0007/0008)。

### C. KV 量化方案对比 (Qwen3.5-27B × 4k-32k context)

| 方案 | 显存 (vs bf16) | 精度 Δ (LongBench/OC, 来源) | DCU 支持 | 工程成本 |
|------|----------------|------------------------------|----------|----------|
| bf16 (baseline) | 1.0× | 0% | 通用 | 0 (vLLM 默认) |
| INT8 per-tensor | 0.5× | Δ 1-3% (Dai 2023 LLM.int8()) | CDNA2+3, aiter 稳定 | 1-2h 接 `--kv-cache-dtype int8` |
| INT8 per-head | 0.5× | Δ < 1% (Cousum 2023) | CDNA2+3, 需 monkey-patch | 4-6h 改 `kv_cache_interface.py` |
| FP8 per-head (FNUZ) | 0.5× | Δ < 1.5% (Kwon 2024, vLLM FP8 报告) | CDNA3 限定, CDNA2 跳过 | 1-2h 接 `--kv-cache-dtype fp8` |
| FP8 per-token + 关键 cache FP16 (KIVI) | ~0.45× (算 scale 元数据) | Δ < 1% (Hooper 2024 KIVI 论文) | CDNA3 限定, 需 L2 路径 | 8-12h 自写 K/V 分组 |
| INT4 + outlier FP16 (KVQuant) | 0.27× | Δ 1-2.5% (Hooper 2024 KVQuant) | CDNA2+3, 精度风险最大 | 12-16h 复杂 outlier 检测 |

选型结论: 必做 **FP8 per-head (CDNA3) + INT8 per-head (CDNA2 退路)**, stretch 选 **KIVI 关键 cache FP16 兜底精度**。INT4 KVQuant 风险 > 收益, 27B 上不做 (跟 ADR 0009 结论一致)。所有 "Δ" 数字来源: KIVI 论文 arXiv:2402.02750 + vLLM FP8 KV cache PR #6978 (2024-Q3) 公开 benchmark — **未验, 待 P0 0.5 + P3 3B.1 on DCU 实测**。

## Consequence

- 三个块互相不冲突, 落地都在 P3 集成日: A 决定 backend 锁法 (L1 flag 写 Dockerfile 还是 L2 改镜像), B 决定优化重点 (带宽 vs 算力), C 决定 KV 量化默认 dtype/粒度
- B 块的瓶颈判别依赖 P2 末 profile 数据 (Task 2.3); 若实测与假设不一致, A/C 块决策不变 (backend 与量化选型对瓶颈类型不敏感)
- A 块的 L2 路径需要镜像外 patch (跟 ADR 0009 L2 路径共享同一 patch 机制), 工程上合成 1 个 entrypoint
- 6 方案中 KIVI 关键 cache FP16 + KVQuant outlier 都属 stretch, 队员 B 工时按 8-12h 估, 优先级在 P3 末根据精度 Δ 决定

## 验证清单 (P0-P3)

- [ ] P0 0.4 (Task 0.4): L1 flag `--attention-backend TRITON_ATTN` 端到端跑通 1 prompt on DCU
- [ ] P0 0.4: dry-run `--attention-backend ROCM_AITER_FA` 验证 aiter import 不挂
- [ ] P0 0.5 (Task 0.5): Triton FP8 store/load 跑通 on DCU, 验证 FP8 per-head scale kernel
- [ ] P2 2.3 (Task 2.3): profile 一次, 拿 prefill vs decode 时间占比 + HBM 带宽利用率, 验证 B 块瓶颈假设
- [ ] P3 3B.1-3 (Task 25 子): 必做 FP8 per-head + KIVI 关键 cache 跑 OpenCompass, Δ 实测
- [ ] P3 集成日: 锁 A 块 (L1 flag 写 Dockerfile) + C 块 (FP8 per-head 默认) 进镜像; 更新本 ADR 状态为 Accepted
- [ ] L3 自写 backend 仅在 aiter 三方对比 (TRITON/AITER_FA/AITER_UNIFIED, ADR 0010) 都不达标时启用
