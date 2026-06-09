# ADR 0009: KV 量化策略

**日期**: 2026-06-09
**状态**: Proposed — 待 P0 0.5 (Triton DCU FP8 跑通) + P2 0.4 (baseline 数字) 验证后改 Accepted
**Owner**: 队员 B (vLLM) 主笔 + 队员 A (Kernel) review + 队长拍板
**关联**:
- AGENTS.md 第 9/21 行: 允许 KV cache 动态量化 + activation 动态量化; 禁持久化量化
- spec §5.1 决策表 (block-size × KV 量化行) + §5.4 (FP8 格式选择)
- ADR 0007 决策矩阵 (5×4 cells) + ADR 0008 block-size 假设
- Plan Stream B (Task 3B.1-3 KV 量化器)

## Context

block-size × 量化粒度 × 数据格式三轴耦合, 20 个组合不可能全测; 需要先拍板:
1. **数据格式**: FP8 FNUZ (CDNA3 限定) / INT8 (CDNA2 + CDNA3 都可) / bf16 (baseline)
2. **粒度**: per-tensor / per-head / per-token / KIVI 2D / KVQuant outlier-aware
3. **vLLM 接入点**: 内置 `--kv-cache-dtype` flag vs 自定义 KV cache hook
4. **DCU 特有坑**: FNUZ vs OCP 不兼容; CDNA2 无原生 FP8; aiter FP8 路径稳定性

9 周预算 + 4 人 5-12h/周, 选"必做 3 + stretch 2" 而不是"全 FP8 重训练"路线。

## Decision

### 必做 (集成日必上)

| 项 | 选择 | 理由 |
|----|------|------|
| 数据格式 (CDNA3) | **FP8 E4M3 FNUZ** (`__hip_fp8_e4m3_fnuz`) | spec §5.4 已验; 比 INT8 动态范围宽, K/V 异常值容得下 |
| 数据格式 (CDNA2) | **INT8 对称 per-head** | CDNA2 无原生 FP8, INT8 是 aiter/FlashAttention-2 fork 都支持的稳定路径 |
| 粒度 (prefill 写) | **per-token scale** | prefill 阶段计算一次 scale, 摊薄开销 |
| 粒度 (decode 读) | **per-head scale** | decode 阶段每 token 算 scale 收益小开销大, per-head 跟 PagedAttention block 头维对齐 |
| vLLM 接入 | **`--kv-cache-dtype fp8` (内置) 优先**; 不行再 monkey-patch `vllm/v1/kv_cache_interface.py` 的 `KVCacheBlock` 写路径 | spec §5.1 决策 + AGENTS.md 边界 (不动 scheduler, 允许 hook) |
| 精度退路 | 关键 cache (首 256 token 或 system prompt) **留 FP16**, KIVI 风格 2D 分组 | 关键 cache 占总 token < 10%, 留高精度是 Δ 兜底 |

### Stretch (做出来加分, 不做不丢分)

1. **aiter FP8 KV 路径**: `VLLM_ROCM_USE_AITER=1` 启 aiter FA + aiter quantize kernel
2. **KVQuant 风格 outlier-aware**: per-channel 阈值, 异常通道 FP16, 其余 INT4

### CDNA2 vs CDNA3 分支

| 维度 | CDNA2 (gfx90a) | CDNA3 (gfx942) |
|------|----------------|----------------|
| FP8 KV | 不支持, 跳过 | FNUZ E4M3, 必做 |
| INT8 KV | **必做** (退路) | 备选 (FP8 失败时降级) |
| aiter FP8 | 不可用 | stretch |
| block-size 配合 | 16 (per-head scale 1 个 head ≈ 128B 量化参数) | 16 或 32 (per-head scale 摊到 32 token) |

### vLLM 0.18.1 接入点

- **L1 (首选)**: `vllm serve --kv-cache-dtype fp8 ...` (vLLM 0.18.1 已支持 `auto`/`fp8`/`int8`)
- **L2 (回退)**: monkey-patch `vllm.v1.kv_cache_interface.KVCacheTensor` 的 `copy_from_blocks` / `copy_to_blocks`, 写自定义 quant/dequant hook, scale 存 per-block metadata
- **L3 (stretch)**: `VLLM_ROCM_USE_AITER=1` + aiter 自带 FP8 KV cache kernel

### 不做什么 (显式排除)

- 全 FP8 重训练 / QLoRA 风格预量化 — 赛题禁止持久化量化
- per-tensor scale — 精度损失太大, 27B 模型 outlier 集中在少数 head/channel
- INT4 KV — 27B 上精度风险 > 收益, 留作 stretch 都不必做

## Consequence

| 维度 | 影响 | 缓解 |
|------|------|------|
| 精度 Δ | FP8 per-head 在 LongBench 上历史报告 Δ < 1.5%, KIVI 关键 cache 留 FP16 进一步压到 < 1% | OpenCompass Δ > 3% 触发回退到 INT8 |
| 显存收益 | FP8 比 bf16 节省 ~50% KV cache 显存; 27B + 32k context 下 HBM 占用降 ~6 GB | 留 1 GB buffer 给 quantization scale 元数据 |
| 解码速度 | FP8 dequant 加 1 个 kernel; per-head scale 在 decode 阶段不增加 per-token 开销 | P3 末 bench 验证 SLA 不破 (TTFT/TPOT P99 ≤ Baseline × 1.5) |
| DCU 风险 | FNUZ 与 OCP 不兼容 — 上游 PyTorch / vLLM 默认 OCP 路径会 crash 或静默错 | 启动时 `assert torch.float8_e4m3fnuz == __hip_fp8_e4m3_fnuz`; P0 0.5 最小 case 跑通 |
| 工程成本 | 队员 B vLLM patch 估 8-12h (P3 Stream B); L1 flag 路线只需 1-2h 验通 | 优先 L1, L2/L3 仅在 L1 失败时启用 |

## 验证清单 (P0-P2 期间)

- [ ] P0 0.5: 最小 Triton FP8 store + load 跑通 on DCU (gfx942)
- [ ] P2 2.2: bf16 baseline 数字到位 (3 档 50 prompts/档)
- [ ] P3 3B.1: `--kv-cache-dtype fp8` 端到端跑通 on DCU, OpenCompass Δ < 3%
- [ ] P3 3B.2: per-head scale 实测 8k-16k 档 SLA 不破
- [ ] P3 3B.3: 关键 cache FP16 (KIVI 风格) Δ 改善验证
- [ ] P3 集成日: 锁进 src/kv_quant/config.py 默认值
