# ADR 0007: Block-size × KV 量化粒度耦合矩阵

**日期**: 2026-06-09
**状态**: 草稿 — 待 P0 验证 + P3 实测填充
**Owner**: 队长 (合) + 队员 B (review)
**关联**:
- spec §5.1 决策表 (block-size + KV 量化行)
- spec §5.4 (FP8 格式选择)
- Task 1.2 ROCm precision-support 总结 (见 `docs/recordings/p1-share-3-dcu-hip.md` 末尾)
- Plan Task 2.2 (3 档 baseline) + Task 3A.1 (block-size 扫描) + Task 3B.1-3 (KV 量化器)

## 背景

Block-size 调参与 KV 量化**不是独立优化**:
- 块管理的分配单位 (block_size tokens) 决定 1 个 block 内能放多少 K/V 值
- KV 量化按 per-head / per-token / per-tensor 计算 scale, scale 粒度跟 block 大小可能错位
- 量化精度 (FP8 FNUZ vs INT8) 跟 block 大小也有耦合 (FP8 在小 block 上 scale 抖动更大)

为避免 P3 期间在 5×4 = 20 个组合上**串行盲测**,先用决策矩阵定义:
1. 哪些组合**不应该测** (P0 0.1 + 0.4 验证后直接砍)
2. 哪些组合**必须测** (高 ROI 候选)
3. 评估指标 + 时机

## 决策矩阵 (5 × 4 = 20 cells, 初始全待测)

| block-size | FP8 per-head scale | FP8 per-token scale | INT8 per-tensor | bf16 (no quant) |
|------------|--------------------|--------------------|-----------------| -----------------|
| 8 | 待测 | 待测 | 待测 | baseline |
| 16 | 待测 | 待测 | 待测 | baseline |
| 32 | 待测 | 待测 | 待测 | baseline |
| 64 | 待测 | 待测 | 待测 | baseline |
| 128 | 待测 | 待测 | 待测 | baseline |

## 预筛 (P0 0.1 验证后填入)

| 决策点 | 当前假设 | 来源 |
|--------|----------|------|
| DCU SKU | 待 P0 0.1 验证 (gfx942 vs gfx90a) | Task 0.1 ADR 0001 |
| FP8 可用? | CDNA3 (gfx942) YES / CDNA2 (gfx90a) NO | spec §5.4 + Task 1.2 |
| Custom backend 路径? | 待 P0 0.4 验证 | Task 0.4 ADR 0004 |
| Triton FP8 已知坑? | 待 P0 0.5 验证 | Task 0.5 ADR 0005 |

基于预筛:
- **CDNA2 + FP8 列** → 整列删除,改测 INT8 only
- **CDNA3 + FP8 列** → 保留
- **per-token scale (FP8)** → 跟 per-block 分配耦合最紧,优先测
- **per-tensor scale (INT8)** → 跟 block-size 解耦,可作 baseline 对照

## 评估指标

| 指标 | 单位 | 工具 | 优先级 |
|------|------|------|--------|
| 显存占用 | GB | `torch.cuda.memory_allocated()` | 1 |
| TTFT P99 | ms | vllm bench serve | 1 |
| TPOT P99 | ms | vllm bench serve | 1 |
| Throughput | tok/s | vllm bench serve | 1 |
| OpenCompass Δ | % | OpenCompass | 2 (P3 末才用) |
| vLLM startup time | s | 启动 log | 3 |
| HBM 带宽利用率 | % | rocprofv3 | 3 (深度优化时) |

## 评估时机

| 阶段 | 测什么 | 谁 |
|------|--------|----|
| P2 末 | 3 档 baseline (bf16, no quant) | 队员 C |
| P3 中 | 5 × 4 矩阵 (DCU 验证后筛过的子集) | 队员 A + 队员 B |
| P3 末 | 集成日: 3 必做项 + 选定矩阵 cell | 全员 |

## 决策准则 (从 spec §11 评分公式 + §10 风险表派生)

1. SLA 不破: TTFT P99 ≤ Baseline × 1.5, TPOT P99 ≤ Baseline × 1.5 (任一破 = 0 分)
2. 精度不塌: OpenCompass Δ ≤ 3% (Δ > 3% 触发回退)
3. 优先 8k-16k 档 (占 50% 权重), 其次 4k-8k (20%) + 16k-32k (30%)
4. 块大小 < 16 在 27B 上 metadata 开销 > 收益, 默认排除
5. per-token scale FP8 在 batch=1 + 长 decode 场景 scale 抖动大, 优先 per-head

## 待办

- [ ] 队长 + 队员 B 联合 review 评估指标 + 时机 (plan Step 2)
- [ ] P0 0.1 验证后, 更新"预筛"段的 SKU 行
- [ ] P0 0.4 + 0.5 验证后, 更新"预筛"段的 backend + Triton 行
- [ ] P3 中填入实测数据 (5×4 子集, 砍过的列)
- [ ] P3 末产出"最终 ROI 表" (1 个 block-size + 1 个量化粒度组合, 走集成日)
