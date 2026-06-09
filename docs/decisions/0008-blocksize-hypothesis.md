# ADR 0008: Block-size 假设 (P2 末 10-min 讨论结论)

**日期**: 2026-MM-DD (P2 末)
**状态**: 草稿 — 待 P3 末 集成日验证
**Owner**: 全员讨论 + 队长拍板
**关联**:
- spec §5.1 决策表 (block-size 行)
- ADR 0007 决策矩阵
- 计划 Task 2.2 (3 档 baseline) + Task 2.3 (profile)
- 计划 Task 3A.1 (block-size 扫描)

## 10-min 讨论输入

1. **ADR 0007 决策矩阵**: 5 个 block-size (8/16/32/64/128) × 4 个量化列
2. **P0 0.1 验证**: DCU SKU (gfx942 vs gfx90a) → 影响 FP8 可用
3. **P2.2 3 档 baseline**: 哪一档是瓶颈? (prefill-bound vs decode-bound)
4. **P2.3 profile**: KV cache 读/写占比 + HBM 带宽利用率

## 假设 (待验证)

> **预测**: `block-size 16` 或 `block-size 32` 最可能赢

### 理由

1. **8 太小**: metadata 开销占比 > 5% (spec §5.1 默认排除 < 16)
2. **128 太大**: 长 prompt (32k) 显存碎片率上升; 短 prompt (4k) block 浪费
3. **16 / 32 sweet spot**: 平衡碎片率 + metadata 开销; Qwen3.5-27B 在 4k-32k 上下文下都能 cover
4. **8k-16k 档 (50% 权重)**: 这一档的 prompt 长度 (12k) 平均分配到 16 / 32 block 上对齐好

### 反对意见 (待 P3 实测)

- 16 在 4k 短 prompt 上 block 浪费可能 > 32
- 32 在 32k 长 prompt 上碎片率可能 > 16

## 决策准则 (用于 P3 实测)

1. **3 档 6 次扫** (5 个 size × 3 档, 砍到 3-4 个高 ROI 后实测)
2. **首选 block-size 16** (跟 12k 平均 prompt 长度对齐)
3. **对比 32** (备选)
4. **最终选 P3 集成日数据最优的** (不是假设, 是实测)

## 关联到 P3

- Task 3A.1 block-size 扫描: 5 个 size, 50 prompts/档, 误差 < 5%
- Task 3A.4 lock best config: 根据扫描结果 + ADR 0007 决策准则
- 集成日: 3 必做项 + 选定 block-size

## 待办

- [ ] 队长组织 10 min 讨论, 填本 ADR 决策段
- [ ] 队员 B 跑 block-size 扫描 (Task 3A.1), 用本假设做 sanity check
- [ ] P3 末 集成日, 根据实测数据更新本 ADR 状态为 "已验证 / 已修正"
