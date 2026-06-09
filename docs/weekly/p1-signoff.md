# P1 CP1 Sign-off

**Phase**: P1 (基础统一, 1.5 周)
**判定标准**: 队长 + 队员 B 双签
**截止**: 2026-MM-DD (P1 末)

## 4 人 quiz 通过名单

| 名字 | 角色 | Quiz 分数 | 通过? | 日期 |
|------|------|-----------|-------|------|
| recoletas | 队长 + Profiling owner | 待填 | ☐ | |
| 队员 A | Kernel owner | 待填 | ☐ | |
| 队员 B | vLLM owner | 待填 | ☐ | |
| 队员 C | 浮动 / QA | 待填 | ☐ | |

通过标准: ≥ 80% (8/10)

## 3 个录屏落 `docs/recordings/`

| # | 主题 | 录制人 | 脚本路径 | 录制状态 | 录屏文件 |
|---|------|--------|----------|----------|----------|
| 1 | Prefill/Decode/KV cache 基础 | 队员 B | `p1-share-1-prefill-decode.md` | 脚本就绪, 待录制 | 待补 |
| 2 | vLLM 0.18.1 架构 | 队长 | `p1-share-2-paged-attention.md` | 脚本就绪, 待录制 | 待补 |
| 3 | DCU/HIP 培训 | 队员 A | `p1-share-3-dcu-hip.md` | 脚本就绪, 待录制 | 待补 |

## ADR 0007 编译决策矩阵

- 路径: `docs/decisions/0007-coupling-matrix.md`
- 状态: 草稿 (P3 中段填充 5×4 实测)
- 决策准则 5 条已立, 待 P0 0.1/0.4/0.5 验证后更新预筛段

5 条决策准则 (来自 `docs/decisions/0007-coupling-matrix.md` §决策准则):

1. SLA 不破: TTFT P99 ≤ Baseline × 1.5, TPOT P99 ≤ Baseline × 1.5 (任一破 = 0 分)
2. 精度不塌: OpenCompass Δ ≤ 3% (Δ > 3% 触发回退)
3. 优先 8k-16k 档 (占 50% 权重), 其次 4k-8k (20%) + 16k-32k (30%)
4. 块大小 < 16 在 27B 上 metadata 开销 > 收益, 默认排除
5. per-token scale FP8 在 batch=1 + 长 decode 场景 scale 抖动大, 优先 per-head

## 双签

- [ ] 队长 (recoletas) 签
- [ ] 队员 B 签

## 失败补救

- Quiz 不通过: 重听 + 重考, 不延期
- 录屏未完成: P1 不开 P2, 延期由全组决定
- ADR 0007 未 review: 队长 + 队员 B 当面过 1 次, 1h 内完成

## 关联文档

- spec §9 (P1 出口 CP1)
- spec §10 (Phase 跳过规则)
- Plan Task 1.1, 1.2, 1.3
- 5 个 ADRs (0001, 0002, 0003, 0006a, 0007)
- 5 个 P0 任务交付 (0001, 0002, 0003, 0006a, 0007)
