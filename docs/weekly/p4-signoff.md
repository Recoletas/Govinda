# P4 CP4 Sign-off

**Phase**: P4 (集成 + 精度, 0.5 周)
**判定标准**: 全员 4 签 (队长 + 队员 A + B + C)
**截止**: 2026-MM-DD (P4 末)

## 3 档集成基准 (待 P4.1 跑分)

每档 100 prompts × 3 次取稳态:

| Tier | n | Throughput (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) | 来源 |
|------|---|--------------------|---------------|---------------|------|
| 4k-8k | 100 | 待填 | 待填 | 待填 | Task 4.1 跑分 |
| 8k-16k | 100 | 待填 | 待填 | 待填 | Task 4.1 跑分 |
| 16k-32k | 100 | 待填 | 待填 | 待填 | Task 4.1 跑分 |

数据源: `benchmarks/optimized/final-3tier/summary.md` (Task 4.1 输出) + `benchmarks/baseline/<tier>-<ts>.json`

**3 档权重** (per spec §9 / ADR 0007 #3): 4k-8k 20% / 8k-16k 50% / 16k-32k 30%
**SLA 不破** (per ADR 0007 #1): TTFT P99 ≤ Baseline × 1.5, TPOT P99 ≤ Baseline × 1.5 (任一破 = 0 分)

## 4 类任务精度验证 (待 P4.2 跑, per spec §11)

| 任务类型 | OpenCompass Δ vs baseline | 阈值 | 状态 |
|----------|---------------------------|------|------|
| QA | 待填 | ≤ 3% | 待 P4.2 跑 |
| 摘要 | 待填 | ≤ 3% | 待 P4.2 跑 |
| 检索 | 待填 | ≤ 3% | 待 P4.2 跑 |
| 聚合 | 待填 | ≤ 3% | 待 P4.2 跑 |

数据源: `reports/accuracy-validation.md` (Task 4.2 输出)
红线 (per spec §10 + ADR 0007 #2): Δ > 3% 触发回退

## 1 次干净全量编译演练 (待 P4.3 跑)

- 路径: `docs/weekly/p4-clean-rebuild-log.md` (Task 4.3 输出)
- 步骤: 删 Docker 镜像/容器/缓存 → `docker compose build` → 跑 1 个最小请求确认服务起来 → 记总耗时
- 状态: 待 P4.3 跑
- 风险对冲 (per spec §10 "编译失败撞 P5 截止" 行): 演练放在 P4 末而非 P5, 保留上次成功构建的 Docker 镜像

## 提交材料定稿 (Task 4.4 已完成)

| 赛题条款 | 提交材料 | Owner | 状态 |
|----------|----------|-------|------|
| §13 环境变量 | `reports/env-vars.md` | 队员 C | 已存在 |
| §14 优化方案 | `reports/optimization-plan.md` | 队长 | 已存在 |
| §15 第三方引用 | `reports/submission-readme.md` | 队长 | 已存在 (草稿, P5 定稿) |

## 是否进入 P5

进入 P5 前,逐条检查:

- [ ] 4 类任务精度 Δ ≤ 3% (per spec §9 CP4 硬卡门 + §10 "KV 量化精度塌方" 行)
- [ ] 3 档 baseline 误差 < 5% (per spec §9 CP2, P4 复用确认无偏移)
- [ ] 全员 4 签齐 (队长 + 队员 A + B + C)
- [ ] 1 次干净全量编译演练完成 (`docs/weekly/p4-clean-rebuild-log.md` 存在)
- [ ] 提交材料 3 份齐 (`reports/env-vars.md` / `optimization-plan.md` / `submission-readme.md`)

**任一精度 Δ > 3%** → spec §10 触发回退, KV 量化退回 bf16, 其他优化点保留, 重跑 P4.2。
**干净编译失败** → 重做 P4.3, 保留上次成功 Docker 镜像作 backup。
**4 签不齐** → 异步 ack 24h 内补签, 不阻塞 P5 启动。
**提交材料缺漏** → 缺哪份补哪份, 不延期。

## 全员 4 签

- [ ] 队长 (recoletas) 签
- [ ] 队员 A (Kernel) 签
- [ ] 队员 B (vLLM) 签
- [ ] 队员 C (浮动) 签

## 失败补救

- 4 类任务任一 Δ > 3%: 退回 KV 量化到 bf16, 其他优化点保留, 重跑 P4.2 验证
- 干净编译演练失败: 重做 P4.3, 用上次成功 Docker 镜像作 backup (per spec §10 "编译失败撞 P5 截止" 行)
- 4 签不齐: 异步 ack 24h 内补签, 不阻塞 P5 启动
- 提交材料缺漏: 缺哪份补哪份, P5 演练前必须 3 份齐
- SLA 任一档破线 (TTFT P99 / TPOT P99 > Baseline × 1.5): 立即回退当周优化点, 不抢进度 (per spec §10 + ADR 0007 #1)

## 关联文档

- spec §9 (P4 出口 CP4 硬卡门)
- spec §10 (精度塌方 / 编译失败 / SLA 破线 3 类风险)
- spec §11 (完工标准 + 4 类任务 + 3 份提交材料)
- Plan Task 4.1 (3 档集成基准) / 4.2 (精度验证) / 4.3 (干净编译演练) / 4.4 (提交材料) / 4.5 (本 sign-off)
- ADR 0007 (coupling matrix + 5 条决策准则)
- ADR 0015 (集成日最终 ROI 表, 待 P3D.1 创建) — 注: P4 仅引用路径, 实际 0015 由 P3D.1 任务产出
- `reports/env-vars.md` / `reports/optimization-plan.md` / `reports/submission-readme.md`
- `docs/weekly/p4-clean-rebuild-log.md` (待 P4.3 创建)
