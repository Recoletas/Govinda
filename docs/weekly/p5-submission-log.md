# P5 Submission Log

**Phase**: P5 (演练 + 提交)
**提交日期**: 待填
**提交人**: 队长 (recoletas)
**提交渠道**: 待填 (赛方 portal / 邮件 / 其他)
**提交总状态**: 待填 (成功 / 失败 / 退回)

> 本文档是提交日的操作 log, 不是 sign-off。预提交 checklist 通过后, 队长按 spec §9 CP5 出口双签 (队长 + 队员 C) 收尾。

## 1. 预提交 checklist

全部 5 条必须为 ✅, 队长方可提交:

- [ ] **P1 CP1 双签齐** — `docs/weekly/p1-signoff.md` 队长 + 队员 B 签 (per spec §9 CP1)
- [ ] **P2 CP2 双签齐** — `docs/weekly/p2-signoff.md` 队长 + 队员 A 签 (per spec §9 CP2)
- [ ] **P3 CP3 双签齐** — `docs/weekly/p3-signoff.md` 队长 + 队员 C 签 (per spec §9 CP3)
- [ ] **P4 CP4 全员 4 签齐** — `docs/weekly/p4-signoff.md` 队长 + A + B + C 签 (per spec §9 CP4)
- [ ] **P5 演练成功** — `docs/weekly/p5-dryrun-log.md` step 1-5 全 PASS, Shutdown 正常 (per spec §9 CP5 + §11 "1 次干净的全量编译 + 跑通演练")

## 2. 提交材料清单 (per spec §11 §12-15)

| 赛题条款 | 文件路径 | 状态 | 备注 |
|----------|----------|------|------|
| §12 源码 | `src/` (完整代码 + 编译脚本 + 注释) | 待填 | 队长合 + 全员贡献, P5 末 |
| §12 源码 README | `src/README.md` 头部 | 待填 | 队长, P5 末 |
| §13 环境变量 | `reports/env-vars.md` | 已存在 | 队员 C, P4 末定稿 |
| §14 优化方案 | `reports/optimization-plan.md` | 已存在 | 队长, P4 末定稿 |
| §15 第三方引用 + 编译步骤 | `reports/submission-readme.md` | 已存在 (草稿, P5 定稿) | 队长, P5 末定稿 |
| §15 主 README | `README.md` | 待填 | 引用 + 编译入口 |

**打包结构** (待填): 压缩包名 / 目录树 / 大小 / 校验和 (sha256)

## 3. 第三方引用

→ 见 `reports/submission-readme.md` §1 第三方引用清单 (软件库 + 数据集)

**额外声明** (如有, 待填): 比赛期间新引入的库 / 数据集 / 工具, 不在 `submission-readme.md` §1 列表内, 需在此处追加并回写 `submission-readme.md`。

## 4. 提交流程时间线

| # | 阶段 | 起始时间 | 结束时间 | 状态 | 操作 / 备注 |
|---|------|----------|----------|------|-------------|
| 1 | 准备材料 | 待填 | 待填 | 待填 | 打包 §12-15 全部, 复查 README 路径 |
| 2 | 打包 | 待填 | 待填 | 待填 | 压缩包 / 上传目录生成, 记 sha256 |
| 3 | 上传 | 待填 | 待填 | 待填 | 渠道: 待填, 进度监控 |
| 4 | 提交 | 待填 | 待填 | 待填 | 提交按钮 / 发送邮件, 记渠道响应 |
| 5 | 确认 | 待填 | 待填 | 待填 | 赛方 portal 状态 / 邮件回执 |
| 6 | 通知全队 | 待填 | 待填 | 待填 | 微信群 / GitHub Issue 通知, 附 log 链接 |

**总耗时**: 待填 (从准备材料到通知全队)

## 5. 提交确认 (附件证据)

- **时间戳**: 待填 (ISO 8601)
- **提交渠道**: 待填 (portal URL / 邮件地址 / 其他)
- **截图路径**: 待填 (附件落 `docs/weekly/p5-submission-attachments/`, 含 portal 状态页 / 邮件回执 / 上传进度)
- **提交编号 / 工单号**: 待填 (如赛方有回执 ID)
- **提交包 sha256**: 待填

## 6. 队长 + 队员 C 双签 (per spec §9 CP5)

- [ ] 队长 (recoletas) 签 — 日期: 待填
- [ ] 队员 C 签 — 日期: 待填

**双签特殊性**: P5 出口 CP5 仅需 队长 + 队员 C (per spec §9 Phase 表), 不需全员 4 签 (全员 4 签是 P4 CP4 的要求, 见 `p4-signoff.md`)。

## 7. 提交后行动

| # | 行动 | 时间点 | 状态 | 备注 |
|---|------|--------|------|------|
| 1 | 等待赛方 ack | 待填 | 待填 | portal 状态轮询 / 邮件 |
| 2 | 答疑群监控 | 待填 | 待填 | 赛方答疑群活跃时段, 4h 内响应 |
| 3 | 复赛通知 | 待填 | 待填 | 复赛时间 / 晋级名单公布 |
| 4 | 决赛时间 (若晋级) | 待填 | 待填 | 决赛时间 / 地点 / 形式 |

**赛方联系方式** (待填): 答疑群 / 邮箱 / 客服电话
**内部通知渠道** (待填): 微信群 / 邮件列表 / GitHub Issue

## 8. 失败补救

- **提交失败 (网络 / 渠道异常)**: 重试 3 次 (间隔 5min), 仍失败则切 backup 渠道 (邮件 / 备用 portal), 同步全队
- **漏材料**: 立即补交 (port 补传 / 邮件附件), 同步在 §2 状态列回写 "补交于 HH:MM"
- **队长缺席**: 队员 C 代理提交 + 异步 ack (per spec §9 CP5 双签授权), 队长 24h 内回 ack 邮件确认
- **赛方退回 (材料不完整 / 格式错)**: 按退回意见改材料, 重新走 §4 时间线 1-6 步, 更新本文档 §2 状态 + §5 时间戳
- **sha256 不一致 / 打包损坏**: 重新打包, 重算 sha256, 再走上传 + 确认

## 9. 关联文档

- spec §9 (P5 出口 CP5: 队长 + 队员 C 双签)
- spec §11 (完工标准 + 赛题 §12-15 提交材料清单)
- spec §10 (风险表: 编译撞 P5 / SLA 破线 / 精度塌方回退)
- plan Task 5.1 (演练: `docs/weekly/p5-dryrun-log.md` + `scripts/dry_run.py`)
- plan Task 5.2 (本任务: 提交)
- `reports/submission-readme.md` (第三方引用 + 编译步骤)
- `reports/env-vars.md` (赛题 §13)
- `reports/optimization-plan.md` (赛题 §14)
- `docs/weekly/p1-signoff.md` ~ `p4-signoff.md` (CP1-4 签收)
- ADR 0001 (DCU SKU) / 0002 (测试集) / 0003 (baseline) / 0006a (Docker) / 0007 (coupling matrix)
