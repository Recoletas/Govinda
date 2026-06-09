# Govinda

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）参赛项目
> **赛题**：基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化

## 项目目标

在国产 DCU 加速卡（单卡 → 决赛多卡）上，把 vLLM 0.18.1 + Qwen3.5-27B 的在线推理服务做长上下文场景下的 TTFT / TPOT / 吞吐优化。

- **初赛**：单卡 DCU，并发 = 1，输入长度分 4k-8k / 8k-16k / 16k-32k 三档（权重 20% / 50% / 30%）
- **决赛**：多卡分布式
- **核心指标**：输出吞吐量（Output Tokens/s），受 TTFT P99 / TPOT P99 SLA 约束（≤ Baseline × 1.5）

## 项目状态

> 注：本表区分 **plan-level 任务交付** (artifacts 落地) 与 **CP gate 状态** (模板是否就绪 + 实际是否签过)。CP0/1/2 不是"通过",是"模板就绪,签名 slot 留好"。

| 阶段 | 任务交付 (plan-level) | CP Gate 状态 |
|------|----------------------|--------------|
| P0 基础统一 | 5/7 交付 (0.4/0.5 DCU 阻塞) | **CP0 未签** ([模板](weekly/p0-offline-log.md)) |
| P1 基础培训 | 4/4 交付 | **CP1 未签** ([模板](weekly/p1-signoff.md), 双签 slot 留好) |
| P2 Baseline 锁定 | 2/4 交付 (2.2/2.3 DCU 阻塞) | **CP2 未签** ([模板](weekly/p2-signoff.md), 4 签 slot 留好) |
| P3 优化试错 | 3/9 交付 (3A/3B.2/3B.3/3C.2/3D.1 DCU 阻塞) | **CP3 未签** ([模板](weekly/p3-signoff.md)) |
| P4 集成精度 | 2/5 交付 (4.1/4.2/4.3 DCU 阻塞) | **CP4 未签** ([模板](weekly/p4-signoff.md)) |
| P5 提交冲刺 | 2/2 交付 (5.1 dry-run 脚本 + 5.2 log 模板) | **CP5 未签** ([模板](weekly/p5-submission-log.md), 实际提交事件在 P5 末) |

**关键阻塞**：LongBench / RULER 测试集需赛方确认（[ADR 0002](decisions/0002-testset-access.md)），3 档 baseline 跑分因此暂缓。

## 文档导航

- **设计**：完整设计文档（Context / 决策 / 边界 / 路线图 / 风险）
- **计划**：实施计划（22 个任务 / 6 个 phase / 9.5 周）
- **决策**：8 个 ADR，按编号顺序
- **录屏**：6 段 P0/P1 知识分享脚本
- **周报**：6 份 phase sign-off / log 模板
- **资源**：[AI 协作约定](AGENTS.md) / [AI Prompts 库](ai-prompts/README.md)

## 仓库布局

```
.
├── README.md              # GitHub 首页
├── AGENTS.md              # AI 助手协作说明
├── LICENSE                # MIT
├── mkdocs.yml             # 文档站配置
├── .github/workflows/     # GitHub Actions (docs 部署)
├── docs/                  # mkdocs 源
│   ├── index.md           # 文档站首页 (本文件)
│   ├── specs/             # 完整设计
│   ├── superpowers/plans/ # 实施计划
│   ├── decisions/         # ADR 决策记录
│   ├── recordings/        # 录屏脚本
│   ├── weekly/            # 周报 / sign-off
│   └── ai-prompts/        # 共享 prompt 库
├── src/                   # 优化代码 (kv_quant / compile)
├── benchmarks/            # 压测 harness + 原始数据
├── tests/                 # 单元测试
├── reports/               # 赛题提交材料 (env-vars / optimization-plan / submission-readme)
├── docker/                # Dockerfile + compose.yml
├── scripts/               # 验证脚本 (DCU / testset / dry-run)
└── qwen_use.pdf           # 赛题原文 (未入仓,赛方重新下载)
```

## 团队

4 人队伍（队长 + 3 队员），详见 [设计文档 §团队分工](specs/2026-06-09-qwen-inference-optimization-design.md)。

## License

MIT — 详见 [LICENSE](https://github.com/Recoletas/Govinda/blob/main/LICENSE)。
