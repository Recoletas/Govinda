# Govinda

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）参赛项目
> **赛题**：基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化

## 项目目标

在国产 DCU 加速卡（单卡 → 决赛多卡）上，把 vLLM 0.18.1 + Qwen3.5-27B 的在线推理服务做长上下文场景下的 TTFT / TPOT / 吞吐优化。

- **初赛**：单卡 DCU，并发 = 1，输入长度分 4k-8k / 8k-16k / 16k-32k 三档（权重 20% / 50% / 30%）
- **决赛**：多卡分布式
- **核心指标**：输出吞吐量（Output Tokens/s），受 TTFT P99 / TPOT P99 SLA 约束（≤ Baseline × 1.5）

## 项目状态

> **2026-07-01**: 官方 baseline 已 Accepted — 4-8K=12.95 / 8-16K=10.03 / 16-32K=5.75 tok/s, 得分 **59.9119** (rank 56/76), SLA=0, 精度=0. 详细见 [ADR 0003](decisions/0003-baseline-source.md).

> **剩余 14 天** (截止 2026-07-15). 优先级: (1) block_size sweep [ADR 0008](decisions/0008-blocksize-hypothesis.md) (2) INT8 KV smoke → 三档 [ADR 0009](decisions/0009-kv-quant-strategy.md) (3) 启动侧大改暂不动.

**历史 P0-P5 阶段表已废弃** (见 git log 2026-06-21 之前的提交). 现在只剩"已 AC baseline" + "14 天优化窗口" 两状态.

## 文档导航

- **队员入门**: [team-onboarding.md](team-onboarding.md) — 当前状态 / 连接 / 分工 / 14 天路线 (新人必读)
- **学习**: [learning.md](learning.md) — vLLM / AMD ROCm / DCU 关键文件 + grep 技巧 + 术语速查
- **设计**: 完整设计文档（Context / 决策 / 边界 / 路线图 / 风险）
- **计划**: 实施计划（22 个任务 / 6 个 phase / 9.5 周）
- **决策**: 8 个 ADR，按编号顺序
- **进度**: [weekly/progress.md](weekly/progress.md) — 4 人每周 standup 模板
- **资源**: [AI 协作约定](AGENTS.md) / [AI Prompts 库](ai-prompts/README.md)

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
│   ├── learning.md        # 找学习资料的方法
│   ├── specs/             # 完整设计
│   ├── superpowers/plans/ # 实施计划
│   ├── decisions/         # ADR 决策记录
│   ├── weekly/progress.md # 轻量 standup 模板
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
