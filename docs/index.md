# Govinda

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）参赛项目
> **赛题**：基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化

## 项目目标

在国产 DCU 加速卡（单卡 → 决赛多卡）上，把 vLLM 0.18.1 + Qwen3.5-27B 的在线推理服务做长上下文场景下的 TTFT / TPOT / 吞吐优化。

- **初赛**：单卡 DCU，并发 = 1，输入长度分 4k-8k / 8k-16k / 16k-32k 三档（权重 20% / 50% / 30%）
- **决赛**：多卡分布式
- **核心指标**：输出吞吐量（Output Tokens/s），受 TTFT P99 / TPOT P99 SLA 约束（≤ Baseline × 1.5）

## 项目状态

> 注：本表区分 **plan-level 任务交付** (artifacts 落地) 与 **CP gate 状态** (实际产出是否达标)。CP 通过靠 bench 跑分 / 精度 / 集成日 ROI, **不靠 4 签 checkbox** (sign-off 模板已删, 见 [Changelog](../AGENTS.md#changelog))。

| 阶段 | 任务交付 (plan-level) | CP Gate 状态 (看实际产出) |
|------|----------------------|--------------------------|
| P0 基础统一 | 5/7 交付 (0.4/0.5 DCU 阻塞) | 待 DCU 跑 P0 0.1 verify_dcu + 0.4 vLLM backend smoke |
| P1 基础培训 | 4/4 交付 | 队员自学, 无 gate 形式 |
| P2 Baseline 锁定 | 2/4 交付 (2.2/2.3 DCU 阻塞) | 待 DCU 跑 3 档 baseline |
| P3 优化试错 | 3/9 交付 (3A/3B.2/3B.3/3C.2/3D.1 DCU 阻塞) | 待 DCU 跑 3 必做项 + 集成日 |
| P4 集成精度 | 2/5 交付 (4.1/4.2/4.3 DCU 阻塞) | 待 DCU 跑集成 + 4 类精度 + 干净编译 |
| P5 提交冲刺 | 2/2 交付 (5.1 dry-run 脚本 + 5.2 log 模板) | 实际提交事件在 P5 末 |

**关键阻塞**：LongBench / RULER 测试集需赛方确认（[ADR 0002](decisions/0002-testset-access.md)），3 档 baseline 跑分因此暂缓。

**进度同步**: 看 [weekly/progress.md](weekly/progress.md) — 4 人每周 1 行 standup (本周 / 阻塞 / 下周)。

## 文档导航

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
