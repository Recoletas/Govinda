# Govinda

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）参赛项目
> **赛题**：基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化

## 项目目标

在国产 DCU 加速卡（单卡 → 决赛多卡）上，把 vLLM 0.18.1 + Qwen3.5-27B 的在线推理服务做长上下文场景下的 TTFT / TPOT / 吞吐优化。

- **初赛**：单卡 DCU，并发 = 1，输入长度分 4k-8k / 8k-16k / 16k-32k 三档（权重 20% / 50% / 30%）
- **决赛**：多卡分布式
- **核心指标**：输出吞吐量（Output Tokens/s），受 TTFT P99 / TPOT P99 SLA 约束（≤ Baseline × 1.5）

## 项目状态

> 注：本表区分 **plan-level 任务交付** (artifacts 落地) 与 **CP gate 状态** (实际产出是否达标)。CP 通过靠 bench 跑分 / 精度 / 集成日 ROI, **不靠 4 签 checkbox** — sign-off 模板已删, 改用 [`learning.md`](learning.md) + [`weekly/progress.md`](weekly/progress.md)。

| 阶段 | 任务交付 (plan-level) | CP Gate 状态 (看实际产出) |
|------|----------------------|--------------------------|
| P0 基础统一 | 8/8 交付 (DCU SKU 已确认 gfx90a, 容器流程对齐官方调试文档) | 待 DCU 上跑通容器实例 + vllm 编译 + 模型 + 3 档 baseline (Task 0.6 Step 1-6) |
| P1 基础培训 | 4/4 交付 | 队员自学, 无 gate 形式 |
| P2 Baseline 锁定 | 1/4 交付 (Task 2.1 改用官方 `run_throughput.sh`) | 待 DCU 上 3 档 baseline 数字落地 (官方脚本直跑) |
| P3 优化试错 | 6/9 交付 (3A/3B.2/3B.3/3C.2/3D.1 DCU 阻塞) | 待 DCU 跑 3 必做项 + 集成日 |
| P4 集成精度 | 2/5 交付 (4.2/4.3 DCU 阻塞) | 待 DCU 跑集成 + 4 类精度 (官方 `run_accuracy.sh`) + 干净编译 |
| P5 提交冲刺 | 2/2 交付 (5.1 dry-run 脚本 + 5.2 提交) | 实际提交事件在 P5 末 |

**关键阻塞已解除 (2026-06-21)**:
- ✅ DCU SKU 已实测: Hygon DCU K100, gfx90a (CDNA2-class), DTK 26.04, [ADR 0001](decisions/0001-dcu-sku.md)
- ✅ 评测环境已对齐: SCNet 容器服务 + image `qwen3.5-dtk26.04:0509` + 官方 `testdata/{start_vllm,run_throughput,run_accuracy}.sh`, [ADR 0006b](decisions/0006b-container-instance.md)
- ✅ 测试集已落实: 3 档吞吐 (4-8K / 8-16K / 16-32K) + 4 类精度 (hotpotqa / gov_report / retrieval_multi_point / aggregation_keyword_aggregation)
- ⏳ 待验证: `vllm_cscc` 是否与 upstream vllm v0.18.1 一致 (Task 0.8)

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
