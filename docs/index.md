# Govinda

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）参赛项目
> **赛题**：基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化

## 项目目标

在国产 DCU 加速卡（单卡 → 决赛多卡）上，把 vLLM 0.18.1 + Qwen3.5-27B 的在线推理服务做长上下文场景下的 TTFT / TPOT / 吞吐优化。

- **初赛**：单卡 DCU，并发 = 1，输入长度分 4k-8k / 8k-16k / 16k-32k 三档（权重 20% / 50% / 30%）
- **决赛**：多卡分布式
- **核心指标**：输出吞吐量（Output Tokens/s），受 TTFT P99 / TPOT P99 SLA 约束（≤ Baseline × 1.5）

## 文档入口

- [完整设计文档](specs/2026-06-09-qwen-inference-optimization-design.md) — Context / 决策 / 边界 / 路线图 / 风险

## 仓库布局（计划）

```
.
├── README.md              # 本文件 GitHub 首页
├── AGENTS.md              # AI 助手协作说明
├── LICENSE                # MIT
├── mkdocs.yml             # 文档站配置
├── docs/                  # mkdocs 源
│   ├── index.md           # 文档站首页
│   └── specs/             # 完整设计 / 实施计划
├── src/                   # 代码（待建）
├── benchmarks/            # 压测原始数据（待建）
├── tests/                 # 单元测试（待建）
└── reports/               # 赛题提交材料（待建）
```

## 团队

4 人队伍（队长 + 3 队员），详见 [设计文档 §团队分工](specs/2026-06-09-qwen-inference-optimization-design.md)。

## License

MIT — 详见 [LICENSE](https://github.com/Recoletas/Govinda/blob/main/LICENSE)。
