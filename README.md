# Govinda

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）参赛项目
> 赛题：基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化

[![Docs](https://img.shields.io/badge/docs-recoletas.github.io%2FGovinda-blue)](https://recoletas.github.io/Govinda/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Competition](https://img.shields.io/badge/赛题-智能计算创新设计赛-orange)](docs/specs/2026-06-09-qwen-inference-optimization-design.md)

## 是什么

在国产 DCU 加速卡（单卡 → 决赛多卡）上，把 vLLM 0.18.1 + Qwen3.5-27B 的在线推理服务做长上下文场景下的 TTFT / TPOT / 吞吐优化。

## 怎么开始

- 文档站：<https://recoletas.github.io/Govinda/>
- 完整设计：[`docs/specs/2026-06-09-qwen-inference-optimization-design.md`](docs/specs/2026-06-09-qwen-inference-optimization-design.md)
- 赛题原始材料：见 `qwen_use.pdf`（未入仓，从 [比赛网站](https://pra.xtnl.org.cn/) 重新下载）

## 项目状态

> 注：本表区分 **plan-level 任务交付** (artifacts 落地) 与 **CP gate 状态** (实际产出是否达标)。CP 通过靠 bench 跑分 / 精度 / 集成日 ROI, **不靠 4 签 checkbox** — sign-off 模板已删, 改用 [`docs/learning.md`](docs/learning.md) + [`docs/weekly/progress.md`](docs/weekly/progress.md)。

| 阶段 | 任务交付 (plan-level) | CP Gate 状态 (看实际产出) |
|------|----------------------|--------------------------|
| P0 基础统一 | 5/7 交付 (0.4/0.5 DCU 阻塞) | 待 DCU 跑 P0 0.1 verify_dcu + 0.4 vLLM backend smoke |
| P1 基础培训 | 4/4 交付 | 队员自学, 无 gate 形式 |
| P2 Baseline 锁定 | 2/4 交付 (2.2/2.3 DCU 阻塞) | 待 DCU 跑 3 档 baseline |
| P3 优化试错 | 3/9 交付 (3A/3B.2/3B.3/3C.2/3D.1 DCU 阻塞) | 待 DCU 跑 3 必做项 + 集成日 |
| P4 集成精度 | 2/5 交付 (4.1/4.2/4.3 DCU 阻塞) | 待 DCU 跑集成 + 4 类精度 + 干净编译 |
| P5 提交冲刺 | 2/2 交付 (5.1 dry-run 脚本 + 5.2 提交) | 实际提交事件在 P5 末 |

详细交付清单 + 阻塞点：见 [设计文档 §路线图](docs/specs/2026-06-09-qwen-inference-optimization-design.md) 和 [实施计划](docs/superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md)。

**关键阻塞**：LongBench / RULER 测试集需赛方确认（[ADR 0002](docs/decisions/0002-testset-access.md)），3 档 baseline 跑分因此暂缓。

**进度同步**: [docs/weekly/progress.md](docs/weekly/progress.md) — 4 人每周 standup。

## 团队

4 人队伍，详见 [设计文档 §团队分工](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。

## 第三方引用

本项目使用 vLLM（Apache-2.0）、PyTorch、Transformers、Qwen3.5-27B（Apache-2.0 权重） 等开源项目，按赛方要求在 `docs/specs/` 和 `reports/` 标注。

## License

MIT — 详见 [LICENSE](LICENSE)。
