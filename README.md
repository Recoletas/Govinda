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
| P0 基础统一 | 8/8 交付 (DCU SKU 已确认 gfx90a, 容器流程对齐官方调试文档) | 待 DCU 上跑通容器实例 + vllm 编译 + 模型 + 3 档 baseline (Task 0.6 Step 1-6) |
| P1 基础培训 | 4/4 交付 | 队员自学, 无 gate 形式 |
| P2 Baseline 锁定 | 1/4 交付 (Task 2.1 改用官方 `run_throughput.sh`) | 待 DCU 上 3 档 baseline 数字落地 (官方脚本直跑) |
| P3 优化试错 | 6/9 交付 (3A/3B.2/3B.3/3C.2/3D.1 DCU 阻塞) | 待 DCU 跑 3 必做项 + 集成日 |
| P4 集成精度 | 2/5 交付 (4.2/4.3 DCU 阻塞) | 待 DCU 跑集成 + 4 类精度 (官方 `run_accuracy.sh`) + 干净编译 |
| P5 提交冲刺 | 2/2 交付 (5.1 dry-run 脚本 + 5.2 提交) | 实际提交事件在 P5 末 |

详细交付清单 + 阻塞点：见 [设计文档 §路线图](docs/specs/2026-06-09-qwen-inference-optimization-design.md) 和 [实施计划](docs/superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md)。

**关键阻塞已解除 (2026-06-21)**:
- ✅ DCU SKU 已实测: Hygon DCU K100, gfx90a (CDNA2-class), DTK 26.04, [ADR 0001](docs/decisions/0001-dcu-sku.md)
- ✅ 评测环境已对齐: SCNet 容器服务 + image `qwen3.5-dtk26.04:0509` + 官方 `testdata/{start_vllm,run_throughput,run_accuracy}.sh`, [ADR 0006b](docs/decisions/0006b-container-instance.md)
- ✅ 测试集已落实: 3 档吞吐 (4-8K / 8-16K / 16-32K) + 4 类精度 (hotpotqa / gov_report / retrieval_multi_point / aggregation_keyword_aggregation)
- ⏳ 待验证: `vllm_cscc` 是否与 upstream vllm v0.18.1 一致 (Task 0.8)

**进度同步**: [docs/weekly/progress.md](docs/weekly/progress.md) — 4 人每周 standup。

## 团队

4 人队伍，详见 [设计文档 §团队分工](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。

## 第三方引用

本项目使用 vLLM（Apache-2.0）、PyTorch、Transformers、Qwen3.5-27B（Apache-2.0 权重） 等开源项目，按赛方要求在 `docs/specs/` 和 `reports/` 标注。

## License

MIT — 详见 [LICENSE](LICENSE)。
