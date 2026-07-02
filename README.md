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

> **2026-07-01**: 官方 baseline 已 Accepted — 4-8K=12.95 / 8-16K=10.03 / 16-32K=5.75 tok/s, 得分 **59.9119** (rank 56/76), SLA=0, 精度=0. 详细见 [ADR 0003](docs/decisions/0003-baseline-source.md).

> **剩余 14 天** (截止 2026-07-15). 优先级: (1) [block_size sweep](docs/decisions/0008-blocksize-hypothesis.md) (2) INT8 KV smoke → 三档 [ADR 0009](docs/decisions/0009-kv-quant-strategy.md) (3) 启动侧大改暂不动.

**P0-P5 历史阶段表已废弃** (见 git log 2026-06-21 之前的提交). 当前操作指南看 [`AGENTS.md` 当前阶段](AGENTS.md) + [HANDOVER.md](../HANDOVER.md) (本地, 不入仓) + [weekly/progress.md](docs/weekly/progress.md).

## 团队

4 人队伍，详见 [设计文档 §团队分工](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。

## 第三方引用

本项目使用 vLLM（Apache-2.0）、PyTorch、Transformers、Qwen3.5-27B（Apache-2.0 权重） 等开源项目，按赛方要求在 `docs/specs/` 和 `reports/` 标注。

## License

MIT — 详见 [LICENSE](LICENSE)。
