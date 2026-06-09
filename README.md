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

当前阶段：**基础周 A**（CP1 待达成）。详见 [设计文档 §路线图](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。

## 团队

4 人队伍，详见 [设计文档 §团队分工](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。

## 第三方引用

本项目使用 vLLM（Apache-2.0）、PyTorch、Transformers、Qwen3.5-27B（Apache-2.0 权重） 等开源项目，按赛方要求在 `docs/specs/` 和 `reports/` 标注。

## License

MIT — 详见 [LICENSE](LICENSE)。
