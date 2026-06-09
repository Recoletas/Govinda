# AGENTS.md — AI 助手协作说明

> 本文件给所有 AI 会话（Claude / Cursor / Copilot / subagent 等）阅读，建立项目级共识。

## 项目是什么

在国产 DCU 加速卡上优化 Qwen3.5-27B 推理服务（vLLM 0.18.1）。完整背景见 [`docs/specs/2026-06-09-qwen-inference-optimization-design.md`](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。

## 赛题硬约束（**不可越界**，AI 给建议前必须自查）

来自 [赛题原文](https://pra.xtnl.org.cn/) 第 7 条：

- ❌ **禁止投机解码**（draft model / MTP / 多头预测 / 外挂小模型 / 自训练预测器 / early-exit draft / 预生成 token 缓存）
- ❌ **禁止持久化量化、结构化剪枝、权重重排、模型图重构、模型格式转换**（包括"权重加载前后"与服务初始化阶段）
- ❌ **禁止任何剪枝**（结构化 / 非结构化 / 动态通道跳过 / 动态层跳过 / 注意力头裁剪 / token pruning / early-exit）
- ❌ **禁止修改 batch scheduler 相关代码和参数**（`--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` 全部锁定）
- ❌ **禁止预缓存测试集与答案、预生成中间结果**
- ❌ **禁止训练/微调/蒸馏/后训练**
- ❌ **禁止绕开统一服务接口、评测流程、资源统计路径**
- ❌ **禁止引入题目规定外的辅助模型**（含外挂小模型投机采样）
- ✅ **允许**：KV cache 动态量化、activation 动态量化、kernel 内低精度、自定义 Python 包与 custom kernel（需在评测容器内可编译）

**SLA 硬约束**（赛题第 9 条第 5 款）：TTFT P99、TPOT P99 任一超 Baseline × 1.5，该档吞吐量得分直接清零。

**精度硬约束**（赛题第 9 条第 6 款）：Δ > 10% → 该类任务不计分（系数 = 0）。

## 当前阶段

参见 [`docs/specs/2026-06-09-qwen-inference-optimization-design.md` §阶段与检查点](docs/specs/2026-06-09-qwen-inference-optimization-design.md)。每次开始会话前先 read 一下。

## AI 使用约定

### 用 AI 做的事
- 解释概念、写脚手架代码、读 vLLM 源码、生成文档初稿、整理调研笔记、写测试
- 提示：项目已装 `superpowers:dispatching-parallel-agents` 可并行调研；`context7` 插件可查 vLLM / PyTorch / Triton 最新 API

### 不让 AI 做的事
- **不**让 AI 决定方案（决策看 spec）
- **不**让 AI 写 custom kernel 不验证就合入
- **不**让 AI 读 PDF 得出"赛题允许 X"——必须查赛题原文（[`qwen_use.pdf`](qwen_use.pdf) 或赛方链接）

### AI 输出验证协议（强制）
所有 AI 生成的代码 / 文档，必须经过 3 道关：

1. **可读性核对** — 人读一遍，确认逻辑符合预期
2. **运行验证** — 跑一个最小用例
3. **回归对比** — 和 baseline 跑同一 bench，确认变化方向对

任何"AI 写 → 跑 → 加速了"的现象都要警惕**测错 / 跳过步骤**。

### 标注
AI 写 ≥ 50 行的代码必须在文件头加注释：
```python
# AI-generated, awaiting verification by <name> on <YYYY-MM-DD>
```
精确含义：代码由 AI 起草并**尚未**经团队成员验证。AI 输出验证协议（下方）通过后，改为 `verified by <name> on <YYYY-MM-DD>`。不要省略 "awaiting"，避免给后来者制造"已验证"的错觉。

### 共享 prompt 库
[`docs/ai-prompts/`](docs/ai-prompts/README.md) 存好用过的 prompt，避免重复造轮子。

## 必读 vLLM 0.18.1 文件清单

（按重要性排序，先读前 3 个）

| 优先级 | 路径 | 作用 |
|--------|------|------|
| ★★★ | `vllm/attention/backends/` | 各种 attention backend 注册；自定义 backend 入口 |
| ★★★ | `vllm/v1/kv_cache_interface.py` | KV cache 块管理 |
| ★★ | `vllm/v1/worker/model_runner.py` | 模型前向主循环 |
| ★★ | `vllm/v1/core/sched/` | 调度器（**只读，不改**） |
| ★ | `vllm/entrypoints/openai/serving_chat.py` | 服务入口 |

## 团队 & 角色

见 spec §团队分工。**联系队长（项目所有者）确认**任何涉及赛题边界、AI 工具栈变更、跨 owner 协作的决策。

## 反馈循环

- 任何新发现（vLLM 行为、DCU 限制、最佳实践） → 写进 [`docs/decisions/`](docs/decisions/) 或 [本文件末尾](#changelog)
- spec 与实际不符 → 在 PR 里指出并提出修改
- AI 给的建议违反赛题边界 → 在 PR / Issue 里 flag，立即停止

## Changelog

<!-- 每次会话后追加一行：日期 / 主要发现 / 决策 -->

- **2026-06-09 (本次会话)**：交付 16 个 CPU 可做 plan-level 任务的 artifacts (代码 + 文档模板 + ADR + 脚本),但 **CP0/1/2/3/4/5 gate 全部未签** — 模板就绪,签名 slot 留好,需队长+队员按 weekly/<p>-signoff.md 流程过。新增 8 份文档(3 份 reports / 5 份 weekly+sign-off / dry-run 脚本 / 提交 log 模板)。修复 ADR 0002 / 0006a / 0006 命名冲突,确认 vLLM 0.18.1 attention backend 真实机制,验证 DCU FP8 矩阵。**关键阻塞**: 测试集访问 (ADR 0002) 待赛方确认。文档站: 新增 GitHub Pages Actions (`.github/workflows/docs.yml`),扩展 mkdocs nav 到 7 个 section,新建 `docs/ai-prompts/`。
- **2026-06-09 (v4 spec)**: 3 轮 subagent 复核 + PDF 关键页重读,移除自相矛盾的"❌ 自定义 kernel",锁定 9 周预算 + 24 个 AttentionBackendEnum 真实位置。**注**: spec 是设计文档,不是状态报告。
- **2026-06-09 (v1 plan)**: 22 任务 / 136 子步 / 6 phase / 9.5 周,带 TDD skip 标记的 DCU 阻塞任务清单。**注**: plan 是路线图,不是已完成清单。
