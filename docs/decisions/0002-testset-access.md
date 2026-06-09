# ADR 0002: 测试集访问验证结果

**日期**: 2026-06-09
**状态**: 阻塞 — 需询问赛方 / 走 fallback
**执行人**: AI agent (team lead 角色)
**关联脚本**: `scripts/verify_testset_access.py`
**原始结果**: `benchmarks/testset_access.json`

## 背景

P0 任务 0.2 要求在跑任何 baseline / 优化前先确认 LongBench 和 RULER 测试集能不能下载。脚本运行后,两个测试集都 FAIL。这是真实信号,不是脚本问题。

## 验证结果(实际跑出)

| 测试集 | 检查项 | 状态 | 原因 |
|--------|--------|------|------|
| LongBench | `xinrongzhang2022/longbench` (NarrativeQA) | **FAIL** | `DatasetNotFoundError`: dataset gated/private,无法无 auth 访问 |
| LongBench | `xinrongzhang2022/longbench` (Qasper) | **FAIL** | 同上 |
| RULER | `https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic.json` | **FAIL** | HTTP 404: 该路径不存在,实际数据由 `scripts/data/synthetic/json/` 下的脚本生成(SQuAD / HotpotQA / Paul Graham essays),不存预置 JSON |

### 失败原因分析

1. **LongBench gated**: 原始仓库 `xinrongzhang2022/longbench` 在 HuggingFace Hub 上已变为 gated / private。即使是 `streaming=True` 也需要登录。`datasets-server.huggingface.co/info?dataset=xinrongzhang2022%2Flongbench` 返回 `401` 等价错误。
2. **RULER 数据结构不同**: 计划里写的 `scripts/data/synthetic.json` 不存在。RULER 的"基准数据"是合成的 — 评测脚本运行时调用 `download_paulgraham_essay.py` 和 `download_qa_dataset.sh` 从 SQuAD / HotpotQA / Paul Graham Blog 拉源数据,生成 NIAH / VT / CWE / QA / RAG 任务的 jsonl。所以"预下载 synthetic.json"这个概念本身在 RULER 里不成立。
3. **`datasets` 库未预装**(环境中性发现): 干净环境需 `pip install datasets` 才能跑 HF 流式访问。赛方评测容器需确认镜像里是否包含。

## 决策

- **不**修改脚本结构(脚本设计目的就是检测这些失败)。
- **不**提交未在赛方/官方授权下镜像的测试集数据。
- **走 fallback**: 自造 100 样本 smoke 测试集,用于 P2 阶段的 bench harness 自检。
- **真实评测集**: 走赛方流程(邮件/IM 群),问三件事:
  1. LongBench 是否由赛方提供镜像/快照?是否需要专用 HF token?
  2. RULER 评测是赛方统一跑还是我们自己跑?如果是后者,需要确认 SQuAD / HotpotQA / Paul Graham 源数据是否允许现场拉取。
  3. 评测容器镜像里预装了什么 Python 包(`datasets` / `transformers` / `vllm`)?

## Fallback 策略: 自造 100 样本

如果 24h 内未得到赛方回复或赛方不提供测试集:

1. **构造脚本**: `scripts/make_synth_bench.py`
   - 从 Wikipedia 公开页面或 CC0 文本(如 `wikimedia/wikipedia` 摘要)抽 100 个 long-context 段(每段 8k-32k tokens)。
   - 每段配 1-2 个 QA 对,答案严格在段内可定位(避免幻觉测量失真)。
2. **目的**: bench harness 自检 + 调优循环 + 回归对比(参照 AGENTS.md §AI 输出验证协议 §2 "运行验证")。
3. **限制**: 不可作为最终提交 score 的依据,只用于研发期。
4. **重做条件**: 赛方提供官方测试集后,改用官方集重跑 P2 §2.2 三档 baseline,自造集归档到 `benchmarks/synth_100/` 留作回归基线。

## 影响

- **P2 任务 2.1 (bench harness TDD)** 不会被卡 — harness 只依赖 OpenAI 兼容接口,数据来源可注入。
- **P2 任务 2.2 (3 档 baseline 跑分)** **会被卡** 直到测试集到位 — 用自造 100 样本先出 1 档 rough baseline。
- **P3+ 任务 (优化)** 不依赖此任务,继续推进。
- **Spec 引用**: 此 ADR 替代了 spec 中"测试集来源"假设,后续 PR 需在 spec 增补一段。

## 待办

- [ ] 队长发邮件/IM 给赛方,引用本 ADR + `benchmarks/testset_access.json`
- [ ] 收到回复后更新本 ADR 的"决策"段
- [ ] 若 24h 无回复,启动 fallback: 写 `scripts/make_synth_bench.py`
