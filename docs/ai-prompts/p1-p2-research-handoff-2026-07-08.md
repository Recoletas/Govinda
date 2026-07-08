# P1/P2 Research Handoff Prompts 2026-07-08

Use these prompts for teammate or subaccount Claude windows. The goal is
research and design first, not direct submission. All workers must read:

- `AGENTS.md`
- `docs/decisions/0015-optimization-ledger-2026-07-08.md`
- `docs/decisions/0016-next-optimization-roadmap-2026-07-08.md`
- `docs/decisions/0013-competition-rules-interpretation.md`

Do not let any worker push or submit without owner review.

## Shared Ground Rules

- Safe scoring line: `vllm_cscc_work` commit
  `2c73297 Capture exact Qwen decode graph sizes`.
- Do not use the dirty AITER default-on diff.
- Do not change scheduler locked behavior:
  `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`.
- Do not change sampling, truncation, prompt filtering, output semantics,
  model structure, or weights.
- Do not use FP8 weight quantization; it already caused catastrophic accuracy
  penalty.
- Do not use direct KV FP8 flag as a scoring shortcut.
- Any code change must stay behind an env gate until smoke proves it.

## P1 Prompt: Decode GEMV / Fusion Research

```text
你负责 P1：decode skinny GEMV / fusion 调研。只读和设计为主，除非我明确让你改代码。项目根目录是 /home/recoletas/Govinda，评分代码树是 /home/recoletas/Govinda/vllm_cscc_work。安全线是 vllm_cscc_work commit 2c73297，不要基于脏 AITER 默认开启 diff 做结论。

必须先读：
1. AGENTS.md 的赛题硬约束
2. docs/decisions/0015-optimization-ledger-2026-07-08.md
3. docs/decisions/0016-next-optimization-roadmap-2026-07-08.md
4. docs/大模型decode访存瓶颈与双缓冲_DCU实测(1)(1).html
5. docs/第三集_带宽利用与算子优化(1).html

背景：
- 当前安全成绩约 79.11：4K-8K=15.97，8K-16K=14.44，16K-32K=11.04，SLA/精度罚 0。
- decode 已有正向来自 gfx936 LLMM1 skinny GEMV，尤其 gate_up/mid_up。
- 已证伪：简单启用 LLMM1 到 down/full_qkv/full_o/lm_head 是负优化或无收益。
- 当前脏 diff 中有 LLMM1Silu 实验，但端到端未证明，不能默认启用。

你的任务：
1. 确认 Qwen3.5 实际 dense MLP 路径：
   - qwen3_5.py 如何调用 Qwen2MoeMLP / Qwen3NextMLP
   - gate_up_proj/down_proj 的实际 weight shape
   - full_attention 层和 linear_attention 层是否共享同一个 MLP shape
2. 评估 LLMM1Silu 是否值得重写：
   - 当前硬编码 shape `(34816,5120)` 是否匹配实际模型
   - 如果不匹配，列出应支持的真实 `(2*intermediate, hidden)` shape
   - 说明需要改哪些文件：csrc/rocm/skinny_gemms.cu、csrc/rocm/ops.h、csrc/rocm/torch_bindings.cpp、vllm/_custom_ops.py、vllm/model_executor/models/qwen2_moe.py
3. 设计 microbench：
   - 基于 tools/codex_llmm1_microbench.py 和 tools/codex_llmm1_shape_sweep.py
   - 对比 torch linear + SiluAndMul vs LLMM1 + SiluAndMul vs LLMM1Silu
   - 必须报告 maxdiff/meandiff，不只看速度
4. 给出是否接入端到端的判据：
   - microbench 至少有稳定正向
   - 只在 `VLLM_GFX936_FUSED_GATE_UP_SILU=1` 下启用
   - 先 4K-8K smoke，再 8K-16K，再 16K-32K

输出格式：
- 结论：可行 / 不可行 / 需要容器验证
- 关键文件和函数
- 真实 shape 表
- microbench 命令和预期输出字段
- 风险清单：精度、SLA、启动、编译、提交污染
- 不要直接提交，不要推 Git。
```

## P2 Prompt: Runtime INT8 KV Cache Research

```text
你负责 P2：runtime INT8 KV cache 可行性调研。只做调研和设计，不要提交实现。项目根目录是 /home/recoletas/Govinda，评分代码树是 /home/recoletas/Govinda/vllm_cscc_work。安全线是 vllm_cscc_work commit 2c73297。

必须先读：
1. AGENTS.md 的赛题硬约束
2. docs/decisions/0009-kv-quant-strategy.md
3. docs/decisions/0012-vllm-cscc-vs-upstream.md
4. docs/decisions/0013-competition-rules-interpretation.md
5. docs/decisions/0015-optimization-ledger-2026-07-08.md
6. docs/decisions/0016-next-optimization-roadmap-2026-07-08.md

硬约束：
- 允许 runtime KV cache 动态量化。
- 禁止权重持久化量化、权重重排、模型格式转换、生成可复用量化权重或缓存文件。
- 禁止改 scheduler 锁参。
- 禁止通过截断/过滤/跳层/跳 token 改语义。

背景：
- 直接 FP8 KV flag 曾失败：TRITON attention 报 `A non 1.0 q_scale is not currently supported`。
- FP8 权重路径吞吐高但精度扣爆，禁止继续。
- ADR 0012 已记录：当前海光 vLLM 没有现成 INT8 KV cache 实现；已有 INT8 ops 多是 activation/communication，不是 KV cache。

你的任务：
1. 读代码确认 KV 写入路径：
   - csrc/cache_kernels.cu
   - csrc/cache_kernels_fused.cu
   - csrc/cache.h
   - vllm/v1/attention/backends/rocm_attn.py
   - vllm/v1/kv_cache_interface.py
2. 读代码确认 KV 读取 / attention 路径：
   - csrc/rocm/attention.cu
   - vllm/v1/attention/ops/chunked_prefill_paged_decode.py
   - vllm/v1/attention/ops/triton_unified_attention.py
   - vllm/v1/attention/backends/triton_attn.py
   - vllm/v1/attention/backends/rocm_attn.py
3. 判断最小可行 INT8 KV 设计：
   - cache storage dtype 怎么表示
   - per-head scale 还是 per-block scale
   - scale tensor 放在哪里，不持久化
   - 写 cache 时 quantize
   - attention 内部 fused dequant，避免单独 dequant kernel
4. 明确需要改的 enum/dispatch：
   - `kv_cache_dtype` 现在哪些值能通过
   - C++/HIP dispatch 是否需要新增 int8 case
   - Python backend 是否现在只认 `startswith("fp8")`
5. 给出最小 correctness 测试：
   - 随机小 tensor reshape_and_cache int8 写入再 dequant 对比 bf16
   - 单层 attention 输出误差对比
   - 端到端只做 smoke，不直接提交

输出格式：
- 可行性等级：短期可做 / 中期可做 / 暂不做
- 必改文件列表
- 数据结构和 scale 粒度建议
- correctness 测试设计
- 性能预期：4K-8K / 8K-16K / 16K-32K 哪档可能收益
- 规则风险自查
- 不要直接提交，不要推 Git。
```

## Owner Review Checklist

When a teammate reports back, check:

- Did they start from safe line `2c73297`?
- Did they avoid AITER default-on dirty diff?
- Did they avoid scheduler and sampling changes?
- Did they provide exact file/function paths?
- Did they distinguish microbench speed from end-to-end score?
- Did they include correctness/error checks, not only throughput?
