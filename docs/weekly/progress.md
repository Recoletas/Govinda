# 进度 (Weekly Standup)

> 轻量进度交流模板。每人每周 1 行追加即可, **不是** sign-off, 不需要 4 签。
> 删掉了之前的 P1-P5 sign-off 模板 + P5 dry-run/submission log (2026-06-09 session 决策)。

## Week of 2026-06-09

### 队长 (recoletas)
- 本周做了什么: spec/plan 文档精修 + 切到学习资料索引 + 写 mkdocs left_toc plugin (50 行) + 补 Owner×Phase + RACI 矩阵
- 阻塞: DCU 硬件未到 (P0 0.1/0.4/0.5 等)
- 下周计划: 跑 §2 4 项验证 (DCU 到位后), 给赛方发测试集访问问询

### 队员 A (Kernel)
- 本周做了什么: (待填)
- 阻塞: (待填)
- 下周计划: (待填)

### 队长 (recoletas)
- 本周做了什么: 读 SCNet 官方调试文档, 把 P0/P2/P4 全部对齐官方流程 (容器服务 + qwen3.5-dtk26.04:0509 + vllm_cscc + ModelScope + 官方 3 脚本). 写 ADR 0001 (DCU 实测: Hygon K100, gfx90a, DTK 26.04) + ADR 0006b (Accepted). 重写 Plan §0.6 (web console 流程) / 新增 §0.8 (vllm_cscc vs upstream diff) / 改 §2.1 §4.2 用官方 run_*.sh
- 阻塞: web shell 2h 超时, 但容器实例在后台持续跑; 只要不点"停止容器", `~/` 下数据不丢
- 下周计划: 容器内跑通 Step 2-6 (vllm 编译 + 模型 + testdata + start_vllm.sh + curl smoke), 跑 baseline 50 prompts/档

### 队员 B (vLLM)
- 本周做了什么: 写 ADR 0009 (KV 量化策略) + ADR 0010 (attention backend 选型) + 实读 vLLM 0.18.1 源码确认 24 enum
- 阻塞: (待填)
- 下周计划: (待填)

### 队员 C (浮动 / QA)
- 本周做了什么: (待填)
- 阻塞: (待填)
- 下周计划: (待填)

### P0 末 CPU 调研输出 (2026-06-09 一次性记录)
- ADR 0009: KV 量化 — FP8 E4M3 FNUZ (CDNA3) + INT8 per-head (CDNA2 退路) + KIVI 关键 cache 留 FP16
- ADR 0010: Attention backend — 默认 `TRITON_ATTN`, P3 集成日视 aiter 实测切
- ADR 0011: P0 末调研 — vLLM 接入点 3 层 (flag / 改 priority / register) + DCU 性能特征 (CDNA2 1.6 TB/s vs CDNA3 3.2 TB/s) + 6 方案 KV 量化横评
- learning.md 新增 2 个 section: "KV 量化" + "Attention backend 选型 (vLLM 0.18.1)"
- src/ 新增 4 个模块 + tests/ 4 个测试 (全部 skip, 等 DCU 验证后开):
  - `src/kv_quant/base.py` Quantizer ABC + `int8_quant.py` INT8PerHeadQuantizer
  - `src/block_size/sweep.py` BlockSizeSweep harness
  - `src/attn_backend/triton_decode.py` TritonDecodeAttention 注册入口
- benchmarks/ 新增 `compare.py` (baseline vs optimized ROI) + `run_bench_3tier.sh` (3 档 × 3 iter bash 入口)
- mkdocs: 删顶部 tabs, 加 left_toc plugin, 单页 H2 全部进 left nav
- 测试: 35 collected, 8 passed, 27 skipped (DCU 验证后开)

## Week of YYYY-MM-DD

### 队长 (recoletas)
- 本周做了什么:
  -
- 阻塞:
  -
- 下周计划:
  -

### 队员 A (Kernel)
- 本周做了什么:
  -
- 阻塞:
  -
- 下周计划:
  -

### 队员 B (vLLM)
- 本周做了什么:
  -
- 阻塞:
  -
- 下周计划:
  -

### 队员 C (浮动 / QA)
- 本周做了什么:
  -
- 阻塞:
  -
- 下周计划:
  -

<!-- 上一周的 progress 折叠到下方, 不要删, git log 找 -->
