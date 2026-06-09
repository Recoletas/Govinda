# ADR 0013: torch.compile + cudagraph ROI 分析

**日期**: 2026-MM-DD (P3 中段)
**状态**: 草稿 — 待 P3 实测填充
**Owner**: 队员 A (Kernel) + 队长 (合)
**关联**:
- spec §5.1 (torch.compile mode 决策 + 图优化边界)
- Task 2.1 (bench harness) + Task 2.2 (3 档 baseline) + Task 3A.1 (block-size 扫描)
- Plan Stream C 全部

## 背景

torch.compile + cudagraph 是必做 2 项,目标:
- 减少 Python 层调度开销 → TTFT 改善
- 录制 kernel launch 序列 → TPOT 改善 (decode 阶段)
- 不动模型权重/结构 → 合规 (赛题 §3.3.(2) "图重构" 限定为权重层级)

## 实验设计

| 实验 | mode | use_cudagraph | enforce_eager | 备注 |
|------|------|---------------|---------------|------|
| baseline (Task 2.2) | (vLLM 默认) | (默认) | (默认) | 对照组 |
| torch.compile default | default | True | False | 必做 2 主实验 |
| torch.compile + cudagraph | reduce-overhead | True | False | 高 ROI 备选 |
| cudagraph only | (eager) | True | False | 对照 #2 |
| max-autotune (排除) | max-autotune | True | False | spec §5.1 排除, 不测 |

每实验 × 3 档 (4k-8k / 8k-16k / 16k-32k) × 50 prompts

## 待 P3 实测填入的指标

| 指标 | 4k-8k | 8k-16k | 16k-32k | 来源 |
|------|-------|--------|---------|------|
| Throughput (tok/s) | 待测 | 待测 | 待测 | benchmarks/optimized/torch-compile-default/ |
| TTFT P99 (ms) | 待测 | 待测 | 待测 | 同上 |
| TPOT P99 (ms) | 待测 | 待测 | 待测 | 同上 |
| 启动时间 (s) | 待测 | 待测 | 待测 | serve log |
| Warmup 耗时 (s) | 待测 | 待测 | 待测 | serve log |
| OpenCompass Δ | 待测 | 待测 | 待测 | P3 末 |

## 决策准则 (P3 实测后填)

1. SLA 不破 (TTFT P99 ≤ Baseline × 1.5, TPOT P99 ≤ Baseline × 1.5)
2. 精度 Δ ≤ 3%
3. 8k-16k 档至少 5% 提速才值得集成
4. 启动时间增加 < 60s 才不阻塞 bench 循环

## 待办

- [ ] 队员 A 跑实验 #1 (default + cudagraph)
- [ ] 队员 A 跑实验 #2 (reduce-overhead + cudagraph)
- [ ] 队员 A 跑实验 #3 (cudagraph only)
- [ ] 队长分析 3 实验的指标对比
- [ ] P3 末 集成日, 选定最终配置, 锁进 src/compile/config.py 默认值
- [ ] 更新本 ADR 状态为 "已决策" + 记录最终配置
