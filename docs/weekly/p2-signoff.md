# P2 CP2 Sign-off

**Phase**: P2 (Baseline 锁定 + 调研, 1.5 周)
**判定标准**: 全员 4 签 (队长 + 队员 A + B + C)
**截止**: 2026-MM-DD (P2 末)

## 3 档 baseline 数字 (误差 < 5%)

每档 100 prompts × 3 次取稳态:

| Tier | n | Throughput (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) | 来源 |
|------|---|--------------------|---------------|---------------|------|
| 4k-8k | 100 | 待填 | 待填 | 待填 | Task 2.2 跑分 |
| 8k-16k | 100 | 待填 | 待填 | 待填 | Task 2.2 跑分 |
| 16k-32k | 100 | 待填 | 待填 | 待填 | Task 2.2 跑分 |

数据源: `benchmarks/baseline/<tier>-<ts>.json` + `benchmarks/baseline/summary.md`

## 1 份 profile 报告

- 路径: `benchmarks/profiles/baseline-profile.md`
- 内容: top-3 decode kernel 时间占比 / top-3 prefill kernel / HBM 带宽利用率 / KV cache 读/写占比
- 工具: torch.profiler + rocprofv3 (P0 0.6 Docker 镜像里)

## ADR 0008: block-size 假设 (10-min 讨论结论)

- 路径: `docs/decisions/0008-blocksize-hypothesis.md`
- 10 min 讨论输入: ADR 0007 矩阵 + P0 0.1 验证结果 + profile 报告
- 输出: 哪个 block-size 最可能赢 + 理由 + 待 P3 实测验证

## 全员 4 签

- [ ] 队长 (recoletas) 签
- [ ] 队员 A (Kernel) 签
- [ ] 队员 B (vLLM) 签
- [ ] 队员 C (浮动) 签

## 失败补救

- Baseline 数字误差 > 5%: 重跑, 不延期
- Profile 工具未装 (rocprofiler): 用 torch.profiler 单工具, P4 末再补
- ADR 0008 讨论超时: 队长决定, 全员 ack

## 关联文档

- spec §9 (P2 出口 CP2)
- spec §10 风险表 baseline 行
- Plan Task 2.1 (bench harness) + 2.2 (baseline) + 2.3 (profile)
- ADR 0001-0007 (前序 6 个 ADR 决策)
