# P3 CP3 Sign-off

**Phase**: P3 (优化试错, 3.5 周)
**判定标准**: 队长单签 + 全员 ack (队员 A / B / C)
**截止**: 2026-MM-DD (P3 末)

## 3 必做项各自的 ROI 文档

每 stream 子任务的 ROI 文档,各自归档到 `docs/decisions/`:

| Stream | 必做项 | ROI 文档路径 | 状态 |
|--------|--------|--------------|------|
| **A: 块管理** | `--block-size` 扫描 | `docs/decisions/0009-block-size-ROI.md` | 待 P3A.Y 创建 |
| **A: 块管理** | prefix-caching | `docs/decisions/0010-prefix-caching-ROI.md` | 待 P3A.Y 创建 |
| **A: 块管理** | chunked-prefill | `docs/decisions/0011-chunked-prefill-ROI.md` | 待 P3A.Y 创建 |
| **B: KV 量化** | FP8 (CDNA3) / INT8 fallback (CDNA2) | `docs/decisions/0012-fp8-kv-ROI.md` (或 `0012a-int8-kv-ROI.md`) | 待 P3B.Y 创建 |
| **C: torch.compile** | `default` mode + cudagraph | `docs/decisions/0013-torch-compile-roi.md` | 已存在, 待 P3C.Y 填充实测 |
| **C: torch.compile** | cudagraph (独立 ROI) | `docs/decisions/0014-cudagraph-ROI.md` | 待 P3C.Y 创建 |

**3 档权重** (per spec §9 / ADR 0007 #3): 4k-8k 20% / 8k-16k 50% / 16k-32k 30%
**SLA 不破** (per ADR 0007 #1): TTFT P99 ≤ Baseline × 1.5, TPOT P99 ≤ Baseline × 1.5 (任一破 = 0 分)

## 集成日最终 ROI 表 (Task 3D.1)

- 路径: `docs/decisions/0015-integration-final.md` (待 P3D.1 创建)
- 内容: 3 档 × 3 必做 = 9 格的 throughput / TTFT P99 / TPOT P99 提升数据
- 数据源: `benchmarks/optimized/integration-<date>/` + `benchmarks/baseline/<tier>-<ts>.json`

## 是否进入 P4 / 是否触发 spec §10 应急

进入 P4 前,逐条检查:

- [ ] 必做 3 项 (Stream A / B / C) 各自至少 1 个 ROI 文档存在且数据齐全
- [ ] 集成日 3 档都达到 ≥ 10% 提升 (相对 baseline 锁定数)
- [ ] 3 项叠加无互相抵消 (Stream A + B + C 集成后, 3 档不出现负收益)
- [ ] 集成日最终 ROI 表 (`0015-integration-final.md`) 写完
- [ ] OpenCompass Δ ≤ 3% (精度未塌)

**任一必做项未达 ≥ 10% 提升** → spec §10 触发"砍必做到 2 项"流程,降级到 2 项推进。
**集成日 3 项叠加互相抵消** → 退回选 2 项叠加,删除第 3 项。
**P3 任何 1 周内 0 bench 进展** → spec §10 立刻开"砍必做到 2 项"应急。

## 全员 4 签

- [ ] 队长 (recoletas) 单签
- [ ] 队员 A (Kernel) ack
- [ ] 队员 B (vLLM) ack
- [ ] 队员 C (浮动) ack

## 失败补救

- 任一必做项未达 ≥ 10% 提升: 写明哪项没达, spec §10 触发"砍必做到 2 项"流程
- 集成日冲突 (3 项叠加互相抵消): 退回选 2 项叠加, 删 1 项
- 队长 + 队员 C 缺席: 异步 ack 24h 内补签, 不阻塞 P4 启动
- OpenCompass Δ > 3%: 退回 KV 量化到 bf16, 其他优化点保留
- ROI 文档未到 24h 未审: revert (per spec §10 "AI 代码未及时审阅合 main" 行)
- DCU 是 CDNA2 (gfx90a) 无原生 FP8: Stream B 改 INT8 fallback, 走 `0012a-int8-kv-ROI.md`

## 关联文档

- spec §9 (P3 出口 CP3)
- spec §10 (Phase 跳过 / 应急规则)
- Plan Task 3A / 3B / 3C / 3D (4 个 stream 全集)
- ADR 0007 (coupling matrix + 5 条决策准则)
- ADR 0008 (block-size 假设)
- ADR 0009 / 0010 / 0011 (Stream A 3 个 ROI, 待 P3A.Y 创建)
- ADR 0012 / 0012a (Stream B ROI, 待 P3B.Y 创建)
- ADR 0013 (Stream C torch.compile ROI, 已存在)
- ADR 0014 (Stream C cudagraph ROI, 待 P3C.Y 创建)
- ADR 0015 (集成日最终 ROI 表, 待 P3D.1 创建)
