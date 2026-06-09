# ADR 0001: DCU SKU 验证结果

**日期**: 2026-MM-DD (待 DCU 上跑后填入)
**状态**: 待确认

## 验证结果

- 设备名: 待 DCU
- GCN arch: 待 DCU (gfx942 或 gfx90a)
- FP8 支持: 待 DCU (FNUZ / OCP / NONE)
- 显存: 待 DCU (GB)

> 上述字段在硬件到位后由 `python3 scripts/verify_dcu.py` 输出回填。脚本结果同时写入 `benchmarks/device_info.json`。

## 决策

- 若为 CDNA3 (gfx942) → KV 量化走 FP8 FNUZ 路线
- 若为 CDNA2 (gfx90a) → KV 量化改 INT8 动态量化 / 保留 bf16

## 影响

- 影响 §5.1 决策表 (CP1 sign-off 之后) 中 KV 量化策略行的具体取值
- 影响 Stream B (P3) 任务 #25 选用的量化内核: FNUZ / OCP FP8 / INT8 / bf16
- 影响 Triton DCU FP8 验证 (P0 任务 #12) 的目标 dtype 路径
