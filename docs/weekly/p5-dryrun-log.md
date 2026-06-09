# P5 Dry Run Log

<!-- AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD> -->

**Phase**: P5 (演练 + 提交)
**演练人**: 待填
**演练日期**: 待填
**总状态**: 待填
**总耗时**: 待填 (秒)
**主机**: 待填
**Tiers**: 4k-8k, 8k-16k, 16k-32k
**Skip accuracy**: 否

> 此模板由 `scripts/dry_run.py` 自动生成。队员 C 在 DCU 上实跑后, 用 `python scripts/dry_run.py --output docs/weekly/p5-dryrun-log.md` 重新生成正文; 现场若有意外, 再手填下面的待填字段 + 补附件。

## Step 1: clean_build

- 状态: 待填
- 起止时间: 待填 → 待填
- 耗时: 待填 (s)
- 错误: 待填
- 输出: `benchmarks/dryrun/logs/step1_clean_build.log`
- 备注: 待填

## Step 2: start_service

- 状态: 待填
- 起止时间: 待填 → 待填
- 耗时: 待填 (s)
- 错误: 待填
- 输出: `benchmarks/dryrun/logs/step2_start_service.log`
- 备注: 待填 (例如: vllm 启动用了 N 分钟, GPU0 HBM 余量 X GiB)

## Step 3: dcu_and_testset_verify

- 状态: 待填
- 起止时间: 待填 → 待填
- 耗时: 待填 (s)
- 错误: 待填
- 输出: `benchmarks/dryrun/logs/step3_verify.log`
- 备注: 待填 (DCU SKU = ?, FP8 = ?, testset LongBench / RULER 状态 = ?)

## Step 4: bench_3tier

- 状态: 待填
- 起止时间: 待填 → 待填
- 耗时: 待填 (s)
- 错误: 待填
- 输出: `benchmarks/dryrun/logs/step4_bench_3tier.log` + `benchmarks/dryrun/<tier>/*.json`
- 备注: 待填 (3 档 throughput / TTFT / TPOT 摘要)

## Step 5: accuracy_validation

- 状态: 待填
- 起止时间: 待填 → 待填
- 耗时: 待填 (s)
- 错误: 待填
- 输出: `benchmarks/dryrun/logs/step5_accuracy.log`
- 备注: 待填 (4 类任务: QA / 摘要 / 检索 / 聚合, 各任务相对 baseline Δ = ?; 任一 Δ > 3% 立即按 spec §10 "KV 量化精度塌方" 流程回退)

## Shutdown

- 状态: 待填
- 输出: `benchmarks/dryrun/logs/shutdown.log`

## TODO after run

- [ ] 队员 C 签
- [ ] 附 DCU 上跑分的截图 / nvidia-smi / rocprofv3 输出到 `docs/weekly/p5-dryrun-attachments/`
- [ ] 把 step1-5 的状态从待填填入实际值
- [ ] 更新 plan Task 5.1 status: completed
- [ ] 若任一 step FAIL, 在 PR 描述里写明补救计划

## 关联文档

- spec §11 (完工标准)
- spec §10 风险表 (KV 量化精度塌方回退流程)
- plan Task 4.2 (精度验证 4 类任务)
- plan Task 5.1 (本任务)
- `scripts/dry_run.py` (orchestrator)
- `scripts/verify_dcu.py`
- `scripts/verify_testset_access.py`
- `benchmarks/run_bench.py`
- `benchmarks/analyze.py`
- `docker/compose.yml`
- `docker/Dockerfile`
