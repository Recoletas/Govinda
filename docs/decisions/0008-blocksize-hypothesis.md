# ADR 0008: Block-size Sweep 计划 (P3 优先级 1)

**日期**: 2026-07-01
**状态**: Accepted (从假设升级为 P3 执行清单)
**Owner**: 队长 recoletas
**关联**: ADR 0013 (规则, `--block-size` 不在 LOCKED), ADR 0014 (bench 命令), `scripts/sweep_block_size_bench.sh`

## 假设阶段 (2026-06-09 草稿, 已合并入下)

> `block_size 16` 或 `block_size 32` 最可能赢. 8 太小 (metadata 开销 > 5%), 128 太大 (长 prompt 碎片率). 16/32 sweet spot 平衡碎片率 + metadata 开销.

## 当前计划 (2026-07-01, P3 优先级 1)

### 参数空间

- `--block-size ∈ {16, 32, 64}`
- 8 不测 (ADR 草稿阶段已排除)
- 128 不测 (长 prompt 碎片率风险, ROI 不明, 14 天窗口不放这里)
- 走 bench 命令 (`start_vllm_bench.sh` + 加 `--block-size`), 不走 dev 命令 (避免 LOCKED flag 污染)

### 测试矩阵

```
       4-8K    8-16K   16-32K
16     [x]     [x]     [skip]
32     [x]     [x]     [skip]
64     [x]     [x]     [skip]
```

- 6 启停 (16/32/64 × 4-8K/8-16K), 每次 ~5-10 min
- 16-32K 历史从未跑通, sweep 阶段不强求, 启挂即 skip
- 10 prompts/dataset (与 baseline 一致; ADR 草稿写的 50 prompts 时间不允许)

### SLA 自检

- TTFT P99 ≤ **4557.3 × 1.5 = 6835.95 ms**
- TPOT P99 ≤ **69.79 × 1.5 = 104.69 ms**
- completed == num_prompts (完成率 100%)
- 任一不达标 cell 标红, 不进入选型

### 选型

- 跨 4-8K + 8-16K 两档, 哪个 block_size **throughput 最高且 SLA 不破** 写入 P3 集成日 `start_vllm_bench.sh`
- 不在 16-32K 单独优选 (历史未跑通)

### 风险

- 单参数改动, 影响面有限
- 不动 LOCKED flag, 不动权重, 不动 scheduler
- 每个 cell 重启 vllm (改 --block-size 是启停参数), 容器负载正常

## Action items

- [x] 修正版 bench-mode sweep 脚本已写 (无 LOCKED flag + 全优化 env)
- [ ] 等 vllm 重启窗口 (bench 容器实例化, 跟 dev baseline 不混)
- [ ] 跑 6 启停 sweep, 记录到 `docs/sweep-results/`
- [ ] 选最优 block_size, 更新 `start_vllm_bench.sh`
- [ ] 后续可选: block_size=128 / 16-32K dataset 二次 sweep (如 14 天内有余量)