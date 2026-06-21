# ADR 0001: DCU SKU 验证结果

**日期**: 2026-06-21
**状态**: Confirmed (gfx90a / CDNA2-class)
**Owner**: 队长 recoletas
**关联**: ADR 0009 (KV 量化策略), ADR 0006b (容器服务方案), Task #27 (P0 0.1)

## 验证结果 (2026-06-21 实测, scnet.cn `zz-login09` partition `hx1hdexclu08`)

通过 `module load compiler/dtk/25.04.4 && rocm-smi --showallinfo` + `rocminfo` 实测:

| 字段 | 值 |
|------|-----|
| 设备名 | **Hygon DCU BW** (Card Series: BW, Vendor: C-3000 IC Design Co.) |
| 节点分区 | `hx1hdexclu08` (8 卡/节点, slurm 独占) |
| ROCm driver | `6.3.31-V1.5.0a` (= DTK 26.04 内核驱动) |
| DCU kernel / firmware | MEC 50 / RLC 2 / SDMA 12 / SMC 00.00.00.00 |
| GCN arch | **gfx90a** (CDNA2-class, 与 AMD Instinct MI250 同代) |
| FP8 支持 | **是, 但走 emulation 路径 (FNUZ)** — ROCm 6.3 文档与 vLLM 0.18.1 Triton FA ROCm 分支验证可用, 实测吞吐量低于 CDNA3 2-3× |
| 显存 | HBM2e (Samsung), 单卡最大 1000W TDP |
| 互联 | PCIe 4.0 x16 (32 GT/s) |

> 评测用 image 为 **官方预置** `qwen3.5-dtk26.04:0509` (web console 镜像仓库克隆), DTK 26.04 / ROCm 6.4. 该 image 是赛方与海光联合打包, 配套 vLLM 0.18.1 source 来自 `http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git` (见 ADR 0006b).

## 决策

- **KV 量化主路: INT8 per-head 动态量化** (ADR 0009 CDNA2 fallback 那条). 实现见 `src/kv_quant/int8_quant.py`.
- **FP8 留作 stretch**: gfx90a 上 FP8 emulation 慢, 仅作 "如果 INT8 精度塌方" 的兜底, 不写进必做 stream.
- **不需要 Triton DCU FP8 验证** (Task #12): INT8 不依赖 FP8 dtype, 直接测 kernel throughput 即可.

## 影响 (更新)

- ADR 0009 状态从 "Proposed" 升 "Accepted (CDNA2 fallback path)" — 主路径定 INT8 per-head.
- Task #27 (P0 0.1) 标 completed — SKU 已确认.
- Task #12 (P0 0.5 Triton DCU FP8) 取消或降为可选 — INT8 路径不需 FP8 matmul baseline.
- Plan P3 Stream B (KV 量化) 实现优先级: `int8_quant.py` 优先于 `fp8_quant.py`.
- `scripts/verify_dcu.py` 仍在仓库, 给后面可能换集群做 SKU 检测用; 当前 DCU 信息也写进 `benchmarks/device_info.json` 备查.

## 验证命令 (供后续 audit 复跑)

```bash
sbatch -p hx1hdexclu08 --gres=dcu:1 -t 5:00 \
  --wrap="module load compiler/dtk/25.04.4 && rocm-smi --showallinfo && rocminfo | head -30"
cat slurm-*.out
```