# 提交材料 §15 · 第三方引用 + 编译步骤 (草稿)

> **提交日期**: 2026-06-09
> **赛方要求**: 赛题 §15
> **状态**: 草稿 (P5 演练后定稿, 见 Plan Task 5.1)
> **Owner**: 队长
> **配套**: [spec §11](../../docs/specs/2026-06-09-qwen-inference-optimization-design.md) / [ADR 0002](../../docs/decisions/0002-testset-access.md) / [ADR 0006a](../../docs/decisions/0006a-docker-build.md) / [Plan Task 5.1](../../docs/superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md)

## 1. 第三方引用清单

### 1.1 软件库 / 工具

| 库 / 工具 | 版本 | 用途 | License | 来源 |
|-----------|------|------|---------|------|
| PyTorch + ROCm | 2.10.0 (基线) | 推理框架 + DCU (HIP) 后端 | BSD-3-Clause | 基础镜像 `rocm/vllm-rocm:v0.18.1` 自带 |
| triton | >= 3.0, < 4.0 | Triton kernel (stretch 项 4-5) | MIT | `docker/requirements.txt` (PyTorch ROCm wheel 路径, **不** `pip install triton[all]`) |
| flash-attn | 2.7.4.post1 | ROCm/flash-attention fork (FA2 only, spec §5.1) | BSD-3-Clause | `docker/requirements.txt` (DCU 容器内编译, **不**外网) |
| aiter | >= 0.1 | ROCm optimized kernels (aiter FA + Triton FP8 后端) | MIT | `docker/requirements.txt` |
| vllm | 0.18.1 | 推理引擎 (赛方 pin) | Apache-2.0 | 赛方 baseline 锁定, `rocm/vllm-rocm:v0.18.1` |
| Qwen3.5-27B | (HuggingFace) | 推理模型 | Apache-2.0 / Tongyi Qianwen 许可 | `Qwen/Qwen3.5-27B` (HuggingFace Hub) |

### 1.2 数据集 / 评测基准

| 数据集 / 基准 | 来源 | 用途 | License / 状态 |
|---------------|------|------|----------------|
| LongBench | `xinrongzhang2022/longbench` (HuggingFace) | 长上下文评测 (spec §2 验证项 3) | **gated, 待赛方 token** (ADR 0002) |
| RULER | NVIDIA `NVIDIA/RULER` GitHub | 长上下文 NIAH / VT / CWE / QA / RAG | **由赛方统一评测**, 自跑需源数据 (ADR 0002) |
| OpenCompass | 0.18.1 | 4 类任务精度基准 (赛题 §9 评分项) | Apache-2.0 |

## 2. 编译步骤 (P5 演练后定稿)

```bash
# 1. 进入 docker 目录
cd docker
# 2. 构建镜像 (基础镜像: rocm/vllm-rocm:v0.18.1)
docker compose build
# 3. 启动容器
docker compose up -d
# 4. 验证 (DCU SKU + FP8 + 测试集访问)
docker compose exec vllm python /workspace/scripts/verify_dcu.py
docker compose exec vllm python /workspace/scripts/verify_testset_access.py
# 5. 跑 3 档 baseline
python benchmarks/run_bench.py --tier 4k-8k --output benchmarks/baseline/4k-8k.json
python benchmarks/run_bench.py --tier 8k-16k --output benchmarks/baseline/8k-16k.json
python benchmarks/run_bench.py --tier 16k-32k --output benchmarks/baseline/16k-32k.json
# 6. 跑优化后基准 (待 P3 + P4 完成后填入)
# 7. 生成 summary
python benchmarks/analyze.py benchmarks/baseline --output benchmarks/baseline/summary.md
```

> **注 1**: 步骤 2 在内网 / daocloud 镜像源环境下可拉取 `rocm/vllm-rocm:v0.18.1` (ADR 0006a); 外网可能 403 Forbidden, 需提前确认镜像可达性。
> **注 2**: 步骤 6 当前为占位, P3 末集成日后填入实际 `--extra-args "<Stream A/B/C 参数>"`。
> **注 3**: `flash-attn` 在 DCU 容器内编译, ROCm 6.0+ / PyTorch 2.2+ (spec §5.3), 需 `--no-build-isolation`。
> **注 4**: warmup_iters ≥ 3 (CompileConfig 默认, spec §5.1), 否则 cudagraph capture 失败。

## 3. 已知问题 (per ADR 0002 + 0006a)

- **LongBench gated**: HuggingFace 仓库 `xinrongzhang2022/longbench` 变为 gated / private, **需**赛方提供 HF token 或官方快照, 否则 P2 末自测走"自造 100 样本 smoke 测试集" (ADR 0002 决策)。
- **RULER 评测由赛方统一跑**: 自跑需 SQuAD / HotpotQA / Paul Graham 源数据现场拉取, 容器内 `datasets` / `transformers` 包预装情况待确认 (ADR 0002 失败 → 询问映射表)。
- **Docker 镜像拉取需内网/daocloud 镜像**: `rocm/vllm-rocm:v0.18.1` 在外网拉取返回 403 Forbidden (ADR 0006a 实测), 需确认赛方提供的镜像仓库地址或内网代理配置。
- **vLLM 全量编译 4-12h** (spec §10 风险表): 演练放在 P4 末而非 P5, 保留上次成功构建的 Docker 镜像。
- **DCU SKU 不确定**: 编译 / 部署动作前先跑 `verify_dcu.py`, 确认 CDNA2 (gfx90a) vs CDNA3 (gfx942) 走不同 KV 量化分支 (spec §5.4 + ADR 0007 决策矩阵预筛)。

## 4. 验证项

- [ ] P4 末 1 次干净全量编译演练 (spec §11)
- [ ] P5 演练按上述步骤跑通 3 档 baseline + summary 生成
- [ ] P5 演练后填入步骤 6 优化后参数 (`--extra-args`)
- [ ] 状态字段由"草稿"改为"定稿"

## 关联文档

- spec §11 (赛题 §12-15 提交材料清单 + 完工标准)
- [ADR 0002: 测试集访问验证结果](../../docs/decisions/0002-testset-access.md) (LongBench gated / RULER 评测由赛方统一)
- [ADR 0006a: vllm-rocm Docker 镜像构建状态](../../docs/decisions/0006a-docker-build.md) (镜像拉取 403)
- [Plan Task 5.1: 材料定稿 + 演练](../../docs/superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md) (P5 演练入口)
