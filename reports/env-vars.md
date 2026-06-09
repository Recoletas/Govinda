# 提交材料 §13 · 改动过的 vLLM 环境变量 / 启动参数

> **提交日期**: 2026-06-09
> **赛方要求**: 赛题 §13
> **状态**: 草稿 (P4 末完成, P5 演练后定稿)
> **Owner**: 队员 C (P4 末) → 队长 (P5 定稿)
> **配套**: [`spec §11`](../../docs/specs/2026-06-09-qwen-inference-optimization-design.md) / [ADR 0007](../../docs/decisions/0007-coupling-matrix.md) / [ADR 0013](../../docs/decisions/0013-torch-compile-roi.md)

## 1. 环境变量 (实际使用, 来自 `docker/compose.yml`)

| 变量名 | 取值 | 作用 | 设置位置 |
|--------|------|------|----------|
| `VLLM_USE_V1` | `1` | 强制 vLLM 走 v1 engine (0.18.1 默认已是 v1, 显式声明以防 fallback) | `docker/compose.yml` |
| `HIP_VISIBLE_DEVICES` | `0` | 限定 DCU 0 可见, 多卡环境避免误用 | `docker/compose.yml` |
| `NCCL_MIN_NCHANNELS` | `112` | NCCL 通道数下界, 应对长上下文 all-reduce; 初赛单卡仍保留以便决赛多卡路径直接复用 | `docker/compose.yml` |

> **未设置但已验证不存在**: `VLLM_ATTENTION_BACKEND` (vLLM 0.18.1 `envs.py` grep 0 次命中) — backend 选择走 `AttentionBackendEnum` + 平台 priority list + `register_backend()`, 不走环境变量 (spec §5.1 决策表 + §5.2)。

## 2. vLLM 启动参数 (实际传入, 来自 `benchmarks/run_bench.py` + `src/compile/config.py`)

| 参数 | 取值 | 作用 | 何时启用 |
|------|------|------|----------|
| `--max-model-len` | `32768` | 模型最大上下文长度, 覆盖 3 档 (4k-8k / 8k-16k / 16k-32k) + 输出余量 | 必启用 (赛题 P5 锁) |
| `--max-num-seqs` | `1` | 单卡单并发 (赛题要求) | 必启用 (赛题 P5 锁) |
| `--served-model-name` | `govinda` | API 暴露名, 与赛方评测口径一致 | 必启用 (赛题 P5 锁) |
| `--port` | `8000` | 监听端口, 与评测脚本期望一致 | 必启用 (赛题 P5 锁) |
| `--enforce-eager` | (无值, flag) | 关闭 cudagraph / torch.compile, 退化 eager 模式 | 条件启用 (Stream C 故障 / 精度回退时, 由 `CompileConfig.enforce_eager=True` 触发) |
| `--compilation-config.use_cudagraph` | `True` | 启用 HIP graph 录制 kernel launch 序列, 减少 Python 调度开销 | 条件启用 (Stream C 主实验, 由 `CompileConfig.use_cudagraph=True` 触发) |

> **注**: `--max-num-batched-tokens` / batch scheduler 类参数在赛题 P5 锁定列表中, **不** 在此表内调整 (spec §4)。

## 3. P3 待实测验证 (Stream A / B / C 必做项预期要用的参数)

> **状态**: 全部"待 P3X.Y 实测" — 仅列出预期值, P3 中段填入实测值后定稿。

| 参数 | 预期取值 | 作用 | 状态 |
|------|----------|------|------|
| `--block-size` | `{8, 16, 32, 64, 128}` 扫描 | KV cache 块大小, 直接影响 cache miss 率与量化粒度耦合 (ADR 0007) | 待 P3A.1 实测 (Stream A) |
| `--enable-prefix-caching` | `True` | 共享 prompt 前缀 KV 复用, 长 system prompt 场景收益大 | 待 P3A.2 实测 (Stream A) |
| `--enable-chunked-prefill` | `True` | 长 prompt 切成多块, 避免 prefill 独占 decode 通道 | 待 P3A.3 实测 (Stream A) |
| `--kv-cache-dtype` | `fp8_e4m3` (CDNA3) / `int8` (CDNA2 fallback) | KV cache 动态量化, 显存降 ~50% | 待 P3B.1-P3B.3 实测 (Stream B, FP8 走 `--compilation-config.kv_cache_dtype`) |
| `--compilation-config.use_cudagraph` | `True` | HIP graph 录制, 已在表中 (复列以强调 P3C 主实验) | 待 P3C.1-P3C.2 实测 (Stream C) |

## 4. 验证项

- [ ] P3A.1 block-size 扫描产出 5 × 4 耦合矩阵 (ADR 0007)
- [ ] P3B 精度 Δ ≤ 3% (spec §10 风险表)
- [ ] P3C warmup_iters ≥ 3 (避免 capture 失败)
- [ ] P4 末 1 次干净全量编译演练通过 (spec §11)

## 5. 变更记录

| 阶段 | 变更 | 触发条件 |
|------|------|----------|
| P0 | (无, 仅 `docker compose up` 验证基础镜像可拉) | 镜像 403 阻塞 (ADR 0006a) |
| P1 | (无) | — |
| P2 | `benchmarks/run_bench.py` 加入 `--max-model-len 32768 --max-num-seqs 1 --served-model-name govinda --port 8000` | 3 档 baseline 跑分 (Task 2.2) |
| P3A | `--block-size {8,16,32,64,128}` + `--enable-prefix-caching` + `--enable-chunked-prefill` | 块管理 3 子任务 (Task 3A.1-A.3) |
| P3B | `--kv-cache-dtype fp8_e4m3` (CDNA3) / `int8` (CDNA2) | KV 动态量化 (Task 3B.1-B.3) |
| P3C | `--compilation-config.use_cudagraph=True` (默认开) / `--enforce-eager` (回退开关) | torch.compile 实验 (Task 3C.1-C.2) |
| P4 | (无新增, 集成日固化配置) | 1 次干净全量编译演练 |
| P5 | (无新增, 演练后定稿) | Task 5.1 |

## 关联文档

- spec §11 (赛题 §12-15 提交材料清单 + §5.1 决策表)
- [ADR 0007: Block-size × KV 量化粒度耦合矩阵](../../docs/decisions/0007-coupling-matrix.md)
- [ADR 0013: torch.compile + cudagraph ROI 分析](../../docs/decisions/0013-torch-compile-roi.md)
- `docker/compose.yml` / `docker/requirements.txt` / `benchmarks/run_bench.py` / `src/compile/config.py`
