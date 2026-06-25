# ADR 0013: 比赛规则解读 + dev/bench 命令区分

**日期**: 2026-06-25
**状态**: Accepted
**Owner**: 队长 recoletas
**关联**: ADR 0006b (容器服务, 附录 A 启动命令), Plan P5 (提交材料)
**来源**: 先导杯 2026 大模型推理服务优化赛技术方案 PDF (2026-06-25 拿到)

## Context

2026-06-25 读到官方《先导杯 2026 大模型推理服务优化赛技术方案》, 跟我们之前 plan/spec 假设有几条**关键冲突** (主要在 vllm serve 锁定参数 + 量化边界 + 完成率熔断). 这条 ADR 收齐解读, 同时定下 dev vs bench 两条命令路径, 避免 P5 提交时踩雷.

## 与之前认知的关键差异

| 项 | 我们旧认知 | 比赛规则实际 | 行动 |
|---|---|---|---|
| PyTorch 版本 | 2.7.1 (推 6.3 wheel) | **2.10.0** (官方镜像预装, **不能改**) | 实跑 `python3 -c "import torch; print(torch.__version__)"` 确认, 不一致则 INT8 path 调试时注意版本差异 |
| `--max-num-seqs` | dev 用 128 | **LOCKED** (§9 (8)) | dev 可以用, bench 提交用 default |
| `--max-num-batched-tokens` | dev 用 4096 | **LOCKED** (§9 (8)) | 同上 |
| `--default-chat-template-kwargs '{"enable_thinking": false}'` | 我们用过 | **chat template LOCKED** (§9 (8)) | **dev 也建议去掉**, 不然跟官方 baseline 不公平 (官方 baseline 默认 thinking on/off 不一致) |
| transformers 版本 | 没明说 | **5.5.0** | 锁版本, INT8 quant 用到 transformers 工具时注意 |
| 投机解码 | 没计划 | **严禁** (§7) | 已对齐 |
| 模型结构 / 权重 | 不动 Qwen | **严禁修改** (§7) | 已对齐 |
| 持久化权重量化 | 没计划 | **严禁** (§7) | 已对齐 |
| KV cache 量化 | Stream B 计划 | **明确允许** (§7) ✓ | 路径合规 |

## 量化边界 (最重要, §7)

**严禁**:
- 权重加载**前后**对模型权重进行持久化量化
- 结构化剪枝 / 权重重排压缩 / 模型图重构 / 模型格式转换
- 生成**可复用**的量化权重缓存

**允许** (这是我们的全部优化空间):
- 推理过程中**非持久化**、算子级低精度计算优化
- **激活值动态量化**
- **KV cache 量化** ← Stream B 主路
- kernel 内部临时类型转换
- 低精度矩阵乘法
- Attention 内核优化

**关键**: INT8 KV cache 我们是 per-request 动态量化, 不写持久化权重, 合规. 但要确保:
- 量化 scale tensor 不持久化
- 量化后的 KV 不写盘 (除了 vllm 必要的 paging, 那是 vllm 内部行为)
- 服务重启不缓存量化结果

## 服务可用性熔断 (§8)

- **完成率 ≥ 99%** (失败 ≤ 1%) — 失败超 1% 该档得分清零
- TTFT/TPOT P99 × 1.5 SLA 违反 → 该档清零
- **双熔断**: 即使吞吐提升明显, 完成率不达标也清零

**影响**: Stream B INT8 KV cache 必须鲁棒. 任何精度塌方 → 立即 revert, 不能硬撑 (双熔断下, 1% 完成率损失 = 该档清零, 比吞吐提升更亏).

## Dev vs Bench 命令路径

**重要**: bench 时 vllm serve 命令跟 dev 不一样. 我们内部 baseline 是用 dev 命令测的 (含锁定参数). 官方 bench 会用纯命令测. **相对提升 = (方案 - Baseline) / Baseline, 都用 bench 命令测, 公平**. 我们的 baseline 数字仅作内部对照 (知道大致在哪), 不作为评测基准.

### Dev 命令 (`scripts/start_vllm_dev.sh` — 已存, 复制自 ADR 0006b 附录 A)

```bash
vllm serve /public/home/xdzs2026_c087/Qwen3.5-27B \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --served-model-name Qwen3.5-27B \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

**用途**: 内部 dev / baseline / P3 调参. 锁定参数可加, 加速试错. 

### Bench 命令 (`scripts/start_vllm_bench.sh` — 新建, 严格对齐规则 §9)

```bash
vllm serve /public/home/xdzs2026_c087/Qwen3.5-27B \
  --port 8001 \
  --max-model-len 32768
```

**用途**: 官方评测 + 提交时 + P3 任何"对比基线"评测. 严格匹配规则 §9 (7) 的命令模板. 我们的 vllm wheel 必须在这个命令下工作, **不能假设** `--max-num-seqs` / `--max-num-batched-tokens` / `--default-chat-template-kwargs` 存在.

**P3 关键约束**: 任何 patch (INT8 KV cache / Triton kernel / torch.compile config) 必须 **default 行为对**:
- 不传 `--max-num-seqs` 时, 默认就是 1 (符合 spec §3 并发=1)
- 不传 `--max-num-batched-tokens` 时, 默认走 vllm 调度, 不能因为我们没传就崩
- 不传 `--default-chat-template-kwargs` 时, 模型走官方 chat template 默认行为 (Qwen3 可能有 thinking, 不能假设 enable_thinking=false)

## 提交材料 (§12-15) 影响

### §13 env vars 文档 (`reports/env-vars.md`)

我们用过的 env var:
- `MODEL_DIR=/public/home/xdzs2026_c087/Qwen3.5-27B` — bench 脚本用来找 tokenizer / model path

后续 P3 可能加的 env var:
- `VLLM_USE_V1=1`
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` / `ROCM_AITER_FA`
- `TORCH_COMPILE_DEBUG=0`
- ...

**所有这些都要写进 `reports/env-vars.md`** (P5 必交付).

### §14 优化方案说明 (`reports/optimization-plan.md`)

按官方要求:
- 优化方法 + 技术路线
- 各项优化对性能提升的贡献分析
- 优化点汇总表 (含关键代码路径 + 性能对比数据)

**已有 P0 末调研** (`docs/learning.md` + 3 ADR) 凑素材, P3 / P4 阶段补完后整理.

### §15 第三方库标头 (`README.md` 头部)

当前 README 已有:
> 本项目使用 vLLM（Apache-2.0）、PyTorch、Transformers、Qwen3.5-27B（Apache-2.0 权重） 等开源项目

**合规**, 不用改. 但 P5 提交前再 check 一次 §15 列名 (HIP / ROCm-DTK / aiter 等是否也要列).

## Action items

- [x] ADR 0013 写完 (本文)
- [x] `scripts/start_vllm_bench.sh` 建好
- [x] Memory `competition_rules.md` 写好 (新规则)
- [ ] 容器内 `python3 -c "import torch; print(torch.__version__)"` + `import transformers; print(transformers.__version__)` — 确认实际版本
- [ ] P3 启动前: 验证 INT8 KV cache patch + bench 命令兼容 (没有 --max-num-seqs 等能跑)
- [ ] Plan P5: 提前起稿 `reports/env-vars.md` + `reports/optimization-plan.md`
- [ ] README.md 头部第三方 attribution check (确认 HIP / DTK / aiter 是否要列)