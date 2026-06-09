# 学习资源与方法

> 给团队 4 人(队长 + 队员 A/B/C)用的索引, 重点是"怎么找" + "去哪读", 不是"读什么结论"。**vLLM / ROCm / DCU 文档更新快, 链接用相对路径 + 入口 URL, 实际章节靠 grep 找**。

## vLLM 0.18.1 (赛方 pin 的版本)

### 源码入口

- 仓库: `vllm-project/vllm` GitHub, 0.18.1 tag
- 装好后源码在 Python 环境中, 例如 `python -c "import vllm; print(vllm.__file__)"` → 然后从那往下看

### 必读模块 (按重要性, 优先级降序)

| 优先级 | 路径 | 作用 |
|--------|------|------|
| ★★★ | `vllm/attention/backends/` | 各种 backend 实现; `__init__.py` + `selector.py` 看注册机制 |
| ★★★ | `vllm/v1/kv_cache_interface.py` | KV cache 块管理 + 量化 hook |
| ★★ | `vllm/v1/worker/model_runner.py` | 模型前向主循环, decode/prefill 切分 |
| ★★ | `vllm/v1/core/sched/` | 调度器(赛题禁止改) |
| ★ | `vllm/entrypoints/openai/serving_chat.py` | 服务入口, 看 API 行为 |
| ★ | `vllm/envs.py` | 所有 VLLM_* 环境变量定义 |

### Grep 技巧 (找代码/定义/默认值)

```bash
# 找某个 attention backend 怎么注册
grep -rn "register_backend\|AttentionBackendEnum" vllm/attention/

# 找某个 vLLM CLI flag 在哪定义
grep -rn "add_cli_args\|--block-size" vllm/engine/arg_utils.py

# 找 FP8 / KV 量化相关代码
grep -rln "fp8\|kv_cache_dtype" vllm/v1/ vllm/model_executor/

# 找某个 env var 在哪里被读
grep -rn "VLLM_USE_V1\|VLLM_ATTENTION_BACKEND" vllm/

# 看 AttentionBackendEnum 全部值
python3 -c "from vllm.attention.backends.registry import AttentionBackendEnum; print(list(AttentionBackendEnum))"

# 看模块级 _get_backend_priorities 实际值 (rocm 平台)
grep -A 30 "_get_backend_priorities" vllm/platforms/rocm.py
```

### 已知结论 (已验, 不用再重读, 标"v3/v4 验证" 在 spec §5)

- attention backend 真实机制: **enum + `register_backend()`**, 不是"丢个 .py 进 `vllm/attention/backends/`"。见 spec §5.2 + ADR 0006
- vLLM 0.18.1 `AttentionBackendEnum` 有 24 个值, 含 `CUSTOM = None` 槽位(spec §5.1 表)
- `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` **赛题禁止改**(spec §4 + AGENTS.md 赛题硬约束)

## AMD ROCm / DCU

### 官方文档

- ROCm docs 入口: https://rocm.docs.amd.com/
- HIP programming guide: https://rocm.docs.amd.com/projects/HIP/
- DCU 兼容矩阵: 找赛方发的"DCU 软件栈兼容表"(per ADR 0001 待 DCU 验证)

### FP8 矩阵 (已验, 标 spec §5.4)

| 架构 | SKU | FP8 原生 | 变体 |
|------|-----|---------|------|
| CDNA2 (gfx90a) | MI210 / MI250 / MI250X | **❌ 不支持** | — |
| CDNA3 (gfx942) | MI300A / MI300X / MI325X | ✅ | **FNUZ** 变体(`__hip_fp8_e4m3_fnuz`) |
| CDNA4 | MI350X / MI355X | ✅ | OCP 变体 |
| RDNA4 | RX 9070 / 9070XT | ✅ | OCP 变体 |

**注意**: FNUZ (Finite + No inf + Unsigned zero) 与 OCP **不兼容**, 同一 bit pattern 解释不同。

### FlashAttention ROCm fork (已验, 标 spec §5.3)

- 上游 FlashAttention 2 ROCm fork 支持 MI200x / MI250x / MI300x / MI355x
- 含 CDNA2(无 FP8)+ CDNA3(有 FP8, 设 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` 启用 FP8 路径)

## Triton + DCU

- Triton 3.0+ 原生支持 AMD GPU
- 已知坑(spec §5.5): 某些 `tl.atomic_*` / `tl.dot` scale 组合编译失败 — P0 期间要跑最小 case 验证
- aiter 库 (AMD 自家优化 kernel 集合): https://github.com/ROCm/aiter

## Qwen3.5-27B 模型

- HuggingFace: `Qwen/Qwen3.5-27B`(27B 参数, Apache-2.0 权重)
- 找模型卡: https://huggingface.co/Qwen/Qwen3.5-27B
- 关键信息: max context length / tokenizer / 推理建议的 dtype

## 工具

### 调研用

- `superpowers:context7` skill: 查 vLLM / PyTorch / Triton 最新 API 文档
- `superpowers:dispatching-parallel-agents` skill: 并行调研多个主题
- `WebSearch` + `WebFetch` (Claude Code 原生): 找最新 issue / 文档

### 验证用

- `scripts/verify_dcu.py`: 跑在 DCU host, 输出 SKU + FP8 支持 (gcnArchName)
- `scripts/verify_testset_access.py`: 跑在能联网环境, 检查 LongBench / RULER 是否可下载
- `scripts/dry_run.py`: P5 末 dry-run 编排(clean build → serve → bench → accuracy → shutdown)
- `python3 -c "import torch; print(torch.cuda.get_device_properties(0))"`: 快速看 DCU 信息

### 跑分用

- `benchmarks/run_bench.py`: 跑单档 bench (3 tiers: 4k-8k / 8k-16k / 16k-32k)
- `benchmarks/analyze.py`: 聚合 JSON → markdown 表格

## 术语速查

| 术语 | 含义 |
|------|------|
| prefill | 处理整个 prompt, 计算所有 token 的 K/V |
| decode | 自回归生成, 每次算 1 个新 token |
| PagedAttention | vLLM 的 KV cache 块管理, 类似 OS 虚拟内存分页 |
| block size | PagedAttention 的 KV 块大小 (--block-size), 单位: token |
| KV cache | prefill 算出的 K/V 矩阵, decode 时复用 |
| cudagraph | CUDA graph capture, 减少 kernel launch 开销 (DCU 上是 HIP graph) |
| torch.compile | PyTorch 的 JIT 编译, 在 vLLM 0.18.1 走 `default` mode (不要 max-autotune) |
| FNUZ | AMD FP8 变体, 跟 NVIDIA OCP 不兼容 |
| GCN arch | AMD GPU 架构代号, `gfx90a` = CDNA2, `gfx942` = CDNA3 |
| AttentionBackendEnum | vLLM 0.18.1 attention backend 的 enum 注册表, 24 个值 |

## 已知不确定 (per ADR 0002 / 0001 / 0006a)

- LongBench 数据集 gated, 需赛方 token — **待赛方确认**
- RULER 评测是赛方统一跑还是我们跑 — **待赛方确认**
- DCU 实际 SKU (gfx90a 还是 gfx942) — **待 P0 0.1 verify_dcu.py 在 DCU 上跑**
- vllm-rocm Docker 基础镜像可拉性 — ADR 0006a, 内网 / daocloud 镜像

## 改这份文档

任何人发现新结论 / 新坑 / 新工具, 直接编辑这份文档, commit 即可。**这是活文档, 不是 1 次写完就锁定的 design doc**。
