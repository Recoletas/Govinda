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

## KV 量化

> 围绕 ADR 0009 的"必做 3 + stretch 2"路线。**赛题禁持久化量化** (AGENTS.md 第 9/21 行), 运行时动态量化是唯一合规路径。spec §5.1 = 接 KV cache hook 不改 scheduler; §5.4 = FP8 FNUZ 已验。

### 粒度选择

- **per-tensor**: 1 scale 整层, LongBench Δ 3-5%, 排除
- **per-head**: 1 scale per head, 跟 PagedAttention 头维对齐, decode 不增 per-token 开销, **必做**
- **per-token**: 1 scale per token, prefill 写 OK, decode 读收益小开销大
- **KIVI 2D**: 关键 cache (首 256 token) 留 FP16, 其余 INT4/FP8 — **stretch**, 占 < 10% token 作 Δ 兜底
- **KVQuant outlier-aware**: 异常通道 FP16 其余 INT4 — **stretch**, 27B outlier 集中在少数 head

### 数据格式对比

| 格式 | 动态范围 | 异常值 | CDNA2 (gfx90a) | CDNA3 (gfx942) | vLLM 0.18.1 |
|------|----------|--------|----------------|----------------|-------------|
| bf16 (baseline) | 宽 | 最优 | 支持 | 支持 | 默认 |
| FP8 E4M3 FNUZ | 中 (~±448) | 中 | **不支持** | **支持** 原生 | `--kv-cache-dtype fp8` |
| INT8 对称 | 窄 (±127) | 差 (饱和) | 支持 (FA2) | 支持 | `--kv-cache-dtype int8` |
| INT4 (stretch) | 极窄 (±7) | 差 | 实验性 | 实验性 | 无内置, 需 hook |

FNUZ (Finite + No inf + Unsigned zero) 跟 NVIDIA OCP FP8 **不兼容** (spec §5.4): 同一 bit pattern 解释不同, DCU 上必须用 `__hip_fp8_e4m3_fnuz` 而非上游 `torch.float8_e4m3fn`。

### 代表方案对比

| 方案 | 格式 | 粒度 | 长 context Δ | 适配 27B / 32k |
|------|------|------|--------------|----------------|
| KIVI (2024) | K-FP16 / V-2D-INT4 | per-token + 关键 cache 留高精度 | < 1% (16k) | 可, stretch 备选 |
| KVQuant (2024) | INT4 + outlier FP16 | per-channel | < 1.5% (32k) | 适合长 context |
| Atom (2024) | INT8 | per-channel | < 1% | 中 context 友好 |
| QServe (2024) | W4A8KV4 | per-head | < 2% | 主打 weight 量化, KV4 副产物 |
| SmoothQuant (2022) | W8A8KV8 | per-tensor | < 0.5% | 27B 不优 (outlier 多) |
| vLLM 0.18.1 内置 | FP8 / INT8 | per-head | **未验, 待 P3 测** | **L1 首选路径** |

### DCU + ROCm 特有坑

- **FNUZ vs OCP 不兼容**: 上游 PyTorch / vLLM 默认 OCP 路径会在 DCU crash 或静默错值; 启动时 `assert torch.float8_e4m3fnuz == __hip_fp8_e4m3_fnuz` 自检
- **CDNA2 (gfx90a) 无原生 FP8**: MI210/MI250/MI250X 走 INT8 退路; FA2 fork 已支持 INT8 路径
- **aiter FP8 KV**: `VLLM_ROCM_USE_AITER=1` 启用, **stretch**, DCU 稳定性未验
- **Triton 已知坑** (spec §5.5): `tl.atomic_*` 跟 FP8 scale 组合编译偶发失败 — P0 0.5 最小 case 验

### vLLM 0.18.1 接入代码片段

```python
# src/kv_quant/hook.py — 队员 B 主笔, AI-generated awaiting verification
# L1: 读内置 flag; 失败时 L2 monkey-patch 写路径
import os
from vllm import LLM

def build_llm_with_kv_quant(model: str, kv_dtype: str = "fp8"):
    llm = LLM(
        model=model,
        kv_cache_dtype=kv_dtype,  # "fp8" → FNUZ on CDNA3, "int8" → 退路
        dtype="bfloat16",          # 权重保持 bf16 (AGENTS.md 禁持久化)
        enforce_eager=False,       # 配合 torch.compile (ADR 0013)
    )
    # L2 回退: 仅在 L1 跑挂时启用
    if os.environ.get("VLLM_KV_QUANT_HOOK") == "1":
        from vllm.v1.kv_cache_interface import KVCacheTensor
        _orig = KVCacheTensor.copy_from_blocks
        def _patched(self, blocks, *a, **kw):
            # 在写 HBM 前 per-head quantize 到 fp8_fnuz
            return _orig(self, blocks, *a, **kw)
        KVCacheTensor.copy_from_blocks = _patched
    return llm
```

> 完整决策 (粒度 × 格式 × 接入点 × CDNA 分支 + 验证清单) 见 [`docs/decisions/0009-kv-quant-strategy.md`](decisions/0009-kv-quant-strategy.md)。

## 已知不确定 (per ADR 0002 / 0001 / 0006a / 0009)

- LongBench 数据集 gated, 需赛方 token — **待赛方确认**
- RULER 评测是赛方统一跑还是我们跑 — **待赛方确认**
- DCU 实际 SKU (gfx90a 还是 gfx942) — **待 P0 0.1 verify_dcu.py 在 DCU 上跑**
- vllm-rocm Docker 基础镜像可拉性 — ADR 0006a, 内网 / daocloud 镜像
- `--kv-cache-dtype fp8` 在 CDNA3 (gfx942) 上 Δ 与 SLA 实测 — **未验, 待 P0 0.5 Triton DCU FP8 跑通后确认**
- aiter FP8 KV 路径 (CDNA3) 稳定性 — **未验, 待 DCU 上手后最小 case 验证**

## Attention backend 选型 (vLLM 0.18.1)

> 围绕 ADR 0010 写的"3 候选 + 接入方式"指南。spec §5.1 决策是"复用 TRITON_ATTN, 不自己写新 backend"; spec §5.2 已验 backend 注册机制; spec §5.3 已验 FlashAttention ROCm fork FP8 路径; ADR 0006 / 0006a 是 Docker 镜像与基础镜像拉取状态。

### 24 个 enum 值分组 (compact 表)

源码 `vllm/v1/attention/backends/registry.py:34-87`, 24 个值, 含义按"用途 + 平台"分类 (路径以模块名前缀 `vllm.v1.attention.backends.` 省略):

| 分组 | 成员 | 平台 | 用途 / 备注 |
|------|------|------|-------------|
| **NVIDIA 通用** | `FLASH_ATTN` `FLASH_ATTN_DIFFKV` `FLASHINFER` `FLASHINFER_MLA` `FLASHINFER_MLA_SPARSE` `FLEX_ATTENTION` | CUDA | FA2/FA3 wrapper + FlashInfer + PyTorch flex_attention; DCU 上 `FLASH_ATTN` 走 ROCm/flash-attention fork (spec §5.3) |
| **NVIDIA Hopper/Blackwell** | `FLASH_ATTN_MLA` `FLASHMLA` `FLASHMLA_SPARSE` `CUTLASS_MLA` | CUDA (SM90/100) | MLA 专用; DCU 不可用, **NVIDIA-only 硬件特性** |
| **ROCm 通用** | `TRITON_ATTN` `ROCM_ATTN` `ROCM_AITER_FA` `ROCM_AITER_UNIFIED_ATTN` `ROCM_AITER_MLA` `ROCM_AITER_TRITON_MLA` `ROCM_AITER_MLA_SPARSE` | ROCm | 跨平台 Triton + AMD aiter 后端; `ROCM_AITER_FA` 限制 `on_mi3xx()` 即 CDNA3 only (源码 `vllm/v1/attention/backends/rocm_aiter_fa.py:789-795`) |
| **XPU 专用** | `XPU_MLA_SPARSE` | Intel XPU | 不可移植 |
| **CPU** | `CPU_ATTN` | x86 CPU | 排除 |
| **其他** | `TORCH_SDPA` (ViT only, 仅 `""` 占位) `TREE_ATTN` (tree-structured) `NO_ATTENTION` (mamba/linear 退化) `CUSTOM` (None 槽位, 须 `register_backend` 登记后才能用) | — | 自定义或非 dense |
| **Mamba 系** (另 enum) | `MAMBA1` `MAMBA2` `SHORT_CONV` `LINEAR` `GDN_ATTN` `CUSTOM` | — | Qwen3.5 MoE 不会触发, 列在 `MambaAttentionBackendEnum` 旁路 |

**DCU 上能用的 3 个核心候选**: `TRITON_ATTN` (默认) / `ROCM_AITER_FA` (开 `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MHA=1`) / `ROCM_AITER_UNIFIED_ATTN` (开 `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`)。源码 `vllm/platforms/rocm.py:309-352`。

### 3 候选对比 (Triton vs AITER-FA vs AITER-Unified)

| 维度 | `TRITON_ATTN` (默认) | `ROCM_AITER_FA` (开 aiter) | `ROCM_AITER_UNIFIED_ATTN` (开 aiter) |
|------|---------------------|---------------------------|-------------------------------------|
| 实现位置 | `vllm/v1/attention/backends/triton_attn.py` | `vllm/v1/attention/backends/rocm_aiter_fa.py` (800+ 行) | `vllm/v1/attention/backends/rocm_aiter_unified_attn.py` (继承 `RocmAttentionBackend`) |
| 平台支持 | 任意 (`supports_compute_capability=True`) | `on_mi3xx()` 即 **CDNA3 only** (gfx942) | 任意 ROCm (无 `supports_compute_capability` 限制) |
| KV cache dtype | fp16/bf16/fp8/fp8_e4m3/fp8_e5m2 | fp16/bf16/fp8/fp8_e4m3/fp8_e5m2 | fp16/bf16/fp8/fp8_e4m3/fp8_e5m2 |
| Head size | >= 32 | `[64, 128, 256]` (Qwen3.5 27B head_size=128 OK) | >= 32 |
| Block size | 16 的倍数 | **仅 [16, 32]** | 16 的倍数 |
| Prefill | Triton FA | aiter `flash_attn_varlen_func` (MHA) | aiter `unified_attention` (Triton) |
| Decode | Triton FA | aiter `paged_attention_v1` (ll4mi 汇编) + 兜底 `unified_attention` (head<64) | aiter `unified_attention` (Triton) |
| FP8 attention | 走 Triton FP8 (设 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` 后, spec §5.3) | aiter 汇编 FP8 (`fp8_e4m3`/`fp8_e5m2`, FNUZ on CDNA3) | aiter Triton FP8 + scale descale |
| cudagraph | 支持 (`UNIFORM_BATCH`) | 支持 | 继承自 ROCM_ATTN, 支持 |
| DCU 实测稳定性 | **未验, 待 P0 0.4 backend smoke** | **未验, 待 P0 0.5 Triton DCU FP8 跑通后** | **未验, 同上** |
| 已知风险 | Triton FP8 路径某些 `tl.atomic_*` 组合在 ROCm 失败 (spec §5.5) | CDNA2 (gfx90a) 上 `on_mi3xx()` 直接 False, 不可用 | 同一 backend 跑 prefill+decode, 单 query/seq 短时不一定比专用路径快 |

### DCU FP8 attention 路径

- **TRITON_ATTN**: 走 `flash_attn_varlen_func` → 内部走 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` 启用的 Triton FP8 (spec §5.3 已验) — CDNA2 (gfx90a) **无 FP8 原生支持** (spec §5.4), 强制 fp16/bf16; CDNA3 (gfx942) 走 FNUZ 变体
- **ROCM_AITER_FA**: aiter 的 MHA 汇编 + `reshape_and_cache_flash` 处理 fp8 KV cache, 内部用 `IS_FNUZ` 标志 (源码 `rocm_aiter_fa.py:213, 299`); **仅 CDNA3**
- **ROCM_AITER_UNIFIED_ATTN**: aiter Triton `unified_attention` + `triton_rope_and_cache` 支持 fp8 KV cache, **ROCm 通用**
- **CDNA2 FP8 限制**: spec §5.4 已验 CDNA2 (gfx90a) 无 FP8 原生硬件, 三家 backend 全部要降级 fp16/bf16

### 接入方式 (3 选 1 推荐顺序)

1. **CLI flag `--attention-backend`**: v0.18.1 `vllm/engine/arg_utils.py:782` 已加, 走 `AttentionConfig.backend`, **最稳, P0 末用** (例: `vllm serve ... --attention-backend TRITON_ATTN`)
2. **`AttentionBackendEnum.register_backend(CUSTOM, "my.path.MyBackend")`**: 源码 `registry.py:210-262`, decorator + 直接调用 2 种 API, 需 import 在 `vllm` 之前; **P3 stretch 自定义 kernel 时用**
3. **改 `_get_backend_priorities()` 源码** (`vllm/platforms/rocm.py:309-352`): 模块级函数, 改 `backends.append(...)` 顺序即可; 不走 vLLM 镜像的话直接改源码, 走镜像就 monkey-patch。**P0 验证后改源码**

> **注意**: `VLLM_ATTENTION_BACKEND` env var **在 v0.18.1 不存在** (spec §5.1 第 4 行"没找到"已验, 源码 `envs.py` 全文 grep 0 命中)。平台选择走 `_get_backend_priorities()`, **不要**找这个 env var。

### 平台 priority list 改法 (举例)

DCU 上验证 `ROCM_AITER_UNIFIED_ATTN` 比 `TRITON_ATTN` 快, 想把 Unified 放第 1 位: 编辑 `vllm/platforms/rocm.py:_get_backend_priorities` 第 332-351 行, 把 `if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:` 块放在最前, 删 `backends.append(AttentionBackendEnum.TRITON_ATTN)` 兜底即可。改前先在 v0.18.1 mirror 上 cp 备份 (`docker cp` 出来), 改坏直接 restore。

## 改这份文档

任何人发现新结论 / 新坑 / 新工具, 直接编辑这份文档, commit 即可。**这是活文档, 不是 1 次写完就锁定的 design doc**。
