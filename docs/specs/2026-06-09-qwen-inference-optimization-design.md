# 2026-06-09 Qwen 推理服务优化比赛 · 设计文档

> **作者**：队长 recoletas · **最后更新**：2026-06-09
> **配套**：[`AGENTS.md`](../../AGENTS.md) · [`README.md`](../../README.md)

---

## 1. Context

参加 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯），赛题为"基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化"。初赛单卡、并发=1、长上下文场景（4k-32k 三档）。**团队 4 人均为新人，9.5 周时间预算（从 2 周后有时间为起算点），目标稳定拿 60-75 / 100 分而非冲击高分**。

**对手画像假设**：先导杯强校如清华/中科院/上交战队多为成熟团队，**不**与头部争 90+ — 60-75 是合理对位，避免中途焦虑动方案。

## 2. 待验证未知项（**Phase 0 硬卡门**，确认前不开 Phase 1）

| 项 | 不确定点 | 影响 | 验证方法 |
|---|---------|------|----------|
| **DCU SKU** | K100 (CDNA2 / gfx90a) 还是 Z100 (CDNA3 / gfx942) | FP8 KV cache 是否可行（CDNA2 无原生 FP8，CDNA3 有） | `rocminfo` 或 `python -c "import torch; print(torch.cuda.get_device_properties(0))"` |
| **官方 baseline 数字** | 赛方是否下发给我们，还是只能自测 | 60-75 估算是否成立；SLA 自比较还是绝对值 | 询问赛方 / 赛方最新通知 |
| **LongBench / RULER 测试集** | 开发期可下载 vs 仅评测平台可见 | 1.05× 提速回退规则能否本地验证 | 询问赛方 / 赛方文档 |
| **vLLM 0.18.1 custom backend 路径** | 机制已源码验证（§5.2：`AttentionBackendEnum` + `register_backend()` + 平台 priority list），**但 DCU 上能否成功 import / 注册 / 端到端跑通未验证** | stretch 4-5 是否可行 | DCU 上手后跑 1 个最小 backend 注册 smoke test（继承 `TritonAttentionBackend` 改 1 行，跑通 `vllm bench serve` 1 个 prompt） |

**任一项不利 → 回到 §5 重新选型。这是 P0 的硬卡门。**

## 3. 用户决策（已确认）

| 项 | 决策 |
|---|---|
| 项目仓库 | `/home/recoletas/Govinda/`（GitHub: `Recoletas/Govinda`） |
| 团队组成 | 4 人队（队长 + 3 队员），全新人（CS 课程级 DL） |
| 时间预算 | 8-10 周，从"队长有时间的那个点"起算；前 2 周完全离线 |
| DCU 硬件 | 1-2 周后到位；之前无硬件 |
| AI 用法 | 当开发助手（写代码/调研/文档），**不接进推理路径** |
| 角色分配 | 队长 = Profiling & 集成 owner（5-10h/周）；其余 3 人分 Kernel / vLLM / 浮动支持 |
| 文档站 | mkdocs-material + GitHub Pages 自动部署 |
| License | MIT |
| 预期分数 | 60-75 / 100，不熔断，精度扣分 ≤ 3% |

## 4. 边界（明确**不**做的）

**赛题原文**（PDF P3 第 7 条）已明确禁止项，spec 不增不减：

- ❌ 修改采样参数、截断输入、跳过长样本、过滤困难样本、**跳过层**（PDF P3.1）
- ❌ 预缓存测试集与答案、预生成中间结果（PDF P3.2）
- ❌ 绕开统一服务接口 / 评测流程 / 资源统计路径；引入外援模型（PDF P3.3）
- ❌ 严禁修改模型结构、替换权重、改变推理语义或输出口径；后训练/蒸馏/微调（PDF P3.3.(1)）
- ❌ **量化操作边界**（PDF P3.3.(2)）：权重加载前/后、服务初始化、正式推理前对权重的**持久化**量化、结构化剪枝、权重重排压缩、**模型图重构**（指**权重层级的图重写**，如 ONNX 重导出 / TensorRT engine 持久化；**不**指运行时 cudagraph / torch.compile graph capture）、模型格式转换、可复用量化权重缓存
- ❌ **任何形式剪枝**（PDF P3.3.(3)）：结构化 / 非结构化 / 动态通道跳过 / 动态层跳过 / 注意力头裁剪 / token pruning / early-exit
- ❌ 推理过程生成可复用新模型 / 量化 / 剪枝 / 压缩权重文件（PDF P3.3.(4)）
- ❌ **batch scheduler 相关**（PDF P5 lock-list）：`--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` / 其它 batch scheduler 参数 + vLLM 内部 `v1/core/sched/*.py` 代码修改
- ❌ 投机解码：draft model / MTP / 多头预测 / 外挂小模型 / 自训练预测器 / 早退 draft / 预生成 token 缓存（PDF P5 锁）
- ❌ 改 bench percentile 统计口径 / 结果保存 / 解析脚本（PDF P5 锁）
- ❌ 改服务接口：served-model-name / API 路径 / 请求响应格式 / host:port（PDF P5 锁）

**赛题原文**（PDF P2 + P3.3.(2) 括号 + P6 + P7.3）已明确**允许**：

- ✅ 选手可叠加自定义层（PDF P2）：**安装 Python 包、编译 custom kernel**（在容器内，**不**从外网下载）
- ✅ **推理过程中非持久化、算子级低精度**（PDF P3.3.(2) 括号）：激活值动态量化、KV Cache 量化、kernel 内部临时类型转换、低精度矩阵乘法、**Attention 内核优化**
- ✅ KV Cache 分配机制、显存预算、块管理策略优化（PDF P6.2）
- ✅ Decode 阶段调度深度定制（PDF P6.1，**注**：与"batch scheduler 代码锁定"是 attention/sampling 调度，非 batch scheduler）
- ✅ DCU 资源组织调度适配（PDF P6.3）

**spec 自身硬约束**（赛题 + 团队新人约束合并）：
- 4 新人时间预算不支持"从零写新 attention backend"；优先复用 vLLM 已注册 backend + 算子级优化
- **不**改 vLLM 0.18.1 核心源码；任何 vLLM 源码 patch 必须有 PR diff 评审 + 至少 1 项精度验证
- 输出必须基于**标准自回归解码流程**（PDF P5 锁）

## 5. 关键技术决策

### 5.1 决策表

| 决策点 | 选择 | 理由 / 风险 |
|--------|------|-------------|
| **Decode attention 提速** | **优先复用 vLLM 已有的 `TRITON_ATTN` backend**（vllm.v1.attention.backends.triton_attn.TritonAttentionBackend），不自己写新 backend | vLLM 0.18.1 `AttentionBackendEnum` 已有 24 个值,含 `TRITON_ATTN` / `ROCM_ATTN` / `ROCM_AITER_FA` / `ROCM_AITER_UNIFIED_ATTN` / `FLASH_ATTN` / `FLASH_ATTN_DIFFKV` / `CPU_ATTN` / `TREE_ATTN` 等。新人写新 backend 风险大；先调通已存在的 TRITON_ATTN |
| **Custom backend 路径** | 真要扩展：调 `AttentionBackendEnum.register_backend()`，**或**改模块级 `_get_backend_priorities()`（非 RocmPlatform 方法）优先级，**或**用 `AttentionBackendEnum.CUSTOM` 槽位（已存在，CUSTOM = None） | 机制是"enum + register_backend()"，**不是**"丢个 .py 进 vllm/attention/backends/"。改平台优先级需改 vLLM 源码或 monkey-patch |
| **FA2 vs FA3** | **FA2 only**（用 ROCm/flash-attention fork） | FA3 是 Hopper WMMA 专用，DCU 不可用。ROCm/flash-attention 覆盖 MI200x/MI250x/MI300x/MI355x；CK backend 默认仅 fp16/bf16，FP8 走 aiter/Triton |
| **KV cache 优化** | **动态 FP8 量化**（per-head/per-token scale，**非持久化**） | 取决于 DCU SKU：CDNA3 (gfx942) 原生支持 FNUZ 变体（`__hip_fp8_e4m3_fnuz`）；CDNA2 (gfx90a) **不支持**。**§2 必查**；不行则改 INT8 或保留 bf16 |
| **torch.compile mode** | **`default` 或 `reduce-overhead`**，**不要 `max-autotune`** | `max-autotune` 在 ROCm 上不完整（很多模板 CUDA-only），可能静默回退 |
| **图优化边界** | **`torch.compile` + cudagraph 允许**（运行时 graph capture，**不**改模型权重/结构） | 赛题 P3.3.(2) "模型图重构" 限定在**权重层级持久化图重写**（ONNX 重导出 / TensorRT engine 持久化等）；**不**含运行时 cudagraph / torch.compile graph capture |
| **块管理调参** | `--block-size` 试 {8, 16, 32, 64, 128}；prefix-caching / chunked-prefill **可开**（非 batch scheduler 参数） | 有效值受硬件/算法约束，不是任意；与 KV 量化分块粒度**耦合**，P1 末必须做耦合矩阵 |
| **Python 路径优化** | `compilation_config.use_cudagraph=True` + `enforce_eager=False`，decode ≥ 3 次 warmup | HIP graph 录制有效；注意 warmup 不足会 capture 失败 |
| **调度** | **不改** vLLM batch scheduler 代码；可优化 Decode 阶段 attention/sampling 调度（赛题 P6.1 允许） | 赛题禁止 + 能力不足 |
| **vLLM 镜像** | 用 `vllm/vllm-rocm` Docker tag，不用 CUDA 镜像；官方 Dockerfile.rocm 用 **ROCm 7.0** 基线（旧分支支持 5.7-6.4） | CUDA image 装 DCU 必失败；ROCm 版本要锁 |
| **不要漏的开关** | `--enable-prefix-caching`、`--enable-chunked-prefill`、`VLLM_USE_V1=1` | ⚠️ **没找到 `VLLM_ATTENTION_BACKEND` env var**（v0.18.1 `envs.py` 全文 grep 0 次命中），后端选择走 platform priority list + `register_backend()` |

### 5.2 vLLM 0.18.1 attention backend 真实机制（已源码验证）

- **枚举**：`AttentionBackendEnum` 在 `vllm/v1/attention/backends/registry.py`，**共 24 个值**
- **关键值**：`TRITON_ATTN` / `ROCM_ATTN` / `ROCM_AITER_FA` / `ROCM_AITER_UNIFIED_ATTN` / `FLASH_ATTN` / `FLASH_ATTN_DIFFKV` / `FLASHINFER` / `FLEX_ATTENTION` / `TORCH_SDPA` / `CPU_ATTN` / `TREE_ATTN` / `CUSTOM` (None)
- **平台选择**：
  - **模块级函数** `_get_backend_priorities()` 在 `vllm/platforms/rocm.py:309-352`
  - `RocmPlatform.get_attn_backend_cls()` 类方法在 `vllm/platforms/rocm.py:~432`
- **平台 plugin**：`PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"` (`vllm/plugins/__init__.py`)，通过 `importlib.metadata.entry_points()` 加载；平台用 `builtin_platform_plugins = {'tpu','cuda','rocm','xpu','cpu'}` + 外部 plugin
- **Pyproject entry-points**：只有 `vllm.general_plugins`（LoRA resolver），**没有** `vllm.platform_plugins` 或 `vllm.attention_backends` 显式声明（意味着第三方可以声明，会被 `load_plugins_by_group` 自动发现）
- **ROCM 默认 backend**：`_get_backend_priorities()` 默认返回 `[TRITON_ATTN]`；开启 `VLLM_ROCM_USE_AITER=1` 才注入 `ROCM_AITER_UNIFIED_ATTN` / `ROCM_AITER_FA`

### 5.3 FlashAttention ROCm 真相（已验证）

- 官方 fork：[ROCm/flash-attention](https://github.com/ROCm/flash-attention)
- **支持架构**：MI200x / MI250x / MI300x / MI355x / RDNA 3/4（**覆盖 CDNA2 gfx90a 和 CDNA3 gfx942**）
- **安装**：`pip install flash-attn --no-build-isolation`（需 ROCm 6.0+、PyTorch 2.2+）
- **两个 backend**：Composable Kernel (CK) 默认 / Triton via `aiter` package
  - **CK backend**：**仅 fp16/bf16**（不支持 FP8）
  - **Triton/aiter backend**：**支持 FP8**（设 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`）
- vLLM 的 `FLASH_ATTN` enum 值在 ROCm 上是这条 fork 的 wrapper

### 5.4 AMD ROCm FP8 支持（已验证）

来源：[AMD ROCm precision-support docs](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)

| 架构 | GPU | FP8 支持 | 格式 |
|------|-----|----------|------|
| CDNA3 (gfx942) | MI300A / MI300X / MI325X | ✅ 原生 | **FNUZ** 变体（`__hip_fp8_e4m3_fnuz`）|
| CDNA4 | MI350X / MI355X | ✅ 原生 | OCP 变体 |
| **CDNA2 (gfx90a)** | **MI210 / MI250 / MI250X** | **❌ 不支持** | — |
| CDNA1 | MI100 | ❌ | — |
| RDNA4 | RX 9070 / 9070XT | ✅ 原生 | OCP 变体 |

**关键**：FP8 FNUZ = **F**inite + **N**o inf + **U**nsigned zero（AMD 文档原文）。和 NVIDIA H100 的 OCP FP8 **不兼容**（无 inf、无 signed zero）。任何"FP8 KV cache" 方案在 DCU 上**需要确认是 FNUZ 还是 OCP 变体**，否则对不上。

### 5.5 Triton + DCU 真相（**部分未验证**，需 P0 跑最小 case）

- Triton 3.x 在 ROCm/HIP 上有 first-class backend（target `gfx90a` / `gfx942` / `gfx950`）
- 装 Triton 走 PyTorch 的 ROCm wheel 路径，**不要** `pip install triton[all]`
- **未验证的已知坑**（无 issue 链接 / 复现步骤，需 P0 跑最小 case 验证）：DCU 上 `tl.atomic_*` for FP8 和部分 `tl.dot` scale 组合的 bug
  - **P0 验证动作**：P0 期间在 DCU 跑 1 个 5 行 `triton.jit` matmul + FP8 store 最小 case，失败即降级 bf16 路线

### 优化方向砍到 **3 必做 + 3 stretch**

**必做**（P3 必须有进展；占 80% 提分空间）：
1. **KV cache 动态量化**（FP8 优先，CDNA2 退化 INT8）
2. **`torch.compile` (default) + `use_cudagraph=True` + 必杀开关全开**
3. **`--block-size` 调参 + prefix caching + chunked prefill**

**Stretch**（P3 末才能开；不及格则放弃）：
4. **Triton decode attention kernel 改造**（vLLM backend 替换）
5. **手写 attention backend 注册为 vLLM backend**（依赖 §2 第 4 项验证通过）
6. **FA2 之外的 attention 内核优化**（高 ROI 但高难度）

**理由**：6 个方向对 4 个新人 + 5-10h/周（队长）太激进，必做 3 项已足够冲击 60-75 分。Stretch 项"做不出来也不丢分"。

## 6. 可用 Skills & Tools

| 工具 / skill | 用途 | 何时用 |
|--------------|------|--------|
| `mmx-cli`（本地已装） | web search、查论文、生成架构图 | 调研期 |
| `context7` 插件 | 实时查 vLLM / PyTorch / Triton / transformers API | 任何要查 API 时 |
| `torch.profiler`（零安装） | 第一线 profile，任何性能问题 | 任何性能异常 |
| `rocprofv3` / `rocsys`（DCU 专用） | kernel 级 + timeline profile，需 `apt install rocprofiler` | 深度 profile |
| `superpowers:dispatching-parallel-agents` | 并行启动 3+ 个调研 subagent | 一次性调研 |
| `superpowers:writing-plans` | spec → 实施计划 | spec 审阅通过后 |
| `superpowers:subagent-driven-development` | subagent 写脚手架 / 测试 / 文档 | 编码期 |
| `superpowers:test-driven-development` | kernel 单元测试先写 | 任何 kernel 改造前 |
| `superpowers:systematic-debugging` | 性能变差 / SLA 熔断排查 | "为什么变慢了"问题 |
| `superpowers:verification-before-completion` | 任何"优化提速了"声称前必验证 | P3+ 提分时 |
| `superpowers:using-git-worktrees` | 3 条 owner 线用 worktree 隔离 | P1 起 |

## 7. 团队 & 角色

| 角色 | 谁 | 周投入 | 主要交付 |
|------|----|----|----------|
| 🅰 队长 + Profiling & 集成 owner | recoletas | 5-10 h | 周主持、决策、3 档压测脚本、最终整合、文档收尾 |
| 🅱 Kernel owner | 队员 A | 8-12 h | Triton kernel 模板（stretch）；P3 进展不行则降级 |
| 🅲 vLLM & KV cache owner | 队员 B | 8-12 h | vLLM 源码笔记、KV 量化、块管理调参 |
| 🅳 浮动支持 / QA | 队员 C | 3-5 h | 跑回归、复核数字、文档校对 |

**4 人 + AI 辅助 ≈ 5-6 人等效生产力**。

### 决策机制
- **技术分歧**：3 owner 投票，多数决；平票 → 队长最终拍板
- **决策超时**：单议题阻塞超过 24h → 升级到下次全组 sync
- **赛题边界**（修改 scheduler / 投机解码 / 持久化量化等）→ 一票否决，**任何 owner 可触发升级**
- **角色变更**（人员退出 / 换 owner）→ 全组 sync 决议

### 协作机制
- **每周 1 次同步会**（30 min）：上周交付、本周计划、阻塞
- **PR 全部进 GitHub**：每个 owner 1 个 worktree
- **轻量进度交流**：每周 standup 写入 `docs/weekly/progress.md`（4 人各 1 行）
- **共享调研笔记 / grep 技巧**：写到 `docs/learning.md`（活文档，持续追加）

## 8. 文档架构

**单层 mkdocs-material 站，部署到 GitHub Pages**。

```
docs/
├── index.md                 # 文档站首页
├── learning.md              # 找学习资料的方法 (vLLM / AMD / DCU 关键文件 + grep 技巧)
├── specs/                   # 设计 / 决策 / 计划
├── decisions/               # ADR —— P0+ 按需建
├── weekly/progress.md       # 轻量 standup 模板 —— P1+ 用
├── ai-prompts/              # 共享 prompt 库 —— P3+ 按需建
└── appendices/              # subagent 调研报告 —— 一次性建
```

> **精简原则**：不预建空 md。内容从 spec 长出，按需建子目录。

## 9. 推进原则（**软** Phase，硬卡门）

### 核心原则
1. **基础优先**：4 人必须都能讲清"Prefill/Decode/KV cache/PagedAttention"，过不了这关不进 P3
2. **ROI 排序**：每个优化点必须先有"为什么能提分"假设 + "怎么测"方案，没假设不开工
3. **小步快跑**：每改一处跑 1 次 bench，找不到 1.05× 提速立刻回退
4. **AI 输出必审**：subagent 写的代码 / 文档必须经过"读 + 跑 + 对比 baseline"三关
5. **集成是瓶颈**：P3 起每周 1 次集成日，所有 owner 合到一起跑 3 档

### Phase 划分（9 周预算）

| Phase | 目标 | 出口（CP） | 时长 |
|-------|------|-----------|------|
| **P0** | §2 4 项未知全部确认 + 建仓库 + 各自看 1 篇 vLLM 论文 + 各 owner 离线练手任务 | **CP0**（硬卡） | 1.5 周（离线） |
| **P1**（基础统一） | 4 人能讲清 LLM 推理基础 + DCU/HIP 区别 + vLLM 0.18.1 架构 + Triton 5 行 FP8 matmul smoke test | **CP1**（硬卡） | 1.5 周 |
| **P2**（baseline 锁定 + 调研） | 三档 baseline 数字锁定，误差 < 5% | **CP2**（硬卡） | 1.5 周 |
| **P3**（优化试错） | 必做 3 项至少有 1 项在 1 档上提分 ≥ 10% | **CP3** | 3.5 周（**锁死**） |
| **P4**（集成 + 精度） | 3 档 + 4 类任务精度扣分 ≤ 3% + 1 次干净全量编译演练 | **CP4** | 0.5 周 |
| **P5**（提交冲刺） | 提交材料齐全，演练成功 | **CP5** | 0.5 周 |
| **buffer** | Phase 滑期用 | — | 0.5 周 |
| **总计** | — | — | **9.5 周** |

**对照用户预算 8-10 周**：spec 落 9.5 周（取下沿），留 0.5-1.5 周 buffer 给突发（队员退 / DCU 不到位 / 编译撞墙）。

### P0 离线期间具体任务
- **队长（5-10h/周 × 1.5 周 = 8-15h）**：通读 `qwen_use.pdf` 1 遍 + 写完 AGENTS.md + 用 `mmx-cli` 查 1 篇 vLLM 论文 + 跑 §2 4 项验证的协调
- **队员 A / Kernel（8-12h/周 × 1.5 周 = 12-18h）**：Triton 官方 tutorial 跑通 vector_add / softmax / fused attention 3 个例子；提交 1 个练习 PR
- **队员 B / vLLM（8-12h/周 × 1.5 周 = 12-18h）**：精读 vLLM 0.18.1 `v1/kv_cache_interface.py` + `attention/backends/`，写 1 页阅读笔记落 `docs/decisions/0006-vllm-readmap.md`
- **队员 C / 浮动（3-5h/周 × 1.5 周 = 5-8h）**：本地 `vllm serve` + `vllm bench serve` 命令跑通（GPU 或 CPU mock），熟悉 vllm bench 输出格式

**注**：P0 任务按 5-12h/周 配比；与 §7 周投入承诺**整体一致**，但**不平衡**（队员 C 1.5 周仅 5-8h 是 bus factor 1，P3 起把"调研笔记 + standup owner"主责任迁到队员 B）。

### Phase 跳过规则（硬化）
- CP0 不通过 → 不开 P1
- CP1 不通过 → P2 推迟（哪怕超周界）
- CP2 不通过 → P3 不开
- P3 任何 1 周内 0 bench 进展 → 立刻开 spec §10 应急"砍必做到 2 项"

## 10. 风险与对策

| 风险 | 信号 | 对策 |
|------|------|------|
| **SLA 超基线（≤ Baseline × 1.5）** — TTFT P99 / TPOT P99 任一超阈 | 评测平台 P99 统计超阈 | 立即回退当周优化点；不抢进度；TTL P99 按档独立判断，TPOT P99 全局统一（PDF P9.3.(6)） |
| **DCU 是 CDNA2 (gfx90a)，无原生 FP8** | §2 验证 | KV cache 改用 INT8 动态量化或保留 bf16；放弃 FP8 路线 |
| **DCU 不是 vLLM 0.18.1 支持的型号** | `vllm serve` 启动报错 | 立即联系赛方，**不绕** —— 赛题第 7 条第 3 款 |
| **vLLM custom backend 接入路径不通** | §2 验证 | 砍掉 stretch 项 4-5；Kernel 路线降级为"torch.compile + 块管理" |
| **官方 baseline 数字不公开**（PDF P9.3.(7) 说"评测时严格锁定"暗示赛方有 baseline） | 询问赛方无回 | CP2 改为"自测 baseline ±5% 复现稳定"；**60-75 估算改为按比例而非绝对分** |
| **LongBench / RULER 测试集开发期不可下载** | 询问赛方无回 | bench script 改用"自造 100 样本本地验证 + 评测平台每周 1 次回归" |
| **Triton kernel 编译/运行失败** | DCU 上 `triton.jit` 异常 | fallback → vLLM `ROCM_ATTN` backend 或 `torch.compile`，**不放弃 Triton 但退守默认 backend** |
| **KV 量化精度塌方** | OpenCompass Δ > 3% | 退回 bf16，保留其他优化点 |
| **ROCm wheel / Docker image 不匹配** | `import torch` 找不到 HIP 后端 | 用 `vllm/vllm-rocm` Docker tag；装 PyTorch 时用 `+rocmX.Y` index URL |
| **队友时间进一步缩水** | 周会 2 周连续 0 出席 | 合并 owner 角色，必做 3 项降级到 2 项 |
| **优化反而变慢** | bench 数据反向 | 48h 找不到原因 → 回退到上一稳定版 |
| **误改赛题锁定参数** | 评测平台拒绝 | PR 检查清单 + 跑通 `vllm serve --max-model-len 32768 --max-num-seqs 1` 锁定参数 |
| **编译失败撞 P5 截止** | vLLM 全量编译 4-12h 实测 | 演练放在 P4 末而非 P5；保留上次成功构建的 Docker 镜像 |
| **决赛多卡扩展**（PDF P11 提决赛多卡分布式） | 初赛结束 | P3 末评估 1 个 stretch 项：tensor parallel 路径调研（不实现） |
| **评测黑盒：请求顺序/时间戳不可见** | PDF P11 第 3 款 | prefix-caching 调参策略要保守；不要假设请求可重放 |
| **队员 C 退出（bus factor 1）** | 队员 C 连续 2 周 0 交付 | P0 末把"调研笔记 + standup owner"主责任迁到队员 B；队员 C 仅保留 QA |
| **AI 代码未及时审阅合 main** | `git log` 找到无 reviewer 的 AI commit | AGENTS.md 已有 3 关（读 / 跑 / 对比 baseline）；spec 加 24h 内未审 = revert |

## 11. 完工标准（Phase 5 结束）

**仓库里**：
- ✅ `AGENTS.md` 完整
- ✅ `mkdocs.yml` + `docs/index.md` 文档站可本地预览
- ✅ GitHub Pages 部署通过 Actions
- ✅ `benchmarks/{baseline,optimized}/` 三档 JSON
- ✅ `src/` 下 KV 量化模块 / torch.compile 配置 / block-size 调参脚本可一键加载
- ✅ 1 次干净的全量编译 + 跑通演练（**P4 末完成，P5 只重跑**）

### 赛题第 12-15 条提交材料清单

| 赛题条款 | 提交材料 | Owner | 截止 | 状态 |
|----------|----------|-------|------|------|
| §12 源码 | 完整源代码 + 编译脚本 + 注释 | 队长（合）+ 全员 | P5 末 | ☐ |
| §13 环境变量 | `reports/env-vars.md`：变量名/取值/作用 | 队员 C | P4 末 | ☐ |
| §14 优化方案 | `reports/optimization-plan.md`：方法/路线/贡献分析/优化点汇总表 | 队长 | P4 末 | ☐ |
| §15 第三方引用 | `README.md` + `reports/submission-readme.md` | 队长 | P5 末 | ☐ |
| §15 源码 README | 源码 `README.md` 头部 | 队长 | P5 末 | ☐ |

**预期分数**：60-75 / 100（赛题第 9 条第 4 款公式估算；**待 §2 第 2 项确认后修正**）。

**Baseline 数字公开 vs 不公开 → 评分场景表**：

| 场景 | 信号 | 分数区间 | 策略 |
|------|------|----------|------|
| **赛方下发官方 baseline** | §2 验证通过 | 60-75 | 按 3 必做 + 1 stretch 推进 |
| **赛方部分公开（如仅给单档）** | §2 部分通过 | 45-65 | 集中攻已公开档，其余按比例估算 |
| **完全自测 baseline**（赛方不公开） | §2 验证失败 | 30-55（**估算失锚**） | 砍必做到 2 项；放弃 8k-16k 档（占 50% 权重） |

## 12. 剩余硬卡门

1. **P0 验证**：DCU 硬件到位后跑 §2 的 4 项（DCU SKU / 官方 baseline / LongBench+RULER 访问 / custom backend 路径）
2. **验证回写**：P0 结果落本 spec §2 表格 + §5 决策表（如选型有变）
3. **P0 末调研**：vLLM backend 接入点 / DCU 性能特征 / KV 量化方案对比 → `docs/appendices/`

