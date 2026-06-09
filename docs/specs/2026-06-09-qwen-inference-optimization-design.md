# 2026-06-09 Qwen 推理服务优化比赛 · 设计文档

> **状态**：v2（subagent 审查后修订）· **作者**：队长 recoletas · **最后更新**：2026-06-09
> **配套**：[`AGENTS.md`](../../AGENTS.md) · [`README.md`](../../README.md) · [v1 commit](https://github.com/Recoletas/Govinda/commit/8b763b4)
> **审查来源**：3 subagent 并行报告（vLLM internals / DCU + Triton + FlashAttention / spec 完整性）

---

## 1. Context

参加 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯），赛题为"基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化"。初赛单卡、并发=1、长上下文场景（4k-32k 三档）。**团队 4 人均为新人，8-10 周时间预算（从 2 周后有时间为起算点），目标稳定拿 60-75 / 100 分而非冲击高分**。

## 2. 待验证未知项（**Phase 0 硬卡门**，确认前不开 Phase 1）

| 项 | 不确定点 | 影响 | 验证方法 |
|---|---------|------|----------|
| **DCU SKU** | K100 (CDNA2 / gfx90a) 还是 Z100 (CDNA3 / gfx942) | FP8 KV cache 是否可行（CDNA2 无原生 FP8，CDNA3 有） | `rocminfo` 或 `python -c "import torch; print(torch.cuda.get_device_properties(0))"` |
| **官方 baseline 数字** | 赛方是否下发给我们，还是只能自测 | 60-75 估算是否成立；SLA 自比较还是绝对值 | 询问赛方 / 赛方最新通知 |
| **LongBench / RULER 测试集** | 开发期可下载 vs 仅评测平台可见 | 1.05× 提速回退规则能否本地验证 | 询问赛方 / 赛方文档 |
| **vLLM 0.18.1 custom backend 路径** | entry-point 插件 vs 目录 drop-in | 整条 Kernel 战略是否成立 | DCU 上手后读 `vllm/platforms/rocm.py` + `vllm/attention/selector.py` |

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

与赛题第 7 条 + 时间约束的合并：

- ❌ 自定义 CUDA / HIP kernel（**仅允许使用 vLLM 已注册 backend 替换点**；如该路径不可行则放弃整条路线）
- ❌ 修改 vLLM 0.18.1 的 `v1/core/sched/*.py`
- ❌ 调整 `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens`
- ❌ 投机解码、MTP、early-exit draft、外挂小模型
- ❌ 持久化量化、结构化/非结构化剪枝、权重重排
- ❌ 训练 / 微调 / 蒸馏 / 后训练
- ❌ 预缓存测试集 / 预生成中间结果
- ❌ 引入题目规定外的辅助模型

✅ **允许**：KV cache 动态量化、activation 动态量化、kernel 内低精度计算、Python 包安装与 custom kernel 编译（在容器内）。

## 5. 关键技术决策

| 决策点 | 选择 | 理由 / 风险 |
|--------|------|-------------|
| **Decode attention 提速** | **Triton kernel + vLLM backend 注册点接入** | Triton 是 DCU (ROCm) **原生支持**的路径，**不是**风险路径。失败 fallback → vLLM `ROCM_ATTN_V1` backend 或 `torch.compile` |
| **FA2 vs FA3** | **FA2 only**（vLLM 已带 ROCm 后端） | FA3 是 Hopper WMMA 专用，DCU 不可用 |
| **KV cache 优化** | **动态 FP8 量化**（per-head/per-token scale，**非持久化**） | 取决于 DCU SKU：CDNA3 (gfx942) 原生支持，CDNA2 (gfx90a) 只能模拟且慢 → **§2 必查**；不行则改 INT8 或保留 bf16 |
| **torch.compile mode** | **`default` 或 `reduce-overhead`**，**不要 `max-autotune`** | `max-autotune` 在 ROCm 上不完整（很多模板 CUDA-only），可能静默回退 |
| **块管理调参** | `--block-size` 试 {8, 16, 32, 64, 128} | 有效值受硬件/算法约束，不是任意 |
| **Python 路径优化** | `compilation_config.use_cudagraph=True` + `enforce_eager=False`，decode ≥ 3 次 warmup | HIP graph 录制有效；注意 warmup 不足会 capture 失败 |
| **调度** | **不改** vLLM scheduler 代码 | 赛题禁止 + 能力不足 |
| **vLLM 镜像** | 用 `vllm/vllm-rocm` Docker tag，不用 CUDA 镜像 | CUDA image 装 DCU 必失败 |
| **不要漏的开关** | `--enable-prefix-caching`、`--enable-chunked-prefill`、`VLLM_USE_V1=1`、`VLLM_ATTENTION_BACKEND` | 这些是 vLLM 0.18.1 现成的开关，先全开再选 |

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
| 🅳 浮动支持 / QA / 录屏 | 队员 C | 3-5 h | 跑回归、复核数字、文档校对、bus factor 录屏 |

**4 人 + AI 辅助 ≈ 5-6 人等效生产力**。

### 决策机制（**之前缺，补上**）
- **技术分歧**：3 owner 投票，多数决；平票 → 队长最终拍板
- **决策超时**：单议题阻塞超过 24h → 升级到下次全组 sync
- **赛题边界**（修改 scheduler / 投机解码 / 持久化量化等）→ 一票否决，**任何 owner 可触发升级**
- **角色变更**（人员退出 / 换 owner）→ 全组 sync 决议

### 协作机制
- **每周 1 次同步会**（30 min）：上周交付、本周计划、阻塞
- **每周 owner 各录 1 段 ≤ 30min 讲解自己负责模块**（落 `docs/recordings/`，bus factor 备份）
- **PR 全部进 GitHub**：每个 owner 1 个 worktree
- **共享笔记**：每周 1 篇 1-page 周报，owner 轮流写

## 8. 文档架构

**单层 mkdocs-material 站，部署到 GitHub Pages**。

```
docs/
├── index.md                 # 文档站首页
├── specs/                   # 设计 / 决策 / 计划
├── decisions/               # ADR —— P0+ 按需建
├── weekly/                  # 周报 —— P1+ 按需建
├── recordings/              # owner 录屏 —— P1+ 按需建
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

### Phase 划分
| Phase | 目标 | 出口（CP） | 时长 |
|-------|------|-----------|------|
| **P0** | **§2 4 项未知全部确认** + 建仓库 + 各自看 1 篇 vLLM 论文 + 各 owner 离线练手任务 | **CP0**（硬卡） | 2 周（离线） |
| **P1**（基础统一） | 4 人能讲清 LLM 推理基础 + DCU/HIP 区别 + vLLM 0.18.1 架构 | **CP1**（硬卡） | 1-2 周 |
| **P2**（baseline 锁定 + 调研） | 三档 baseline 数字锁定，误差 < 5% | **CP2**（硬卡） | 1-2 周 |
| **P3**（优化试错） | 必做 3 项至少有 1 项在 1 档上提分 ≥ 10% | **CP3** | 3-4 周 |
| **P4**（集成 + 精度） | 3 档 + 4 类任务精度扣分 ≤ 3% | **CP4** | 1 周 |
| **P5**（提交冲刺） | 提交材料齐全，1 次演练成功 | **CP5** | 1 周 |

### P0 离线期间具体任务（**之前缺，现在补**）
- **队长**：通读 `qwen_use.pdf` 1 遍 + 写完 AGENTS.md + 用 `mmx-cli` 查 1 篇 vLLM 论文
- **队员 A**：Triton 官方 tutorial 跑通 vector_add / softmax / fused attention 3 个例子；提交 1 个练习 PR
- **队员 B**：精读 vLLM 0.18.1 `v1/kv_cache_interface.py` + `attention/backends/`，写 1 页阅读笔记落 `docs/decisions/0005-vllm-readmap.md`
- **队员 C**：本地 `vllm serve` + `vllm bench serve` 命令跑通（GPU 或 CPU mock），熟悉 vllm bench 输出格式

### Phase 跳过规则（硬化）
- CP0 不通过 → 不开 P1
- CP1 不通过 → P2 推迟（哪怕超周界）
- CP2 不通过 → P3 不开

## 10. 风险与对策

| 风险 | 信号 | 对策 |
|------|------|------|
| **DCU 是 CDNA2 (gfx90a)，无原生 FP8** | §2 验证 | KV cache 改用 INT8 动态量化或保留 bf16；放弃 FP8 路线 |
| **DCU 不是 vLLM 0.18.1 支持的型号** | `vllm serve` 启动报错 | 立即联系赛方，**不绕** —— 赛题第 7 条第 3 款 |
| **vLLM custom backend 接入路径不通** | §2 验证 | 砍掉 stretch 项 4-5；Kernel 路线降级为"torch.compile + 块管理" |
| **官方 baseline 数字不公开** | 询问赛方无回 | CP2 改为"自测 baseline ±5% 复现稳定"，60-75 估算修正 |
| **LongBench / RULER 测试集开发期不可下载** | 询问赛方无回 | bench script 改用"自造 100 样本本地验证 + 评测平台每周 1 次回归" |
| **Triton kernel 编译/运行失败** | DCU 上 `triton.jit` 异常 | fallback → vLLM `ROCM_ATTN_V1` backend 或 `torch.compile`，**不放弃 Triton 但退守默认 backend** |
| **KV 量化精度塌方** | OpenCompass Δ > 3% | 退回 bf16，保留其他优化点 |
| **ROCm wheel / Docker image 不匹配** | `import torch` 找不到 HIP 后端 | 用 `vllm/vllm-rocm` Docker tag；装 PyTorch 时用 `+rocmX.Y` index URL |
| **队友时间进一步缩水** | 周会 2 周连续 0 出席 | 合并 owner 角色，必做 3 项降级到 2 项 |
| **优化反而变慢** | bench 数据反向 | 48h 找不到原因 → 回退到上一稳定版 |
| **误改赛题锁定参数** | 评测平台拒绝 | PR 检查清单 + 跑通 `vllm serve --max-model-len 32768` 锁定参数 |
| **编译失败** | 评测平台报错 | 每周 1 次 clean rebuild 演练 |

## 11. 完工标准（Phase 5 结束）

**仓库里**：
- ✅ `AGENTS.md` 完整
- ✅ `mkdocs.yml` + `docs/index.md` 文档站可本地预览
- ✅ GitHub Pages 部署通过 Actions
- ✅ `benchmarks/{baseline,optimized}/` 三档 JSON
- ✅ `src/` 下 KV 量化模块 / torch.compile 配置 / block-size 调参脚本可一键加载
- ✅ 1 次干净的全量编译 + 跑通演练

### 赛题第 12-15 条提交材料清单（**之前缺 owner，现在补**）

| 赛题条款 | 提交材料 | Owner | 截止 | 状态 |
|----------|----------|-------|------|------|
| §12 源码 | 完整源代码 + 编译脚本 + 注释 | 队长（合）+ 全员 | P5 末 | ☐ |
| §13 环境变量 | `reports/env-vars.md`：变量名/取值/作用 | 队员 C | P4 末 | ☐ |
| §14 优化方案 | `reports/optimization-plan.md`：方法/路线/贡献分析/优化点汇总表 | 队长 | P4 末 | ☐ |
| §15 第三方引用 | `README.md` + `reports/submission-readme.md` | 队长 | P5 末 | ☐ |
| §15 源码 README | 源码 `README.md` 头部 | 队长 | P5 末 | ☐ |

**预期分数**：60-75 / 100（赛题第 9 条第 4 款公式估算；**待 §2 第 2 项确认后修正**）。

## 12. 下一步

1. **本会话内**：v2 spec 提交 + 等用户审阅
2. **用户审阅通过后** → 调 `superpowers:writing-plans` 转实施计划
3. **实施计划批准后** → 启动 Phase 0（含 §2 4 项验证）
4. **P0 验证结果回写本 spec**（如选型有变）→ §5 决策表更新
5. **P0 末** → subagent 并行调研（vLLM backend 接入点 / DCU 性能特征 / KV 量化方案对比），输出落 `docs/appendices/`

---

## 修订记录

- **v1** (`8b763b4`)：初稿
- **v2**（本版）：3 subagent 审查后修订
  - **新增 §2 待验证未知项 4 项硬卡门**（C B1-B4 + B FP8-SKU 风险）—— Phase 0 入口
  - **反转 §5 Triton fallback 方向**（B blocker：Triton 是支持路径，CUDA-only 才不可用）
  - **砍优化方向 6 → 3 必做 + 3 stretch**（C I1：4 新人 + 5-10h/周太激进）
  - **§5 `torch.compile` mode 改 `default`**（A + B：max-autotune 在 ROCm 不完整）
  - **§5 加 FA2 only 限定、vllm/vllm-rocm Docker**（B）
  - **§6 加 profiler 行**（B：`torch.profiler` / `rocprofv3` / `rocsys`）
  - **§7 加决策机制**（C B2：之前缺）
  - **§9 P0 加离线期间具体任务**（C B1：之前太空）
  - **§9 CP1/CP2 硬化**（C I3：可跳过是隐患）
  - **§11 加赛题 §12-15 owner 表**（C I2：之前缺）
  - **§10 加新风险**（ROCm wheel 不匹配 / CDNA SKU / baseline 不可见 / 测试集不可下载）
  - **§7 加 bus factor 录屏机制**（C I4）
