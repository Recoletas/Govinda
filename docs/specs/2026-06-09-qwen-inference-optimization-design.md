# 2026-06-09 Qwen 推理服务优化比赛 · 设计文档

> **状态**：草案 v1 · **作者**：队长（recoletas）· **最后更新**：2026-06-09
> **配套文档**：[`AGENTS.md`](../../AGENTS.md) · [`README.md`](../../README.md)

---

## 1. Context

参加 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯），赛题为"基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化"。初赛单卡、并发=1、长上下文场景（4k-32k 三档）。**团队 4 人均为新人，8-10 周时间预算（从 2 周后有时间为起算点），目标稳定拿 60-75 / 100 分而非冲击高分**。详见 [`qwen_use.pdf`](https://pra.xtnl.org.cn/) 赛题原始材料。

## 2. 用户决策（已确认）

| 项 | 决策 |
|---|---|
| 项目仓库 | `/home/recoletas/Govinda/`（GitHub: `Recoletas/Govinda`） |
| 团队组成 | 4 人队（队长 + 3 队员），全新人（CS 课程级 DL） |
| 时间预算 | 8-10 周，从"用户有时间的那个点"起算；前 2 周完全离线 |
| DCU 硬件 | 1-2 周后到位；之前无硬件 |
| AI 用法 | 当开发助手（写代码/调研/文档），**不接进推理路径** |
| 角色分配 | 队长 = Profiling & 集成 owner（5-10h/周）；其余 3 人分 Kernel / vLLM / 浮动支持 |
| 文档站 | mkdocs-material + GitHub Pages 自动部署 |
| License | MIT |
| 预期分数 | 60-75 / 100，不熔断，精度扣分 ≤ 3% |

## 3. 边界（明确**不**做的）

来自 [赛题第 7 条](https://pra.xtnl.org.cn/) 加上时间约束的合并：

- ❌ 自定义 CUDA / HIP kernel（高时间投入，新人易翻车；用 vLLM 已注册 backend 替换点）
- ❌ 修改 vLLM 0.18.1 的 `v1/core/sched/*.py`（赛题第 7 条 + 能力不足）
- ❌ 调整 `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens`（赛题锁定）
- ❌ 投机解码、MTP、early-exit draft、外挂小模型（赛题第 7 条第 1 款）
- ❌ 持久化量化、结构化/非结构化剪枝、权重重排（赛题第 7 条第(2)/(3) 款）
- ❌ 训练 / 微调 / 蒸馏 / 后训练（赛题第 7 条第(1) 款）
- ❌ 预缓存测试集 / 预生成中间结果（赛题第 7 条第 2 款）
- ❌ 引入题目规定外的辅助模型（赛题第 7 条第 3 款）

✅ **允许**：KV cache 动态量化、activation 动态量化、kernel 内低精度计算、Python 包安装与 custom kernel 编译（在容器内）。

## 4. 关键技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Decode attention 提速 | **Triton standalone kernel** + 通过 vLLM `attention/backends/` 注册点接入 | 不动 vLLM 主体；Triton 跨平台适合新人 |
| KV cache 优化 | **动态 FP8 量化**（per-head / per-token scale，**非持久化**） | 显存省、速度换；FP8 DCU 支持好；非持久化不违规 |
| 块管理调参 | 调整 `--block-size`（默认 16 → 试 8/32/64） | 调参成本低，长上下文可能受益 |
| 算子融合 | `torch.compile(mode="max-autotune")` 不开 dynamo cache | vLLM 0.18.1 已支持，零代码改动 |
| Python 路径优化 | 预录制 **CUDA / HIP graph** 录制 decode 步骤 | TTFT 主要对手是 launch 延迟 |
| 调度 | **不改** vLLM scheduler 代码 | 赛题禁止 + 能力不足 |
| 并行策略 | 单卡（初赛赛题要求） | 不涉及张量并行 / 流水线并行 |

## 5. 可用 Skills & Tools

| 工具 / skill | 用途 | 何时用 |
|--------------|------|--------|
| `mmx-cli` (本地装) | web search、查论文、生成架构图 | 调研期 |
| `context7` 插件 | 实时查 vLLM / PyTorch / Triton / transformers API | 任何要查 API 时 |
| `superpowers:dispatching-parallel-agents` | 并行启动 3 个调研 subagent | 一次性调研 |
| `superpowers:writing-plans` | 把 spec 转成可执行实施计划 | spec 审阅通过后 |
| `superpowers:subagent-driven-development` | subagent 写脚手架 / 测试 / 文档 | 编码期 |
| `superpowers:test-driven-development` | kernel 单元测试先写 | 任何 kernel 改造前 |
| `superpowers:systematic-debugging` | 性能变差 / SLA 熔断时排查 | 任何"为什么变慢了"问题 |
| `superpowers:verification-before-completion` | 任何"优化提速了"声称前的强制验证 | W6+ 提分时 |
| `superpowers:using-git-worktrees` | W4+ 三条 owner 线用 worktree 隔离 | W4 起 |

## 6. 团队 & 角色

| 角色 | 谁 | 周投入 | 主要交付 |
|------|----|----|----------|
| 🅰 队长 + Profiling & 集成 owner | recoletas（用户） | 5-10 h | 周主持、决策、3 档压测脚本、最终整合、文档收尾 |
| 🅱 Kernel owner | 队员 A | 8-12 h | Triton kernel 模板 + attention backend 接入 |
| 🅲 vLLM & KV cache owner | 队员 B | 8-12 h | vLLM 源码笔记、KV FP8 量化、块管理调参 |
| 🅳 浮动支持 / QA | 队员 C | 3-5 h | 跑回归、复核数字、文档校对、bus factor |

**4 人 + AI 辅助 ≈ 5-6 人等效生产力**。

### 协作机制
- **每周 1 次同步会**（30 min，固定时间）：上周交付、本周计划、阻塞
- **PR 全部进 GitHub**：每个 owner 1 个 worktree
- **共享笔记**：每周 1 篇 1-page 周报，owner 轮流写

## 7. 文档架构

**单层 mkdocs-material 站，部署到 GitHub Pages**。

```
docs/
├── index.md                 # 文档站首页
└── specs/                   # 设计 / 决策 / 计划文档
    └── 2026-06-09-qwen-inference-optimization-design.md   # 本文件
```

> **精简原则**：不预建 50 个空 md。新内容从 spec 链接长出，按需建子页。`AGENTS.md` 顶在仓库根，给 AI 助手看。

### GitHub Pages 自动部署
- 配置：`.github/workflows/docs.yml`（推 main → 部署）
- 部署前需 GitHub 仓库 Settings → Pages → Source: GitHub Actions 一次性设置

## 8. 推进原则（**软**的 Phase 划分，不画甘特图）

### 核心原则
1. **基础优先**：3 周内 4 人必须都能讲清"Prefill/Decode/KV cache/PagedAttention"，过不了这关不进 Phase 2
2. **ROI 排序**：每个优化点必须先有"为什么能提分"假设 + "怎么测"方案，没假设不开工
3. **小步快跑**：每改一处跑 1 次 bench，找不到 1.05× 提速立刻回退
4. **AI 输出必审**：subagent 写的代码 / 文档必须经过"读 + 跑 + 对比 baseline"三关（见 `AGENTS.md`）
5. **集成是瓶颈**：Phase 3 起每周 1 次集成日，所有 owner 合到一起跑 3 档

### Phase 划分
| Phase | 目标 | 出口（CP） |
|-------|------|-----------|
| **P0**（现在到 DCU 到位） | 建仓库 + AGENTS.md + 各自看 1 篇 vLLM 论文 | 仓库可 mkdocs build，文档站可本地预览 |
| **P1**（基础统一） | 4 人能讲清 LLM 推理基础 + DCU/HIP 区别 + vLLM 0.18.1 架构 | **CP1** |
| **P2**（baseline 锁定 + 调研） | 三档 baseline 数字锁定，误差 < 5% | **CP2** |
| **P3**（优化试错） | 至少 1 个优化点在 1 档上提分 ≥ 10% | **CP3** |
| **P4**（集成 + 精度） | 3 档 + 4 类任务精度扣分 ≤ 3% | **CP4** |
| **P5**（提交冲刺） | 提交材料齐全，1 次演练成功 | **CP5** |

**允许**：跳过某 Phase、合并、回头补——基于每周 sync 会的实际状况调整。

## 9. 风险与对策

| 风险 | 信号 | 对策 |
|------|------|------|
| DCU 平台 Triton 不兼容 | standalone kernel 跑失败 | 立即放弃 Triton 路线，资源全转 KV 量化 + torch.compile |
| KV 量化精度塌方 | OpenCompass Δ > 3% | 退回 bf16 量化，保留其他优化点 |
| 队友时间进一步缩水 | 周会 2 周连续 0 出席 | 合并 owner 角色，Kernel 路线降级为"torch.compile + 块管理" |
| 编译失败 | 评测平台报错 | 每周 1 次 clean rebuild 演练 |
| 优化反而变慢 | bench 数据反向 | 48h 找不到原因则回退到上一稳定版 |
| 误改赛题锁定参数 | 评测平台拒绝 | PR 检查清单 + 跑通 `vllm serve --max-model-len 32768` 锁定参数 |

## 10. 完工标准（Phase 5 结束）

**仓库里**：
- ✅ `AGENTS.md` 完整（赛题约束 + AI 约定 + 必读文件）
- ✅ `mkdocs.yml` + `docs/index.md` 文档站可本地预览
- ✅ `benchmarks/{baseline,optimized}/` 三档 JSON
- ✅ `src/` 下自定义 attention backend + KV 量化模块可一键加载
- ✅ `reports/{optimization-plan.md, env-vars.md, submission-readme.md}` 齐全
- ✅ GitHub Pages 部署通过 Actions
- ✅ 1 次干净的全量编译 + 跑通演练

**预期分数**：60-75 / 100（赛题第 9 条第 4 款公式估算）。

## 11. 下一步

1. **本次会话内**：git init + 首次 commit（包含本 spec）
2. **并行启动 3 个调研 subagent**（A: vLLM 0.18.1 必读文件清单 / B: DCU + Triton / FlashAttention 选型 / C: KV cache FP8 在 DCU 上的可行性）
3. subagent 报告回写进 spec 附录（`docs/specs/appendices/`）
4. 用户审阅本 spec → 调 `superpowers:writing-plans` 转实施计划
5. Phase 0 启动
