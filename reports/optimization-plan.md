# 提交材料 §14 · 优化方案 (方法 / 路线 / 贡献分析 / 优化点汇总)

> **提交日期**: 2026-06-09
> **赛方要求**: 赛题 §14
> **状态**: 草稿 (P4 末完成, P5 演练后定稿)
> **Owner**: 队长
> **配套**: [spec §3 / §5 / §10 / §11](../../docs/specs/2026-06-09-qwen-inference-optimization-design.md) / [ADR 0007](../../docs/decisions/0007-coupling-matrix.md) / [ADR 0008](../../docs/decisions/0008-blocksize-hypothesis.md) / [ADR 0013](../../docs/decisions/0013-torch-compile-roi.md)

## 1. 方法 / 路线

团队 4 人 (3 队员 + 队长), 9.5 周预算 (P0-P5 + buffer, spec §9), 锁定 **vLLM 0.18.1** 与 **DCU (HIP-based ROCm 7.0)** 平台。**AI (mmx-cli / subagent) 作为开发助手**, 不接入推理路径, 任何 AI 输出必经"读 + 跑 + 对比 baseline"三关 (spec §9 核心原则 4)。优化方向砍到 **3 必做 + 3 stretch** (必做占 80% 提分空间, stretch 失败不扣分): 必做聚焦 KV 量化、torch.compile + cudagraph、块管理调参, stretch 留给 Triton decode kernel 改造 / 自定义 attention backend 注册 / FA2 之外 attention 内核优化 (spec §5.1)。对位先导杯头部战队 (清华 / 中科院 / 上交) 假设 60-75 / 100 是合理目标分 (spec §1), 不与头部争 90+。

## 2. 贡献分析 (3 必做项)

### 2.1 Stream A · 块管理调参 (队员 B / vLLM owner)

通过 `--block-size` 扫描 (候选 {8, 16, 32, 64, 128}) + `--enable-prefix-caching` + `--enable-chunked-prefill` 三项组合, 降低 KV cache miss 率、复用 system prompt 公共前缀、避免长 prompt prefill 独占 decode 通道。块大小与 KV 量化粒度存在耦合 (5 × 4 = 20 cells 决策矩阵, ADR 0007), P0 0.1 DCU SKU 验证后预筛列再实测, 砍过的列不再测。预期 8k-16k 档 (50% 权重) 提分 5-15%, 4k-8k 与 16k-32k 档提分 3-8%。

### 2.2 Stream B · KV cache 动态量化 (队员 B / vLLM owner)

按 DCU SKU 分两路: **CDNA3 (gfx942) 走 FP8 FNUZ 动态量化**, per-head scale 优先 (per-token scale 在 batch=1 + 长 decode 场景抖动大, ADR 0007 决策准则 5); **CDNA2 (gfx90a) 无原生 FP8, 走 INT8 per-tensor fallback**。量化限定为**运行时非持久化** (赛题 P3.3.(2) 括号允许), 不修改权重文件 / 不结构化剪枝 / 不图重构 (spec §4)。OpenCompass Δ > 3% 立即回退 (spec §10 风险表)。预期显存降 30-50%, 长上下文档 (16k-32k) decode 阶段 HBM 带宽压力缓解。

### 2.3 Stream C · torch.compile + cudagraph (队员 A / Kernel owner)

`torch.compile(mode="default")` + `compilation_config.use_cudagraph=True` + `enforce_eager=False`, 搭配 `warmup_iters ≥ 3` (避免 capture 失败, spec §5.1)。**不**用 `max-autotune` (ROCm 上不完整, 可能静默回退)。cudagraph 录制 kernel launch 序列, 减少 Python 层调度开销 (TTFT 改善) 与 decode 阶段 kernel launch 延迟 (TPOT 改善)。5 实验设计见 ADR 0013, 每实验 × 3 档 × 50 prompts。8k-16k 档至少 5% 提速才值得集成 (ADR 0013 决策准则 3)。

## 3. 优化点汇总表 (3 档 × 3 必做项 = 9 格)

> **状态**: 全部 P3 中段填入, 当前"提升 %"列全"待填 P3 实测"。

| Tier | Stream | 优化点 | 状态 | 提升 % | 风险 | 备注 |
|------|--------|--------|------|--------|------|------|
| 4k-8k (20%) | A | block-size + prefix-caching + chunked-prefill | 待 P3A 实测 | 待填 | 块过小 metadata 开销 > 收益 (ADR 0007 #4) | 默认排除 block-size < 16 |
| 4k-8k (20%) | B | KV 动态量化 (FP8 per-head / INT8 per-tensor) | 待 P3B 实测 | 待填 | CDNA2 无 FP8, 退回 INT8 | 精度 Δ ≤ 3% 红线 |
| 4k-8k (20%) | C | torch.compile (default) + cudagraph | 待 P3C 实测 | 待填 | warmup 不足 capture 失败 | warmup_iters ≥ 3 |
| 8k-16k (50%) | A | block-size + prefix-caching + chunked-prefill | 待 P3A 实测 | 待填 | 同上 | 优先档 |
| 8k-16k (50%) | B | KV 动态量化 | 待 P3B 实测 | 待填 | 同上 | 优先档, per-head 优先 |
| 8k-16k (50%) | C | torch.compile (default) + cudagraph | 待 P3C 实测 | 待填 | 同上 | 优先档 |
| 16k-32k (30%) | A | block-size + prefix-caching + chunked-prefill | 待 P3A 实测 | 待填 | 长 prompt prefix 复用率需验 | 长 context 收益可能更大 |
| 16k-32k (30%) | B | KV 动态量化 | 待 P3B 实测 | 待填 | FP8 显存降 ~50% 是 HBM 救命 | per-token scale 抖动大 |
| 16k-32k (30%) | C | torch.compile (default) + cudagraph | 待 P3C 实测 | 待填 | 长 context graph capture 内存占用 | 启动时间 + 内存 trade-off |

## 4. 3 档权重 (per spec §9 / ADR 0007 #3)

| 档 | 占比 |
|----|------|
| 4k-8k | 20% |
| 8k-16k | 50% |
| 16k-32k | 30% |

**策略**: 8k-16k 档优先投入, 是 60-75 / 100 目标分的最大变量; 4k-8k 档提分空间小 (输出短) 但易保分; 16k-32k 档提分空间大 (KV 量化 + prefix-caching 双重收益) 但风险高, 走 P3 中段实测再定夺。

## 5. SLA 与精度约束 (per ADR 0007 #1 #2)

| 约束项 | 阈值 | 触发动作 |
|--------|------|----------|
| TTFT P99 | ≤ Baseline × 1.5 (按档独立, spec §10) | 破阈 = 该档 0 分, 立即回退当周优化点 |
| TPOT P99 | ≤ Baseline × 1.5 (全局统一, PDF P9.3.(6)) | 破阈 = 0 分, 立即回退 |
| OpenCompass Δ | ≤ 3% | Δ > 3% 触发回退, 保留其他优化点 |
| 启动时间 | + < 60s (ADR 0013 决策准则 4) | 阻塞 bench 循环的红线 |

## 6. 已知风险与对策 (per spec §10 风险表, 4 行摘要)

- **SLA 上界破阈**: TTFT P99 / TPOT P99 任一超 Baseline × 1.5 → 立即回退当周优化点, 48h 找不到原因则退回上一稳定版。
- **决赛多卡扩展**: 初赛单卡, 决赛多卡分布式 (PDF P11) — P3 末评估 1 个 stretch 项: tensor parallel 路径调研 (不实现), `NCCL_MIN_NCHANNELS=112` 已留好。
- **评测黑盒 (请求顺序/时间戳不可见)**: prefix-caching 调参策略保守, **不**假设请求可重放; bench script 走 50-100 prompts 单轮, 不复用 prompt 顺序。
- **AI review SLA**: 24h 内未审 = revert (spec §10), AGENTS.md 已有"读 + 跑 + 对比 baseline"三关, 强制 owner 在 PR 描述里附 baseline 对比数字。

## 7. 验证项

- [ ] P3A.1-A.3 块管理 3 子任务全部跑通 + 填入上表
- [ ] P3B.1-B.3 KV 量化 3 子任务 + 精度 Δ 验证
- [ ] P3C.1-C.2 torch.compile + cudagraph 5 实验
- [ ] P3 末集成日 9 格填入实测提升%
- [ ] P4 末 1 次干净全量编译演练 (spec §11)

## 关联文档

- spec §3 (用户决策) / §5 (技术决策) / §10 (风险与对策) / §11 (完工标准 + §12-15 提交材料清单)
- [ADR 0007: Block-size × KV 量化粒度耦合矩阵](../../docs/decisions/0007-coupling-matrix.md)
- [ADR 0008: block-size 假设与 ROI 估算](../../docs/decisions/0008-blocksize-hypothesis.md)
- [ADR 0013: torch.compile + cudagraph ROI 分析](../../docs/decisions/0013-torch-compile-roi.md)
- Plan `docs/superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md` (Task 3A / 3B / 3C / Task 4.4 / Task 5.1)
