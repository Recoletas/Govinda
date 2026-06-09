# P1 知识分享 2: vLLM 0.18.1 架构总览

**录制人**: 队长 recoletas
**目标听众**: 4 人队
**时长**: 30 min
**前置阅读**: 分享 1 + spec §5.2
**录制日期**: P1 第 1 周 (分享 1 之后 1-2 天)

> **录制前必做**: 先去 https://github.com/vllm-project/vllm/releases 看 v0.18.1 release notes, 找 2-3 个对架构有影响的变化, 录的时候直接念出来。本稿下面的 "v0.18.1 vs v0.17 变化" 段先用占位符, 录之前补全。

---

## 0:00 - 5:00 — 整体组件图

### 5 大核心组件 (2 min)

vLLM 0.18.1 推理服务的 5 大组件 (从外到内):

1. **API Server** (FastAPI / OpenAI 兼容)
   - 入口, 收 HTTP 请求
   - 跑在主进程 (或独立进程, 取决于部署)

2. **Engine** (vLLM 核心, V1 架构)
   - 协调者: 接收请求, 调度, 输出
   - 维护请求生命周期

3. **Worker**
   - GPU 上的执行单元
   - 每个 GPU 一个 worker 进程 (tensor parallel 下还有 sub-worker)
   - V1 引擎下是异步架构

4. **ModelRunner**
   - 跑在 Worker 里
   - 负责 model forward 实际调用
   - KV cache 写回的入口

5. **Scheduler**
   - 决定哪些请求这步跑
   - 决定 prefill / decode 怎么混合
   - V1 下 chunked prefill 是默认行为

### 组件关系图 (1.5 min) — **建议画板书**

```
HTTP request
    |
    v
[API Server]  -- tokenize -->  [Engine]
                                    |
                                    v
                              [Scheduler]  <-- FIFO + prefill/decode 混合
                                    |
                                    v
                                [Worker]  (per-GPU)
                                    |
                                    v
                              [ModelRunner]
                                    |
                                    v
                              [Model forward]
                                    |
                                    v
                              [KVCacheManager]  <-- 块分配/回收
                                    |
                                    v
                              [Attention Backend]  (Triton / FlashInfer / XFormers / ...)
```

关系:
- Scheduler 调 Worker, Worker 调 ModelRunner
- ModelRunner 跟 KVCacheManager 交互 (拿 / 还 block)
- ModelRunner 选 Attention Backend (spec §5.2 决策表)

### 进程模型 (1.5 min) — V1 引擎

- vLLM 0.18.1 默认开 V1 引擎 (实验性 → 稳定的过渡, 看 release notes 确认)
- 主进程: API Server + EngineCore
- 每个 GPU: 1 个 Worker 进程 (也可能多 GPU shared)
- TP (tensor parallel) > 1 时: Worker 内部有多个 rank, 走 NCCL / RCCL
- DCU 上等价于 RCCL

---

## 5:00 - 15:00 — 1 个请求的端到端路径

### 步骤拆解 (8 min)

用户发 `POST /v1/chat/completions` 携带 prompt = "Hello", max_tokens = 100。

**Phase 1: API 接收 (1 min)**
- API Server 解析 JSON → 转成 vLLM 内部 `Request` 对象
- tokenize (BPE) → token ids
- 入 Engine 的请求队列

**Phase 2: Engine 调度 (1 min)**
- Scheduler 决定这步处理哪些请求
- 新请求触发 prefill, 已在跑的请求继续 decode
- 调用 Worker.execute_model()

**Phase 3: Prefill (3 min) — 时间点标注**
- Worker 调 ModelRunner
- ModelRunner 调 model.forward(input_ids=prompt_tokens)
- 内部步骤:
  1. Embedding lookup
  2. 逐层 transformer block
  3. 每层 attention: 算 Q, K, V → attention score → 写 KV cache → 算 output
  4. 最后一层 → LM head → logits
  5. sample 第一个 token
- **写 KV cache**: KVCacheManager 给这个请求分配 block, ModelRunner 把 K, V 写到对应 block
- **时间点**: 这一段就是 "首字延迟" (TTFT)

**Phase 4: Decode 循环 (2 min) — 时间点标注**
- Scheduler 把这个请求标记成 "running"
- 每步:
  1. Worker.execute_model() 调 ModelRunner
  2. ModelRunner 拿当前请求的最新 1 个 token → forward
  3. 读 KV cache (这个请求的所有历史 K, V)
  4. 算 logits, sample 下一个 token
  5. KVCacheManager 把新 1 个 token 的 K, V 追加到 block
- **时间点**: 每步的时间 = TPOT, 整段时间 ≈ 100 × TPOT

**Phase 5: 流式响应 (1 min)**
- Engine 把每步 sample 出的 token 走 streaming 返回 API Server
- API Server 通过 SSE 流给客户端
- 满足 stop 条件 (max_tokens / EOS) → Engine 收尾, 释放 block

### 关键点 (1 min)

- **prefill 写一次, decode 每步追加**: KV cache 是单调增长的
- **block 释放时机**: 请求结束才还, 中间不还
- **batch=1 场景下**: prefill 和 decode 不会并发, 但 vLLM 会把多个请求的 decode batch 起来 (continuous batching)
- **DCU 上**: prefill 跑 DCU compute units, decode 主要吃 HBM 带宽

---

## 15:00 - 25:00 — 关键模块深入

### KVCacheManager: 块分配/回收逻辑 (3 min)

职责:
- 维护 free block 池
- 分配: 请求需要新 block 时, 从 free list 拿; 满了报 OOM
- 回收: 请求结束时, 把所有 block 还给 free list
- 复制 (prefix caching): 命中已缓存前缀时, refcount 共享 block

数据:
- `block_size`: 每块多少 token (默认 16, spec §5.1 决策表)
- `num_blocks`: 总 block 数, 启动时按显存算
- 实际显存公式: `num_blocks × block_size × per_token_kv_size × num_layers`

我们 3 必做项里的 "块管理" 主要改这里 — 调 block_size, 调分配策略, 加 prefix cache hint。

### Scheduler: 请求调度 (3 min)

职责:
- 维护 waiting / running 队列
- 每步选哪些请求跑 (decode 优先, prefill 看 chunked)
- V1 默认开 chunked prefill: prefill 拆小块, 跟 decode 步交错

策略:
- FIFO 是基础
- continuous batching: decode 步可以动态进出
- chunked prefill: 长 prefill 不阻塞 decode
- 优先级 / preemption 策略 (一般不开, 复杂)

跟 3 必做项的映射:
- chunked prefill 默认开, **但参数要调** (max_num_batched_tokens 等)
- torch.compile 对 scheduler 影响小, 主要在 model runner

### Attention backend 选择 (2 min) — spec §5.2 决策表

vLLM 0.18.1 的 backend 机制:
- `AttentionBackendEnum`: 枚举所有 backend (FLASH_ATTN, FLASHINFER, XFORMERS, TRITON_ATTN, ROCM_ATTN, ...)
- `register_backend(name, cls)`: 注册新 backend
- `_get_backend_priorities()`: 平台返回优先级列表
- DCU 平台 (rocm.py) 返回的 priority list 决定默认选哪个

我们的 attention kernel 怎么落:
1. 继承 `TritonAttentionBackend` 改几行
2. 写个 register 函数
3. 通过环境变量 / 配置文件让 vLLM 选我们的 backend

**这块是 spec §5.2 的核心, 录的时候直接对着 spec 念, 不要自由发挥**。

### 我们的 3 必做项在哪些模块落地 (2 min)

| 必做项 | 主要改的模块 | 次要影响 |
|--------|------------|---------|
| **KV cache 动态量化** | KVCacheManager + Attention backend (读量化数据) | Memory profiling |
| **torch.compile** | ModelRunner (model forward) | 启动时间, 编译 cache |
| **块管理调参** | KVCacheManager (block_size, 分配策略) | Scheduler (调度粒度) |

---

## 25:00 - 30:00 — v0.18.1 vs v0.17 变化

> **录制前必做**: 查 https://github.com/vllm-project/vllm/releases v0.18.1 release notes
> **本段先用占位符, 录之前补全**

### 主要变化 (3 min) — 录制时填写

**变化 1**: [填]

举例占位:
- V1 引擎转 stable / 默认开启
- 新增 XX backend
- Chunked prefill 默认行为变化
- KV cache 管理 API 重构

**变化 2**: [填]

**变化 3**: [填]

### 对我们的优化方向有什么影响 (1.5 min)

[根据上面查到的 3 个变化, 录的时候口头分析]

大致方向:
- 如果 V1 引擎转默认 → 我们的代码要在 V1 路径下验证, V0 可能 deprecated
- 如果 backend 注册 API 变了 → spec §5.2 的方案要调整
- 如果 chunked prefill 行为变了 → 我们调参起点要重测

### Q&A (0.5 min)

---

## 录制注意事项 (给队长自己)

1. **录之前先查 release notes**, 不要现场编 — release notes 是权威, 我自己的记忆可能过时
2. v0.18.1 vs v0.17 段先填占位符, 查完再补 — 录制时这段一定要念 release notes 原文, 不要自由发挥
3. spec §5.2 决策表那段, 念 spec 不要自由发挥 — 决策表是已经复核过的, 改动要写 ADR
4. 进程模型那段, 如果不确定 V1 vs V0 细节, 老实说 "V1 架构下大致是..., 具体看 release notes"
5. 板书建议画两张图: 组件关系图 + 请求时序图 (prefill 一次, decode 循环)

## 录制后自检清单

- [ ] 5 大组件 (API / Engine / Worker / ModelRunner / Scheduler) 都讲了
- [ ] 端到端路径从 OpenAI API 到 KV cache 写完讲完
- [ ] Prefill / Decode 时间点明确标注
- [ ] KVCacheManager / Scheduler / Attention backend 3 个模块都深入到了
- [ ] 3 必做项在哪些模块落地的映射表有
- [ ] v0.18.1 vs v0.17 段不是空, 引了 release notes
- [ ] spec §5.2 决策表没有自由发挥
- [ ] 总时长 30 min ± 2 min
