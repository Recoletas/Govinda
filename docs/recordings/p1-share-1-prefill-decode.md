# P1 知识分享 1: Prefill vs Decode + KV cache 基础

**录制人**: 队员 B
**目标听众**: 4 人队 (含 2 个 0 LLM 推理基础)
**时长**: 30 min
**前置阅读**: spec §1 + §5.1 决策表
**录制日期**: P1 第 1 周

---

## 0:00 - 5:00 — 为什么需要 KV cache

### 开场 (1 min)
- 自我介绍
- 这次分享的目的: 给后面所有优化打基础。我们后面要做的所有事 — KV cache 量化、块管理调参、chunked prefill — 都跟今天这 4 个概念强相关
- 今天路线图: 为什么 → 阶段差异 → 块管理 → 收口

### 自回归生成的 O(n²) 复杂度 (1.5 min)

LLM 生成是**自回归**的: 每生成 1 个 token, 都把当前整段序列重新过一遍 attention。

生成 n 个 token, 第 t 步要算的 attention 长度是 t, 所以总计算量是 O(1 + 2 + ... + n) = **O(n²)**。

举例: 生成 1024 个 token, 不做缓存 → 大约 1024 × 1024 / 2 ≈ 500K 次 attention score 计算。
实际感受: 不缓存的话, 生成 1024 token 的延迟是 1 token 时的 500 倍级别。

### KV cache 的核心思想 (1.5 min)

每层 attention 算的是 `softmax(Q @ K^T / sqrt(d)) @ V`。
生成第 t 步时, Q 是新 token (1 个), K 和 V 是前 t-1 个 token (要重新算)。

**关键观察**: K 和 V 跟输入有关, 一旦输入定下来就不变。把每层的 K, V 缓存下来, 下次直接读 → 避免重算。

代价: 每 1 个新 token, 都要在 HBM 里多存 `2 × num_layers × num_heads × head_dim` 个元素。

存储: 不缓存 = O(n²) FLOPs, 缓存后 = O(n) FLOPs, 但空间从 O(1) 变成 O(n)。

### 27B 模型 KV cache 占用估算 (1 min) — **近似值, 下面标注**

**假设 Qwen3.5-27B 类似 Qwen2.5-27B 架构** (40 层, 32 注意力头, head_dim=128; 但 Qwen3.5 可能用 GQA, 见末尾注意):

每层每 token KV cache (bf16) = 2 × num_kv_heads × head_dim × 2 bytes

- 全 MHA (num_kv_heads=32): 2 × 32 × 128 × 2 = **16,384 bytes/token/layer**
- 如果是 GQA (num_kv_heads=8, 推测): 2 × 8 × 128 × 2 = **4,096 bytes/token/layer** (1/4)

**全 MHA 假设下 4 档上下文 KV cache 占用对照表 (近似)**:

| 上下文长度 | 单层 KV cache | 40 层总占用 | 备注 |
|-----------|--------------|------------|------|
| 4k  (4096 tokens)  | 64 MB        | ~2.6 GB     | 接近 27B bf16 权重 (54 GB) 的 5% |
| 8k  (8192 tokens)  | 128 MB       | ~5.2 GB     | 10% |
| 16k (16384 tokens) | 256 MB       | ~10.5 GB    | 19% |
| 32k (32768 tokens) | 512 MB       | ~21 GB      | 39% — 接近权重一半 |

**注意 (GQA 不确定)**:
- 实际 Qwen3.5-27B 是否用 GQA — 我没在 spec 里找到确认, **录制时如果不确定, 老实说"这是按 MHA 估算的, 实际可能要除以 N (GQA group 数)"**
- 实际数字以官方 config.json 为准: `num_hidden_layers` / `num_attention_heads` / `num_key_value_heads`
- 16k 上下文 ~10 GB, 32k ~21 GB — **这个量级是要重点关注的**, 直接决定能跑多少并发

### 收口 (0.5 min)
- KV cache = 拿空间换时间
- 27B 在 32k 已经吃 ~21 GB 显存 (近似), 单卡就快只剩不下多少给 batch 了
- 后面所有优化的核心问题: **怎么在保质量前提下, 把这张表里的数压下来**

---

## 5:00 - 15:00 — Prefill vs Decode 阶段差异

### 两阶段定义 (2 min)

一个 LLM 推理请求分两阶段:

1. **Prefill**: 处理整个 prompt, 并行算所有 token 的 K, V, Q, attention。一次 forward 完。
2. **Decode**: 自回归生成阶段, 每步 1 个 token, 反复 forward。

举例: 用户发 "Write a poem about DCU" (8 tokens), 模型生成 200 token 回应:
- Prefill 阶段: 1 次 forward, 输入 8 tokens
- Decode 阶段: 200 次 forward, 每次 1 token

### Prefill 特性: compute-bound (2 min)

- 一次性处理整个 prompt, **GEMM 是大矩阵乘** (seq_len × hidden_dim @ hidden_dim × hidden_dim)
- 算术强度高: FLOPs / bytes 比高
- 瓶颈在 GPU 算力 (DCU 的 MFU)
- TTFT (Time To First Token) 主要由 prefill 决定 — 用户感受到的"首字延迟"

### Decode 特性: memory-bound (3 min)

- 每步只生成 1 个 token, 但要读**整个** KV cache 做 attention
- 算的: `1 × hidden_dim @ hidden_dim × seq_len` (Q 是 1 行) → 小矩阵乘
- 数据量: 每步要从 HBM 读 `2 × num_layers × num_kv_heads × head_dim × seq_len` bytes
- 算术强度低: 大量时间花在**搬数据**, 不在算

**为什么对长上下文不友好 (batch=1)**:
- 32k 上下文, 每 decode 一步要读 ~21 GB (近似, 见上表)
- HBM 带宽假设 1.5 TB/s (DCU 典型): 单步光读 KV 就 ~14 ms
- 实际 TPOT 会被这个下界卡死

**TTFT vs TPOT 的根本原因**:
- TTFT ≈ prefill 时间 = f(prompt 长度, 模型 FLOPs) → **算力 bound**
- TPOT ≈ decode 每步时间 = f(单步 FLOPs, KV cache 大小, HBM 带宽) → **带宽 bound**
- 这是为什么优化方向不同: prefill 看 compute 效率, decode 看 memory 效率

### 为什么 batch=1 + 长上下文对 decode 不友好 (2.5 min)

- 算 1 个 token 的 FLOPs 极小 (跟 prefill 比), GPU 利用率天然低
- 唯一能摊薄单 token 成本的是**把多个请求的 KV 一起读** (continuous batching)
- batch=1 时, 没有摊薄; 长上下文时, 每次要读大量 KV
- 两害叠加: 长上下文 + batch=1 = 极端 memory-bound 场景
- **赛题场景就是 batch=1 + 长上下文** — 所以我们的优化必须直面这个

### 衔接 (0.5 min)
- "好, 现在我们知道 decode 是 memory-bound, KV cache 占用是关键 — 那 vLLM 怎么管理这块显存?"

---

## 15:00 - 25:00 — PagedAttention 块管理

### 传统连续显存分配的问题 (3 min)

朴素方法: 每个请求预分配一段连续显存 = `max_seq_len × per_token_kv_size × num_layers`。

问题:
1. **外部碎片**: 分配/释放顺序不同时, 中间空隙没法用
2. **内部碎片**: 用户实际只用 1024 token, 但预分配了 32k 的空间 — 浪费 31/32
3. **预分配 vs 动态**: 不知道用户会用多长, 预多了浪费, 预少了 OOM
4. **batch=1 场景下, 每个请求独占一段连续空间** — 显存利用率极低

### PagedAttention 思想 (3 min)

来自 vLLM 0.2 (SOSP'23), 借鉴 OS 虚拟内存 + 页表:

- **固定大小 block** (典型 16 tokens 的 KV)
- 请求的 KV cache 是一组 block, 编号是**逻辑的**
- 物理 block 在显存里**不需要连续**
- 一张**块表** (block table) 维护逻辑到物理的映射
- 物理 block 用完再分配, 用完即释放

类比 OS:
- 逻辑 block 编号 = 虚拟页号
- 物理 block 编号 = 物理页号
- block table = 页表

### 块大小对碎片率的影响 (2.5 min)

- 块越小 → 内部碎片越小 (一个请求末尾浪费的 token 数 = 块大小 - 1)
- 块越大 → metadata 开销小 (block table 项少), 但内部碎片大
- **存在最优值**, 经验上 16 tokens 是常见默认值
- 我们调参时要权衡: 在 DCU 上 HBM 访问是否对齐 (块大小是否 2 的幂且 ≥ 16)

### 块大小对 HBM 访问对齐的影响 (1.5 min)

- attention 计算时, 一次会读一个或多个 block 的 K/V
- 块大小如果跟 attention kernel 的 tile size 对齐 → 访存效率高
- 块过小 → kernel launch / metadata 开销占比上升
- 块过大 → 不必要的内存读 (末尾 block 多数时候不满)
- 这是 spec §5.1 决策表里 block_size 调参要回答的问题

### 衔接 (0.5 min)
- "PagedAttention 解决了分配碎片, 但**显存总量**还是要看 KV cache 实际数据大小。下一阶段我们讲量化怎么压这块"

---

## 25:00 - 30:00 — Q&A + 总结

### 总结: 我们的优化方向如何映射到这些基础 (4 min)

把今天讲的 4 个概念映射到 3 必做项:

| 优化项 | 作用层 | 直接效果 |
|--------|-------|---------|
| **KV cache 动态量化** (FP8/INT8) | 数据精度 | **直接减 KV cache 占用** — 32k 上下文从 ~21 GB → ~10 GB (近似) |
| **block_size 调参** | 块管理 | 平衡碎片率 + metadata 开销, 间接提升 batch 利用率 |
| **chunked prefill** | 调度 | 改善 prefill + decode 混合场景的 TTFT 和 TPOT |

具体说:
- **量化** = 把上面那张 4 档对照表里的数字除以 2 (INT8) 或 4 (FP8), 32k 从 ~21 GB → ~10 GB / ~5 GB
- **block_size 调参** = 在碎片率和 metadata 之间找平衡, 让长上下文请求能跑起来
- **chunked prefill** = 缓解 "prefill 把 decode 挤死" 的问题, 在 batch=1 场景下能稳 TPOT

### Q&A (1 min)
- 留时间给队员提问
- 不确定的问题老实说 "需要查 spec / 源码 / 跑实验确认", 不要瞎猜

---

## 录制注意事项 (给队员 B)

1. 数字是近似 — 录制时如果不确定 Qwen3.5-27B 是否 GQA, 老实说"按 MHA 估算, 实际要查 config.json"
2. 27B 在 32k 吃 ~21 GB 这个量级一定要讲到, 后面所有优化都跟这个数字有关
3. 听众有 2 个 0 基础, 算术强度 / memory-bound 这些概念要举生活化例子 (比如 "搬桌子比搬桌子腿慢" 这种)
4. 控制好时间, 每个 5 min 块不要超; Q&A 留至少 1 min, 不然 quiz 通过率会难看
5. 录完后自己回看一遍, 数字 / 术语有口误就在本文件改, 然后通知队长更新

## 录制后自检清单

- [ ] 4 档上下文档位 (4k/8k/16k/32k) 都给出了 KV cache 占用近似值
- [ ] 标注了 "近似" 和 GQA 不确定
- [ ] Prefill compute-bound / Decode memory-bound 都讲了
- [ ] TTFT vs TPOT 区分讲了
- [ ] PagedAttention 跟 OS 虚拟内存类比了
- [ ] block_size 调参的权衡 (碎片率 + metadata) 讲了
- [ ] 3 必做项跟今天概念的映射表有了
- [ ] 总时长控制在 30 min ± 2 min
