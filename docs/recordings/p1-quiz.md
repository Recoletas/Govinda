# P1 知识分享 Quiz (闭卷)

**出题人**: 队长 recoletas
**闭卷**: 是, 不允许看分享文字稿
**时长**: 20 min
**通过标准**: 80% (8/10)
**失败处理**: 重听分享 1 + 分享 2, 次日重考
**覆盖范围**: 分享 1 (Prefill/Decode/KV cache) + 分享 2 (vLLM 0.18.1 架构)

---

## 题目

### Q1. 为什么 decode 阶段是 memory-bound 而不是 compute-bound?

**期望答案要点** (3 选 2 即满分):
- 每步只生成 1 个 token, Q 是 1 行, GEMM 算术强度低
- 每步要从 HBM 读整个 KV cache (跟当前序列长度成正比), 数据量大
- 算力利用率低, GPU 大部分时间在等 HBM 数据
- 跟 prefill 对比: prefill 是大矩阵乘, compute-bound; decode 是小矩阵乘 + 大数据读, memory-bound

### Q2. Qwen3.5-27B 在 32k 上下文下的 KV cache 大约占用多少显存? (近似即可)

**期望答案**:
- 近似 **21 GB** (按 MHA 假设: 40 层, 32 KV heads, head_dim=128, bf16)
- 公式: 2 × 32 × 128 × 32768 × 2 × 40 = ~21 GB
- 如果答 "GQA 假设下要除以 N" 也算对, 但要说明不确定 GQA group 数
- 容许范围: 15-25 GB (按 bf16 + 假设架构)

### Q3. PagedAttention 的 "页" 对应 vLLM 中的什么概念?

**期望答案**:
- 对应 **block** (或 "KV cache block" / "physical block")
- 固定大小 (默认 16 tokens), 物理显存里的最小分配单位
- 类比 OS 虚拟内存: 块 = 页, 块表 = 页表

### Q4. block_size 越大越好还是越小越好? 为什么?

**期望答案要点** (3 选 2 即满分):
- **不是单调关系**, 存在最优值
- 越小 → 内部碎片小, 但 metadata 开销 (block table 大) + 访存对齐差
- 越大 → metadata 小, 对齐好, 但内部碎片大 (末尾 block 浪费)
- 默认 16 tokens 是经验最优附近, 调参要测

### Q5. 列出 vLLM 0.18.1 中你记得的 3 个 attention backend enum 值。

**期望答案** (3 选即可):
- `FLASH_ATTN` (FlashAttention)
- `FLASHINFER`
- `XFORMERS`
- `TRITON_ATTN`
- `ROCM_ATTN` (ROCm 平台用)
- 评分: 记对 1 个得 1/3, 全对满分, 错的不倒扣

### Q6. (代码阅读) vLLM 中 `_get_backend_priorities()` 的作用是什么?

**期望答案**:
- 平台 (CPU / CUDA / ROCm / DCU) 通过这个方法告诉 vLLM 哪些 backend 优先级高
- vLLM 选 backend 时按 priority list 选第一个能用的
- DCU 平台在 `platforms/rocm.py` 里实现, 录的时候 spec §5.2 已经验证
- 我们的 custom backend 怎么落: 改这个 priority list 或 register 到最高优先级

### Q7. Prefill 和 Decode 在 batch=1 + 长上下文场景下, 哪个对 KV cache 量化更敏感?

**期望答案**:
- **Decode 更敏感**
- 原因: decode 每步读整个 KV cache, 量化误差在每步累积, 影响每个生成 token 的质量
- Prefill 只算一次, 量化误差影响一次 (虽然首字质量有损, 但影响面小)
- 长上下文放大 decode 敏感度: 32k 上下文每步都读 ~21 GB (量化前), 量化误差被反复消费

### Q8. vLLM 0.18.1 中 V1 引擎和 V0 引擎的主要区别是什么? (如不确定可写 "未查")

**期望答案** (诚实评分):
- 主要区别 (举例, 2 选 1 即满分):
  - 架构: V0 是同步架构, V1 是异步架构 (EngineCore 跟 Worker 通信用 asyncio)
  - Chunked prefill: V0 默认不开, V1 默认开
  - Performance: V1 普遍更优 (尤其混合 prefill/decode 场景)
- 答 "未查" → 给 1/2 分 (不是 0, 鼓励诚实)

### Q9. KV cache 量化后, 如果 decode 出现数值塌方 (生成重复 / 无意义 token), 最可能的原因是什么?

**期望答案要点** (2 选 1 即满分):
- **量化粒度不对**: per-tensor 量化在长上下文下动态范围太大, 精度不够; 应该用 per-token 或 per-head
- **scale/zero-point 计算有偏**: 训练时算的 scale 跟推理时分布不匹配, 导致反量化后值偏
- **outlier token 没保护**: 少数 token 的 KV 值远大于其他, 量化被它们 dominate, 其他 token 精度塌方
- 解法: dynamic quantization (每步重算 scale) 或 per-head + outlier protection

### Q10. (开放题) 我们的 3 必做项 (KV 量化 / torch.compile / 块管理) 中, 哪个对 8k-16k 这一档 (占 50% 权重) ROI 最高? 为什么?

**评分说明**: 这是开放题, **看推理过程, 没有标准答案**。可以从以下角度任答:

**参考维度**:
- **8k-16k 占比 50%** (spec §3 用户决策): 优化空间最大的一档
- **量化效果**: 8k 从 ~5 GB → ~2.5 GB (INT8) 或 ~1.3 GB (FP8), 16k 从 ~10 GB → ~5 GB / ~2.6 GB
  - 8k-16k 这一档用 INT8 量化基本无损 (Qwen2.5 系列经验), FP8 需要看 DCU 是否支持
- **torch.compile**: 对 prefill compute 提升大, 对 decode memory-bound 帮助小; 8k-16k 仍然 prefill + decode 混合, 都有收益
- **块管理**: 8k-16k 段 block 数适中, 调 block_size 影响有限; 32k 才更需要精细调

**参考答案 (之一, 不唯一)**:
- 多数人会选 **KV 量化**: 直接减显存 → 提升 batch 上限 → 8k-16k 段是 batch 友好档, 收益最直接
- 也有合理回答选 **torch.compile**: 8k-16k 仍有大量 prefill, compile 收益稳定, 风险小
- 选 块管理 的要解释清楚: 8k-16k 为什么调 block_size 比 4k/32k 收益大

**评分原则**:
- 有推理过程 + 引用 spec 决策点 + 给出风险评估 → 满分
- 只给结论没说为什么 → 1/2
- 三个都写 "都好" 没区分 → 1/3

---

## 出题思路 (给队长参考, 不发给队员)

- Q1-Q4 测分享 1 基础概念 (decode 特性, KV cache 占用, PagedAttention, block_size)
- Q5-Q8 测分享 2 架构 (attention backend, _get_backend_priorities, V1 vs V0)
- Q9 测推理能力 (量化常见坑, 不直接来自分享, 测思考)
- Q10 测综合判断 (把分享内容映射回我们的项目决策)

难度: 中等偏上. 通过率预期 60-70%, 不通过的 1-2 人重听重考即可。

## 阅卷注意事项

1. Q8 答 "未查" 给 1/2 分 — 鼓励诚实
2. Q9 不要扣太严, 答出 1 个合理原因即给满分
3. Q10 重点看推理, 结论不唯一 — 不要按唯一答案批
4. 不通过 (≤ 7 题对) → 安排 1-on-1 复盘 + 次日重考
