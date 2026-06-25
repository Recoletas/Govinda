# INT8 KV Cache Patch MVP · 设计文档

> **作者**：队长 recoletas (与 Claude 会话协作)
> **日期**：2026-06-25
> **范围**：P3 Stream B MVP (patcher 真文件验证)
> **关联**：[ADR 0009](../../decisions/0009-kv-quant-strategy.md) · [ADR 0012](../../decisions/0012-vllm-cscc-vs-upstream.md) · [ADR 0013](../../decisions/0013-competition-rules-interpretation.md) · [AGENTS.md](../../../AGENTS.md)

---

## 1. Context

P3 Stream B (INT8 KV cache 量化, ADR 0009 主路) 需要让 vLLM 在 CDNA2 (gfx90a, 无原生 FP8) 上跑 INT8 per-head 对称 KV cache。代码骨架已存在:

- `src/kv_quant/int8_quant.py` — `INT8PerHeadQuantizer` (numpy, per-(B,H,T) scale, banker's rounding, 无 clip), 已通过 unit test
- `src/kv_quant/patch_kv_cache_interface.py` — 幂等 patcher 脚本, 改 `KVCacheSpec` 加 `kv_cache_dtype` + `kv_cache_int8_per_head` 字段 + 改 `page_size_bytes` 加 INT8 分支
- `tests/test_patch_kv_cache_interface.py` — 基于 SYNTHETIC fixture (单 `page_size_bytes`, 无 frozen, 无 AttentionSpec)

**关键 gap**: patcher 从未在真 `~/vllm_cscc/vllm/v1/kv_cache_interface.py` (499 行, vllm 0.18.1) 上验证过。读完真文件后, 发现 patcher 有 **3 个设计 bug**:

1. 改错位置 — patcher 替换 `page_size_bytes` (line 72, `AttentionSpec` 的 wrapper, 调 `real_page_size_bytes`), 但实际存储大小由 `real_page_size_bytes` (line 80) 决定。改 wrapper 不改底层公式 → INT8 半内存效果不出来
2. 公式错 — 即使改对位置, patcher 新 body 用 `head_size * block_size * num_kv_heads * head_size`, 缺 `2 *` (K + V 各一份)
3. 变量名遮蔽 — body 里 `head_size = get_dtype_size(torch.int8)` 局部变量, 遮住 `self.head_size` 类字段

加之 `page_size_bytes` 在真文件出现 7 次 (`KVCacheSpec` / `AttentionSpec` / `FullAttentionSpec` / `MLAAttentionSpec` / `ChunkedLocalAttentionSpec` / `MambaSpec` / `UniformTypeKVCacheSpecs`), patcher 的 "first match" 会改到 `KVCacheSpec.page_size_bytes` (line 29, abstract `raise NotImplementedError`), 改完无意义。

**SYNTHETIC fixture 测试会过** (因为里面只有 1 个 `page_size_bytes`), 但真文件上 patcher 行为错。AGENTS.md 警告的 "AI 写 → 跑 → 通过 → 实际没改对" 经典场景。

## 2. Goals (MVP "完工" 定义)

1. patcher 在真 `vllm_cscc/vllm/v1/kv_cache_interface.py` 上 dry-run, 产出的 unified diff **人眼可读 + 正确**
2. patcher 加的字段在 patched Python 文件里能 `ast.parse`, 能 `dataclasses.fields()` 列出
3. INT8 模式下 `AttentionSpec.real_page_size_bytes` / `FullAttentionSpec.real_page_size_bytes` / `MLAAttentionSpec.real_page_size_bytes` 返回正确 (1 byte/element) 的字节数
4. bf16 / fp16 默认模式下 (即 `kv_cache_dtype="auto"`) 各 spec 输出与 unpatched 一致 — **无回归**
5. 幂等: 跑两遍结果 = 跑一遍
6. `MambaSpec` / `UniformTypeKVCacheSpecs` / `ChunkedLocalAttentionSpec` / `SlidingWindowSpec` **未被改动**

## 3. Non-Goals (明确不做)

- 远程真 apply + rebuild wheel + vllm serve 启动验证 — 是写操作, 团队约定后单独走
- DCU 上 INT8 路径吞吐 / Δ 精度实测 — 要 DCU 时间 + 模型 + testset
- Bench 命令整合 (env var / flag / 默认值) — 跟 ADR 0013 §9 命令模板独立议题
- KIVI 关键 cache FP16 保留 (前 256 token 高精度) — 留后续 spec
- FP8 `DynamicQuantizer` 从旧 `KVQuantizer` ABC 迁新 `Quantizer` — 留后续 spec
- 远程 `vllm serve --kv-cache-dtype int8` 启动 smoke — 要 GPU

## 4. Approach

### 4.1 Patcher 改造 (`src/kv_quant/patch_kv_cache_interface.py`)

**保留**:
- `patch_kv_spec_fields()` — 已能正确在 `KVCacheSpec` 类内加字段
- 文件级 idempotent marker 机制

**替换**:
- `patch_page_size()` → **`patch_real_page_size()`**: 匹配 `class AttentionSpec` 内的 `def real_page_size_bytes(self) -> int:` body, 把 `get_dtype_size(self.dtype)` 替换为 `get_dtype_size(self.kv_dtype())`, 保留 `2 *` 系数和其余乘法
- 新增 **`patch_full_attention_real_page_size()`**: 锚定 `FullAttentionSpec.real_page_size_bytes` (line 181). 同模式, 同 marker 独立幂等
- 新增 **`patch_mla_real_page_size()`**: 锚定 `MLAAttentionSpec.real_page_size_bytes` (line 196). 同模式

**只这 3 处需要改**。其他 attention 子类 (ChunkedLocalAttentionSpec / SlidingWindowSpec / SinkFullAttentionSpec) 通过继承拿 `real_page_size_bytes`, 不需要单独 patch。Qwen3.5-27B 用 GQA 走 `FullAttentionSpec` 路径, 后两个 patch 是为通用性。

**新增字段 / 方法**:
- `KVCacheSpec` 加方法 `kv_dtype(self) -> torch.dtype`: 返回 `torch.int8 if self.kv_cache_dtype == "int8" else self.dtype`. 这样 INT8 路径调用点只需写 `get_dtype_size(self.kv_dtype())`, 不写三元表达式

### 4.2 测试 fixture (`tests/fixtures/kv_cache_interface_v0181.py.snapshot`)

- 从 `~/vllm_cscc/vllm/v1/kv_cache_interface.py` 复制 499 行
- 文件头加注释: source = SCNet vllm_cscc (vllm v0.18.1), SHA 留空 (无需严格校验, vllm_cscc 是 grafted single-commit 锁版本, 不会变)
- 测试从此 fixture 出发 → CI 不依赖 vllm 安装

### 4.3 测试扩展 (`tests/test_patch_kv_cache_interface.py`)

**保留**现有 SYNTHETIC 测试 (3 个): 防止 fixture 引入的 bug 被错过

**新增** (基于真 fixture):
- `test_real_fixture_first_run_inserts_fields` — patched file 含新字段 + `kv_dtype` 方法
- `test_real_fixture_idempotent` — 跑两次 = 跑一次
- `test_real_fixture_syntax_valid` — `ast.parse(patched)` 成功
- `test_real_fixture_attention_spec_int8_size` — `AttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16, kv_cache_dtype="int8").real_page_size_bytes == 32768`
- `test_real_fixture_attention_spec_default_no_regression` — `kv_cache_dtype="auto"` 输出与 unpatched 一致 (`2 * 16 * 8 * 128 * 2 = 65536` for bf16)
- `test_real_fixture_full_attention_spec_int8_size` — FullAttentionSpec (line 181) 同行为
- `test_real_fixture_mla_attention_spec_int8_size` — MLAAttentionSpec (line 196) 同行为
- `test_real_fixture_inherited_specs_inherit_int8_path` — SlidingWindowSpec / ChunkedLocalAttentionSpec 通过继承 AttentionSpec 拿 INT8 路径, SinkFullAttentionSpec 继承 FullAttentionSpec
- `test_real_fixture_mamba_spec_untouched` — MambaSpec 没改 (line 283 page_size_bytes 是 state-space, 不走 INT8)
- `test_real_fixture_uniform_type_spec_untouched` — UniformTypeKVCacheSpecs 没改

## 5. Components 改动清单

| 文件 | 改动类型 | 大致行数 |
|---|---|---|
| `src/kv_quant/patch_kv_cache_interface.py` | 替换 `patch_page_size` → `patch_real_page_size` + 加 `kv_dtype` 字段 patch + 2 个新 spec 锚定函数 (FullAttentionSpec + MLAAttentionSpec) | ~80 行替换 |
| `tests/fixtures/kv_cache_interface_v0181.py.snapshot` | 新增 (复制粘贴 499 行) | +499 行 |
| `tests/test_patch_kv_cache_interface.py` | 加 ~9 个新测试 | +120 行 |
| `~/vllm_cscc/vllm/v1/kv_cache_interface.py` | **不改** (MVP 范围 = dry-run review) | 0 |

## 6. Data Flow

```
[真 upstream file] 
       │
       ▼
[patch_kv_spec_fields] ──→ KVCacheSpec 加 kv_cache_dtype + kv_cache_int8_per_head + kv_dtype() 方法
       │
       ▼
[patch_real_page_size] ──→ AttentionSpec.real_page_size_bytes 用 self.kv_dtype()
       │
       ▼
[patch_full_attention_real_page_size] ──→ FullAttentionSpec 同上 (Qwen3.5-27B GQA 走这条)
       │
       ▼
[patch_mla_real_page_size] ──→ MLAAttentionSpec 同上 (通用性, Qwen3.5-27B 不用)
       │
       ▼
[patched file] (dry-run: stdout, --apply: write + .bak)
```

每步带独立 marker, 任一步已 patch 过就跳过。整体 idempotent。

## 7. Error Handling

- 真文件里 `class AttentionSpec` 不存在 → `patch_real_page_size` 返回 None, CLI 退码 2 + stderr 提示
- `def real_page_size_bytes(self)` 找不到 → 同上
- `--target` 文件不存在 → CLI 退码 1
- `--apply` 时磁盘满 / 权限错 → Python `IOError` 自然抛出, 不吞 (patcher 不该吞异常)
- 幂等性破坏 (patched file 已被外部改坏) → marker 找不到 → 返回 None + warn

## 8. Testing + Validation

### 8.1 自动测试

```
pytest tests/test_patch_kv_cache_interface.py -v
```

期望: 现有 3 个 SYNTHETIC 测试 + 新增 ~9 个真 fixture 测试全绿。

### 8.2 手动验证协议 (人执行, MVP 阶段不进 CI)

```bash
# 1. 远程 dry-run, 看 diff
python src/kv_quant/patch_kv_cache_interface.py \
    --target ~/vllm_cscc/vllm/v1/kv_cache_interface.py

# → 输出 unified diff. 人眼 review:
#   - KVCacheSpec 加 kv_cache_dtype + kv_cache_int8_per_head + kv_dtype()
#   - AttentionSpec.real_page_size_bytes 用 self.kv_dtype()
#   - FullAttentionSpec / MLAAttentionSpec / ChunkedLocalAttentionSpec 同上
#   - 没动 MambaSpec / UniformTypeKVCacheSpecs / KVCacheSpec.page_size_bytes

# 2. 跑自动测试 (本地 sandbox, 不需 DCU)
pytest tests/test_patch_kv_cache_interface.py -v

# 3. (可选, MVP 不强求) 真 apply + rebuild
python src/kv_quant/patch_kv_cache_interface.py \
    --target ~/vllm_cscc/vllm/v1/kv_cache_interface.py --apply
cd ~/vllm_cscc && python setup.py bdist_wheel
cd dist && pip install vllm-*.whl --no-deps
python -c "from vllm.v1.kv_cache_interface import KVCacheSpec; import dataclasses; print([f.name for f in dataclasses.fields(KVCacheSpec)])"
```

### 8.3 AGENTS.md 三道关

1. **可读性** — dry-run diff 人眼 review (本次走流程)
2. **运行验证** — `pytest tests/test_patch_kv_cache_interface.py -v` 全绿
3. **回归** — bf16 默认模式下 patched file 各 spec `real_page_size_bytes` 与 unpatched 一致 (`test_real_fixture_*_default_no_regression`)

## 9. Risks + Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| patcher regex 在 vllm 0.18.1 patch 后失效 | Low (锁版本) | fixture 是真文件 snapshot, 测试锁住现状; vllm 升级是单独 spec |
| 真 apply 后 vllm import 失败 | Medium | MVP 不真 apply; 若团队同意才走, 应单独 plan (rebuild wheel ~2min + import smoke) |
| INT8 路径精度塌方 (Δ > 10%) | Unknown (要 DCU 实测) | MVP 不碰; P4 精度阶段单独验 |
| 与 vllm 上游未来 `--kv-cache-dtype int8` (L1 路径) 冲突 | Low | 如果 vllm 0.18.1 已支持 `--kv-cache-dtype int8`, patcher 路径可废弃 — 这是 ADR 0009 L1 vs L2 选择, 跟 MVP 独立 |
| 远程 `dubious ownership` (git 命令在 remote 失败) | Confirmed | 用文件读 (`sed` / `cat`) 而非 git 命令 |

## 10. Compliance (§7 + §8)

- **§7 严禁**: 投机解码 ✗ 不涉及, 持久化权重量化 ✗ 不涉及, 剪枝 ✗ 不涉及, scheduler 参数改 ✗ 不涉及, 训练 ✗ 不涉及, 预缓存 ✗ 不涉及
- **§7 允许**: KV cache 动态量化 ✓ (本 MVP 实现路径), activation 动态量化 ✗ 不涉及 (后续), kernel 内低精度 ✗ 不涉及 (后续), 自定义 Python 包 ✓ (patcher 本身)
- **§8 SLA**: MVP 不跑 vllm, 无 TTFT/TPOT 影响
- **§8 完成率**: MVP 不改服务行为, 无失败率影响
- **§8 精度**: MVP 不动量化计算逻辑 (只改 size 公式), 无 Δ 影响

合规无歧义。
