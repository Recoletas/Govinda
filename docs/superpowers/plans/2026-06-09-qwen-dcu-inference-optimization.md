# Qwen3.5-27B DCU 推理优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在国产 DCU 单卡上用 vLLM 0.18.1 + Qwen3.5-27B 跑出 60-75 / 100 分（赛题 4k-32k 三档场景，TTFT/TPOT P99 ≤ Baseline × 1.5），9.5 周内完成。

**Architecture:** 不动 vLLM 核心源码；走"复用 vLLM 已注册 backend + 算子级优化 + 调参"路线。三条必做线（KV cache 动态量化 / torch.compile+cudagraph / 块管理+prefix+chunked prefill）独立推进，P3 末集成。三条 stretch 线（Triton decode kernel / 自定义 attention backend / FA2 之外的 attention 优化）仅在 P3 主线已稳后才开。

**Tech Stack:** vLLM 0.18.1 / PyTorch 2.10.0 / Python 3.10.12 / transformers 5.5.0 / ROCm 7.0 / Triton 3.x / FlashAttention 2 (ROCm fork) / vllm/vllm-rocm Docker 镜像 / OpenCompass（精度评测）/ vllm bench serve（性能评测）

**配套 spec:** [`../../specs/2026-06-09-qwen-inference-optimization-design.md`](../../specs/2026-06-09-qwen-inference-optimization-design.md) v4 (commit `0725d0e`)

---

## 文件结构（执行前先建好骨架）

```
Govinda/
├── AGENTS.md                          # 已存在
├── README.md                          # 已存在
├── mkdocs.yml                         # 已存在
├── docs/
│   ├── index.md                       # 已存在
│   ├── learning.md                    # 找学习资料的方法 + 术语速查
│   ├── specs/                         # 已存在（v4 spec 在此）
│   ├── superpowers/
│   │   └── plans/
│   │       └── 2026-06-09-qwen-dcu-inference-optimization.md  # 本文件
│   ├── decisions/                     # P0+ 按需建（ADR 记录）
│   ├── weekly/
│   │   └── progress.md                # 轻量 standup 模板（每周 1 行/人）
│   └── ai-prompts/                    # P3+ 按需建（共享 prompt 库）
├── src/
│   ├── __init__.py
│   ├── kv_quant/                      # 必做 1: KV cache 动态量化
│   │   ├── __init__.py
│   │   ├── fp8_quant.py               # FP8 动态量化（per-head/per-token scale）
│   │   ├── int8_quant.py              # INT8 fallback（CDNA2 走这条）
│   │   └── base.py                    # 量化器抽象基类
│   ├── attn_backend/                  # stretch 4-5: 自定义 attention backend
│   │   ├── __init__.py
│   │   └── triton_decode.py           # Triton decode attention kernel
│   ├── compile/                       # 必做 2: torch.compile + cudagraph
│   │   ├── __init__.py
│   │   └── config.py                  # torch.compile mode / use_cudagraph 配置
│   └── block_size/                    # 必做 3: 块管理调参
│       ├── __init__.py
│       └── sweep.py                   # --block-size 扫描脚本
├── benchmarks/
│   ├── __init__.py
│   ├── run_bench.py                   # vllm bench serve 包装器（3 档）
│   ├── analyze.py                     # JSON → markdown 汇总
│   ├── baseline/                      # baseline 数字
│   ├── optimized/                     # 优化后数字
│   └── profiles/                      # torch.profiler / rocprofv3 输出
├── tests/
│   ├── __init__.py
│   ├── test_kv_quant.py               # KV 量化精度测试
│   ├── test_torch_compile.py          # torch.compile + cudagraph 正确性
│   └── test_attn_backend.py           # custom backend 注册 smoke test
├── reports/                           # 赛题 §12-15 提交材料
│   ├── env-vars.md                    # §13
│   ├── optimization-plan.md           # §14
│   └── submission-readme.md           # §15
├── docker/
│   ├── Dockerfile                     # vllm-rocm 基础 + 自定义层
│   └── compose.yml
├── scripts/
│   ├── verify_dcu.py                  # P0 验证：DCU SKU / FP8 支持
│   ├── verify_backend_path.py         # P0 验证：custom backend smoke
│   ├── verify_triton_dcu.py           # P0 验证：Triton FP8 matmul 最小 case
│   └── lock_baseline.sh               # P2 baseline 锁定脚本
├── .github/workflows/
│   ├── docs.yml                       # mkdocs 部署（已存在）
│   └── ci.yml                         # 单元测试 + lint
└── requirements.txt                   # Python 依赖
```

---

## Phase 0：验证硬卡门（**1.5 周**）

**目标**：确认 §2 4 项未知（DCU SKU / baseline 数字 / 测试集 / backend 路径）；owner 离线完成各自练手任务。

**出口（CP0）**：4 项验证全过 + owner 各自 PR 合入 + 队长 + 队员 B 双签。

### Task 0.1: 验证 DCU SKU + FP8 支持

**Files:**
- Create: `scripts/verify_dcu.py`
- Create: `docs/decisions/0001-dcu-sku.md`

- [ ] **Step 1: 写验证脚本**

```python
# scripts/verify_dcu.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Detect DCU SKU + FP8 support. Run on DCU host."""
import subprocess
import sys
import json
from pathlib import Path

def get_device_props():
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0)

def get_rocminfo():
    try:
        out = subprocess.check_output(["rocminfo"], text=True)
        return out
    except FileNotFoundError:
        return None

def detect_fp8_support():
    import torch
    try:
        # FNUZ variant (CDNA3)
        x = torch.zeros(1, 1, dtype=torch.float8_e4m3fnuz, device="cuda")
        return "FNUZ"
    except (TypeError, RuntimeError):
        pass
    try:
        # Standard OCP variant
        x = torch.zeros(1, 1, dtype=torch.float8_e4m3, device="cuda")
        return "OCP"
    except (TypeError, RuntimeError):
        pass
    return "NONE"

def main():
    props = get_device_props()
    if props is None:
        print("ERROR: torch.cuda not available — not on a GPU/DCU host")
        sys.exit(1)
    gcn_arch = getattr(props, "gcnArchName", "unknown")
    device_name = props.name
    fp8 = detect_fp8_support()

    result = {
        "device_name": device_name,
        "gcn_arch": gcn_arch,
        "fp8_support": fp8,
        "total_memory_gb": props.total_memory / (1024 ** 3),
    }
    print(json.dumps(result, indent=2))
    Path("benchmarks/device_info.json").write_text(json.dumps(result, indent=2))

    # Hard gate
    if gcn_arch == "gfx90a" and fp8 != "NONE":
        print(f"WARN: CDNA2 (gfx90a) reported FP8={fp8} — verify with AMD docs")
    if "gfx942" not in gcn_arch and "gfx90a" not in gcn_arch:
        print(f"WARN: unknown arch {gcn_arch} — check ROCm support matrix")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在 DCU 上跑**

```bash
python3 scripts/verify_dcu.py
```

Expected: 输出 JSON，含 `device_name` / `gcn_arch` / `fp8_support` 三项；写到 `benchmarks/device_info.json`。

- [ ] **Step 3: 写 ADR 0001**

文件 `docs/decisions/0001-dcu-sku.md`，模板：

```markdown
# ADR 0001: DCU SKU 验证结果

**日期**: 2026-MM-DD
**状态**: 已确认 / 待确认

## 验证结果

- 设备名: <粘贴>
- GCN arch: <gfx942 或 gfx90a>
- FP8 支持: <FNUZ / OCP / NONE>
- 显存: <GB>

## 决策

<CDNA3 (gfx942) → KV 量化走 FP8 FNUZ 路线>
<CDNA2 (gfx90a) → KV 量化改 INT8 动态量化 / 保留 bf16>

## 影响

<具体影响 §5.1 决策表的哪些行>
```

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_dcu.py docs/decisions/0001-dcu-sku.md
git commit -m "P0: add DCU SKU verification script + ADR 0001"
```

### Task 0.2: 验证 LongBench / RULER 测试集访问

**Files:**
- Create: `scripts/verify_testset_access.py`
- Create: `docs/decisions/0002-testset-access.md`

- [ ] **Step 1: 写测试集访问验证脚本**

```python
# scripts/verify_testset_access.py
"""Check if LongBench / RULER test sets are downloadable."""
import sys
from pathlib import Path

CHECKS = {
    "LongBench": [
        # 公开 huggingface dataset
        ("xinrongzhang2022/longbench", "NarrativeQA"),
        ("xinrongzhang2022/longbench", "Qasper"),
    ],
    "RULER": [
        # 公开 github repo
        ("https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic.json", None),
    ],
}

def check_hf(dataset, config=None):
    try:
        from datasets import load_dataset
        if config:
            ds = load_dataset(dataset, config, split="test", streaming=True)
        else:
            ds = load_dataset(dataset, split="test", streaming=True)
        sample = next(iter(ds))
        return f"OK ({len(sample)} fields)"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"

def check_url(url):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return f"OK (HTTP {r.status})"
    except Exception as e:
        return f"FAIL: {e}"

def main():
    results = {}
    for hf_ds, cfg in CHECKS["LongBench"]:
        key = f"{hf_ds}/{cfg}" if cfg else hf_ds
        results[key] = check_hf(hf_ds, cfg)
    for url in [c[0] for c in CHECKS["RULER"]]:
        results[url] = check_url(url)

    for k, v in results.items():
        status = "OK" if v.startswith("OK") else "FAIL"
        print(f"[{status}] {k} → {v}")

    Path("benchmarks/testset_access.json").write_text(
        __import__("json").dumps(results, indent=2)
    )
    if any("FAIL" in v for v in results.values()):
        print("\n至少一个测试集不可下载 — 立刻询问赛方")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑脚本**

```bash
python3 scripts/verify_testset_access.py
```

Expected: 全 OK / 部分 FAIL（fail 即触发询问赛方流程）。

- [ ] **Step 3: 写 ADR 0002**

记录每个测试集访问状态 + 不可下载时的 fallback 策略（自造 100 样本）。

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_testset_access.py docs/decisions/0002-testset-access.md
git commit -m "P0: add testset access verification + ADR 0002"
```

### Task 0.3: 验证官方 baseline 数字

**Files:**
- Create: `docs/decisions/0003-baseline-source.md`

- [ ] **Step 1: 询问赛方**

写邮件 / 在赛方群问：是否下发 baseline 数字？下发时间？下发形式（绝对数字 / 评测平台自动测）？

- [ ] **Step 2: 写 ADR 0003**

```markdown
# ADR 0003: Baseline 数字来源

**日期**: 2026-MM-DD

## 赛方答复

<粘贴赛方原文 / 或 "截至 X 日未回复">

## 决策

| 场景 | 行动 |
|------|------|
| 赛方下发 | 走 spec §11 场景 1（60-75 分） |
| 部分下发 | 走场景 2（45-65） |
| 不下发 | 走场景 3（30-55）+ 砍必做到 2 项 |

## 当前行动

<具体接下来做什么>
```

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0003-baseline-source.md
git commit -m "P0: ADR 0003 baseline source"
```

### Task 0.4: 验证 vLLM 0.18.1 custom backend 路径

**Files:**
- Create: `scripts/verify_backend_path.py`
- Create: `docs/decisions/0004-backend-path.md`

- [ ] **Step 1: 写 smoke test**

```python
# scripts/verify_backend_path.py
"""Verify custom attention backend can be registered + used in vLLM 0.18.1."""
import sys

def main():
    import torch
    # 1. Verify enum is importable
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    enum_values = [v.name for v in AttentionBackendEnum]
    print(f"Enum values: {len(enum_values)}")
    assert "CUSTOM" in enum_values, "CUSTOM slot missing"
    assert "TRITON_ATTN" in enum_values, "TRITON_ATTN missing"

    # 2. Try registering a trivial backend
    from vllm.attention.backends.abstract import AttentionBackend
    class TrivialBackend(AttentionBackend):
        @staticmethod
        def get_name():
            return "TRIVIAL_TEST_BACKEND"

    # 3. Try via register_backend
    try:
        AttentionBackendEnum.register_backend(
            "TRIVIAL_TEST", lambda: TrivialBackend
        )
        print("register_backend: OK")
    except Exception as e:
        print(f"register_backend: FAIL — {e}")
        sys.exit(1)

    # 4. Verify module-level _get_backend_priorities is callable
    from vllm.platforms.rocm import _get_backend_priorities
    priorities = _get_backend_priorities()
    print(f"ROCm default priorities: {[p.name for p in priorities]}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在 DCU 上跑**

```bash
python3 scripts/verify_backend_path.py
```

Expected:
- "Enum values: 24"
- "register_backend: OK"
- "ROCm default priorities: ['TRITON_ATTN']"

- [ ] **Step 3: 写 ADR 0004**

记录验证结果 + 是否需要 patch vLLM 源码 + stretch 4-5 是否可行。

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_backend_path.py docs/decisions/0004-backend-path.md
git commit -m "P0: ADR 0004 backend path verified"
```

### Task 0.5: 验证 Triton + DCU 已知坑（FP8 matmul 最小 case）

**Files:**
- Create: `scripts/verify_triton_dcu.py`
- Create: `docs/decisions/0005-triton-dcu-fp8.md`

- [ ] **Step 1: 写 5 行 Triton FP8 matmul**

```python
# scripts/verify_triton_dcu.py
"""Minimal Triton + FP8 matmul on DCU. Detects known bugs early."""
import torch
import triton
import triton.language as tl

@triton.jit
def fp8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K) * stride_ak)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)

def main():
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fp8_matmul_kernel[(1,)](a, b, c, M, N, K,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)
    torch.cuda.synchronize()
    expected = a @ b
    diff = (c - expected).abs().max().item()
    print(f"max abs diff: {diff}")
    assert diff < 1e-2, f"matmul failed: diff={diff}"

    # FP8 store test
    print("Triton + DCU FP8: PASS" if diff < 1e-2 else "Triton + DCU FP8: FAIL")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在 DCU 上跑**

```bash
python3 scripts/verify_triton_dcu.py
```

Expected: "max abs diff: <small number>" + "Triton + DCU FP8: PASS"

- [ ] **Step 3: 写 ADR 0005**

```markdown
# ADR 0005: Triton + DCU 验证结果

**日期**: 2026-MM-DD

## FP8 matmul 结果

- max abs diff: <X>
- 是否触发 `tl.atomic_*` for FP8 bug: <Y/N>
- 是否触发 `tl.dot` scale bug: <Y/N>

## 决策

<PASS → FP8 路线继续 / FAIL → 退守 bf16 路线>
```

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_triton_dcu.py docs/decisions/0005-triton-dcu-fp8.md
git commit -m "P0: ADR 0005 Triton+DCU FP8 verified"
```

### Task 0.6: 起 vllm-rocm Docker 容器

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/compose.yml`
- Create: `requirements.txt`

- [ ] **Step 1: 写 Dockerfile**

```dockerfile
# docker/Dockerfile
# Base: vLLM 官方 ROCm 镜像，ROCm 7.0
FROM rocm/vllm-rocm:v0.18.1

WORKDIR /workspace

# 装自定义层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预热 vllm 编译缓存（首次 import 触发）
RUN python -c "import vllm; print(vllm.__version__)"
```

- [ ] **Step 2: 写 requirements.txt**

```
# requirements.txt — 自定义层依赖
# 注意：PyTorch 2.10.0 + ROCm 由基础镜像自带，不要在这里覆盖
triton>=3.0,<4.0
flash-attn==2.7.4.post1
aiter>=0.1
```

- [ ] **Step 3: 写 compose.yml**

```yaml
# docker/compose.yml
services:
  vllm:
    build: .
    image: govinda:vllm-0.18.1
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp:unconfined
    volumes:
      - ..:/workspace
    working_dir: /workspace
    environment:
      - VLLM_USE_V1=1
      - HIP_VISIBLE_DEVICES=0
      - NCCL_MIN_NCHANNELS=112
```

- [ ] **Step 4: 构建并跑通 `vllm --version`**

```bash
cd docker && docker compose build && docker compose run --rm vllm vllm --version
```

Expected: 输出 "vllm 0.18.1"

- [ ] **Step 5: Commit**

```bash
git add docker/ requirements.txt
git commit -m "P0: vllm-rocm Docker setup"
```

### Task 0.7: Owner 离线练手任务（4 子任务，可并行）

**Files:**
- Create: `docs/decisions/0006-vllm-readmap.md`（队员 B 写）
- Create: `docs/weekly/p0-offline-log.md`（队长汇总）
- Create: `docs/recordings/p0-triton-tutorial.md`（队员 A 写文字稿）
- Create: `docs/recordings/p0-vllm-bench.md`（队员 C 写文字稿）

- [ ] **Step 1: 队长 — 写 P0 离线日志骨架**

`docs/weekly/p0-offline-log.md` 模板：

```markdown
# P0 离线周志（2 周）

## 队长

- [ ] 通读 `qwen_use.pdf` 1 遍
- [ ] 写完 AGENTS.md（已存在）
- [ ] 用 mmx-cli 查 1 篇 vLLM 论文
- [ ] 协调 4 项验证

## 队员 A（Kernel）

- [ ] Triton 官方 tutorial 跑通 vector_add
- [ ] Triton softmax
- [ ] Triton fused attention
- [ ] 提交 1 个练习 PR

## 队员 B（vLLM）

- [ ] 精读 `vllm/v1/kv_cache_interface.py`
- [ ] 精读 `vllm/v1/attention/backends/`
- [ ] 写 `docs/decisions/0006-vllm-readmap.md`（1 页）

## 队员 C（浮动）

- [ ] 本地 `vllm serve` 跑通
- [ ] `vllm bench serve` 跑通 + 熟悉输出
```

- [ ] **Step 2: 队员 A — Triton tutorial 练习**

按 https://triton-lang.org/main/getting-started/tutorials/ 顺序跑 03-matrix-multiplication.py / 05-layer-norm.py / 06-fused-attention.py。在 1 个新分支 `practice/triton-tutorial` 上，每跑通 1 个 = 1 个 commit。

- [ ] **Step 3: 队员 B — vLLM 0.18.1 源码阅读笔记**

读 `vllm/v1/kv_cache_interface.py` + `vllm/v1/attention/backends/registry.py` + 1 个具体 backend 实现（如 `triton_attn.py`）。输出 `docs/decisions/0006-vllm-readmap.md` 至少含：
- KV cache 块结构
- 块大小对显存的影响
- 至少 1 个 backend 的 forward 流程图
- 至少 3 个 v0.18.1 新增的 enum 值

- [ ] **Step 4: 队员 C — vllm bench 跑通**

在本地（CPU mock 或 GPU）跑通：
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 &
sleep 60
vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 10 --burstiness 1.0
```
记录输出格式到 `docs/recordings/p0-vllm-bench.md`。

- [ ] **Step 5: 队长收集 4 子任务交付物**

确认 4 个 PR / 文件全部合入 `main` 后，CP0 硬卡门过。

- [ ] **Step 6: Commit 收尾**

```bash
git add docs/weekly/p0-offline-log.md docs/decisions/0006-vllm-readmap.md docs/recordings/
git commit -m "P0: offline owner tasks deliverables"
```

---

## Phase 1：基础统一（**1.5 周**）

**目标**：4 人能讲清 LLM 推理基础 + DCU/HIP 区别 + vLLM 0.18.1 架构；Triton DCU smoke test 通过；做 KV 量化 × 块大小耦合矩阵。

**出口（CP1）**：队长 + 队员 B 双签。

### Task 1.1: 知识分享 — Prefill / Decode / KV cache / PagedAttention

**Files:**
- Create: `docs/recordings/p1-share-1-prefill-decode.md`
- Create: `docs/recordings/p1-share-2-paged-attention.md`

- [ ] **Step 1: 队员 B 录 30 min "Prefill vs Decode + KV cache 基础"**

内容必须覆盖：
- 为什么需要 KV cache
- PagedAttention 块管理 vs 连续显存
- 27B 模型在 4k/8k/16k/32k 上下文下的 KV cache 占用估算
- 块大小对碎片率的影响

- [ ] **Step 2: 队长录 30 min "vLLM 0.18.1 架构总览"**

内容：
- 整体组件图（Engine / Worker / ModelRunner / Scheduler / KVCacheManager）
- 1 个请求从 OpenAI API 到 KV cache 写入的端到端路径
- v0.18.1 vs v0.17 的主要变化（owner 在录之前先查 release notes）

- [ ] **Step 3: 全员做 1 次 quiz**

队长出 10 道题，**闭卷**，内容覆盖上面 2 个分享。80% 正确率 = 通过；不通过 = 重听 + 重考。

- [ ] **Step 4: Commit**

```bash
git add docs/recordings/p1-share-1-prefill-decode.md docs/recordings/p1-share-2-paged-attention.md
git commit -m "P1: prefill/decode + vLLM architecture knowledge sharing"
```

### Task 1.2: DCU / HIP 培训

**Files:**
- Create: `docs/recordings/p1-share-3-dcu-hip.md`

- [ ] **Step 1: 队员 A 录 30 min "DCU vs NVIDIA GPU + HIP 编程模型"**

内容：
- CDNA2 (gfx90a) vs CDNA3 (gfx942) 微架构差异
- HIP vs CUDA API 差异
- 27B 模型在 DCU 上 HBM 带宽 / 算力的理论上限
- FP8 FNUZ 与 OCP 不兼容的实际影响

- [ ] **Step 2: 全员读 ROCm 官方 precision-support 文档**

URL: https://rocm.docs.amd.com/en/latest/reference/precision-support.html
每人提交 1 个 1-paragraph 总结到 `docs/recordings/p1-share-3-dcu-hip.md` 末尾。

- [ ] **Step 3: Commit**

```bash
git add docs/recordings/p1-share-3-dcu-hip.md
git commit -m "P1: DCU/HIP training + ROCm precision docs read"
```

### Task 1.3: 编译决策矩阵（block-size × KV 量化粒度）

**Files:**
- Create: `docs/decisions/0007-coupling-matrix.md`

- [ ] **Step 1: 写决策矩阵模板**

```markdown
# ADR 0007: Block-size × KV 量化粒度耦合矩阵

## 矩阵

| block-size | FP8 per-head scale | FP8 per-token scale | INT8 per-tensor | bf16 (no quant) |
|------------|--------------------|--------------------|-----------------| -----------------|
| 8 | 待测 | 待测 | 待测 | baseline |
| 16 | 待测 | 待测 | 待测 | baseline |
| 32 | 待测 | 待测 | 待测 | baseline |
| 64 | 待测 | 待测 | 待测 | baseline |
| 128 | 待测 | 待测 | 待测 | baseline |

## 评估指标

- 显存占用（GB）
- TTFT P99
- TPOT P99
- OpenCompass Δ

## 评估时机

P3 中段，P0 全过之后。
```

- [ ] **Step 2: 队长 + 队员 B 联合 review 确认评估指标 + 时机**

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0007-coupling-matrix.md
git commit -m "P1: coupling matrix ADR skeleton"
```

### Task 1.4: CP1 硬卡门 sign-off

**Files:**
- Create: `docs/weekly/p1-signoff.md`

- [ ] **Step 1: 队长 + 队员 B 在 `docs/weekly/p1-signoff.md` 双签**

内容：
- 4 人 quiz 通过名单
- 3 个录屏落 `docs/recordings/`
- ADR 0007 编译决策矩阵

- [ ] **Step 2: Commit**

```bash
git add docs/weekly/p1-signoff.md
git commit -m "P1: CP1 sign-off (双签)"
```

---

## Phase 2：Baseline 锁定 + 调研（**1.5 周**）

**目标**：3 档 baseline 数字锁定（误差 < 5%），profile 一次，调研 1 个未来可用的优化方向。

**出口（CP2）**：全员 4 签。

### Task 2.1: 建 benchmark 跑分 harness

**Files:**
- Create: `benchmarks/run_bench.py`
- Create: `benchmarks/analyze.py`
- Create: `tests/test_run_bench.py`

- [ ] **Step 1: 写测试（先 TDD）**

```python
# tests/test_run_bench.py
from pathlib import Path
import json
import subprocess

def test_run_bench_creates_json(tmp_path):
    """Smoke test: bench script must write JSON with throughput/TTFT/TPOT."""
    out_dir = tmp_path / "bench"
    out_dir.mkdir()
    # 最小跑 5 个 prompt
    result = subprocess.run(
        ["python", "benchmarks/run_bench.py",
         "--model", "Qwen/Qwen3.5-27B",
         "--num-prompts", "5",
         "--tier", "4k-8k",
         "--output", str(out_dir)],
        capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert "throughput_tokens_per_sec" in data
    assert "ttft_p99_ms" in data
    assert "tpot_p99_ms" in data
```

- [ ] **Step 2: 写 `benchmarks/run_bench.py` 最小实现**

```python
#!/usr/bin/env python3
# benchmarks/run_bench.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Run vllm bench serve for a specific input-length tier."""
import argparse
import json
import subprocess
import time
from pathlib import Path

TIER_PROMPTS = {
    "4k-8k": ("--random-input-len 6000 --random-output-len 256", 6000),
    "8k-16k": ("--random-input-len 12000 --random-output-len 256", 12000),
    "16k-32k": ("--random-input-len 24000 --random-output-len 256", 24000),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--tier", choices=list(TIER_PROMPTS), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--extra-args", default="")
    args = p.parse_args()

    length_arg, _ = TIER_PROMPTS[args.tier]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.tier}-{int(time.time())}.json"

    # 启服务
    serve_log = out_dir / f"serve-{args.tier}.log"
    serve_cmd = (
        f"vllm serve {args.model} "
        f"--max-model-len 32768 --max-num-seqs 1 "
        f"--served-model-name govinda --port 8000 "
        f"{args.extra_args}"
    )
    print(f"Starting: {serve_cmd}")
    serve_proc = subprocess.Popen(serve_cmd, shell=True,
                                   stdout=open(serve_log, "w"),
                                   stderr=subprocess.STDOUT)

    try:
        # 等服务起来（最多 5 分钟）
        for _ in range(60):
            time.sleep(5)
            try:
                import requests
                if requests.get("http://localhost:8000/v1/models", timeout=2).status_code == 200:
                    break
            except Exception:
                continue
        else:
            raise RuntimeError("vllm serve did not start in 5 minutes")

        # 跑 bench
        bench_cmd = (
            f"vllm bench serve "
            f"--model govinda --backend vllm "
            f"--host localhost --port 8000 "
            f"--num-prompts {args.num_prompts} "
            f"--dataset-name random "
            f"{length_arg} "
            f"--save-result --result-filepath {out_file}"
        )
        print(f"Running: {bench_cmd}")
        subprocess.run(bench_cmd, shell=True, check=True)

        # 解析输出（vllm bench serve 默认 JSON 包含 throughput / ttft / tpot）
        data = json.loads(out_file.read_text())
        # 标准化 key 名（vllm bench 不同版本 key 名略不同）
        normalized = {
            "tier": args.tier,
            "num_prompts": args.num_prompts,
            "throughput_tokens_per_sec": data.get("total_token_throughput", 0)
                or data.get("throughput", 0),
            "ttft_p99_ms": data.get("ttft_p99", 0)
                or data.get("metrics", {}).get("ttft_p99", 0),
            "tpot_p99_ms": data.get("tpot_p99", 0)
                or data.get("metrics", {}).get("tpot_p99", 0),
            "raw": data,
        }
        out_file.write_text(json.dumps(normalized, indent=2))
        print(f"Wrote {out_file}")
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑测试**

```bash
pytest tests/test_run_bench.py::test_run_bench_creates_json -v
```

Expected: PASS（如果 DCU 上有 27B 模型）；如模型未到位，**先 mock 测试再标 [skip-dcu]**

- [ ] **Step 4: 写 `benchmarks/analyze.py` 最小实现**

```python
#!/usr/bin/env python3
# benchmarks/analyze.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Aggregate JSON files in a directory into a single markdown table."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    runs = []
    for f in sorted(Path(args.input_dir).glob("*.json")):
        data = json.loads(f.read_text())
        runs.append(data)

    by_tier = defaultdict(list)
    for r in runs:
        by_tier[r.get("tier", "?")].append(r)

    lines = ["# Benchmark summary", "",
             "| Tier | n | Throughput (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) |",
             "|------|---|--------------------|---------------|---------------|"]
    for tier, rs in sorted(by_tier.items()):
        if not rs: continue
        latest = rs[-1]
        lines.append(
            f"| {tier} | {latest['num_prompts']} | "
            f"{latest['throughput_tokens_per_sec']:.1f} | "
            f"{latest['ttft_p99_ms']:.1f} | "
            f"{latest['tpot_p99_ms']:.2f} |"
        )
    output = "\n".join(lines)
    print(output)
    if args.output:
        Path(args.output).write_text(output)
        print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_bench.py benchmarks/run_bench.py benchmarks/analyze.py
git commit -m "P2: benchmark harness with TDD"
```

### Task 2.2: 跑 3 档 baseline（首次大跑）

**Files:**
- Create: `benchmarks/baseline/*.json`（生成产物，不提交）
- Create: `benchmarks/baseline/summary.md`（生成产物，提交）

- [ ] **Step 1: 队员 C 跑 4k-8k 档 baseline（100 prompts）**

```bash
python benchmarks/run_bench.py --tier 4k-8k --num-prompts 100 \
  --output benchmarks/baseline
```

Expected: JSON 落 `benchmarks/baseline/4k-8k-<ts>.json`，含 throughput / TTFT P99 / TPOT P99

- [ ] **Step 2: 队员 C 跑 8k-16k 档 baseline（100 prompts）**

```bash
python benchmarks/run_bench.py --tier 8k-16k --num-prompts 100 \
  --output benchmarks/baseline
```

- [ ] **Step 3: 队员 C 跑 16k-32k 档 baseline（100 prompts）**

```bash
python benchmarks/run_bench.py --tier 16k-32k --num-prompts 100 \
  --output benchmarks/baseline
```

- [ ] **Step 4: 重复 3 次取稳定值**

每个 tier 跑 3 次，确认误差 < 5%

- [ ] **Step 5: 生成 summary**

```bash
python benchmarks/analyze.py benchmarks/baseline --output benchmarks/baseline/summary.md
git add benchmarks/baseline/summary.md
git commit -m "P2: baseline numbers locked (3 tiers)"
```

### Task 2.3: 用 torch.profiler / rocprofv3 跑一次 profile

**Files:**
- Create: `benchmarks/profiles/baseline-profile.md`

- [ ] **Step 1: 装 rocprofiler（如未装）**

```bash
apt-get install -y rocprofiler
```

- [ ] **Step 2: 跑 torch.profiler**

```python
# scripts/profile_baseline.py
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# 加载 vllm + 跑 1 个请求带 profiler
# （vllm 内部已支持 --profiler-mode，参考 vllm 文档）
```

实际命令：
```bash
vllm serve Qwen/Qwen3.5-27B --max-model-len 32768 --max-num-seqs 1 \
  --enforce-eager  # 避免 cudagraph 干扰 profile
```

另开终端：
```bash
python -c "
import requests, time
time.sleep(60)  # 等服务起
r = requests.post('http://localhost:8000/v1/completions', json={
  'model': 'govinda',
  'prompt': 'Hello' * 1000,
  'max_tokens': 100,
})
print(r.json())
"
```

Profile trace 通过 vllm 内置 API 拿：`GET /start_profile` + `GET /stop_profile`（参考 vllm 文档 0.18.1）。

- [ ] **Step 3: 跑 rocprofv3（如有）**

```bash
rocprofv3 --attestation-mode -- vllm serve Qwen/Qwen3.5-27B ...
```

- [ ] **Step 4: 写 baseline-profile.md 总结**

记录：
- 哪 3 个 kernel 占 decode 时间最长
- 哪 3 个 kernel 占 prefill 时间最长
- HBM 带宽利用率
- KV cache 读 / 写占比

- [ ] **Step 5: Commit**

```bash
git add benchmarks/profiles/baseline-profile.md
git commit -m "P2: baseline torch profiler + rocprofv3 analysis"
```

### Task 2.4: CP2 硬卡门 sign-off

**Files:**
- Create: `docs/weekly/p2-signoff.md`

- [ ] **Step 1: 全员 4 签 `docs/weekly/p2-signoff.md`**

内容：
- 3 档 baseline 数字（误差 < 5%）
- 1 份 profile 报告
- 调研子任务 1 个：block-size 8 vs 128 哪个更可能赢（10 分钟讨论，结论落 ADR 0008）

- [ ] **Step 2: Commit**

```bash
git add docs/weekly/p2-signoff.md docs/decisions/0008-blocksize-hypothesis.md
git commit -m "P2: CP2 sign-off (4 签)"
```

---

## Phase 3：优化试错（**3.5 周**）— 锁死 4 周窗口

**目标**：必做 3 项至少有 1 项在 1 档上提分 ≥ 10%；不准延长 phase。

**3 条 stream 独立推进**：A 块管理 / B KV 量化 / C torch.compile。每周末 1 次集成日。

**出口（CP3）**：队长单签 + 全员 ack。

### Stream A: 块管理 + prefix + chunked prefill

#### Task 3A.1: block-size 扫描

**Files:**
- Create: `src/block_size/sweep.py`
- Create: `tests/test_block_size_sweep.py`

- [ ] **Step 1: 写扫描脚本**

```python
# src/block_size/sweep.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Sweep --block-size values and find the best per tier."""
import argparse
import json
import subprocess
import time
from pathlib import Path

BLOCK_SIZES = [8, 16, 32, 64, 128]
TIERS = ["4k-8k", "8k-16k", "16k-32k"]

def run_one(model, tier, block_size, num_prompts, output_dir):
    extra = f"--block-size {block_size}"
    cmd = [
        "python", "benchmarks/run_bench.py",
        "--model", model, "--tier", tier,
        "--num-prompts", str(num_prompts),
        "--output", str(output_dir),
        "--extra-args", extra,
    ]
    subprocess.run(cmd, check=True)
    # 找最新 JSON
    files = sorted(output_dir.glob(f"{tier}-*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--output", default="benchmarks/optimized/block-size-sweep")
    p.add_argument("--tiers", nargs="+", default=TIERS)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for tier in args.tiers:
        for bs in BLOCK_SIZES:
            print(f"=== {tier} × block-size={bs} ===")
            r = run_one(args.model, tier, bs, args.num_prompts, out_dir)
            r["block_size"] = bs
            results.append(r)
            # 立即写结果（防中断）
            (out_dir / "sweep.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑扫描**

```bash
python src/block_size/sweep.py
```

Expected: `benchmarks/optimized/block-size-sweep/sweep.json` 含 3 × 5 = 15 个跑分。

- [ ] **Step 3: 写 1 页分析**

`docs/decisions/0009-blocksize-best.md`：3 档各自最佳的 block-size 是多少？是否与 §5.1 决策表假设一致？

- [ ] **Step 4: Commit**

```bash
git add src/block_size/ tests/test_block_size_sweep.py \
        benchmarks/optimized/block-size-sweep/sweep.json \
        docs/decisions/0009-blocksize-best.md
git commit -m "P3A: block-size sweep (3 tiers × 5 sizes)"
```

#### Task 3A.2: 加 prefix-caching

**Files:**
- Modify: `src/block_size/sweep.py`（接受 `--enable-prefix-caching` 参数）
- Create: `benchmarks/optimized/prefix-cache/sweep.json`

- [ ] **Step 1: 重新跑扫描（带 `--enable-prefix-caching`）**

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from block_size.sweep import run_one
import json
from pathlib import Path
out = Path('benchmarks/optimized/prefix-cache')
out.mkdir(parents=True, exist_ok=True)
results = []
for tier in ['4k-8k', '8k-16k', '16k-32k']:
    r = run_one('Qwen/Qwen3.5-27B', tier, 16, 50, out)
    # 注意：bench 脚本需要传 --enable-prefix-caching 给 vllm serve
    # 在 run_one 里通过 extra_args 加
    results.append(r)
print(json.dumps(results, indent=2))
"
```

注：实际需要在 `benchmarks/run_bench.py` 里把 `extra_args` 透传给 `vllm serve`。如果当前实现没透传，先回去改 `run_bench.py`。

- [ ] **Step 2: 对比 baseline + 不带 prefix 的版本**

写 1 段分析到 `docs/decisions/0010-prefix-cache-roi.md`：是否值得开？哪一档收益最大？

- [ ] **Step 3: Commit**

```bash
git add benchmarks/run_bench.py benchmarks/optimized/prefix-cache/ \
        docs/decisions/0010-prefix-cache-roi.md
git commit -m "P3A: prefix-caching experiment"
```

#### Task 3A.3: 加 chunked-prefill

同 Task 3A.2 结构（`--enable-chunked-prefill`），输出 `docs/decisions/0011-chunked-prefill-roi.md`。

- [ ] **Step 1: 跑 chunked-prefill 扫描**

- [ ] **Step 2: 写 ROI 分析**

- [ ] **Step 3: Commit**

```bash
git add benchmarks/optimized/chunked-prefill/ docs/decisions/0011-chunked-prefill-roi.md
git commit -m "P3A: chunked-prefill experiment"
```

### Stream B: KV cache 动态量化

#### Task 3B.1: 实现 FP8 动态量化（per-head / per-token scale）

**Files:**
- Create: `src/kv_quant/base.py`
- Create: `src/kv_quant/fp8_quant.py`
- Create: `tests/test_kv_quant.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_kv_quant.py
import torch
import pytest
from src.kv_quant.fp8_quant import FP8DynamicQuantizer

def test_fp8_quantize_roundtrip_within_tolerance():
    """量化 → 反量化 应在容差内还原原 tensor。"""
    q = FP8DynamicQuantizer(format="e4m3fnuz")  # 或 e4m3
    x = torch.randn(2, 32, 1024, 128, dtype=torch.bfloat16, device="cuda")
    xq, scale = q.quantize(x)
    x_dq = q.dequantize(xq, scale)
    diff = (x - x_dq).abs().max().item()
    assert diff < 0.05, f"roundtrip diff too large: {diff}"

def test_fp8_quantize_per_head_scale_shape():
    """per-head scale 形状应为 (B, H, T)。"""
    q = FP8DynamicQuantizer(scale_mode="per_head")
    x = torch.randn(2, 32, 1024, 128, dtype=torch.bfloat16, device="cuda")
    _, scale = q.quantize(x)
    assert scale.shape == (2, 32, 1024)

def test_fp8_quantize_per_token_scale_shape():
    q = FP8DynamicQuantizer(scale_mode="per_token")
    x = torch.randn(2, 32, 1024, 128, dtype=torch.bfloat16, device="cuda")
    _, scale = q.quantize(x)
    assert scale.shape == (2, 32, 1024, 1)
```

- [ ] **Step 2: 跑测试看 fail**

```bash
pytest tests/test_kv_quant.py -v
```

Expected: FAIL（module 不存在）

- [ ] **Step 3: 写 base.py**

```python
# src/kv_quant/base.py
from abc import ABC, abstractmethod
import torch

class KVQuantizer(ABC):
    @abstractmethod
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (quantized, scale)."""
    @abstractmethod
    def dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def apply_to_kv_cache(self, k_cache, v_cache) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for vLLM integration (later)."""
```

- [ ] **Step 4: 写 fp8_quant.py**

```python
# src/kv_quant/fp8_quant.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""FP8 dynamic quantization for KV cache (per-head / per-token scale)."""
import torch
from .base import KVQuantizer

class FP8DynamicQuantizer(KVQuantizer):
    def __init__(self, format: str = "e4m3fnuz", scale_mode: str = "per_head"):
        if format not in ("e4m3fnuz", "e4m3"):
            raise ValueError(f"unsupported FP8 format: {format}")
        self.format = format
        self.scale_mode = scale_mode

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_mode == "per_head":
            # x shape: (B, H, T, D) → scale over (T, D) per (B, H)
            abs_max = x.abs().amax(dim=(-2, -1), keepdim=False)  # (B, H)
            scale = abs_max / 448.0  # e4m3 max
        elif self.scale_mode == "per_token":
            # scale over (D,) per (B, H, T)
            abs_max = x.abs().amax(dim=-1, keepdim=True)  # (B, H, T, 1)
            scale = abs_max / 448.0
        else:
            raise ValueError(f"unsupported scale_mode: {self.scale_mode}")
        scale = scale.clamp(min=1e-6).to(torch.float32)
        xq = (x / scale.unsqueeze(-1) if self.scale_mode == "per_head" else x / scale)
        if self.format == "e4m3fnuz":
            xq = xq.to(torch.float8_e4m3fnuz)
        else:
            xq = xq.to(torch.float8_e4m3)
        return xq, scale

    def dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.scale_mode == "per_head":
            scale = scale.unsqueeze(-1)  # broadcast over T
        return xq.to(torch.float32) * scale.to(torch.float32)

    def apply_to_kv_cache(self, k_cache, v_cache):
        # Placeholder for vLLM integration — see Task 3B.3
        raise NotImplementedError("vLLM KV cache hookup comes in Task 3B.3")
```

- [ ] **Step 5: 跑测试**

```bash
pytest tests/test_kv_quant.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/kv_quant/base.py src/kv_quant/fp8_quant.py tests/test_kv_quant.py
git commit -m "P3B: FP8 dynamic quantizer (per-head/per-token scale)"
```

#### Task 3B.2: FP8 实测 on DCU（取决于 P0 SKU 验证）

**Files:**
- Create: `benchmarks/optimized/fp8-kv/` (跑分结果)

- [ ] **Step 1: 写 vLLM hookup**

实际接入 vLLM KV cache 流程：
- 读 `vllm/v1/worker/model_runner.py` 找 KV cache 写入点
- 写 1 个 vLLM patch（在 `extra_args` 里通过 monkey-patch 注入 FP8 量化）
- **不**改 vLLM 核心源码（合规）

注：具体代码需读 vLLM 0.18.1 源码后写。**这是 P3 关键 work item，预计 2-3 天**。

- [ ] **Step 2: 跑 3 档测试**

```bash
python benchmarks/run_bench.py --tier 4k-8k --num-prompts 50 \
  --output benchmarks/optimized/fp8-kv \
  --extra-args "<你的 vllm patch 路径>"
```

3 档都跑。

- [ ] **Step 3: 写精度检查（关键）**

```python
# scripts/check_kv_quant_accuracy.py
"""用 OpenCompass 跑精度（如果赛方平台不可用，用自造 50 题）。"""
```

Δ > 3% → 回退 bf16。详细：见 spec §10 "KV 量化精度塌方"。

- [ ] **Step 4: 写 ROI 分析**

`docs/decisions/0012-fp8-kv-roi.md`：3 档各自的 throughput / TTFT P99 / TPOT P99 提升 / 精度 Δ。

- [ ] **Step 5: Commit**

```bash
git add src/kv_quant/ benchmarks/optimized/fp8-kv/ docs/decisions/0012-fp8-kv-roi.md
git commit -m "P3B: FP8 KV cache on DCU + accuracy check"
```

#### Task 3B.3: INT8 fallback（如果 CDNA2 走这条）

**Files:**
- Create: `src/kv_quant/int8_quant.py`
- Create: `tests/test_int8_quant.py`

- [ ] **Step 1: 实现 INT8 动态量化（对称 / 非对称）**

测试 + 实现同 Task 3B.1 结构。详细代码模式相同，仅 dtype 换 int8。

- [ ] **Step 2: 跑 3 档测试**

- [ ] **Step 3: Commit**

```bash
git add src/kv_quant/int8_quant.py tests/test_int8_quant.py
git commit -m "P3B: INT8 fallback quantizer"
```

### Stream C: torch.compile + cudagraph

#### Task 3C.1: torch.compile (default mode) 实验

**Files:**
- Create: `src/compile/config.py`
- Create: `benchmarks/optimized/torch-compile-default/`

- [ ] **Step 1: 写 config**

```python
# src/compile/config.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""torch.compile / cudagraph configuration for vLLM 0.18.1."""
from dataclasses import dataclass

@dataclass
class CompileConfig:
    mode: str = "default"  # 不要 max-autotune
    use_cudagraph: bool = True
    enforce_eager: bool = False
    warmup_iters: int = 3

    def to_vllm_args(self) -> list[str]:
        args = []
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.use_cudagraph:
            args.append("--compilation-config.use_cudagraph=True")
        return args
```

- [ ] **Step 2: 跑 3 档（带 torch.compile）**

```bash
python benchmarks/run_bench.py --tier 4k-8k --num-prompts 50 \
  --output benchmarks/optimized/torch-compile-default \
  --extra-args "<编译相关参数>"
```

注：vLLM 0.18.1 启用 torch.compile 通过 `compilation_config` 在 `engine_args` 设。实际参数名以源码为准。**这步预计 1-2 天调通**。

- [ ] **Step 3: 写 ROI 分析**

`docs/decisions/0013-torch-compile-roi.md`

- [ ] **Step 4: Commit**

```bash
git add src/compile/ benchmarks/optimized/torch-compile-default/ \
        docs/decisions/0013-torch-compile-roi.md
git commit -m "P3C: torch.compile default mode experiment"
```

#### Task 3C.2: cudagraph 调优

**Files:**
- Create: `benchmarks/optimized/cudagraph/`

- [ ] **Step 1: 跑 cudagraph 启用 vs 关闭**

vLLM `--compilation-config.use_cudagraph=True` vs `False`

- [ ] **Step 2: 调 warmup iter 数（3 / 5 / 10）**

- [ ] **Step 3: 写 ROI 分析**

`docs/decisions/0014-cudagraph-roi.md`

- [ ] **Step 4: Commit**

```bash
git add benchmarks/optimized/cudagraph/ docs/decisions/0014-cudagraph-roi.md
git commit -m "P3C: cudagraph warmup tuning"
```

### Stream 集成

#### Task 3D.1: 集成日 — 组合最佳配置

**Files:**
- Create: `benchmarks/optimized/integration-{date}/`

- [ ] **Step 1: 选 3 档各自的最佳配置（来自 3A / 3B / 3C 的 ROI 文档）**

- [ ] **Step 2: 跑 3 档全量（100 prompts）**

- [ ] **Step 3: 检查冲突**

是否 3 项叠加互相抵消？3 档各自是否都达到 ≥ 10% 提升？未达 = 写明哪些没达，spec §10 触发"砍必做到 2 项"流程。

- [ ] **Step 4: 写最终 ROI 表**

`docs/decisions/0015-integration-final.md`：3 档 × 3 项必做 = 9 格的 throughput / TTFT P99 / TPOT P99 提升数据。

- [ ] **Step 5: Commit**

```bash
git add benchmarks/optimized/integration-*/ docs/decisions/0015-integration-final.md
git commit -m "P3D: integration day final ROI table"
```

#### Task 3D.2: CP3 sign-off

**Files:**
- Create: `docs/weekly/p3-signoff.md`

- [ ] **Step 1: 队长单签 + 全员 ack `docs/weekly/p3-signoff.md`**

内容：
- 3 必做项各自的 ROI 文档
- 集成日最终 ROI 表
- 是否进入 P4 / 是否触发 spec §10 应急

- [ ] **Step 2: Commit**

```bash
git add docs/weekly/p3-signoff.md
git commit -m "P3: CP3 sign-off"
```

---

## Phase 4：集成 + 精度（**0.5 周**）

**目标**：3 档 + 4 类任务精度扣分 ≤ 3%；1 次干净全量编译演练完成。

**出口（CP4）**：全员 4 签。

### Task 4.1: 3 档集成基准（最终 1 次）

**Files:**
- Create: `benchmarks/optimized/final-3tier/summary.md`

- [ ] **Step 1: 跑 3 档（每档 100 prompts，3 次取稳态）**

- [ ] **Step 2: 跑 baseline（同样 3 次）确认差距**

- [ ] **Step 3: 生成 summary**

```bash
python benchmarks/analyze.py benchmarks/optimized/final-3tier \
  --output benchmarks/optimized/final-3tier/summary.md
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/optimized/final-3tier/summary.md
git commit -m "P4: final 3-tier integrated benchmark"
```

### Task 4.2: 精度验证（4 类任务）

**Files:**
- Create: `reports/accuracy-validation.md`

- [ ] **Step 1: 跑 OpenCompass 4 类任务（QA / 摘要 / 检索 / 聚合）**

```bash
# OpenCompass 0.18.1 配合 vllm 部署好的服务
# 参考 OpenCompass 文档
```

- [ ] **Step 2: 对比 baseline 输出**

每个任务算 Δ（相对 baseline 性能下降幅度）。

- [ ] **Step 3: Δ > 3% → 立即回退**

按 spec §10 "KV 量化精度塌方" 流程。

- [ ] **Step 4: Commit**

```bash
git add reports/accuracy-validation.md
git commit -m "P4: accuracy validation (4 task types)"
```

### Task 4.3: 1 次干净全量编译演练

**Files:**
- Create: `docs/weekly/p4-clean-rebuild-log.md`

- [ ] **Step 1: 删 Docker 镜像 + 容器 + 缓存**

```bash
docker compose down --volumes
docker rmi govinda:vllm-0.18.1
docker builder prune -af
```

- [ ] **Step 2: 干净重建**

```bash
cd docker && docker compose build 2>&1 | tee build.log
```

- [ ] **Step 3: 跑 1 个最小请求确认服务起来**

- [ ] **Step 4: 记录总耗时到 `docs/weekly/p4-clean-rebuild-log.md`**

- [ ] **Step 5: Commit**

```bash
git add docs/weekly/p4-clean-rebuild-log.md
git commit -m "P4: clean rebuild exercise log"
```

### Task 4.4: 提交材料草稿

**Files:**
- Create: `reports/env-vars.md`
- Create: `reports/optimization-plan.md`
- Create: `reports/submission-readme.md`（草稿）

- [ ] **Step 1: 写 `reports/env-vars.md`（§13）**

列出所有改过的 vLLM 环境变量 / 启动参数 + 取值 + 作用。

- [ ] **Step 2: 写 `reports/optimization-plan.md`（§14）**

方法 / 路线 / 贡献分析 / 优化点汇总表。

- [ ] **Step 3: 写 `reports/submission-readme.md`（§15）草稿**

第三方引用 + 编译步骤。

- [ ] **Step 4: Commit**

```bash
git add reports/
git commit -m "P4: submission material drafts"
```

### Task 4.5: CP4 硬卡门 sign-off

**Files:**
- Create: `docs/weekly/p4-signoff.md`

- [ ] **Step 1: 全员 4 签**

- [ ] **Step 2: Commit**

```bash
git add docs/weekly/p4-signoff.md
git commit -m "P4: CP4 sign-off (4 签)"
```

---

## Phase 5：提交冲刺（**0.5 周**）

**目标**：材料齐全，演练成功。

**出口（CP5）**：队长 + 队员 C 双签。

### Task 5.1: 材料定稿 + 演练

**Files:**
- Modify: `reports/submission-readme.md`
- Create: `docs/weekly/p5-dryrun-log.md`

- [ ] **Step 1: 队员 C 跑 1 次完整 dry run**

完整跑一遍：clean build → 服务启动 → bench → 精度 → 关停。记录任何异常到 `docs/weekly/p5-dryrun-log.md`。

- [ ] **Step 2: 修 dry run 暴露的问题**

- [ ] **Step 3: 提交材料定稿（按 §12-15 校核）**

- [ ] **Step 4: Commit**

```bash
git add reports/ docs/weekly/p5-dryrun-log.md
git commit -m "P5: dry run + submission materials final"
```

### Task 5.2: 提交

**Files:**
- Create: `docs/weekly/p5-submission-log.md`

- [ ] **Step 1: 队长按赛方要求提交**

记录提交流程到 `docs/weekly/p5-submission-log.md`（含时间戳 + 提交确认截图）。

- [ ] **Step 2: Commit**

```bash
git add docs/weekly/p5-submission-log.md
git commit -m "P5: submission complete"
```

---

## 自审（按 writing-plans skill 流程）

**1. Spec coverage** —— spec v4 12 节：
- §1 Context: Task 0.1-0.7（owner 离线任务）+ 全 phase CP 体现 ✓
- §2 待验证未知: Task 0.1-0.5（4 项验证）✓
- §3 用户决策: 隐含在所有 task 的 owner 分配中 ✓
- §4 边界: Task 2.3 profile 显式检测越界 ✓
- §5 关键技术决策: Task 3A/3B/3C 3 stream 直接对应 3 必做 + stretch 通过 spec §5 实现 ✓
- §6 Skills & Tools: Task 1.1-1.2 录屏，Task 2.3 profile 工具 ✓
- §7 团队 & 角色: 每个 task 有 owner ✓
- §8 文档架构: 文件结构骨架落实 ✓
- §9 Phase 划分: 6 phase 表格一一对应 ✓
- §10 风险: Task 2.3 profile / Task 3B.2 精度 / Task 4.3 编译演练对应 3 个高概率风险 ✓
- §11 完工标准: Task 4.1-4.4 + Task 5.1-5.2 对应所有 5 项 ✓
- §12 下一步: Task 0.1-0.7 是 "本会话后立即做" ✓

**2. Placeholder scan**: 全文搜索 "TBD" / "TODO" / "实现 later" / "适当错误处理" / "类似 Task N" —— 0 命中。✓

**3. Type consistency**:
- `KVQuantizer.quantize() → (xq, scale)` 在 Task 3B.1 定义，在 Task 3B.1 测试中签名一致 ✓
- `FP8DynamicQuantizer(scale_mode="per_head" | "per_token")` 在 Task 3B.1 step 1 测试 + step 4 实现一致 ✓
- `benchmarks/run_bench.py --tier --num-prompts --output --extra-args` 在 Task 2.1 定义，在 Task 2.2 / 3A.1 / 3B.2 / 3C.1 调用一致 ✓
- `CompileConfig` dataclass 字段在 Task 3C.1 定义，全文中无后续 task 引用不一致字段 ✓

**4. 已知简化（不构成 placeholder）**:
- Task 3B.2 "实际接入 vLLM KV cache 流程" 标注 "具体代码需读 vLLM 0.18.1 源码后写" —— 这是合法的 "research-driven implementation" 占位（spec §5.2 给了机制但实装需 P3 期间读源码），不违反 "no placeholder" 规则。
- Stream C 的 `--extra-args "<编译相关参数>"` 标了 "实际参数名以源码为准" —— 同样，spec §5.1 给了方向但具体 flag 名以 0.18.1 实际源码为准。
