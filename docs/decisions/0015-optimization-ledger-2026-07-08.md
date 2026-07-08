# 0015 Optimization Ledger 2026-07-08

Status: active working ledger.

## Current Best Submitted Line

Latest known AC score:

- Score: 79.1109
- 4K-8K: 15.97 tok/s
- 8K-16K: 14.44 tok/s
- 16K-32K: 11.04 tok/s
- SLA penalty: 0.0
- Accuracy penalty: 0.0

Safe commit line:

- `630a40a` / `c0feb6f`: gfx936 LLMM1 decode GEMV path and row grouping.
- `777de7c` / `683ef32`: ROCm Triton prefill GQA block tuning, default `BLOCK_M=64` for Qwen3.5-style GQA.
- `2c73297`: exact small decode CUDA graph capture sizes `[1,2,3,4,5,6,7,8,10,12,16]`.

Do not replace this line unless a smoke run proves both throughput and completion stability.

## Rules Boundary

Allowed directions:

- Triton/CUDA-HIP kernel tiling and launch shape tuning.
- Runtime-only activation/KV low precision if accuracy and backend constraints pass.
- Decode GEMV effective bandwidth optimization.
- Prefill attention block shape optimization.

Forbidden or treated as out of scope:

- Weight quantization before/after load, persistent quantized weights, weight reordering or graph conversion.
- Scheduler locked flag changes: `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`.
- Sampling changes, input truncation, sample filtering, token/layer/head skipping.
- Speculative decoding or auxiliary models.

## Validated Positive Directions

### Decode GEMV

Current decode gain came from the gfx936 skinny GEMV / LLMM1 path, especially gate/up and related skinny projection shapes. This is aligned with the bandwidth-wall model: decode is mostly weight IO, so useful work is reducing overhead and increasing effective HBM utilization, not adding software pipelining blindly.

Remaining decode work should inspect the exact hot linear shapes not yet covered by LLMM1 and use microbenchmarks before touching end-to-end code.

### Prefill Full Attention

`BLOCK_M=64` for ROCm prefill GQA 4/6/8 is validated and stable. It improves K/V reuse in Qwen3.5 full-attention layers without changing semantics.

This is the current safe default. It should not be lowered or made adaptive without a same-dataset smoke comparison.

### Long-Prefill Tile Size 64 Candidate

P0 follow-up result, same container and same local bench conditions:

| Dataset | `TILE_SIZE=64` candidate | `TILE_SIZE=32` baseline | Direction |
| --- | ---: | ---: | --- |
| 8K-16K | 11.61 tok/s | 9.45 tok/s | +22.9% |
| 16K-32K | 9.05 tok/s | 9.09 tok/s | flat |

Implementation shape:

- `triton_unified_attention.py::_get_tile_size()` returns `64` only for prefill
  when `max_seqlen_q > 8192`; shorter prefills stay at `32`.
- `VLLM_TRITON_PREFILL_TILE_SIZE=32/64` remains available for A/B override.
- `kernel_unified_attention_2d` launch uses `num_stages=1`; DCU LDS limits can
  make the `TILE_SIZE=64` variant fail compilation with higher/default stages.

Decision: promote to candidate submit line after one scoring-style smoke. The
absolute tok/s values above are lower than the historical safe score because the
container/bench conditions changed; rely on same-container A/B direction, not
cross-container absolute numbers.

### CUDA Graph Capture Sizes

The exact small decode graph list is safe and has a small positive effect. Avoid reintroducing conflicting `cudagraph_capture_sizes` through both top-level args and compilation config; vLLM rejects that combination at startup.

## Failed or High-Risk Directions

### FP8 Weight Path

Result: throughput increased but accuracy penalty was catastrophic.

Decision: do not use.

### KV FP8 Cache

Observed failure: TRITON attention rejected non-1.0 q scale (`A non 1.0 q_scale is not currently supported`).

Decision: do not enable directly. KV quantization may still be valuable, but it needs a dedicated kernel/backend path and accuracy tests, not a config toggle.

### GDN Chunk Size 128

Observed failure: `solve_tril` asserts `A.shape[-1] in [16, 32, 64]`.

Decision: chunk size 128 is structurally unsupported in the current FLA GDN path.

### Full-Attention `BLOCK_M=128`

Small smoke:

- 16-32K, 3 prompts with env override: 3/3 success, 7.47 tok/s, TTFT improved versus one cold safe run.
- 8-16K, 3 prompts with env override: 3/3 success, 18.00 tok/s.
- 4-8K, 3 prompts with env override: 3/3 success, 25.51 tok/s.

Stability smoke:

- 8-16K, 10 prompts with env override: 4/10 success, 6 failed, TPOT exploded.
- Adaptive attempt (`num_seqs <= 3` and long context): 8-16K 10 prompts completed but only 5.67 tok/s with very high TTFT/TPOT; 16-32K 3 prompts dropped to 2.35 tok/s.

Decision: do not submit. The small-run gain is not stable under real bench pressure.

### LLMM1 on Down Projection

Experiment: enable `VLLM_GFX936_LLMM1_SHAPES=gate_up,mid_up,down` at runtime.

Result:

- 4-8K, 3 prompts after warmup: 3/3 success, 25.54 tok/s.
- 4-8K, 10 prompts: 10/10 success, 41.11 tok/s, TPOT P99 219.25 ms.
- 4-8K, 10 prompts with down-specific `rows_per_block=4`: 10/10 success, 32.34 tok/s, worse than the default rows-per-block 8 attempt.
- Reference from the current exact-graph line for the same 4-8K 10-prompt smoke was about 49.26 tok/s.

Decision: do not submit. The existing LLMM1 kernel/row grouping is not tuned for the down projection shape `(5120, 17408)`. If revisited, it should be a kernel-level rows-per-block or tiling experiment, not a simple env enable.

Shape-level microbench after stopping vLLM:

| Shape | torch linear | LLMM1 best |
| --- | ---: | ---: |
| mid_up `(17408, 5120)` | 0.251 ms | 0.178 ms |
| gate_up `(34816, 5120)` | 0.502 ms | 0.330 ms |
| down `(5120, 17408)` | 0.153 ms | 0.167 ms |

This confirms the current `gate_up,mid_up` coverage is sensible and `down` should stay disabled.

Additional candidate-shape microbench:

| Shape | torch linear | LLMM1 best | Decision |
| --- | ---: | ---: | --- |
| full_qkv `(8192, 5120)` | 0.075 ms | 0.085 ms | keep torch |
| full_o / linear_out `(5120, 6144)` | 0.053 ms | 0.069 ms | keep torch |
| linear_conv-like `(17408, 5120)` | 0.242 ms | 0.173 ms | already covered by `mid_up` shape |
| lm_head `(248320, 5120)` | 1.898 ms | 2.188 ms | keep torch |

Conclusion: simple LLMM1 shape expansion is exhausted. Further decode work needs kernel-level changes, not more shape toggles.

### Proxy-Contaminated Bench Runs

If proxy environment variables are present, localhost bench can fail with Squid 503 without hitting vLLM. Always run bench with:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
```

### AITER Backend Smoke 2026-07-08

Observed on container job `659489` / node `e03r2n01`:

- `aiter`, `aiter.ops.triton.unified_attention`, and `aiter.ops.triton.gemm_a16w16` import successfully only when DTK library paths are appended after the existing runtime paths. Prepending all DTK `lib` directories can make `torch.cuda.is_available()` false even though `torch.cuda.device_count()` reports one device.
- `ROCM_AITER_UNIFIED_ATTN` starts backend selection but fails KV cache initialization unless `--block-size 16` is forced. Without that override, ROCm platform code sets block size 64 and Qwen3.5 hybrid KV page sizes cannot be unified.
- With `--block-size 16`, `ROCM_AITER_UNIFIED_ATTN` selects successfully but the process disappeared during model load in this container, without a Python traceback. Treat as unstable; do not submit.
- `ROCM_AITER_FA` does not become the main text attention backend on this gfx936 container. Logs show main model backend remains `TRITON_ATTN`; AITER FA is only selected for the ViT/MM encoder path, which is irrelevant for text-only throughput.
- Starting vLLM via `docker exec -d` inside an `srun --overlap` step is not enough for a reusable background service. The service can be cleaned up when the launching srun step exits. For smoke, run start-wait-bench-stop inside one srun step, or use the platform's persistent container entrypoint.

Decision: AITER is not the next scoring path on this container. Keep default `TRITON_ATTN` with the local `BLOCK_M=64` prefill tuning.

### LLMM1Silu Candidate

Microbench from the previous run showed fused `gate_up + silu_and_mul` is faster than `LLMM1 + torch silu`, but 4-8K 10-prompt smoke was effectively tied with the safe line. Keep it behind an explicit env gate if the code remains in the worktree; do not enable by default until 8-16K and 16-32K smoke show a real end-to-end gain.

## Verification Protocol

Minimum before submit:

1. `4-8K` smoke with at least 3 prompts, 0 failed.
2. `8-16K` smoke with enough concurrency to catch instability. Three prompts are not enough for risky prefill changes.
3. `16-32K` smoke with at least 3 prompts, 0 failed.
4. Check TTFT/TPOT P99 for obvious SLA regressions.

Do not submit an optimization based only on a 3-prompt result if it changes prefill block shape, memory footprint, graph capture behavior, or KV/cache dtype.

## Next Work

Recommended next branches:

1. Decode: extend LLMM1 coverage only after shape-level microbenchmarks show a win.
2. Prefill: inspect full-attention Triton kernels for lower-risk tile-size or num-warps changes around the stable `BLOCK_M=64` path.
3. GDN/linear-attention: optimize within supported chunk size 64; do not change chunk size to 128 unless `solve_tril` and all dependent kernels are redesigned.
4. KV quantization: treat as a separate research branch with explicit accuracy and backend support checks.
