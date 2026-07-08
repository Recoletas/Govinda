# 0016 Next Optimization Roadmap 2026-07-08

Status: active roadmap.

Owner: recoletas + agents.

Purpose: container is temporarily unavailable, so this document consolidates the
local workspace state, verified results, failed paths, and the next engineering
directions. The goal is to prevent another cycle of parameter fiddling, unstable
backend switching, or accidental submission of dirty experiments.

## Current Workspace Map

There are two separate layers in `/home/recoletas/Govinda`:

- Outer project workspace: docs, scripts, smoke helpers, ADRs, and local
  coordination files.
- Scoring code tree: `vllm_cscc_work/`, the vLLM source that should be pushed
  for scoring after verification.

Current scoring code safe line:

- Repo: `vllm_cscc_work`
- Safe HEAD: `2c73297 Capture exact Qwen decode graph sizes`
- Latest known safe score: `79.1109`
- Throughput: `4K-8K=15.97`, `8K-16K=14.44`, `16K-32K=11.04`
- Penalties: `SLA=0.0`, `accuracy=0.0`

Do not submit dirty worktree state until each diff below is classified and
either verified or reverted.

## Dirty Diff Classification

Current dirty files in `vllm_cscc_work`:

| File | Classification | Decision |
| --- | --- | --- |
| `vllm/envs.py` | AITER default-on experiment | Revert before submit. AITER was unstable and not the current scoring path. |
| `vllm/v1/attention/ops/vit_attn_wrappers.py` | AITER ViT fallback experiment | Revert unless a later AITER branch is explicitly revived. Text benchmark does not need this. |
| `csrc/rocm/ops.h` | `LLMM1Silu` custom op declaration | Keep only in a separate decode-fusion branch. Not verified for submit. |
| `csrc/rocm/skinny_gemms.cu` | `LLMM1Silu` custom kernel | Keep only behind env gate and microbench. Not default. |
| `csrc/rocm/torch_bindings.cpp` | `LLMM1Silu` binding | Same as above. |
| `vllm/_custom_ops.py` | `LLMM1Silu` Python wrapper | Same as above. |
| `vllm/model_executor/models/qwen2_moe.py` | MLP fused path experiment | Keep only if shape probe confirms Qwen3.5 dense MLP shape and smoke improves. Current hard-coded `(34816,5120)` is not safe as a general default. |

Outer workspace dirty files:

| Path | Classification | Decision |
| --- | --- | --- |
| `docs/decisions/0015-optimization-ledger-2026-07-08.md` | Useful ledger | Keep. |
| `docs/decisions/0016-next-optimization-roadmap-2026-07-08.md` | This roadmap | Keep. |
| HTML tutorial files in `docs/` | Reference material from user | Keep. |
| `tools/codex_*` | Local smoke/microbench helpers | Keep local unless needed in final report. |
| `scripts/run_fp8_smoke.sh` | FP8 KV smoke script modified from `fp8_e4m3fnuz` to `fp8` | Do not use for submit; FP8 KV direct path is high risk. |
| `scripts/run_today_work.sh` | Local orchestration helper | Keep local; review before use. |

## Rule Boundary

Allowed and still worth pursuing:

- Triton/HIP kernel tiling, block shape, launch shape, and memory-access
  optimization.
- Runtime-only activation/KV cache low precision if it does not persist
  transformed weights or reusable outputs.
- Custom kernels compiled in the scoring container.
- CUDA graph capture size changes if they do not conflict with vLLM config and
  are verified with bench command behavior.

Forbidden or too risky for this contest line:

- Persistent weight quantization or weight rewrite/reorder/compression.
- FP8 weight path: already showed high throughput but catastrophic accuracy
  penalty.
- Scheduler locked flag changes: `--max-model-len`, `--max-num-seqs`,
  `--max-num-batched-tokens`, or equivalent behavioral changes.
- Skipping layers/tokens/heads, speculative decoding, auxiliary models,
  answer/testset caching, truncating samples, or filtering hard cases.

## Evidence From User Tutorials

The DCU decode tutorial supports two conclusions that match our measurements:

1. Decode is weight-IO bound. A single-token GEMV spends almost all time reading
   weights from HBM; adding software double buffering does not help if achieved
   bandwidth is already near peak.
2. Prefill is compute/attention bound. Weight IO is amortized across many
   tokens, and long-context full attention grows roughly as `O(S^2)`.

The bandwidth-utilization tutorial gives the useful checklist:

- If achieved bandwidth is far below the read-only probe, optimize coalescing,
  alignment, kernel specialization, and fusion.
- If a kernel is already near bandwidth peak, stop adding concurrency or double
  buffering; reduce bytes or fuse surrounding launches.
- For decode GEMV, generic BLAS may be bad for `M=1`; custom skinny GEMV is
  justified only for shapes whose microbench proves it beats torch/rocBLAS.

Local evidence agrees:

- LLMM1 helps `gate_up` and `mid_up`.
- LLMM1 hurts `down`, `full_qkv`, `full_o`, and `lm_head` in current form.
- Prefill `BLOCK_M=64` produced the largest stable improvement so far.
- `BLOCK_M=128` looked good on 3 prompts but failed real stability smoke.

## Direction A: Prefill Attention Kernel, Main Scoring Path

Priority: P0.

Why:

- It is the only path that produced the jump from roughly high-60s to 79.
- It targets the long-context bottleneck directly.
- It is compliant kernel tuning, not semantic change.

Files to inspect first:

- `vllm_cscc_work/vllm/v1/attention/ops/prefix_prefill.py`
- `vllm_cscc_work/vllm/v1/attention/ops/triton_prefill_attention.py`
- `vllm_cscc_work/vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm_cscc_work/vllm/v1/attention/backends/rocm_attn.py`

Next experiments:

1. Keep safe default `BLOCK_M=64`.
2. Tune around it, not away from it:
   - `BLOCK_N`
   - `num_warps`
   - `num_stages`
   - cache/unroll constants for GQA K/V reuse
3. Check that all full-attention prefill variants actually hit the tuned path:
   - pure prefill
   - chunked prefill
   - prefix/paged prefill
4. Add shape-specific branches only for Qwen3.5 GQA conditions, not broad
   global defaults.

Do not:

- Re-enable `BLOCK_M=128` by default.
- Accept 3-prompt success as proof for this path.
- Change scheduler limits to make prefill look better.

Minimum validation after container returns:

- 4K-8K 3 prompts: startup and completion smoke.
- 8K-16K 10 prompts: catches the `BLOCK_M=128` style instability.
- 16K-32K 3 prompts: checks long-context TTFT/SLA direction.

Current P0 candidate:

- Keep `BLOCK_M=64`.
- Add `VLLM_TRITON_PREFILL_TILE_SIZE=32/64` override.
- Use `TILE_SIZE=64` only when
  `8192 < max(max_seqlen_q, max_seqlen_k) <= 16384`; keep 4K-8K and 16K-32K on
  `32` by default.
- Launch the 2D unified attention kernel with `num_stages=1` to avoid DCU LDS
  resource failures.
- Same-container A/B showed `8K-16K` improving from `9.45` to `11.61` tok/s
  and `16K-32K` flat at about `9.05` tok/s. Cross-container absolute values are
  not comparable to the historical `79.1109` safe score.
- Rationale for the narrowed band: the broad KV-length trigger submitted as
  `1b7f156` improved official 8K-16K throughput but added `0.5974` accuracy
  penalty and did not move official 16K-32K throughput. Do not expose the long
  band to a different online-softmax tile partition unless it proves a real
  score gain.

## Direction B: Decode Skinny GEMV and Fusion

Priority: P1.

Why:

- Decode remains important, but current simple shape expansion is exhausted.
- The correct target is effective HBM utilization and launch/fusion overhead,
  not generic parameter tuning.

Known good:

- `gate_up (34816, 5120)` with LLMM1.
- `mid_up (17408, 5120)` with LLMM1.

Known bad or not worth enabling with current kernel:

- `down (5120, 17408)`
- `full_qkv (8192, 5120)`
- `full_o / linear_out (5120, 6144)`
- `lm_head (248320, 5120)`

Important model-path finding:

- `qwen3_5.py` imports `Qwen2MoeMLP as Qwen3NextMLP`.
- Therefore `qwen2_moe.py` MLP changes can affect Qwen3.5 dense layers.
- The current `LLMM1Silu` experiment is hard-coded for `(34816,5120)` and
  should not be assumed correct until the live model config and weight shapes
  are probed again inside the container.

Next experiments:

1. Probe exact Qwen3.5 dense MLP shapes from live model config and loaded
   module weights.
2. Microbench `LLMM1Silu` for those exact shapes.
3. If microbench is positive, keep it behind
   `VLLM_GFX936_FUSED_GATE_UP_SILU=1`.
4. Smoke end-to-end before making it default.

Do not:

- Enable LLMM1 on `down` again without a different kernel design.
- Add more LLMM1 shapes only because they look similar.
- Submit `LLMM1Silu` unless 8K-16K and 16K-32K show real end-to-end gain.

## Direction C: Runtime INT8 KV Cache

Priority: P2 research branch, not immediate submit path.

Why:

- It is compliant and could help long-context memory pressure.
- It is not a config-toggle feature in this tree.
- The risk is correctness/accuracy and dequant overhead.

Current code evidence:

- `csrc/cache_kernels.cu` has `reshape_and_cache_flash` with `k_scale/v_scale`
  and supports scale shapes `[1]` or `[num_heads]`.
- `csrc/rocm/attention.cu` ROCm attention dispatch explicitly handles `auto`
  and `fp8/fp8_e4m3`, not a complete INT8 KV path.
- `vllm/v1/attention/backends/rocm_attn.py` checks quantized cache mostly via
  `kv_cache_dtype.startswith("fp8")`.
- ADR 0012 already found no hidden Hygon INT8 KV cache implementation; existing
  INT8 ops are generic activation/communication utilities, not KV cache.

Correct implementation shape:

1. Cache write path:
   - bf16 K/V -> int8 K/V
   - per-head or per-block scale
   - no disk persistence
2. Attention read path:
   - fused int8 dequant inside attention kernel
   - avoid a separate dequant kernel
3. Metadata:
   - scale tensors are runtime-only
   - do not create reusable quantized cache files

Expected benefit:

- Mostly 16K-32K.
- 4K-8K may show little change because decode is dominated by weight IO.

Do not:

- Use FP8 weight quantization.
- Use direct `--kv-cache-dtype fp8` as a shortcut; it already failed on q-scale
  constraints and FP8 on this card/path is risky.
- Treat `scripts/run_fp8_smoke.sh` as a submit path.

## Direction D: AITER Backend

Priority: paused.

Why:

- Import can be fixed with careful `LD_LIBRARY_PATH`, but backend behavior was
  not stable enough.
- `ROCM_AITER_FA` did not become the main text backend in observed logs.
- `ROCM_AITER_UNIFIED_ATTENTION` needs `--block-size 16` to get past one KV
  issue, then still disappeared during model load.

Current decision:

- Revert default-on AITER changes before any submit.
- Only revisit in a clean branch and clean container with a single
  start-wait-bench-stop command, never as mixed dirty default.

## Direction E: GDN / Linear Attention

Priority: P3.

Why:

- Qwen3.5 hybrid architecture has many linear-attention/GDN layers, so this may
  matter.
- But the obvious chunk-size 128 attempt is structurally unsupported:
  `solve_tril` asserts chunk sizes in `[16, 32, 64]`.

Next experiments:

- Stay within chunk size 64.
- Inspect supported FLA/GDN kernels for launch overhead or shape-specific paths.
- Only proceed after prefill and decode-fusion branches are cleaner.

## Container Recovery Execution Order

When a new container is available:

1. Copy or mount only the safe line first:
   - `vllm_cscc_work` at `2c73297`
   - no dirty AITER default-on files
2. Confirm model config and actual loaded MLP shapes:
   - run `tools/codex_qwen35_config_probe.py`
   - add a small module-shape probe if needed
3. Run one baseline smoke using the safe line:
   - same bench command family as official scoring
   - proxy env unset
4. Start P0 prefill branch:
   - only change one kernel parameter group around `BLOCK_M=64`
   - validate with short smoke, then 8K-16K 10 prompts
5. Start P1 decode-fusion branch only after P0 is either positive or exhausted.

## 2026-07-09 Active Submit And Experiment Lines

Scoring `main` on GitLab:

- `3e2df97 Add long prefill tile64 policy switch`
- Default behavior keeps the broad long-prefill `TILE_SIZE=64` policy
  (`max(max_seqlen_q, max_seqlen_k) > 8192`).
- Runtime policy switch:
  - `VLLM_TRITON_PREFILL_TILE64_POLICY=broad` default, use 64 for all >8K.
  - `VLLM_TRITON_PREFILL_TILE64_POLICY=mid`, use 64 only for 8K-16K.
  - `VLLM_TRITON_PREFILL_TILE64_POLICY=long`, use 64 only for >16K.
  - `VLLM_TRITON_PREFILL_TILE64_POLICY=off`, keep prefill tile 32.

Latest official feedback on this line:

- Score: `79.0567`
- Throughput: `4K-8K=15.97`, `8K-16K=14.82`, `16K-32K=11.04`
- Penalties: `SLA=0.0`, `accuracy=0.5974`
- Working hypothesis: the broad `TILE_SIZE=64` policy changes the online
  softmax reduction partition for long prefill. It improves the 8K-16K
  throughput band, but the official 16K-32K throughput did not move, so exposing
  the 16K-32K band to a different reduction order is not currently justified.

Accuracy-risk reduction branch:

- Branch: `experiment/tile64-mid-default-20260709`
- Commit: `3b0e230 Default long prefill tile64 policy to mid`
- Change: keep the same policy switch, but default to `mid` and fall back to
  `mid` for invalid env values.
- Intended behavior: `TILE_SIZE=64` only for `8192 < max(q_len, kv_len) <= 16384`;
  4K-8K and 16K-32K use `32` by default.
- Status: static syntax/import check only. Needs container smoke before promote
  to GitLab `main`.

Unsubmitted experiment branch:

- `experiment/gdn-causal-conv-block-20260709`
- Commit: `76827e1 Experiment GDN causal conv token block`
- Change: makes GDN prefill `causal_conv1d` token block selectable via
  `VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=8/16/32`, defaulting to `16`.
- Do not promote this branch to `main` until at least a 4K-8K smoke completes.
- Existing FLA env candidate: `FLA_GDN_FIX_BT=1` forces GDN `chunk_fwd_o` to use
  `BT=64` instead of adapting to `min(64, max(16, next_power_of_2(T)))`.
  This should be tested as an env-only A/B before adding new GDN code.

Stacked GDN experiment branch:

- Branch: `experiment/tile64-mid-gdn-conv-20260709`
- Commit: `f05e2bb Experiment GDN causal conv token block`
- Base: `3b0e230` (`mid` tile64 default), not the broad policy.
- Change: makes GDN prefill causal-conv token block selectable via
  `VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=8/16/32`, defaulting to `16`.
- Status: static syntax/import check only. Do not promote before smoke.

Stacked GDN chunk experiment branch:

- Branch: `experiment/tile64-mid-gdn-chunk-20260709`
- Commit: `0f49eea Add runtime GDN chunk size knob`
- Base: `f05e2bb` (`mid` tile64 + GDN causal-conv block knob).
- Change: makes GDN/FLA chunk size selectable via
  `VLLM_GDN_CHUNK_SIZE=16/32/64`, defaulting to `64`.
- Risk: chunk size changes the GDN recurrence grouping and can affect both
  performance and numerics. Treat this as smoke-only until it beats the mid
  default on the same container.

Env-only FLA solve-tril candidate:

- Existing code already supports `FLA_TRIL_PRECISION=ieee/tf32` in
  `vllm/model_executor/layers/fla/ops/solve_tril.py`.
- This is an import/start-time environment knob, so it must be set before
  starting vLLM, not only before running `vllm bench serve`.
- Expected upside is limited but cheap to test on the GDN path. Accuracy risk is
  non-zero because it changes triangular-solve math precision, so do not make
  `tf32` default without official-style smoke and a no-penalty submission check.

Fast recovery smoke sequence after container access returns:

```bash
# Preferred one-shot sequence. This restarts vLLM between cases because all
# tested env knobs are server-import-time settings.
RUN_PROFILE=quick bash /public/home/xdzs2026_c087/Govinda/tools/codex_run_p0_ab_sequence.sh

# To focus only on GDN after the mid branch survives:
RUN_PROFILE=gdn bash /public/home/xdzs2026_c087/Govinda/tools/codex_run_p0_ab_sequence.sh

# Manual commands remain below for single-case debugging.

# Inside container. These env vars are read when the vLLM server process imports
# the modules, so set them before starting vLLM. Setting them only before
# `vllm bench serve` does not change the already-running server.
#
# After deploying the selected vLLM source and starting vLLM, the smoke helper
# prints both the running vLLM PID and the server-side env values from
# /proc/$PID/environ to avoid false A/B tests.
#
# Confirm the current GitLab main candidate first.
RANGE=4-8K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh

# Then test the accuracy-risk reduction branch:
#   REF=experiment/tile64-mid-default-20260709
#   server env: VLLM_TRITON_PREFILL_TILE64_POLICY=mid
RANGE=4-8K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh

# If startup and 4K-8K survive, test the official-benefit band:
RANGE=8-16K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh

# If testing the GDN experiment branch, restart vLLM with each server env below,
# then run the same smoke command. First run baseline behavior on that branch:
#   REF=experiment/tile64-mid-gdn-conv-20260709
#   server env: VLLM_TRITON_PREFILL_TILE64_POLICY=mid VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=8
RANGE=4-8K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh

# Then test the experimental default:
#   server env: VLLM_TRITON_PREFILL_TILE64_POLICY=mid VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=16
RANGE=4-8K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh

# Env-only GDN output-kernel BT test, no source change beyond current branch:
#   server env: VLLM_TRITON_PREFILL_TILE64_POLICY=mid VLLM_GDN_CAUSAL_CONV1D_BLOCK_M=16 FLA_GDN_FIX_BT=1
RANGE=4-8K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh

# Only if 4K-8K does not regress or crash:
#   server env: same winning 4K-8K setting
RANGE=8-16K NUM_PROMPTS=3 bash /public/home/xdzs2026_c087/Govinda/tools/codex_smoke_p0_gdn.sh
```

## Suggested Prompts For Claude/Teammates

Prompt for prefill worker:

```text
你只研究 vllm_cscc_work 的 prefill attention kernel，不碰 AITER、不碰 scheduler flag、不碰权重/采样。安全线是 2c73297，当前稳定优化是 Qwen3.5 full-attention prefill BLOCK_M=64。请阅读 prefix_prefill.py、triton_prefill_attention.py、triton_unified_attention.py、rocm_attn.py，找出哪些 prefill 路径没有吃到 BLOCK_M=64 或 K/V reuse 优化。输出具体文件/函数/候选改动和最小 smoke 方案，不要直接改代码。
```

Prompt for decode worker:

```text
你只研究 decode skinny GEMV/fusion，不碰 prefill、不碰 AITER、不碰 KV dtype。安全线是 2c73297。已知 LLMM1 对 gate_up/mid_up 有效，对 down/full_qkv/full_o/lm_head 无效。请确认 Qwen3.5 实际 MLP shape 和 qwen3_5.py -> Qwen2MoeMLP 调用链，评估 LLMM1Silu 是否应该按真实 shape 重写。输出 microbench 脚本和接入点，不要默认启用。
```

Prompt for KV quant worker:

```text
你只做 INT8 KV cache 可行性调研，不写持久化量化、不碰权重、不改 scheduler。请阅读 csrc/cache_kernels.cu、csrc/rocm/attention.cu、vllm/v1/attention/backends/rocm_attn.py、vllm/v1/kv_cache_interface.py，判断要实现 runtime per-head/per-block INT8 KV 需要改哪些 cache 写入和 attention 读取路径。输出设计和最小 correctness 测试，不接入提交线。
```

## Submit Gate

Before any GitLab submit:

1. `git -C vllm_cscc_work status --short` must contain only intended diffs.
2. `vllm/envs.py` must not default-enable AITER.
3. No FP8 weight path or direct risky KV FP8 flag should be default.
4. Smoke must include at least:
   - 4K-8K quick completion
   - 8K-16K 10 prompts for stability
   - 16K-32K quick completion
5. Keep official-result files separate from `_nonbench_` or local smoke result
   files.
