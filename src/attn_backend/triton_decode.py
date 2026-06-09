# AI-generated, awaiting verification by <team-lead> on 2026-06-09
"""Decode-phase Triton attention backend (stretch-5 registration stub).

Spec context (2026-06-09 §2 verification item 4): on DCU we must be
able to register a custom attention backend by subclassing
``vllm.attention.backends.triton_attn.TritonAttentionBackend`` and
making ``vllm bench serve`` run a single prompt through it. Stretch 5
turns the stub into a real flash-decoding-style split-K kernel for
the decode path; the prefill path is inherited unchanged.

Why this file exists now (P0 0.4): the team needs to verify the
*vLLM-side import path and registration call* on the DCU before
investing in the kernel rewrite. So this module does only what is
required for that smoke test:

  1. import :class:`TritonAttentionBackend` from the canonical vLLM
     0.18.1 location (the *current* path; spec confirms this is
     where vLLM keeps the parent class — re-verify on the DCU
     image, vendored builds sometimes lag upstream);
  2. define a subclass with the same public surface so vLLM's
     ``register_backend`` call accepts it;
  3. document explicitly that the kernel override is intentionally
     deferred to the P3 stretch.

If vllm is not importable (e.g. on the WSL2 dev container), the
class still defines itself by aliasing the parent type to a local
placeholder. That keeps ``import src.attn_backend.triton_decode``
side-effect-free in environments without vllm, so other modules can
depend on it without forcing vllm to be installed.
"""
from __future__ import annotations

# We try the canonical vLLM 0.18.1 import first. If vllm is not
# installed (CPU / WSL2 dev box), fall back to ``object`` as the base
# so this module is still importable. Tests do not exercise vllm;
# they only check that *if* vllm is present, our subclass is a real
# subclass of the canonical backend.
try:
    from vllm.attention.backends.triton_attn import TritonAttentionBackend
except Exception:  # pragma: no cover - exercised only on DCU
    TritonAttentionBackend = object  # type: ignore[assignment,misc]


class TritonDecodeAttention(TritonAttentionBackend):
    """Decode-stage attention override for Qwen3.5-27B on DCU.

    Behavioural contract (target, not yet implemented — see stretch 5):
      * Prefill: delegate to the parent TritonAttentionBackend (single
        fused attention kernel over the full prompt).
      * Decode: route to a split-K ("flash-decoding") kernel that
        parallelises the KV reduction over CUDA-grid Y, then merges
        partial softmaxes on host. This is the only way to keep
        TTFT-P99 stable as context length crosses ~8k on a single
        DCU.

    Current state (P0 0.4): this class is a *registration stub* only.
    The kernel override is intentionally left to the parent; the
    class exists so vLLM's ``register_backend`` accepts it and the
    import path is exercised on DCU.
    """

    # Sentinel used by tests to distinguish "subclass of the real
    # vllm backend" from "registration stub on a CPU box without
    # vllm". The ``name`` property is the registration key; keep it
    # stable across P3 — Stream A's autotune cache keys on it.
    BACKEND_NAME: str = "TRITON_DECODE"

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.BACKEND_NAME
