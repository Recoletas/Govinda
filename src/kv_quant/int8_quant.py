# AI-generated, awaiting verification by <team-lead> on 2026-06-09
"""INT8 symmetric per-head quantizer (CDNA2 fallback path).

Why this exists: CDNA2 (gfx90a, our DCU) lacks FP8 tensor-core
acceleration, so the ``FP8DynamicQuantizer`` we plan to use on
CDNA3+ / H100 must have a non-FP8 sibling. INT8 symmetric with
per-head scale is the standard "boring works everywhere" choice —
the same kernel pattern (``xq = round(x / s)``) compiles on every
backend we care about (CPU numpy, torch eager, triton).

The shape convention follows vLLM's KV layout: ``(B, H, T, D)``.
Per-head scale is therefore a 1-D tensor over the last two dims,
i.e. one scale per ``(b, h, t)`` — this matches the per-token
granularity we use for activations elsewhere and is what
:attr:`QuantizedTensor.granularity` reports.
"""
from __future__ import annotations

import numpy as np

from .base import QuantizedTensor, Quantizer, Tensor


class INT8PerHeadQuantizer(Quantizer):
    """Symmetric INT8 quantizer with per-(B, H, T) scale.

    Why per-head-but-with-T: in the KV cache the D axis is fixed
    (head_dim) and only T grows over time, so giving every (b, h, t)
    slice its own scale tracks the running activation distribution
    better than a single per-head scalar. The ``head_dim`` arg is
    retained for shape validation only — quantization does not look
    at D.

    Numerical contract: round-to-nearest, ties-to-even (numpy default
    via ``np.round``) with no clipping beyond the int8 range. We do
    *not* clip outliers because that biases the dynamic-range
    estimator and costs accuracy on tails; vLLM's calibration does
    the equivalent of clipping at the model side.
    """

    def __init__(self, head_dim: int, eps: float = 1e-8) -> None:
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive (used as scale floor), got {eps}")
        self.head_dim = head_dim
        self.eps = float(eps)
        # Pre-compute the int8 symmetric range once. Symmetric around 0
        # means zero_point is unused; we still set it in the output for
        # uniform contract with the dataclass.
        self._qmax = np.int8(127)

    # ------------------------------------------------------------------
    # Quantizer API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "int8_sym_per_head"

    @property
    def dtype(self) -> str:
        return "int8"

    def quantize(self, x: Tensor) -> QuantizedTensor:
        """Quantize ``x`` of shape ``(B, H, T, D)`` to int8.

        Returns a :class:`QuantizedTensor` with:
          * ``data`` — int8 ndarray of the same shape as ``x``
          * ``scale`` — float32 ndarray of shape ``(B, H, T)``
          * ``zero_point`` — None (symmetric)
          * ``granularity`` — ``"per_head"`` (one scale per (B, H, T))
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[-1] != self.head_dim:
            raise ValueError(
                f"expected shape (B, H, T, D={self.head_dim}); got {arr.shape}"
            )

        # Per-(B, H, T) absmax over the D axis. This is the same
        # operation torch.amax(dim=-1) would do, kept numpy-side so
        # the CPU test path is GPU-free.
        absmax = np.abs(arr).max(axis=-1)  # (B, H, T)
        scale = np.maximum(absmax / float(self._qmax), self.eps)  # (B, H, T)

        # Broadcast scale over D, round, cast. ``np.round`` defaults to
        # banker's rounding, which matches torch's behavior.
        divided = arr / scale[..., np.newaxis]
        quantized = np.round(divided).clip(
            float(self._qmax), -float(self._qmax)
        ).astype(np.int8)

        return QuantizedTensor(
            data=quantized,
            scale=scale.astype(np.float32),
            zero_point=None,
            granularity="per_head",
        )

    def dequantize(self, q: QuantizedTensor) -> Tensor:
        """Inverse of :meth:`quantize` — int8 → float32, scale broadcast."""
        if q.granularity != "per_head":
            raise ValueError(
                f"INT8PerHeadQuantizer expects per_head granularity, "
                f"got {q.granularity}"
            )
        if q.scale.ndim != 3:
            raise ValueError(
                f"scale must be (B, H, T); got shape {q.scale.shape}"
            )
        return (q.data.astype(np.float32) * q.scale[..., np.newaxis])

    # ------------------------------------------------------------------
    # Test helper (not part of the public Quantizer contract)
    # ------------------------------------------------------------------

    def _mock_x(
        self,
        batch: int = 2,
        num_heads: int = 4,
        seq_len: int = 8,
        seed: int = 0,
    ) -> np.ndarray:
        """Deterministic input generator for unit tests.

        Bounded within ``[-3, 3]`` so that the per-(B, H, T) absmax is
        non-trivial but well within int8's representable range after
        scaling. Returns float32 to match what the production pipeline
        would feed in (bf16 / fp16 from the model → cast → quantize).
        """
        if batch <= 0 or num_heads <= 0 or seq_len <= 0:
            raise ValueError("batch / num_heads / seq_len must all be positive")
        rng = np.random.default_rng(seed)
        return rng.uniform(-3.0, 3.0, size=(batch, num_heads, seq_len, self.head_dim)).astype(
            np.float32
        )
