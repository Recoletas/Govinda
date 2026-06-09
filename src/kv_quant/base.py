# AI-generated, awaiting verification by <team-lead> on 2026-06-09
"""Abstract base for KV-cache quantizers (P3 Stream B / spec §2).

Defines the public contract every quantizer (FP8 / INT8 / future variants)
must implement. The contract is intentionally tensor-library-agnostic so
that the same Python class can be driven by torch, numpy, or triton
depending on the runtime — important for our DCU (torch) / CPU (numpy)
test matrix.

The companion class ``KVQuantizer`` below is the *legacy* ABC that the
AI-drafted ``fp8_quant.FP8DynamicQuantizer`` already inherits from. It
stays put for now; ``Quantizer`` is the new, stricter contract that
P3 implementations should adopt going forward.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# Tensor is a generic type alias so static checkers see a meaningful
# shape while runtime stays duck-typed. We deliberately avoid importing
# numpy / torch at module load — see AGENTS.md "AI 输出验证协议".
Tensor = Any


@dataclass
class QuantizedTensor:
    """Result of a quantization step.

    Why a dataclass: scale and zero_point are part of the value — they
    travel with the data through the model, so a tuple is too easy to
    transpose by accident. A named container makes call sites
    self-documenting (``.data`` / ``.scale``).

    ``granularity`` is kept as a string rather than an enum to stay
    zero-dep: ``"per_tensor"``, ``"per_token"``, ``"per_head"``,
    ``"per_channel"`` are the only legal values; subclasses validate.
    """

    data: Any
    scale: Any
    zero_point: Any | None = None
    granularity: str = "per_tensor"


class Quantizer(ABC):
    """Public quantizer contract.

    Implementations must:
      * produce a :class:`QuantizedTensor` whose ``.data`` dtype matches
        ``self.dtype`` (a string label, not a real dtype — keeps the
        interface import-light);
      * make :meth:`dequantize` an inverse up to the numerical error of
        the target format (verified by per-class round-trip tests).
    """

    @abstractmethod
    def quantize(self, x: Tensor) -> QuantizedTensor:
        """Encode ``x`` into a low-precision representation."""

    @abstractmethod
    def dequantize(self, q: QuantizedTensor) -> Tensor:
        """Inverse of :meth:`quantize`, up to rounding error."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human label, e.g. ``"fp8_e4m3_per_head"``."""

    @property
    @abstractmethod
    def dtype(self) -> str:
        """One of ``"fp8_e4m3"``, ``"fp8_e5m2"``, ``"int8"``, etc."""


# ---------------------------------------------------------------------------
# Legacy ABC (kept for backwards compatibility with the AI-drafted FP8
# quantizer that pre-dates the dataclass refactor). New code should use
# ``Quantizer`` above.
# ---------------------------------------------------------------------------


class KVQuantizer(ABC):
    """Pre-2026-06-09 quantizer interface.

    Still in use by ``src/kv_quant/fp8_quant.FP8DynamicQuantizer``. Do
    not delete until that class is migrated to :class:`Quantizer`.
    """

    @abstractmethod
    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (quantized, scale)."""

    @abstractmethod
    def dequantize(self, xq: Tensor, scale: Tensor) -> Tensor:
        """Inverse of :meth:`quantize`."""

    @abstractmethod
    def apply_to_kv_cache(self, k_cache: Any, v_cache: Any) -> tuple[Tensor, Tensor]:
        """Hook for vLLM integration (later)."""
