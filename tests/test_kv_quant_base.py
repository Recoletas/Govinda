# AI-generated, awaiting verification by <team-lead> on 2026-06-09
"""Tests for the ``Quantizer`` abstract base (src.kv_quant.base).

What we cover:
  * ``Quantizer`` cannot be instantiated directly (it's an ABC).
  * Subclasses that omit any of the four abstract members fail
    ``TypeError`` at construction, not at call time — this matters
    because vLLM's backend loader inspects classes at import time.

DCU verification gates the activation of the smoke tests, per
plan P0 0.4 + P3 stretch 5.
"""
import pytest

pytestmark = pytest.mark.skip(reason="DCU 验证后开 — 跟 plan P0 0.4 / P3 stretch 5")


def test_quantizer_cannot_be_instantiated_directly():
    from src.kv_quant.base import Quantizer

    with pytest.raises(TypeError, match="abstract"):
        Quantizer()  # type: ignore[abstract]


def test_subclass_missing_quantize_cannot_instantiate():
    from src.kv_quant.base import Quantizer

    class IncompleteQuantizer(Quantizer):
        @property
        def name(self) -> str:
            return "incomplete"

        @property
        def dtype(self) -> str:
            return "int8"

        # NOTE: quantize / dequantize intentionally omitted

    with pytest.raises(TypeError, match="abstract"):
        IncompleteQuantizer()  # type: ignore[abstract]


def test_subclass_missing_dequantize_cannot_instantiate():
    from src.kv_quant.base import Quantizer

    class IncompleteQuantizer(Quantizer):
        @property
        def name(self) -> str:
            return "incomplete"

        @property
        def dtype(self) -> str:
            return "int8"

        def quantize(self, x):  # noqa: D401 - abstract override
            return None

        # dequantize omitted

    with pytest.raises(TypeError, match="abstract"):
        IncompleteQuantizer()  # type: ignore[abstract]


def test_subclass_missing_properties_cannot_instantiate():
    from src.kv_quant.base import Quantizer

    class IncompleteQuantizer(Quantizer):
        # No name / dtype / quantize / dequantize at all
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteQuantizer()  # type: ignore[abstract]


def test_complete_subclass_instantiates_and_satisfies_contract():
    from src.kv_quant.base import Quantizer

    class MinimalQuantizer(Quantizer):
        @property
        def name(self) -> str:
            return "minimal"

        @property
        def dtype(self) -> str:
            return "int8"

        def quantize(self, x):
            return None

        def dequantize(self, q):
            return None

    instance = MinimalQuantizer()
    assert instance.name == "minimal"
    assert instance.dtype == "int8"


def test_legacy_kvquantizer_still_defined():
    """Pre-2026-06-09 ABC must remain importable for FP8DynamicQuantizer.

    The refactor adds ``Quantizer``; it does *not* remove the legacy
    ``KVQuantizer`` because ``src.kv_quant.fp8_quant`` still inherits
    from it. Removing it would silently break that module.
    """
    from src.kv_quant.base import KVQuantizer, Quantizer

    assert issubclass(KVQuantizer, object)
    assert issubclass(Quantizer, object)
    # The two are independent — there is no inheritance relationship.
    assert not issubclass(Quantizer, KVQuantizer)
    assert not issubclass(KVQuantizer, Quantizer)
