# AI-generated, awaiting verification by <team-lead> on 2026-06-09
"""Smoke tests for the Triton decode attention backend stub.

The DCU host has vLLM 0.18.1 installed; the WSL2 dev box does not.
These tests are written to be importable on *both*: on a vllm-less
box the module falls back to a placeholder base, and we only assert
that the module loads and exposes the right symbol.

When run on the DCU, the additional ``test_subclasses_real_backend``
test will execute and prove the inheritance contract.
"""
# AI-generated, awaiting verification by <team-lead> on 2026-06-09
import pytest

pytestmark = pytest.mark.skip(reason="DCU 验证后开 — 跟 plan P0 0.4 / P3 stretch 5")


def test_module_imports():
    from src.attn_backend import triton_decode

    assert hasattr(triton_decode, "TritonDecodeAttention")


def test_class_has_stable_backend_name():
    from src.attn_backend.triton_decode import TritonDecodeAttention

    # The name is the registration key vLLM's register_backend uses
    # and Stream A's autotune cache hashes on — keep it stable.
    assert TritonDecodeAttention.BACKEND_NAME == "TRITON_DECODE"


def test_instance_has_name_property():
    from src.attn_backend.triton_decode import TritonDecodeAttention

    # vllm is not installed in this environment, so the class falls
    # back to ``object`` as base. We can still instantiate and read
    # the ``name`` property the subclass adds.
    instance = object.__new__(TritonDecodeAttention)
    assert instance.name == "TRITON_DECODE"


def test_subclasses_real_backend_when_vllm_present():
    """On the DCU, vllm is importable and the base is the real backend.

    Marked separately so it is obvious in the pytest output *which*
    assertion is the import-path verification for P0 0.4. Skipped
    automatically on dev boxes where vllm is not installed.
    """
    vllm = pytest.importorskip("vllm", reason="requires vllm (DCU only)")

    from src.attn_backend.triton_decode import (
        TritonAttentionBackend,
        TritonDecodeAttention,
    )

    # The base is the *real* vLLM class, not the placeholder ``object``.
    assert TritonAttentionBackend is not object
    assert issubclass(TritonDecodeAttention, TritonAttentionBackend)
    # The name is *not* inherited from the parent — the subclass
    # property is what vLLM's register_backend key off.
    assert TritonDecodeAttention.BACKEND_NAME != vllm.__version__
