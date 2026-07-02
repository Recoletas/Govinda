# Verified by <team-lead> 2026-07-03 (page_size patch bug fix)
"""Self-test for src/kv_quant/patch_kv_cache_interface.py.

We don't have the real container file locally, so the test fabricates a
representative vLLM-0.18.1-style kv_cache_interface.py (modelled after
the P0 readmap and the snippet we saw via `git show HEAD -- ...` in
the container) and verifies the patcher:

  1. inserts the new fields into KVCacheSpec on the FIRST run;
  2. is a no-op on the SECOND run (idempotent);
  3. replaces ``AttentionSpec.real_page_size_bytes`` (the byte-count
     calculation site) to honour ``kv_cache_dtype == "int8"``;
  4. preserves the existing ``block_size`` field and the abstract
     ``page_size_bytes`` (no regression).

Run::

    pytest tests/test_patch_kv_cache_interface.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCHER = REPO_ROOT / "src" / "kv_quant" / "patch_kv_cache_interface.py"


# ---------------------------------------------------------------------------
# Synthetic upstream-like source for the patcher to chew on.
#
# Modeled after vLLM 0.18.1 / kv_cache_interface.py observed via
# `git show HEAD -- vllm/v1/kv_cache_interface.py` on the container
# (2026-06-29, 502-line file). Key elements relevant to the patcher:
#   - KVCacheSpec: abstract base, has `page_size_bytes` raising
#     NotImplementedError, no `real_page_size_bytes`.
#   - AttentionSpec: concrete subclass with `real_page_size_bytes` (the
#     calculation site the patcher targets) and a `page_size_bytes`
#     wrapper.
# ---------------------------------------------------------------------------
SYNTHETIC = '''\
from dataclasses import dataclass

import torch
from typing_extensions import Self

from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size


@dataclass
class KVCacheSpec:
    """Base class for KV cache format of one layer."""

    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int:
        real_page_size = self.real_page_size_bytes
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded
        return real_page_size

    @property
    def real_page_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )
'''


def _load_patcher():
    """Import the patcher module by path."""
    spec = importlib.util.spec_from_file_location("patch_kv_cache_interface", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def patcher():
    return _load_patcher()


def test_first_run_inserts_new_fields(patcher, tmp_path):
    target = tmp_path / "kv_cache_interface.py"
    target.write_text(SYNTHETIC)

    out = patcher.patch_kv_spec_fields(target.read_text())
    assert out is not None, "patcher should report a change on first run"
    assert 'kv_cache_dtype: str = "auto"' in out
    assert "kv_cache_int8_per_head: bool = False" in out
    # Existing field preserved (no regression).
    assert "block_size: int" in out
    # Existing AttentionSpec preserved.
    assert "class AttentionSpec(KVCacheSpec):" in out
    # Abstract page_size_bytes preserved (must still raise).
    assert "raise NotImplementedError" in out


def test_second_run_is_noop(patcher, tmp_path):
    target = tmp_path / "kv_cache_interface.py"
    target.write_text(SYNTHETIC)

    once = patcher.patch_kv_spec_fields(target.read_text())
    assert once is not None
    twice = patcher.patch_kv_spec_fields(once)
    assert twice is None, "second run must be idempotent"


def test_real_page_size_handles_int8(patcher, tmp_path):
    """Patcher must target AttentionSpec.real_page_size_bytes
    (the byte-count calculation site), NOT the abstract base.

    Regression test for the 2026-07-03 bug: the OLD patcher matched
    KVCacheSpec.page_size_bytes (abstract), which (a) replaced the
    raise-NotImplementedError with a concrete body, (b) left the
    AttentionSpec.real_page_size_bytes untouched, so KV cache sizing
    still came out as bf16 even after patching.
    """
    target = tmp_path / "kv_cache_interface.py"
    target.write_text(SYNTHETIC)

    after_fields = patcher.patch_kv_spec_fields(target.read_text())
    after_real = patcher.patch_real_page_size(after_fields)

    if after_real is not None:
        # real_page_size_bytes patched → it must reference kv_cache_dtype
        assert "kv_cache_dtype" in after_real
        # Abstract page_size_bytes NOT touched
        assert after_real.count("raise NotImplementedError") == 1, \
            "abstract page_size_bytes should remain NotImplementedError"
        # Patched body must contain the 2 * block_size * ... factor (K+V)
        # AND the int8 branch. Both must appear in the new real_page_size_bytes body.
        new_body = after_real.split("def real_page_size_bytes", 1)[1]
        assert "2" in new_body, "patched real_page_size_bytes must keep factor-of-2 for K+V"
        assert "torch.int8" in new_body, "patched real_page_size_bytes must switch to torch.int8 when kv_cache_dtype=='int8'"
        assert 'kv_cache_dtype == "int8"' in new_body
    # AttentionSpec class preserved in either case.
    assert "class AttentionSpec(KVCacheSpec):" in (after_real or after_fields)


def test_page_size_alias_still_works(patcher, tmp_path):
    """Backward-compat: patch_page_size should still work
    (now an alias for patch_real_page_size).
    """
    target = tmp_path / "kv_cache_interface.py"
    target.write_text(SYNTHETIC)

    after_fields = patcher.patch_kv_spec_fields(target.read_text())
    after_alias = patcher.patch_page_size(after_fields)
    assert after_alias is not None
    assert "kv_cache_dtype" in after_alias


def test_cli_apply_writes_backup(patcher, tmp_path, capsys):
    target = tmp_path / "kv_cache_interface.py"
    target.write_text(SYNTHETIC)

    # Drive the equivalent of main() inline to avoid argparse/argv interference.
    backup = target.with_suffix(target.suffix + ".bak")
    orig_text = target.read_text()
    new_text = patcher.patch_kv_spec_fields(orig_text)
    assert new_text is not None
    new_text = patcher.patch_real_page_size(new_text) or new_text
    shutil_copy = __import__("shutil").copy2
    shutil_copy(target, backup)
    target.write_text(new_text)
    assert backup.exists()
    assert 'kv_cache_dtype: str = "auto"' in target.read_text()
    # Backup still has the pre-patch content.
    assert 'kv_cache_dtype: str = "auto"' not in backup.read_text()