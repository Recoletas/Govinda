# AI-generated, awaiting verification by <team-lead> on 2026-06-22
"""Idempotent patcher: add INT8 KV cache dtype support to vLLM's
``kv_cache_interface.py``.

Why a script and not a git diff:
- We don't have the exact file content on hand (the container copy lives
  on SCNet, not in this repo). A Python patcher can match by class name
  / field shape instead of by line numbers, so it works against minor
  vLLM revisions.
- Idempotent: re-running it on an already-patched file is a no-op.

What it changes (matches ADR 0009 + 0012):

1. New ``kv_cache_dtype`` field on :class:`KVCacheSpec` — lets the user
   pick ``"auto" / "int8" / "fp16" / "bf16"`` at startup. Default
   unchanged (``"auto"`` → falls back to model dtype).

2. New ``kv_cache_int8_per_head`` flag — when ``True`` and
   ``kv_cache_dtype == "int8"``, each attention spec gets a
   per-(B, H, T) scale tensor alongside the INT8 K/V tensors (matches
   ``INT8PerHeadQuantizer`` in ``src/kv_quant/int8_quant.py``).

3. Updated :meth:`page_size_bytes` — INT8 halves the per-token K/V
   footprint (vs bf16), so the page size halves too.

Usage::

    # Dry-run (default — prints diff, doesn't write)
    python src/kv_quant/patch_kv_cache_interface.py \\
        --target ~/vllm_cscc/vllm/v1/kv_cache_interface.py

    # Apply for real (writes <target>.bak first)
    python src/kv_quant/patch_kv_cache_interface.py \\
        --target ~/vllm_cscc/vllm/v1/kv_cache_interface.py \\
        --apply

After patching, rebuild the wheel::

    cd ~/vllm_cscc && python setup.py bdist_wheel
    cd dist && pip install vllm-*.whl --no-deps
"""
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch fragments
# ---------------------------------------------------------------------------

# Inserted into KVCacheSpec body (between existing fields).
KV_SPEC_NEW_FIELDS = """\

    # ---- added: kv_cache_dtype field (src/kv_quant/patch_kv_cache_interface.py) ----
    # KV cache storage dtype. "auto" keeps upstream behavior (== model
    # weight dtype). "int8" enables per-(B, H, T) symmetric quant via
    # src/kv_quant/int8_quant.py. See ADR 0009 + 0012.
    kv_cache_dtype: str = "auto"
    kv_cache_int8_per_head: bool = False
    # --------------------------------------------------------------------------
"""

# Replacement for the page_size_bytes property. We don't know its exact
# body in vLLM 0.18.1; we patch by NAME match on the property and replace
# the docstring + body lines conservatively. If the upstream signature
# changes, the script will skip this step with a warning.
PAGE_SIZE_NEW_BODY = '''    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        INT8 KV cache halves the per-token footprint vs fp16 / bf16
        (added: page_size_bytes INT8 branch —
        src/kv_quant/patch_kv_cache_interface.py, see ADR 0009 + 0012).

        Returns:
            The page size
        """
        if self.kv_cache_dtype == "int8":
            head_size = get_dtype_size(torch.int8)
        else:
            head_size = get_dtype_size(self.dtype)
        return head_size * self.block_size * self.num_kv_heads * self.head_size
'''


# ---------------------------------------------------------------------------
# Matchers — operate on file text, line-based, idempotent
# ---------------------------------------------------------------------------

KV_SPEC_CLASS_RE = re.compile(
    r"^(@dataclass\([^\)]*\)\s*\n)?class\s+KVCacheSpec\b[\s\S]*?^class\s+\w+",
    re.MULTILINE,
)


# Distinct markers so each patcher can be idempotent independently
# (the field-injection marker is distinct from the page_size marker).
FIELD_MARKER = "added: kv_cache_dtype field (src/kv_quant/patch_kv_cache_interface.py)"
PAGE_MARKER = "added: page_size_bytes INT8 branch (src/kv_quant/patch_kv_cache_interface.py)"


def _field_already_patched(text: str) -> bool:
    return FIELD_MARKER in text


def _page_size_already_patched(text: str) -> bool:
    return PAGE_MARKER in text


def patch_kv_spec_fields(text: str) -> str | None:
    """Insert kv_cache_dtype + kv_cache_int8_per_head into KVCacheSpec.

    Returns the new text, or None if no change was made.
    """
    if _field_already_patched(text):
        return None

    # Find the KVCacheSpec dataclass block: from "@dataclass" (if present)
    # through to the end of the class body. We anchor on the next `class`
    # line after KVCacheSpec.
    m = re.search(
        r"(@dataclass(?:\([^\)]*\))?\s*\n)?class\s+KVCacheSpec\b",
        text,
    )
    if not m:
        return None

    # Find the END of the class body: the next top-level `class` line OR
    # the matching dedent after `class KVCacheSpec:`.
    start = m.end()
    # Search forward for next `^class ` (column 0) — that's the next
    # class definition at module scope.
    next_class = re.search(r"^class\s+\w+", text[start:], re.MULTILINE)
    if next_class:
        end = start + next_class.start()
    else:
        # Fallback: insert before last `^[^ \t]` (e.g. end of file)
        end = len(text)

    # Within the class body, find the last field (a line beginning with
    # an identifier and `:` and either no default or with default) and
    # insert AFTER it.
    body = text[start:end]
    field_lines = re.findall(r"^[ \t]+(\w+)\s*:\s*[^\n]+", body, re.MULTILINE)
    if not field_lines:
        return None

    # Find the offset of the last field line in the original text.
    last_field_name = field_lines[-1]
    last_field_pat = re.compile(
        rf"^[ \t]+{re.escape(last_field_name)}\s*:\s*[^\n]+",
        re.MULTILINE,
    )
    fm = last_field_pat.search(body)
    if not fm:
        return None
    insert_pos_in_body = fm.end()

    new_body = body[:insert_pos_in_body] + KV_SPEC_NEW_FIELDS + body[insert_pos_in_body:]
    return text[:start] + new_body + text[end:]


def patch_page_size(text: str) -> str | None:
    """Replace KVCacheSpec.page_size_bytes to handle INT8 dtype.

    Conservative: matches the property by name and replaces through
    the next top-level boundary (next class, next def/@ at the same
    indent, or end of file). If the structure doesn't match what we
    expect, we skip with a warning.
    """
    if _page_size_already_patched(text):
        return None

    # Find `def page_size_bytes(self)` plus its docstring. Body
    # extends until the next sibling method, decorator, top-level
    # ``class`` line, or end-of-file. We stop BEFORE the blank lines
    # that typically separate methods/classes so the replacement
    # preserves at least the PEP-8 2-blank-line separator.
    pat = re.compile(
        r"(    def page_size_bytes\(self\)[^\n]*:\s*\n)"
        r"((?:        \"\"\"[\s\S]*?\"\"\"\s*\n)?)"
        r"([\s\S]*?)"
        r"(?=\n[ \t]*$|"
        r"^[ \t]*(?:class |def |@)|\Z)",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        print(
            "  WARN: could not auto-patch page_size_bytes; "
            "you'll need to add INT8 handling manually "
            "(see PAGE_SIZE_NEW_BODY in src/kv_quant/patch_kv_cache_interface.py).",
            file=sys.stderr,
        )
        return None

    # Idempotency: if the existing body already returns from `int8`
    # branch, skip.
    if "kv_cache_dtype" in m.group(0):
        return None

    new_method = PAGE_SIZE_NEW_BODY
    return text[: m.start()] + new_method + text[m.end():]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, type=Path,
                    help="Path to vllm/v1/kv_cache_interface.py inside the container")
    ap.add_argument("--apply", action="store_true",
                    help="Write the patched file (default: dry-run, prints unified diff)")
    args = ap.parse_args()

    target: Path = args.target
    if not target.exists():
        print(f"ERROR: target file not found: {target}", file=sys.stderr)
        return 1

    orig = target.read_text()
    patched = patch_kv_spec_fields(orig)
    if patched is None:
        if _field_already_patched(orig) and _page_size_already_patched(orig):
            print(f"OK: already patched, no change needed: {target}")
            return 0
        print(f"ERROR: could not find KVCacheSpec class in {target}; "
              "share the relevant lines and we'll refine.", file=sys.stderr)
        return 2

    patched = patch_page_size(patched) or patched

    if orig == patched:
        print(f"OK: no change needed (already patched or nothing to do): {target}")
        return 0

    diff = difflib.unified_diff(
        orig.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=str(target),
        tofile=str(target) + ".patched",
    )
    sys.stdout.write("".join(diff))

    if not args.apply:
        print("\n--- dry-run only; pass --apply to write ---", file=sys.stderr)
        return 0

    backup = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, backup)
    target.write_text(patched)
    print(f"OK: wrote {target} (backup: {backup})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())