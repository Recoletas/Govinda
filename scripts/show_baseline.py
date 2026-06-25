"""Print the headline metrics from a vLLM-bench-serve result.json.

Usage::

    python3 scripts/show_baseline.py <path/to/result.json> [<result.json> ...]
    python3 scripts/show_baseline.py          # scans ~/testdata/test/*_throughput/result.json

Designed to be pasted into a web e-shell with no extra deps.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Keys we display, in display order. Anything missing is shown as "n/a".
FIELDS: list[tuple[str, str, str]] = [
    ("output_throughput", "output",        "tok/s"),
    ("ttft_p50",          "TTFT  P50",     "ms"),
    ("ttft_p99",          "TTFT  P99",     "ms"),
    ("tpot_p50",          "TPOT  P50",     "ms"),
    ("tpot_p99",          "TPOT  P99",     "ms"),
    ("itl_p50",           "ITL   P50",     "ms"),
    ("itl_p99",           "ITL   P99",     "ms"),
    ("e2el_p50",          "E2EL  P50",     "ms"),
    ("e2el_p99",          "E2EL  P99",     "ms"),
]


def _label_for(path: Path) -> str:
    # Strip "_throughput" suffix and use the tier name as the heading.
    name = path.parent.name
    return name.replace("_throughput", "")


def show_one(path: Path) -> None:
    if not path.exists():
        print(f"=== {path.parent.name} ===  MISSING")
        return
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"=== {path.parent.name} ===  bad JSON: {e}")
        return
    print(f"=== {_label_for(path)} ===")
    for key, label, unit in FIELDS:
        if key in d and isinstance(d[key], (int, float)):
            print(f"  {label:14s} {d[key]:8.2f} {unit}")
        else:
            print(f"  {label:14s}       n/a")


def _default_paths() -> list[Path]:
    """Scan ~/testdata/test/*_throughput/result.json by default.

    Tries several candidate roots in order — useful on SCNet containers
    where the shell runs as root but the actual user data lives at
    ``/public/home/<account>/`` (not the root user's ``/root/``).

    Order:
      1. ``$GOVINDA_TESTDATA`` env var (set explicitly if you like)
      2. ``<cwd>/testdata/test`` (when run from the repo root)
      3. ``~/testdata/test``
      4. ``/public/home/$USER/testdata/test``
    """
    candidates: list[Path] = []
    env_root = os.environ.get("GOVINDA_TESTDATA")
    if env_root:
        candidates.append(Path(env_root) / "test")
    candidates.append(Path.cwd() / "testdata" / "test")
    candidates.append(Path(os.path.expanduser("~/testdata/test")))
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "xdzs2026_c087"
    candidates.append(Path(f"/public/home/{user}/testdata/test"))

    for base in candidates:
        if base.is_dir():
            return sorted(base.glob("*_throughput/result.json"))
    return []


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        paths = [Path(p) for p in argv[1:]]
    else:
        paths = _default_paths()
        if not paths:
            print("no input files; pass result.json paths or have ~/testdata/test/*_throughput/result.json",
                  file=sys.stderr)
            return 1
    for p in paths:
        show_one(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))