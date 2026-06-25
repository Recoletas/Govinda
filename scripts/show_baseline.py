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


# Fields we display, in display order. Each entry lists candidate key
# names — we pick the first one that's actually present in the JSON so
# this works across vLLM versions where the schema has shifted
# (e.g. v0.18.1 uses ``output_throughput`` + ``p99_ttft_ms`` rather
# than the older ``ttft_p50`` / ``ttft_p99`` flat layout).
FIELDS: list[tuple[str, list[str], str]] = [
    ("output",        ["output_throughput", "output_tok_per_s",
                       "total_token_throughput"],                       "tok/s"),
    ("req/s",         ["request_throughput", "num_requests"],           "req/s"),
    ("TTFT  mean",    ["mean_ttft_ms", "ttft_mean", "ttft_avg"],        "ms"),
    ("TTFT  median",  ["median_ttft_ms", "ttft_median",
                       "ttft_p50"],                                     "ms"),
    ("TTFT  p99",     ["p99_ttft_ms", "ttft_p99"],                      "ms"),
    ("TPOT  mean",    ["mean_tpot_ms", "tpot_mean", "tpot_avg"],        "ms"),
    ("TPOT  median",  ["median_tpot_ms", "tpot_median",
                       "tpot_p50"],                                     "ms"),
    ("TPOT  p99",     ["p99_tpot_ms", "tpot_p99"],                      "ms"),
    ("ITL   mean",    ["mean_itl_ms", "itl_mean", "itl_avg"],          "ms"),
    ("ITL   median",  ["median_itl_ms", "itl_median",
                       "itl_p50"],                                      "ms"),
    ("ITL   p99",     ["p99_itl_ms", "itl_p99"],                        "ms"),
    ("E2EL  mean",    ["mean_e2el_ms", "e2el_mean", "e2el_avg"],      "ms"),
    ("E2EL  median",  ["median_e2el_ms", "e2el_median",
                       "e2el_p50"],                                     "ms"),
    ("E2EL  p99",     ["p99_e2el_ms", "e2el_p99"],                      "ms"),
]


def _lookup(d: dict, candidates: list[str]):
    """Pick the first candidate key that's actually in d.

    Returns the value or None.
    """
    for k in candidates:
        if k in d:
            return d[k]
    # vLLM 0.18.1 also nests under "metrics" for some fields
    if "metrics" in d and isinstance(d["metrics"], dict):
        for k in candidates:
            if k in d["metrics"]:
                return d["metrics"][k]
    return None


def _label_for(path: Path) -> str:
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
    found_any = False
    for label, candidates, unit in FIELDS:
        v = _lookup(d, candidates)
        if isinstance(v, (int, float)):
            print(f"  {label:14s} {v:8.2f} {unit}")
            found_any = True
    if not found_any:
        # Helpful when we don't know the schema yet
        print("  (no matching fields — raw top-level keys:)")
        for k, v in d.items():
            if isinstance(v, (int, float, str)):
                print(f"    {k}: {v}")


def _default_paths() -> list[Path]:
    """Find testdata/test/*_throughput/result.json by trying several roots.

    Resolution order:
      1. ``$GOVINDA_TESTDATA`` env var (explicit override)
      2. Walk up from cwd looking for a ``testdata/test/`` sibling.
      3. ``~/testdata/test`` (ordinary user home)
      4. ``/public/home/$USER/testdata/test`` (SCNet container, root shell)
    """
    candidates: list[Path] = []
    env_root = os.environ.get("GOVINDA_TESTDATA")
    if env_root:
        candidates.append(Path(env_root) / "test")
    cur = Path.cwd().resolve()
    for _ in range(6):
        candidates.append(cur / "testdata" / "test")
        if cur.parent == cur:
            break
        cur = cur.parent
    candidates.append(Path(os.path.expanduser("~/testdata/test")))
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "xdzs2026_c087"
    candidates.append(Path(f"/public/home/{user}/testdata/test"))

    seen: set[Path] = set()
    out: list[Path] = []
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if base.is_dir():
            out.extend(sorted(base.glob("*_throughput/result.json")))
            if out:
                return out
    return out


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