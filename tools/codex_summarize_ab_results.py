# AI-generated, awaiting verification by recoletas on 2026-07-09.
"""Summarize Codex smoke A/B result directories.

Usage:
    python3 tools/codex_summarize_ab_results.py /path/to/codex_p0_ab_xxx

The input directory is expected to contain the `summary.tsv` written by
`codex_run_p0_ab_sequence.sh`. Each row's result_dir should contain
`<range>_n<num_prompts>/result.json`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "completed",
    "failed",
    "output_throughput",
    "ttft_p99",
    "tpot_p99",
    "itl_p99",
    "e2el_p99",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_rows(root: Path) -> list[dict[str, str]]:
    summary = root / "summary.tsv"
    if not summary.exists():
        raise FileNotFoundError(f"missing summary.tsv: {summary}")
    with summary.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _result_path(row: dict[str, str]) -> Path:
    result_dir = Path(row["result_dir"])
    return result_dir / f"{row['range']}_n{row['num_prompts']}" / "result.json"


def _base_key(label: str, range_name: str) -> tuple[str, str]:
    if "base" in label:
        return ("explicit", range_name)
    if label.startswith("mid_") and not any(
        marker in label for marker in ("gdn", "llmm", "attn", "combo")
    ):
        return ("mid", range_name)
    if label.startswith("combo_base"):
        return ("combo", range_name)
    return ("", range_name)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: codex_summarize_ab_results.py OUT_ROOT", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).expanduser().resolve()
    rows = _load_rows(root)

    records: list[dict[str, Any]] = []
    baselines: dict[tuple[str, str], float] = {}

    for row in rows:
        rec: dict[str, Any] = dict(row)
        path = _result_path(row)
        rec["result_json"] = str(path)
        if path.exists():
            data = _read_json(path)
            for key in METRIC_KEYS:
                rec[key] = data.get(key)
            throughput = data.get("output_throughput")
            label = row.get("case", "")
            base_key = _base_key(label, row.get("range", ""))
            if throughput is not None and base_key[0]:
                baselines[base_key] = float(throughput)
        else:
            rec["missing_result"] = True
        records.append(rec)

    print(
        "\t".join(
            (
                "case",
                "range",
                "status",
                "tok/s",
                "delta_vs_base",
                "completed",
                "failed",
                "ttft_p99",
                "tpot_p99",
                "result_json",
            )
        )
    )

    for rec in records:
        label = rec.get("case", "")
        range_name = rec.get("range", "")
        throughput = rec.get("output_throughput")
        delta = "-"
        if throughput is not None:
            base = None
            if label.startswith("combo_"):
                base = baselines.get(("combo", range_name))
            if base is None:
                base = baselines.get(("explicit", range_name))
            if base is None:
                base = baselines.get(("mid", range_name))
            if base:
                delta = f"{(float(throughput) / base - 1.0) * 100:+.2f}%"

        print(
            "\t".join(
                (
                    label,
                    range_name,
                    rec.get("status", ""),
                    _fmt_float(throughput),
                    delta,
                    _fmt_float(rec.get("completed"), 0),
                    _fmt_float(rec.get("failed"), 0),
                    _fmt_float(rec.get("ttft_p99")),
                    _fmt_float(rec.get("tpot_p99")),
                    rec.get("result_json", ""),
                )
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
