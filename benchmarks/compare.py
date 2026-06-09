#!/usr/bin/env python3
# benchmarks/compare.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Compare two benchmark directories (baseline vs optimized) and emit an ROI report.

Pure stdlib + dataclass. No torch / vllm imports — safe to run on CPU.
Pairs JSON reports by (tier, metric), computes delta %, and checks SLA:
  - TTFT P99 / TPOT P99: regression of > 50% fails
  - throughput_tokens_per_s: regression (Δ <= 0) fails
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

TIERS: tuple[str, ...] = ("4k-8k", "8k-16k", "16k-32k")
METRICS: tuple[str, ...] = ("ttft_p99_ms", "tpot_p99_ms", "throughput_tokens_per_s")
LATENCY_METRICS: frozenset[str] = frozenset({"ttft_p99_ms", "tpot_p99_ms"})
LATENCY_SLA_PCT: float = 50.0


@dataclass(frozen=True)
class BenchRow:
    """One (tier, metric) pairing with delta and SLA verdict."""

    tier: str
    metric: str
    baseline: float
    optimized: float
    delta_pct: float
    pass_sla: bool


@dataclass(frozen=True)
class CompareResult:
    """Aggregate of all pairings plus headline speedup + overall SLA verdict."""

    rows: list[BenchRow]
    sla_pass: bool
    throughput_speedup: float


def _load_dir(d: Path) -> dict[tuple[str, str], float]:
    """Walk a directory of bench JSONs; return {(tier, metric): latest_value}."""
    out: dict[tuple[str, str], float] = {}
    if not d.is_dir():
        raise FileNotFoundError(f"benchmark dir not found: {d}")
    for f in sorted(d.glob("*.json")):
        data = json.loads(f.read_text())
        tier = data.get("tier")
        if tier not in TIERS:
            continue
        for metric in METRICS:
            value = data.get(metric)
            if isinstance(value, (int, float)):
                out[(tier, metric)] = float(value)
    return out


def _delta_pct(baseline: float, optimized: float) -> float:
    """Δ% = (optimized - baseline) / baseline * 100. Returns 0.0 if baseline == 0."""
    if baseline == 0:
        return 0.0
    return (optimized - baseline) / baseline * 100.0


def _pass_sla(metric: str, delta_pct: float) -> bool:
    """Latency metrics: |Δ| ≤ 50% pass. Throughput: Δ > 0 pass."""
    if metric == "throughput_tokens_per_s":
        return delta_pct > 0.0
    return abs(delta_pct) <= LATENCY_SLA_PCT


def _throughput_speedup(
    baseline: dict[tuple[str, str], float],
    optimized: dict[tuple[str, str], float],
) -> float:
    """Aggregate throughput speedup = sum(opt) / sum(base) across tiers."""
    b_sum = sum(baseline.get((t, "throughput_tokens_per_s"), 0.0) for t in TIERS)
    o_sum = sum(optimized.get((t, "throughput_tokens_per_s"), 0.0) for t in TIERS)
    if b_sum == 0.0:
        return 0.0
    return o_sum / b_sum


def compare_dirs(baseline_dir: Path, optimized_dir: Path) -> CompareResult:
    """Pair JSON reports from two dirs by (tier, metric); compute deltas + SLA."""
    baseline = _load_dir(Path(baseline_dir))
    optimized = _load_dir(Path(optimized_dir))
    rows: list[BenchRow] = []
    for tier in TIERS:
        for metric in METRICS:
            b = baseline.get((tier, metric))
            o = optimized.get((tier, metric))
            if b is None or o is None:
                continue
            d = _delta_pct(b, o)
            rows.append(
                BenchRow(
                    tier=tier,
                    metric=metric,
                    baseline=b,
                    optimized=o,
                    delta_pct=d,
                    pass_sla=_pass_sla(metric, d),
                )
            )
    return CompareResult(
        rows=rows,
        sla_pass=all(r.pass_sla for r in rows) if rows else False,
        throughput_speedup=_throughput_speedup(baseline, optimized),
    )


def to_markdown(result: CompareResult) -> str:
    """Render the comparison result as a markdown report (per-tier tables + summary)."""
    lines: list[str] = ["# Baseline vs Optimized — ROI Report", ""]
    for tier in TIERS:
        tier_rows = [r for r in result.rows if r.tier == tier]
        if not tier_rows:
            continue
        lines += [
            f"## {tier}",
            "",
            "| Metric | Baseline | Optimized | Δ% | SLA |",
            "|--------|----------|-----------|----|-----|",
        ]
        for r in tier_rows:
            verdict = "PASS" if r.pass_sla else "FAIL"
            lines.append(
                f"| {r.metric} | {r.baseline:.2f} | {r.optimized:.2f} | "
                f"{r.delta_pct:+.1f}% | {verdict} |"
            )
        lines.append("")
    verdict = "PASS" if result.sla_pass else "FAIL"
    lines += [
        "## Summary",
        "",
        f"- Throughput speedup (aggregate): **{result.throughput_speedup:.2f}x**",
        f"- Overall SLA: **{verdict}**",
        "",
    ]
    return "\n".join(lines)


def to_json(result: CompareResult) -> str:
    """Serialize the result to JSON (json-friendly dict)."""
    return json.dumps(
        {
            "sla_pass": result.sla_pass,
            "throughput_speedup": result.throughput_speedup,
            "rows": [asdict(r) for r in result.rows],
        },
        indent=2,
    )


def main() -> None:
    """CLI entry: --baseline DIR --optimized DIR [--output PATH]."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--optimized", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="Write markdown report here (in addition to stdout).")
    args = p.parse_args()

    result = compare_dirs(args.baseline, args.optimized)
    md = to_markdown(result)
    print(md)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
