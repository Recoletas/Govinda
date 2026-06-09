#!/usr/bin/env python3
# benchmarks/analyze.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Aggregate JSON files in a directory into a single markdown table."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    runs = []
    for f in sorted(Path(args.input_dir).glob("*.json")):
        data = json.loads(f.read_text())
        runs.append(data)

    by_tier = defaultdict(list)
    for r in runs:
        by_tier[r.get("tier", "?")].append(r)

    lines = ["# Benchmark summary", "",
             "| Tier | n | Throughput (tok/s) | TTFT P99 (ms) | TPOT P99 (ms) |",
             "|------|---|--------------------|---------------|---------------|"]
    for tier, rs in sorted(by_tier.items()):
        if not rs: continue
        latest = rs[-1]
        lines.append(
            f"| {tier} | {latest['num_prompts']} | "
            f"{latest['throughput_tokens_per_sec']:.1f} | "
            f"{latest['ttft_p99_ms']:.1f} | "
            f"{latest['tpot_p99_ms']:.2f} |"
        )
    output = "\n".join(lines)
    print(output)
    if args.output:
        Path(args.output).write_text(output)
        print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()
