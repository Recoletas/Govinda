# AI-generated, verified by <team-lead> on 2026-06-09
"""Tests for the benchmark harness (Task 2.1).

The smoke test (test_run_bench_creates_json) requires a DCU + the 27B model
+ vllm installed. It is marked @pytest.mark.skip_dcu and is verified manually
on the DCU host. The unit tests below cover the testable pure-Python logic
and pass in any environment.
"""
from pathlib import Path
import json
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Smoke test (DCU-only)
# ---------------------------------------------------------------------------

# Custom marker so the test is NOT silently PASSed in environments without
# vllm + a 27B model + a DCU. The team runs it on the DCU host.
skip_dcu = pytest.mark.skip(
    reason="skip-dcu: requires DCU + Qwen3.5-27B model + vllm. Run on DCU host.",
)


@skip_dcu
def test_run_bench_creates_json(tmp_path):
    """Smoke test: bench script must write JSON with throughput/TTFT/TPOT."""
    out_dir = tmp_path / "bench"
    out_dir.mkdir()
    # 最小跑 5 个 prompt
    result = subprocess.run(
        ["python", "benchmarks/run_bench.py",
         "--model", "Qwen/Qwen3.5-27B",
         "--num-prompts", "5",
         "--tier", "4k-8k",
         "--output", str(out_dir)],
        capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert "throughput_tokens_per_sec" in data
    assert "ttft_p99_ms" in data
    assert "tpot_p99_ms" in data


# ---------------------------------------------------------------------------
# Unit tests (run anywhere)
# ---------------------------------------------------------------------------

# Add the benchmarks dir to sys.path so we can import run_bench / analyze
# directly. This keeps the unit tests independent of vllm / torch / etc.
BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def test_tier_prompts_table_complete():
    """TIER_PROMPTS must contain all three required tiers with the right keys."""
    import run_bench

    expected = {"4k-8k", "8k-16k", "16k-32k"}
    assert set(run_bench.TIER_PROMPTS.keys()) == expected

    # Each entry is a 2-tuple (length_arg, prompt_len).
    for tier, entry in run_bench.TIER_PROMPTS.items():
        assert isinstance(entry, tuple) and len(entry) == 2
        length_arg, prompt_len = entry
        assert isinstance(length_arg, str)
        assert "--random-input-len" in length_arg
        assert isinstance(prompt_len, int) and prompt_len > 0

    # Sanity check: prompt lengths grow with tier
    assert run_bench.TIER_PROMPTS["4k-8k"][1] < run_bench.TIER_PROMPTS["8k-16k"][1]
    assert run_bench.TIER_PROMPTS["8k-16k"][1] < run_bench.TIER_PROMPTS["16k-32k"][1]


def test_analyze_aggregates_by_tier(tmp_path, capsys):
    """analyze.py groups JSON files by tier and emits a markdown table."""
    # Build synthetic bench results for two tiers.
    runs = [
        {"tier": "4k-8k", "num_prompts": 5, "throughput_tokens_per_sec": 100.0,
         "ttft_p99_ms": 50.0, "tpot_p99_ms": 1.5, "raw": {}},
        {"tier": "8k-16k", "num_prompts": 5, "throughput_tokens_per_sec": 80.0,
         "ttft_p99_ms": 80.0, "tpot_p99_ms": 2.0, "raw": {}},
    ]
    for i, r in enumerate(runs):
        (tmp_path / f"run-{i}.json").write_text(json.dumps(r))

    # Invoke analyze.main() programmatically.
    import analyze

    rc = analyze.main.__wrapped__ if hasattr(analyze.main, "__wrapped__") else None
    # Simpler: build a Namespace and call the body via subprocess + argparse.
    # Use the CLI so we exercise the real entry point.
    result = subprocess.run(
        [sys.executable, str(BENCH_DIR / "analyze.py"), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    out = result.stdout
    assert "4k-8k" in out
    assert "8k-16k" in out
    assert "100.0" in out  # throughput of first row
    assert "80.0" in out   # throughput of second row
    assert "Throughput" in out and "TTFT" in out and "TPOT" in out


def test_analyze_handles_empty_dir(tmp_path):
    """analyze.py must not crash when the input directory has no JSON files."""
    result = subprocess.run(
        [sys.executable, str(BENCH_DIR / "analyze.py"), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    out = result.stdout
    # The header lines should still be emitted.
    assert "Throughput" in out
    assert "TTFT" in out
    assert "TPOT" in out
