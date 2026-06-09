# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""TDD tests for benchmarks.compare (Task 2.2 baseline-vs-optimized).

Marked skip until DCU verification unblocks — see plan P2 2.2.
"""
from pathlib import Path
import json
import sys

import pytest

pytestmark = pytest.mark.skip(reason="DCU 验证后开 — 跟 plan P2 2.2 baseline 比对")

# Add benchmarks dir to sys.path so we can import compare directly.
BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _write_run(d: Path, tier: str, suffix: str, **metrics: float) -> None:
    """Write a minimal bench JSON compatible with compare._load_dir."""
    payload = {"tier": tier, "num_prompts": 50, **metrics, "raw": {}}
    (d / f"{tier}-{suffix}.json").write_text(json.dumps(payload))


def test_compare_matching_tiers(tmp_path: Path) -> None:
    """2 dirs × 3 tiers × 3 metrics → 9 BenchRow entries."""
    from compare import compare_dirs

    base = tmp_path / "base"
    opt = tmp_path / "opt"
    base.mkdir()
    opt.mkdir()
    for tier in ("4k-8k", "8k-16k", "16k-32k"):
        _write_run(base, tier, "1",
                   ttft_p99_ms=100.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)
        _write_run(opt, tier, "1",
                   ttft_p99_ms=120.0, tpot_p99_ms=2.2, throughput_tokens_per_s=1500.0)

    result = compare_dirs(base, opt)
    assert len(result.rows) == 9  # 3 tiers × 3 metrics


def test_compare_sla_pass(tmp_path: Path) -> None:
    """Baseline TTFT 100ms → optimized 130ms: Δ=30% ≤ 50% → pass_sla True."""
    from compare import compare_dirs

    base = tmp_path / "base"
    opt = tmp_path / "opt"
    base.mkdir()
    opt.mkdir()
    _write_run(base, "8k-16k", "1",
               ttft_p99_ms=100.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)
    _write_run(opt, "8k-16k", "1",
               ttft_p99_ms=130.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)

    result = compare_dirs(base, opt)
    assert result.sla_pass is True
    ttft_row = next(r for r in result.rows if r.metric == "ttft_p99_ms")
    assert ttft_row.pass_sla is True
    assert ttft_row.delta_pct == pytest.approx(30.0)


def test_compare_sla_fail(tmp_path: Path) -> None:
    """Baseline TTFT 100ms → optimized 200ms: Δ=100% > 50% → pass_sla False."""
    from compare import compare_dirs

    base = tmp_path / "base"
    opt = tmp_path / "opt"
    base.mkdir()
    opt.mkdir()
    _write_run(base, "16k-32k", "1",
               ttft_p99_ms=100.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)
    _write_run(opt, "16k-32k", "1",
               ttft_p99_ms=200.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)

    result = compare_dirs(base, opt)
    assert result.sla_pass is False
    ttft_row = next(r for r in result.rows if r.metric == "ttft_p99_ms")
    assert ttft_row.pass_sla is False
    assert ttft_row.delta_pct == pytest.approx(100.0)


def test_compare_to_markdown(tmp_path: Path) -> None:
    """to_markdown must include a per-tier table for each of the 3 tiers."""
    from compare import compare_dirs, to_markdown

    base = tmp_path / "base"
    opt = tmp_path / "opt"
    base.mkdir()
    opt.mkdir()
    for tier in ("4k-8k", "8k-16k", "16k-32k"):
        _write_run(base, tier, "1",
                   ttft_p99_ms=100.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)
        _write_run(opt, tier, "1",
                   ttft_p99_ms=110.0, tpot_p99_ms=2.1, throughput_tokens_per_s=1200.0)

    md = to_markdown(compare_dirs(base, opt))
    assert "## 4k-8k" in md
    assert "## 8k-16k" in md
    assert "## 16k-32k" in md
    assert "Throughput speedup" in md
    assert "Overall SLA" in md


def test_compare_to_json(tmp_path: Path) -> None:
    """to_json round-trip preserves rows + throughput_speedup + sla_pass."""
    from compare import compare_dirs, to_json

    base = tmp_path / "base"
    opt = tmp_path / "opt"
    base.mkdir()
    opt.mkdir()
    _write_run(base, "4k-8k", "1",
               ttft_p99_ms=100.0, tpot_p99_ms=2.0, throughput_tokens_per_s=1000.0)
    _write_run(opt, "4k-8k", "1",
               ttft_p99_ms=120.0, tpot_p99_ms=2.0, throughput_tokens_per_s=2000.0)

    result = compare_dirs(base, opt)
    payload = json.loads(to_json(result))
    assert payload["sla_pass"] is result.sla_pass
    assert payload["throughput_speedup"] == result.throughput_speedup
    assert len(payload["rows"]) == len(result.rows)
    for got, want in zip(payload["rows"], result.rows):
        assert got["tier"] == want.tier
        assert got["metric"] == want.metric
        assert got["baseline"] == want.baseline
        assert got["optimized"] == want.optimized
        assert got["delta_pct"] == want.delta_pct
        assert got["pass_sla"] == want.pass_sla
