# AI-generated, awaiting verification by <team-lead> on 2026-06-09
"""Tests for the block-size sweep harness (src.block_size.sweep).

CPU-only: ``BlockSizeSweep.run`` falls back to a deterministic mock
when no real ``run_bench_fn`` is injected. This lets us exercise
the "SLA met → pick winner" and "SLA missed → return -1" branches
without standing up vLLM.

DCU verification (plan P0 0.4) will replace the mock path with the
real ``benchmarks.run_bench`` driver; the test surface here is
intentionally narrow so the swap is mechanical.
"""
# AI-generated, awaiting verification by <team-lead> on 2026-06-09
import pytest

pytestmark = pytest.mark.skip(reason="DCU 验证后开 — 跟 plan P0 0.4 / P3 stretch 5")


def test_run_with_default_block_sizes_returns_one_row_per_size():
    from src.block_size.sweep import BlockSizeSweep

    sweep = BlockSizeSweep()
    results = sweep.run()
    # DEFAULT_BLOCK_SIZES = (8, 16, 32, 64, 128) — see module docstring.
    assert [r.block_size for r in results] == [8, 16, 32, 64, 128]
    for r in results:
        assert r.throughput_tokens_per_s > 0
        assert r.fragmentation_pct >= 0
        assert r.kv_cache_used_gb > 0


def test_recommend_picks_throughput_winner_under_sla():
    from src.block_size.sweep import BlockSizeSweep

    # Mock table (see BlockSizeSweep._mock_run) gives the highest
    # throughput at block_size=32. SLA floor is 80ms TTFT-P99 /
    # 2.0ms TPOT-P99 → cap is 120 / 3.0. block_size=32 sits inside
    # the cap (85 / 1.9); block_size=64 and 128 do not.
    sweep = BlockSizeSweep()
    results = sweep.run()
    pick = sweep.recommend(results)
    assert pick == 32


def test_recommend_returns_minus_one_when_no_size_meets_sla():
    from src.block_size.sweep import BlockSizeSweep

    # Baseline so tight that nothing fits.
    sweep = BlockSizeSweep(baseline_ttft_p99_ms=1.0, baseline_tpot_p99_ms=0.1)
    results = sweep.run()
    assert sweep.recommend(results) == -1


def test_recommend_raises_on_empty_results():
    from src.block_size.sweep import BlockSizeSweep

    sweep = BlockSizeSweep()
    with pytest.raises(ValueError, match="non-empty"):
        sweep.recommend([])


def test_run_rejects_non_positive_block_size():
    from src.block_size.sweep import BlockSizeSweep

    sweep = BlockSizeSweep()
    with pytest.raises(ValueError, match="positive"):
        sweep.run(block_sizes=[0, 16])


def test_run_rejects_non_power_of_two_block_size():
    from src.block_size.sweep import BlockSizeSweep

    sweep = BlockSizeSweep()
    with pytest.raises(ValueError, match="power of two"):
        sweep.run(block_sizes=[12])


def test_run_uses_injected_bench_function():
    from src.block_size.sweep import BlockSizeResult, BlockSizeSweep

    seen: list[int] = []

    def fake_bench(sizes):
        seen.extend(sizes)
        return [
            BlockSizeResult(
                block_size=s,
                ttft_p50_ms=10.0,
                ttft_p99_ms=20.0,
                tpot_p50_ms=1.0,
                tpot_p99_ms=1.0,
                throughput_tokens_per_s=100.0 * s,
                kv_cache_used_gb=1.0,
                fragmentation_pct=1.0,
            )
            for s in sizes
        ]

    sweep = BlockSizeSweep(run_bench_fn=fake_bench)
    results = sweep.run(block_sizes=[8, 32])
    assert seen == [8, 32]
    assert [r.block_size for r in results] == [8, 32]
    # All in SLA → recommend picks the larger throughput.
    assert sweep.recommend(results) == 32


def test_recommend_tiebreaks_by_lower_fragmentation():
    from src.block_size.sweep import BlockSizeResult, BlockSizeSweep

    results = [
        BlockSizeResult(
            block_size=8,
            ttft_p50_ms=10.0,
            ttft_p99_ms=20.0,
            tpot_p50_ms=1.0,
            tpot_p99_ms=1.0,
            throughput_tokens_per_s=100.0,
            kv_cache_used_gb=1.0,
            fragmentation_pct=5.0,  # less fragmented
        ),
        BlockSizeResult(
            block_size=16,
            ttft_p50_ms=10.0,
            ttft_p99_ms=20.0,
            tpot_p50_ms=1.0,
            tpot_p99_ms=1.0,
            throughput_tokens_per_s=100.0,  # tie on throughput
            kv_cache_used_gb=1.0,
            fragmentation_pct=9.0,  # more fragmented
        ),
    ]
    sweep = BlockSizeSweep()
    assert sweep.recommend(results) == 8
