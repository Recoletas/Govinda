# AI-generated, awaiting verification by recoletas on 2026-07-07.
import time

import torch

from vllm import _custom_ops as ops


def bench_shape(name: str, m: int, k: int, rpb: int) -> None:
    w = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)

    for _ in range(5):
        y_torch = torch.nn.functional.linear(x, w)
        y_llmm1 = ops.LLMM1(w, x, rpb)
    torch.cuda.synchronize()

    diff = (y_torch - y_llmm1).abs().float()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    t0 = time.perf_counter()
    for _ in range(100):
        y = torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 10

    t0 = time.perf_counter()
    for _ in range(100):
        y = ops.LLMM1(w, x, rpb)
    torch.cuda.synchronize()
    llmm1_ms = (time.perf_counter() - t0) * 10

    print(
        f"{name:7s} shape=({m},{k}) rpb={rpb} "
        f"torch_ms={torch_ms:.4f} llmm1_ms={llmm1_ms:.4f} "
        f"speed={torch_ms / llmm1_ms:.3f} "
        f"maxdiff={max_diff:.4g} meandiff={mean_diff:.4g}"
    )
    del w, x, y_torch, y_llmm1, y


def bench_batch_shape(name: str, n: int, m: int, k: int, rpb: int) -> None:
    w = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    for _ in range(5):
        y_torch = torch.nn.functional.linear(x, w)
        y_llmmn = ops.LLMMN(w, x, rpb)
    torch.cuda.synchronize()

    diff = (y_torch - y_llmmn).abs().float()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    t0 = time.perf_counter()
    for _ in range(50):
        y = torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 20

    t0 = time.perf_counter()
    for _ in range(50):
        y = ops.LLMMN(w, x, rpb)
    torch.cuda.synchronize()
    llmmn_ms = (time.perf_counter() - t0) * 20

    print(
        f"{name:12s} n={n} shape=({m},{k}) rpb={rpb} "
        f"torch_ms={torch_ms:.4f} llmmn_ms={llmmn_ms:.4f} "
        f"speed={torch_ms / llmmn_ms:.3f} "
        f"maxdiff={max_diff:.4g} meandiff={mean_diff:.4g}"
    )
    del w, x, y_torch, y_llmmn, y


def bench_fused_silu(scale: float) -> None:
    m = 34816
    k = 5120
    rpb = 4
    w = (torch.randn((m, k), device="cuda", dtype=torch.float32) * scale).to(
        torch.bfloat16
    )
    x = (torch.randn((1, k), device="cuda", dtype=torch.float32) * scale).to(
        torch.bfloat16
    )

    for _ in range(5):
        gate_up = ops.LLMM1(w, x, 8)
        y_ref = torch.nn.functional.silu(gate_up[:, : m // 2]) * gate_up[:, m // 2 :]
        y_fused = ops.LLMM1Silu(w, x, rpb)
    torch.cuda.synchronize()

    diff = (y_ref - y_fused).abs().float()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    t0 = time.perf_counter()
    for _ in range(100):
        gate_up = ops.LLMM1(w, x, 8)
        y = torch.nn.functional.silu(gate_up[:, : m // 2]) * gate_up[:, m // 2 :]
    torch.cuda.synchronize()
    split_ms = (time.perf_counter() - t0) * 10

    t0 = time.perf_counter()
    for _ in range(100):
        y = ops.LLMM1Silu(w, x, rpb)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 10

    print(
        f"fused_silu scale={scale:g} shape=({m},{k}) rpb={rpb} "
        f"split_ms={split_ms:.4f} fused_ms={fused_ms:.4f} "
        f"speed={split_ms / fused_ms:.3f} "
        f"maxdiff={max_diff:.4g} meandiff={mean_diff:.4g}"
    )
    del w, x, gate_up, y_ref, y_fused, y


def main() -> None:
    torch.manual_seed(0)
    shapes = [
        ("q_rpb2", 12288, 5120, 2),
        ("q_rpb4", 12288, 5120, 4),
        ("q_rpb8", 12288, 5120, 8),
        ("k_rpb2", 1024, 5120, 2),
        ("k_rpb4", 1024, 5120, 4),
        ("k_rpb8", 1024, 5120, 8),
        ("o_rpb2", 5120, 6144, 2),
        ("o_rpb4", 5120, 6144, 4),
        ("o_rpb8", 5120, 6144, 8),
        ("down_rpb2", 5120, 17408, 2),
        ("down_rpb4", 5120, 17408, 4),
        ("down_rpb8", 5120, 17408, 8),
        ("down_rpb16", 5120, 17408, 16),
        ("gate_up", 34816, 5120, 8),
        ("mid_up", 17408, 5120, 4),
    ]
    for shape in shapes:
        bench_shape(*shape)
    for n in (2, 4, 8):
        bench_batch_shape("gate_up_N", n, 34816, 5120, 8)
        bench_batch_shape("mid_up_N", n, 17408, 5120, 4)
    for scale in (1.0, 0.1, 0.02):
        bench_fused_silu(scale)


if __name__ == "__main__":
    main()
