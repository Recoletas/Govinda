# AI-generated, awaiting verification by recoletas on 2026-07-07.
import time

import torch

from vllm import _custom_ops as ops


def bench(name: str, m: int, k: int, rpbs: list[int]) -> None:
    print(f"-- {name} shape=({m},{k})", flush=True)
    w = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)

    for _ in range(5):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(40):
        y_ref = torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000 / 40
    print(f"torch {torch_ms:.4f} ms", flush=True)

    for rpb in rpbs:
        for _ in range(5):
            y = ops.LLMM1(w, x, rpb)
        torch.cuda.synchronize()
        diff = (y_ref - y).abs().float()
        t0 = time.perf_counter()
        for _ in range(40):
            y = ops.LLMM1(w, x, rpb)
        torch.cuda.synchronize()
        llmm_ms = (time.perf_counter() - t0) * 1000 / 40
        print(
            f"rpb={rpb:<2} {llmm_ms:.4f} ms "
            f"speed={torch_ms / llmm_ms:.3f} "
            f"maxdiff={float(diff.max()):.4g} "
            f"mean={float(diff.mean()):.4g}",
            flush=True,
        )

    del w, x, y_ref, y
    torch.cuda.empty_cache()


def main() -> None:
    torch.manual_seed(0)
    for args in [
        ("gate_up", 34816, 5120, [4, 8, 16]),
        ("mid_up", 17408, 5120, [2, 4, 8, 16]),
        ("down", 5120, 17408, [1, 2, 4, 8, 16]),
        ("q", 12288, 5120, [2, 4, 8, 16]),
        ("o", 5120, 6144, [1, 2, 4, 8]),
        ("lm_head", 248320, 5120, [8, 16, 32]),
    ]:
        bench(*args)


if __name__ == "__main__":
    main()
