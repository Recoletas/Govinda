# AI-generated, awaiting verification by recoletas on 2026-07-07.
import time

import torch

from vllm import _custom_ops as ops
from vllm.utils.platform_utils import num_compute_units


def bench(name: str, n: int, m: int, k: int) -> None:
    cu = num_compute_units()
    print(f"-- {name} n={n} shape=({m},{k}) cu={cu}", flush=True)
    w = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    for _ in range(5):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        y_ref = torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000 / 30

    for _ in range(5):
        y = ops.wvSplitK(w, x, cu)
    torch.cuda.synchronize()
    diff = (y_ref - y).abs().float()
    t0 = time.perf_counter()
    for _ in range(30):
        y = ops.wvSplitK(w, x, cu)
    torch.cuda.synchronize()
    wv_ms = (time.perf_counter() - t0) * 1000 / 30
    print(
        f"torch={torch_ms:.4f}ms wv={wv_ms:.4f}ms "
        f"speed={torch_ms / wv_ms:.3f} "
        f"maxdiff={float(diff.max()):.4g} mean={float(diff.mean()):.4g}",
        flush=True,
    )
    del w, x, y_ref, y
    torch.cuda.empty_cache()


def main() -> None:
    torch.manual_seed(1)
    shapes = [
        ("gate_up", 34816, 5120),
        ("mid_up", 17408, 5120),
        ("down", 5120, 17408),
        ("q", 12288, 5120),
        ("o", 5120, 6144),
    ]
    for n in [1, 2, 4]:
        for name, m, k in shapes:
            bench(name, n, m, k)


if __name__ == "__main__":
    main()
