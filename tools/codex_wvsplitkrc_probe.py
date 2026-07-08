# AI-generated, awaiting verification by recoletas on 2026-07-07.
import time

import torch

from vllm import _custom_ops as ops
from vllm.utils.platform_utils import num_compute_units


def run_one(op_name: str, fn, m: int, k: int) -> None:
    cu = num_compute_units()
    w = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.02
    for _ in range(5):
        y_ref = torch.nn.functional.linear(x, w)
        y = fn(w, x, cu)
    torch.cuda.synchronize()
    diff = (y_ref - y).abs().float()
    t0 = time.perf_counter()
    for _ in range(50):
        y = fn(w, x, cu)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 50
    print(
        f"{op_name} shape=({m},{k}) {ms:.4f}ms "
        f"maxdiff={float(diff.max()):.6g} mean={float(diff.mean()):.6g}",
        flush=True,
    )


def main() -> None:
    for name, fn in [("wv", ops.wvSplitK), ("wvrc", ops.wvSplitKrc)]:
        run_one(name + "_gate", fn, 34816, 5120)
        run_one(name + "_mid", fn, 17408, 5120)


if __name__ == "__main__":
    main()
