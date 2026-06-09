# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""FP8 dynamic quantization for KV cache (per-head / per-token scale)."""
import torch
from .base import KVQuantizer

class FP8DynamicQuantizer(KVQuantizer):
    def __init__(self, format: str = "e4m3fnuz", scale_mode: str = "per_head"):
        if format not in ("e4m3fnuz", "e4m3"):
            raise ValueError(f"unsupported FP8 format: {format}")
        self.format = format
        self.scale_mode = scale_mode

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_mode == "per_head":
            # x shape: (B, H, T, D) → scale over (T, D) per (B, H)
            abs_max = x.abs().amax(dim=(-2, -1), keepdim=False)  # (B, H)
            scale = abs_max / 448.0  # e4m3 max
        elif self.scale_mode == "per_token":
            # scale over (D,) per (B, H, T)
            abs_max = x.abs().amax(dim=-1, keepdim=True)  # (B, H, T, 1)
            scale = abs_max / 448.0
        else:
            raise ValueError(f"unsupported scale_mode: {self.scale_mode}")
        scale = scale.clamp(min=1e-6).to(torch.float32)
        xq = (x / scale.unsqueeze(-1) if self.scale_mode == "per_head" else x / scale)
        if self.format == "e4m3fnuz":
            xq = xq.to(torch.float8_e4m3fnuz)
        else:
            xq = xq.to(torch.float8_e4m3)
        return xq, scale

    def dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.scale_mode == "per_head":
            scale = scale.unsqueeze(-1)  # broadcast over T
        return xq.to(torch.float32) * scale.to(torch.float32)

    def apply_to_kv_cache(self, k_cache, v_cache):
        # Placeholder for vLLM integration — see Task 3B.3
        raise NotImplementedError("vLLM KV cache hookup comes in Task 3B.3")
