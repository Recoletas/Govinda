from abc import ABC, abstractmethod
import torch

class KVQuantizer(ABC):
    @abstractmethod
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (quantized, scale)."""
    @abstractmethod
    def dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def apply_to_kv_cache(self, k_cache, v_cache) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for vLLM integration (later)."""
