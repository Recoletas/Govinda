# src/compile/config.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""torch.compile / cudagraph configuration for vLLM 0.18.1."""
from dataclasses import dataclass

@dataclass
class CompileConfig:
    mode: str = "default"  # 不要 max-autotune
    use_cudagraph: bool = True
    enforce_eager: bool = False
    warmup_iters: int = 3

    def to_vllm_args(self) -> list[str]:
        args = []
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.use_cudagraph:
            args.append("--compilation-config.use_cudagraph=True")
        return args
