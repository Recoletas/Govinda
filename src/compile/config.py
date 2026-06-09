# src/compile/config.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""torch.compile / cudagraph configuration for vLLM 0.18.1.

相关 ADR: 0010 (attention backend 选型), 0013 (compile ROI 假设).
本配置默认 mode = "default" (非 max-autotune), 跟 spec §5.1 决策一致;
attention_backend 字符串字段供 vLLM `--attention-backend` flag 注入用 (P3 集成日改, 跟 ADR 0010).
"""
from dataclasses import dataclass

@dataclass
class CompileConfig:
    """默认 mode = "default" (非 max-autotune, 跟 spec §5.1 决策一致)."""
    mode: str = "default"  # 不要 max-autotune
    use_cudagraph: bool = True
    enforce_eager: bool = False
    warmup_iters: int = 3
    attention_backend: str = "TRITON_ATTN"  # P3 集成日改 (ADR 0010), 当前跟 vLLM ROCm 默认一致

    def to_vllm_args(self) -> list[str]:
        args = []
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.use_cudagraph:
            args.append("--compilation-config.use_cudagraph=True")
        return args
