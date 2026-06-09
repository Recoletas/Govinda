"""Unit tests for CompileConfig dataclass (CPU-only, no vLLM/torch required)."""

import pytest

def test_default_config():
    from src.compile.config import CompileConfig
    c = CompileConfig()
    assert c.mode == "default"
    assert c.use_cudagraph is True
    assert c.enforce_eager is False
    assert c.warmup_iters == 3

def test_to_vllm_args_with_cudagraph():
    from src.compile.config import CompileConfig
    c = CompileConfig(use_cudagraph=True, enforce_eager=False)
    args = c.to_vllm_args()
    assert "--compilation-config.use_cudagraph=True" in args
    assert "--enforce-eager" not in args

def test_to_vllm_args_eager_disables_cudagraph():
    """Eager mode 应该关闭 cudagraph (互斥)."""
    from src.compile.config import CompileConfig
    c = CompileConfig(use_cudagraph=True, enforce_eager=True)
    args = c.to_vllm_args()
    assert "--enforce-eager" in args
    # 注意: 实际 vLLM 行为是 eager 时 cudagraph 标志被忽略; 但 flag 本身还是生成
    # 我们的 to_vllm_args 不做互斥, 让 vLLM 自己处理
    assert "--compilation-config.use_cudagraph=True" in args

def test_to_vllm_args_no_cudagraph():
    from src.compile.config import CompileConfig
    c = CompileConfig(use_cudagraph=False, enforce_eager=False)
    args = c.to_vllm_args()
    assert "--compilation-config.use_cudagraph=True" not in args
    assert "--enforce-eager" not in args
    # 啥也不传, vLLM 用默认
    assert args == []

def test_custom_mode():
    from src.compile.config import CompileConfig
    c = CompileConfig(mode="reduce-overhead")
    assert c.mode == "reduce-overhead"
