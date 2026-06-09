# ADR 0006: vLLM 0.18.1 源码阅读笔记

**状态**: 待 队员 B 填充
**截止**: P0 末
**Owner**: 队员 B (vLLM, 6 h/周 × 1.5 周)

## PENDING — assigned to 队员 B

队员 B 在精读 vLLM 0.18.1 源码后，填充以下内容（来自 plan Task 0.7 Step 3）：

- KV cache 块结构
- 块大小对显存的影响
- 至少 1 个 backend 的 forward 流程图
- 至少 3 个 v0.18.1 新增的 enum 值

**目标文件**（需精读）:
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/attention/backends/registry.py`
- 1 个具体 backend 实现（如 `triton_attn.py`）
