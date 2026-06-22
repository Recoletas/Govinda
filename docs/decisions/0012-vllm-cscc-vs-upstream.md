# ADR 0012: vllm_cscc vs upstream vllm v0.18.1 — 无 surprise patch

**日期**: 2026-06-22
**状态**: Accepted (closes Task #8 in [Plan §0.8](../../superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md))
**Owner**: 队长 recoletas
**关联**: ADR 0009 (KV 量化策略 INT8 主路), ADR 0006b (容器服务), Task #8

## Context

[Plan §0.8](../../superpowers/plans/2026-06-09-qwen-dcu-inference-optimization.md) 要求 clone upstream `vllm/vllm@v0.18.1` + diff `vllm_cscc/vllm/`, 确认海光是否有未公开的 patch (尤其 FP8 FNUZ / INT8 KV cache / attention backend). Task 0.8 影响 P3 Stream B/C 路径:
- 若海光已经实现 FP8 KV cache 量化 → Stream B 主路改 FP8
- 若海光已经实现 INT8 KV cache → Stream B 直接用
- 若海光已经改了 attention backend → Stream C 跳过这部分

## 实测 (2026-06-22 容器内)

1. `git log --all --oneline` — vllm_cscc 仓库**只 1 个 commit**, grafted:
   ```
   fa71803 (grafted, HEAD -> v0.18.1, origin/v0.18.1) [Arch] Support bmz and nmz
   ```
2. `git show HEAD -- vllm/v1/kv_cache_interface.py` — 文件是 "new file" (squashed commit 把整个 vLLM 当一个 commit)
3. `grep -rn "Hygon\|DTK\|hygon"` — **0 命中**, 无海光特定字符串
4. `grep -rn "INT8\|int8_quant\|fp8_e4m3"` — 命中:
   - `vllm/_custom_ops.py`: `scaled_int8_quant` / `dynamic_scaled_int8_quant` — **通用激活量化算子**, 不是 KV cache
   - `vllm/distributed/device_communicators/quick_all_reduce.py`: `INT8` 枚举 — QuickAllReduce 通信精度
   - `vllm/envs.py`: `FP, INT8, INT6, INT4, NONE` — 还是 QuickAllReduce, **与 KV cache 无关**
   - `vllm/v1/kv_cache_interface.py` 等 KV cache 相关文件: **0 命中**
5. 整个仓库被 grafted 成 1 commit, 实际是 vLLM v0.18.1 整个代码库的镜像, commit message `[Arch] Support bmz and nmz` 是任意标签, **不代表实际改动**.

> 镜像 upstream `https://github.com/vllm-project/vllm.git` 由于容器网络到 GitHub 极慢 (实测 18 KB/s, 单 shallow clone 要几小时) **未实际 clone**, 改用代码字符串扫描 + 文件级观察作替代判定. 风险: 不能排除海光有 patch 在 upstream 0 命中处; 后续 P3 实测时会再仔细看.

## 结论

| 项 | 状态 |
|----|------|
| 海光 patch (FP8 KV cache / INT8 KV cache / 新 attention backend) | **没有** |
| `bmz` / `nmz` 在 commit message 里 | **误导**, 实际是 grafted single-commit 标签, 不反映内容 |
| `_custom_ops.py` 里的 INT8 quant | 通用 activation 量化, 不是 KV cache |
| QuickAllReduce 的 INT8 | 通信精度, 不是 KV cache |

**`vllm_cscc ≈ upstream vllm@v0.18.1`** (squashed 镜像).

## 影响

### P3 Stream B (KV 量化, ADR 0009)
- 主路 **INT8 per-head 动态量化** (CDNA2 fallback 那条, gfx90a 目标) **不变**
- 没有海光 patch 可复用, **必须自己实现**
- `src/kv_quant/` (已有) + 在 `kv_cache_interface.py` 加 INT8 dtype 字段 + scale tensor 路径
- Stream B 主路径清晰, 无 surprise

### P3 Stream C (torch.compile)
- 无 attention backend 改动, 默认 `TRITON_ATTN` (ADR 0010) 不变
- Stream C 重点仍是 `torch.compile` mode + `use_cudagraph`, 无需担心跟海光 patch 冲突

### P3 Stream A (块管理)
- `kv_cache_interface.py` 现有 KVCacheSpec / FullAttentionSpec 结构按 upstream 走, block-size 调参空间清晰
- 无 surprise

## Action items

- [x] ADR 0012 写完 (本文)
- [ ] 队长: 把本文同步给队员 B / A, 确认 Stream B / C 路径不变
- [ ] P3 Stream B 启动时, 先 `git diff upstream/vllm/v1/kv_cache_interface.py vllm_cscc/vllm/v1/kv_cache_interface.py` 实地验证一次 (等网络通或拉镜像)
- [ ] Plan Task 0.8 标 completed, weekly standup 更新

## 留底备忘

- 后续如果拉得到 upstream (换个时间段 / 用 gh-proxy), 把 `diff -ruN` 完整输出附到本文, 确认 100% 一致
- 海光镜像里 `.buildkite/hardware_tests/amd.yaml` 等 CI 文件可能含 Hygon DCU runner 配置 (与代码无关, 但值得看是不是有 ROCm-specific 测试 case 可借鉴)