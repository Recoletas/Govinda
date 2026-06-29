# ADR 0014: DCU 冷启动优化 + dev/bench 脚本分清

**日期**: 2026-06-25
**状态**: Accepted
**Owner**: 队长 recoletas
**关联**: ADR 0006b (容器服务, 附录 A), ADR 0013 (比赛规则解读), `scripts/start_vllm_dev.sh` + `scripts/start_vllm_bench.sh`

## Context

读了 CSDN 文章「海光 DCU 上 vLLM 部署 Qwen 大模型: 从开机到服务最快流程」. 关键收获:

1. **冷启动 15min → 6min** 通过: 持久化 vLLM wheel / 模型 / 三类缓存 + `--load-format runai_streamer` + CUDA Graph 裁剪.
2. **DCU 特有 env 变量** 避免常见坑 (gfx 误判, 显存碎片, kernel launch 开销).
3. **三板斧** (文章): 持久化 + 缓存复用 + 直读 + 裁剪.

我们场景高度匹配:
- DCU 容器实例**有运行时限** (官方 2h web shell, 实例也会被回收)
- 每次重开都要 ~15min 冷启动
- vLLM 0.18.1 + Qwen3.5-27B (bf16) — 完全相同 stack

**采纳优化 + 分两套脚本**:

| 优化项 | 类型 | dev / bench | 合规 (§9) |
|---|---|---|---|
| `VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` / `MIOPEN_USER_DB_PATH` 持久化 | env | **都用** | ✓ |
| `HSA_OVERRIDE_GFX_VERSION=9.0.0` | env | **都用** | ✓ |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | env | **都用** | ✓ |
| `HIP_FORCE_DEV_KERNARG=1` | env | **都用** | ✓ |
| `SAFETENSORS_FAST_GPU=1` | env | **都用** | ✓ |
| `HIP_VISIBLE_DEVICES=0` | env | **都用** | ✓ |
| `VLLM_USE_TRITON_FLASH_ATTN=1` | env | **都用** | ✓ |
| `VLLM_ROCM_USE_AITER=0` | env | **都用** | ✓ |
| `--load-format runai_streamer` | flag | **都用** | ✓ loader 格式允许 |
| `--compilation-config '{"cudagraph_capture_sizes":...}'` | flag | **都用** | ✓ torch.compile config 允许 |
| `--gpu-memory-utilization 0.95` | flag | dev | ✓ dev 用, bench 不加避免多算不算 "调整参数" |
| `--dtype bfloat16` | flag | dev | ✓ dev 用; bench 走 vllm 默认 (bf16) |
| `--tensor-parallel-size 1` | flag | dev | 单卡, 仅 dev 显式标; bench 默认 1 |
| `pip install runai-model-streamer` | 一次性 | **都用** | ✓ 装一次, 复用 |
| `--max-num-seqs 128` | flag | ❌ 都不加 | §9(8) **LOCKED** |
| `--max-num-batched-tokens 4096` | flag | ❌ 都不加 | §9(8) **LOCKED** |
| `--default-chat-template-kwargs '{"enable_thinking": false}'` | flag | ❌ 都不加 | §9(8) chat template **LOCKED** |
| `--tool-call-parser` / `--reasoning-parser` / `--enable-auto-tool-choice` | flag | ❌ 都不加 | 评测集不用, 多余 |
| `--served-model-name Qwen3.5-27B` | flag | ❌ 都不加 | §9(8) 服务接口 **LOCKED** |
| `--port 8001` | flag | ❌ 都不加 | §9(8) host:port 平台固定 |

## 决策

**两套脚本分清**:
- `scripts/start_vllm_dev.sh` — **dev 内部用**, 全部优化 + LOCKED-flag 不加. 基准测量 + P3 调参迭代.
- `scripts/start_vllm_bench.sh` — **P5 提交 + 官方评测用**, 仅 `--max-model-len 32768` + LOCKED-flag 不加 + 优化 env 全开.

**Q3-A / 优化触发条件**:
- env 变量**都**开 (不违反 §9 LOCKED 列表, 全是 "vLLM 内部行为" / "硬件/缓存" 类)
- `--load-format` / `--compilation-config` / `--dtype bfloat16` / `--tensor-parallel-size 1` **都加** (不修改模型行为, 不改 chat template, 不改 scheduler batch 参数)
- ⚠️ **不**加任何 LOCKED-flag. 即使技术上不算违规, 加了可能被评测脚本 warn / 拒.

**安装前置 (一次性)**:
```bash
pip install --no-deps "$Q/vllm_wheel/vllm-0.18.*.whl"
pip install runai-model-streamer
# vLLM 0.18.1 wheel 我们已经能从 sourcefind.cn 源码编译 (Task 0.6 step 2)
```

**重开容器后恢复 (~6min)**:
```bash
# 1. 重装 vLLM (~1min)
pip install --no-deps $Q/vllm_wheel/vllm-0.18.*.whl
pip install -q runai-model-streamer

# 2. 启动 (后台, ~6min 第一次, ~1min 之后 — 缓存命中)
nohup bash $Q/Govinda/scripts/start_vllm_dev.sh > $Q/vllm_serve.log 2>&1 &
disown

# 3. 等就绪
while ! curl -s --noproxy 127.0.0.1 http://127.0.0.1:8001/health >/dev/null 2>&1; do
  sleep 30
done
echo "ready"
```

## 加速效果预估

| 阶段 | 默认 | 优化后 | 节省 |
|---|---|---|---|
| 模型准备 (cp 到本地) | 1-3min | 0 (直读) | 1-3min |
| 权重加载 (load_format=auto) | 525s | 60s (runai) | ~465s |
| torch.compile 首次 | 100s | 100s | 0 (首次必编) |
| torch.compile 后续 | 100s | ~0 (命中缓存) | ~100s |
| CUDA Graph capture 默认 | 1-4min | ~1min (裁剪到 [1]) | 1-3min |
| pip install vLLM | 1-5min | 1min (--no-deps + 持久 whl) | 1-4min |
| **端到端冷启动** | **~15min** | **~6min 首次, ~1min 后续** | **~9-14min** |

**对吞吐的副作用**: 缓存命中后启动快 ≠ 跑得快. 4-8K baseline 12.16 tok/s 应当**几乎不变** (优化只在启动阶段). 后续 P3 INT8 KV cache 实测才能看到吞吐变化.

## Action items

- [x] ADR 0014 写完 (本文)
- [x] `scripts/start_vllm_dev.sh` 加全优化
- [x] `scripts/start_vllm_bench.sh` 精简到 LOCKED-clean
- [x] Memory `competition_rules.md` 已含 LOCKED 列表, 队友下 session 不会重复问
- [ ] 容器里: `pip install runai-model-streamer` (一次性)
- [ ] 容器里: 编译 `vllm-0.18.*.whl` 落 `$Q/vllm_wheel/`
- [ ] 容器里: 重启 vLLM 用 `start_vllm_dev.sh`, 确认缓存目录有内容 (`du -sh $Q/vllm_cache`)
- [ ] 验证 4-8K baseline 数字 ≤ 与未优化前差 5% (确认优化不影响吞吐)
- [ ] P3 INT8 KV cache 实测时, 同样跑 `start_vllm_dev.sh`, 对比 INT8 patch vs baseline

## 留底备忘

- HSA_OVERRIDE_GFX_VERSION=9.0.0 对 gfx90a (CDNA2) 安全; 若未来换 CDNA3 (gfx942) 改 9.4.x
- BLAS 后端 (TORCH_BLAS_PREFER_HIPBLASLT=1/0) 文章说 A/B, 我没自动开 — 留 TODO, 等 P3 大 batch prefill 测试时实测
- `nohup ... &` + `disown` 双重保险防 web shell 2h 超时杀掉
- 缓存命中要求 vLLM 版本 / 模型路径 / `--compilation-config` 一致 — 我们 dev/bench 命令一致, 缓存能复用