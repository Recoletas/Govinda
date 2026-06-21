# ADR 0006b: 容器服务 + qwen3.5-dtk26.04:0509 + vllm_cscc 编译

**日期**: 2026-06-21
**状态**: Accepted
**Owner**: 队长 recoletas + 队员 C (运维)
**取代**: [ADR 0006a](0006a-docker-build.md) (Docker 自建路线)
**关联**: ADR 0001 (DCU SKU, gfx90a/CDNA2), ADR 0009 (KV 量化 INT8 主路), ADR 0006 (vLLM 0.18.1 读图), Plan P0 #30 / P2 / P4

## Context

SCNet 官方《选手测试调试文档》(2026-06) 给出确切运行环境. 与原 plan 假设的 "Docker 自建 ROCm 镜像 + docker compose" 不同, 实际是:

1. **scnet.cn web console** (不是 ssh 直接登计算节点)
2. **容器服务** (不是 docker / k8s, 是云平台托管的容器实例)
3. **官方预置 image** `qwen3.5-dtk26.04:0509` (海光 DTK 26.04 + ROCm 6.4 + vLLM 0.18.1 编译工具链, 不含 vLLM wheel 本身)
4. **vLLM 源码** 不在 Docker Hub, 而在 `http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git` (国源码平台), tag `v0.18.1`
5. **模型** 走 ModelScope (`Qwen/Qwen3.5-27B`), 不是 HuggingFace
6. **测试数据集** 由赛方打包在 `https://zzefile.scnet.cn:65011/...` 下载, 含 3 档吞吐 + 4 类精度 + `start_vllm.sh` / `run_throughput.sh` / `run_accuracy.sh` 3 个官方脚本
7. **容器实例** 持久化用户家目录, 非家目录数据重启后丢失 — **所有改动必须放 `~/`**

## Decision

**完全按官方调试文档流程走**, 不自建 Docker, 不试图从 docker.io 拉 `rocm/vllm-rocm`. 具体步骤 (文档顺序):

### Step 1: 创建容器实例 (web console, 一次性)
1. scnet.cn 登录 → 右上角"控制台"
2. 点"容器服务" → "镜像管理" → "镜像仓库" → 选"核心节点分区一"
3. 搜索 `qwen3.5-dtk26.04:0509`, 点"克隆到我的镜像"
4. "容器实例" → "返回旧版" → "创建容器"
5. 选 `hx1hdexclu08` 队列, 开发工具 "SSH", 镜像选刚克隆的
6. 创建, 等待状态 Running → 点 "SSH" 进入

### Step 2: 容器内编译 vLLM (家目录下, **每次重启都需重做**)
```bash
cd ~
git clone -b v0.18.1 --depth 1 http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git
cd vllm_cscc
python setup.py bdist_wheel       # 首次 ~10min, 后续 ~2min
cd dist
pip install vllm-*.whl --no-deps
```

> 选手对 vLLM 的任何优化 (kernel / 调度 / 算子融合 / 编译参数) **必须**能重新编译出 `dist/vllm-*.whl` 并正常安装, 评测脚本以此为基线.

### Step 3: 下载模型 (ModelScope, 家目录)
```bash
cd ~
pip install modelscope
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./Qwen3.5-27B
cp -r ./Qwen3.5-27B /root/Qwen3.5-27B    # /root 加载更快
```

### Step 4: 下载测试数据集 + 官方脚本
```bash
cd ~
curl -f -C - -o testdata.tar.gz \
  https://zzefile.scnet.cn:65011/efile/s/d/c2N5MTE1OTkxMDU1OQ==/a927e65672549b46
mkdir -p ./testdata
tar -xzf testdata.tar.gz -C ./testdata --strip-components=1
cd testdata && chmod +x *.sh
```

### Step 5: 启动 + smoke test
```bash
./start_vllm.sh     # 后台启 vLLM, 监听 127.0.0.1:8001
hy-smi              # 看 DCU 占用
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"你好, 简单回复一句话。"}],"temperature":0.0,"max_tokens":64}'
```

### Step 6: 吞吐 + 精度
```bash
./run_throughput.sh [all | 4-8K | 8-16K | 16-32K] [N]
./run_accuracy.sh   [hotpotqa | gov_report | retrieval_multi_point | aggregation_keyword_aggregation] [N]
```

## 评估指标 (官方明确)

- **输出吞吐量 (Output Tokens/s)** — 主指标
- **首 token 时延 (TTFT) P99**
- **每 token 时延 (TPOT) P99**

3 档 (4-8K / 8-16K / 16-32K), 权重 20% / 50% / 30% (来自 spec §1, 文档未列权重 — 以 spec 为准).

## 数据集含义

吞吐 (`*-throughput.jsonl`) — 长上下文 prompt 测试集, 与赛题 "并发=1" 场景一致.

精度 4 类 (对应 spec §3 提到的 LongBench 风格):
- `hotpotqa` — 多文档问答 (F1)
- `gov_report` — 长文摘要 (ROUGE)
- `retrieval_multi_point` — 多点检索 (赛方自定义, 不用 OpenCompass 汇总分, 单条匹配)
- `aggregation_keyword_aggregation` — 多答案聚合 (赛方自定义)

`retrieval_multi_point` + `aggregation_keyword_aggregation` 走"模型输出 → 答案列表 → 逐项匹配", 不是 OpenCompass 的标准 path. spec 旧版没区分这点, 已 update 进 §3.

## Consequence

- **`docker/Dockerfile`, `docker/requirements.txt`, `docker/compose.yml`** — 留作本地 CPU smoke 用, **不**作 P3 必做路径前置. ADR 0006a 仍记录其历史.
- **Plan P0 0.6** 任务 #30 改写: "vllm-rocm Docker 构建" → "容器实例创建 + qwen3.5-dtk26.04:0509 克隆"
- **Plan P0 新增 0.8**: clone vllm_cscc + 比对 upstream vllm/vllm v0.18.1 看海光是否有 patch
- **Plan P2 2.1 bench harness**: 改用官方 `run_throughput.sh` 出数, `benchmarks/compare.py` 仅做 baseline vs optimized 对比 + ROI 报告
- **Plan P4 4.1 精度回归**: 改用官方 `run_accuracy.sh` 跑 4 类任务, 输出落到 `benchmarks/accuracy/`
- **`reports/optimization-plan.md` (§14 提交材料)**: 描述里需要写明 "评测环境基于容器实例 image `qwen3.5-dtk26.04:0509` + 自编译 vllm wheel from sourcefind.cn v0.18.1 + DTK 26.04 / ROCm 6.4 / gfx90a DCU"
- **`docker/`** 目录: 暂不删, 等 P3 末确认本地 dev loop 不需要再删

## 不做什么

- **不**试图本地 docker pull `rocm/vllm-rocm` — 平台不通, 时间沉没成本
- **不**自己从 ROCm 编译 flash-attn — image 已预置编译好的 ROCm 版
- **不**用 HuggingFace 下载模型 — 走 ModelScope (赛方默认)

## 待办

- [ ] 队长: 走 Step 1-6 实跑 1 次, 确认全流程通, 把"实际启动 vLLM 用了多少秒" / "模型加载用了多少秒" append 到本 ADR 末
- [ ] 队员 B: clone vllm_cscc 后 `git diff vllm/vllm v0.18.1` 比对海光 patch (若有, 影响 P3 Stream B/C 路径, 例如海光可能 patch 了 V0.18.1 不支持的 INT8 path)
- [ ] 队员 A: 比对 `qwen3.5-dtk26.04:0509` 的 Python site-packages vs 官方 `rocm/vllm-rocm:latest`, 看 image 是否已含 torch / vllm 任何 wheel (文档没明说, 取决于 image 内部)
- [ ] Plan P0 Task 0.6 / P2 2.1 / P4 4.1 改写后, update weekly/progress.md 本周条目