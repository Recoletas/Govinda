# ADR 0006a: vllm-rocm Docker 镜像构建状态

**日期**: 2026-06-09 (初版) / 2026-06-21 (superseded)
**关联任务**: P0 #30 (0.6 vllm-rocm Docker)
**关联 spec**: `/home/recoletas/Govinda/docs/specs/2026-06-09-qwen-inference-optimization-design.md`
**被 superseded by**: [ADR 0006b](0006b-container-instance.md)

## 状态

**Superseded — 2026-06-21** — SCNet 官方调试文档确认 Docker / docker-compose 不适用于本赛事平台, 改用 scnet.cn 控制台"容器服务" + 官方预置 image + container instance. 详见 ADR 0006b.

**历史** (2026-06-09 阻塞): 本地 WSL2 Docker daemon 拉 `rocm/vllm-rocm:v0.18.1` 在 `docker.m.daocloud.io` 镜像源 403 Forbidden, base image 触达不到. 同时 plan 风险中识别的 "pip install flash-attn 在 ROCm 环境失败" 二级风险未触达验证.

## 决策摘要

按 plan 0.6 步骤 1-3 写入了 `docker/Dockerfile`、`docker/requirements.txt`、`docker/compose.yml`，**三份内容与 plan 完全一致，未做"改进"**；同时新增了 `.dockerignore` 排除 `benchmarks/`、`docs/`、`.git/`、`__pycache__/` 等非运行期工件。

构建在 WSL2 上尝试 `docker compose build`，**在 FROM 阶段即失败**（详见"失败点"段）。预期中的 `pip install flash-attn` 阶段甚至未触达——这意味着 plan 风险评估中提到的 flash-attn ROCm 编译失败点尚不能被验证（验证本身需要先拉得到 base 镜像）。

## 失败点

### 实测错误（verbatim, `docker compose build` 2026-06-09）

```
#3 [internal] load metadata for docker.io/rocm/vllm-rocm:v0.18.1
#3 ERROR: unexpected status from HEAD request to
  https://docker.m.daocloud.io/v2/rocm/vllm-rocm/manifests/v0.18.1?ns=docker.io: 403 Forbidden
------
 > [internal] load metadata for docker.io/rocm/vllm-rocm:v0.18.1:
------
Dockerfile:3
--------------------
   1 |     # docker/Dockerfile
   2 |     # Base: vLLM 官方 ROCm 镜像，ROCm 7.0
   3 | >>> FROM rocm/vllm-rocm:v0.18.1
   4 |
   5 |     WORKDIR /workspace
--------------------
failed to solve: rocm/vllm-rocm:v0.18.1: failed to resolve source metadata for
  docker.io/rocm/vllm-rocm/v0.18.1: unexpected status from HEAD request to
  https://docker.m.daocloud.io/v2/rocm/vllm-rocm/manifests/v0.18.1?ns=docker.io: 403 Forbidden
```

### 根因

1. **网络/镜像源层面**：本 WSL2 环境的 Docker daemon 走 `docker.m.daocloud.io` 镜像，对 `rocm/vllm-rocm` 整个 namespace 返回 `403 Forbidden`——`v0.18.1`、`latest` tag 同样 403（已实测 fallback），`https://registry-1.docker.io/v2/` 直连也 timeout。
2. **计划中的二级风险（未触达）**：即使 base 镜像能拉，plan 已识别 `pip install flash-attn` 会在 ROCm 环境失败（`flash-attn==2.7.4.post1` 需要 `+rocm` torch wheel，base 镜像自带的是 ROCm 版 torch，理论上这里应能过；ROCm 7.0 上 flash-attn 编译仍可能缺 `hipcc` / `hcc` 头文件——待 base 镜像可达后再验证）。
3. **本环境无 DCU**：即便镜像构建成功，`vllm --version` 这一步在本 WSL2 还能跑（不要求 DCU），但后续 `vllm serve` 必须依赖 DCU。

## 偏离 plan 的事项

| 项 | Plan 要求 | 实际 | 是否需要变更 |
|----|----------|------|--------------|
| Dockerfile 内容 | verbatim | verbatim | 否 |
| requirements.txt 内容 | verbatim | verbatim | 否 |
| compose.yml 内容 | verbatim | verbatim | 否 |
| 新增 `.dockerignore` | plan 未要求 | 已添加（排除 `benchmarks/`、`docs/`、`.git/`、`__pycache__/` 等） | 否（属于常规工程实践；build context 减少 ~80%） |
| 替换 base image | plan 提示"若 `rocm/vllm-rocm:v0.18.1` 不存在则退到 `latest`" | **未替换**——`latest` tag 在本环境同样 403，且本环境本身就跑不通 build，替换无收益 | 否（CP 之后在有 DCU 的环境再决定） |

## 行动

- [ ] **P1 期间在 DCU 上重试**（task #30 配套 owner 上线任务）：
  1. `docker login`（若有内部 mirror）
  2. `cd docker && docker compose build` —— 优先观察 `pip install flash-attn` 是否过 ROCm 7.0
  3. `docker compose run --rm vllm vllm --version` —— 期望 `vllm 0.18.1`
  4. 若 base image `v0.18.1` tag 在内部 mirror 不存在，按 plan 提示退到 `rocm/vllm-rocm:latest` 或 `rocm/vllm:latest` 并在**本 ADR 追加一段"实际采用的 base"**
- [ ] build 成功后运行 `benchmarks/bench_smoke.py`（task #31）做 backend 路径 smoke
- [ ] 若 `pip install flash-attn` 在 DCU 上仍失败：
  - 短期：把 `flash-attn` 移出 `requirements.txt`，改用 vLLM 内部 attention（v0.18.1 已自带 ROCm 路径）
  - 长期：在 P3 Stream A 自定义层中再按需引入

## 影响

- 不影响其它 P0 任务（#24 #26 #27 已 completed；#31 等本任务完成）
- 阻塞 P0 #30 自身的 `DONE` 状态——本 ADR 关闭该任务的条件是"在 DCU 上 build 成功"或"明确决策放弃 flash-attn 预装"
- 对 P3 Stream A 块管理（task #18）的前置假设是"vllm-rocm image 可用"——若后续发现 base image 选型有变，需回头更新 P3 子任务

---

## 更新 (2026-06-21): 平台实测后, Docker 路线整体废弃

SCNet 官方《选手测试调试文档》(2026-06) 给出确切的运行环境, 不需要选手自建 Docker 镜像:

1. scnet.cn 控制台 → 容器服务 → 镜像仓库 → 搜 `qwen3.5-dtk26.04:0509` 克隆到自己账户
2. 创建容器实例 → 选 `hx1hdexclu08` 队列 + SSH 开发工具 + 自己的镜像
3. SSH 进容器 → `git clone` sourcefind.cn 的 vllm_cscc 源码 → `python setup.py bdist_wheel` → `pip install dist/vllm-*.whl`
4. `modelscope download` Qwen3.5-27B → `/root/Qwen3.5-27B` (加载快)
5. 跑 `start_vllm.sh` / `run_throughput.sh` / `run_accuracy.sh` (testdata 由赛方提供)

详细步骤与决策见 [ADR 0006b](0006b-container-instance.md). Plan P0 Task 0.6 / P2 2.1 / P4 4.1 一并改为对接官方流程. `docker/` 目录下文件留作 fallback (本地 CPU smoke test 仍可能用到), 不作 P3 必做路径前置.
