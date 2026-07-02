# Govinda 团队 Onboarding.

## 0. TL;DR

- baseline 已 AC: 4-8K = 12.95, 8-16K = 10.03, 16-32K = 5.75 tok/s, 最终得分 59.9119, 排名 56/76, SLA 扣分 0, 精度扣分 0
- DCU 容器是单一脆弱的集成环境. 不要当普通共享开发服务器用.
- 队长是默认**容器操作员**: 启 vLLM, 跑 benchmark, 推提交分支
- 队员通过 GitLab MR / patch / review notes / 可复现本地分析工作

## 1. 当前比赛状态

我们在 Hygon DCU 上优化 vLLM 0.18.1 + Qwen3.5-27B. 目标是在 SLA + 精度约束内提 4K-8K / 8K-16K / 16K-32K 三档 throughput.

官方 AC baseline:

| 档位 | 官方 throughput | SLA 扣分 | 精度扣分 |
| --- | ---: | ---: | ---: |
| 4K-8K | 12.95 tok/s | 0.0 | 0.0 |
| 8K-16K | 10.03 tok/s | 0.0 | 0.0 |
| 16K-32K | 5.75 tok/s | 0.0 | 0.0 |

baseline 是回退目标. 任何风险 SLA / 精度的优化必须 revert.

## 2. 协作模式

### 推荐: MR 协作

每名队员用自己的 GitLab 账号, 以 MR 或 patch 提交改动. 队长 review diff, 应用到提交仓库, 跑 DCU 验证, 决定是否提交.

为什么这是默认:
- 容器 job ID, 节点, IP 每次重启都变
- 只能有 1 个 vLLM 进程实际占 DCU
- benchmark 慢, 易被 stale 进程或旧结果污染
- 误操作 `pkill`, 环境变量, 覆盖 result 会浪费几小时
- GitLab history 给可追溯 + 可回滚

### 避免: 共享容器自由接入

不要几个人登同一比赛账号独立操作容器. 会产生不可追溯状态, benchmark 结果不可信.

短时配对调试可以共享队长账号, 但队长必须在线, 1 个人明确主导.

### 可选: 子账号

如果 SCNet 支持子账号, 用它们做只读或低风险工作:
- 读 log
- 看文件
- 准备命令
- 检查仓库状态

除非队长分配了操作窗口, 不要用子账号跑 vLLM benchmark.

## 3. 操作员规则

只有当前操作员可以:
- 启停 vLLM
- 跑 `run_throughput.sh` 或长时间 `vllm bench serve`
- kill 容器内进程
- 覆盖 `test/*/result.json`
- 推最终 GitLab 提交分支

操作员移交前, 记录:
- Slurm job ID
- compute node
- 容器 ID
- 当前 vLLM PID
- 当前 log 路径
- 正在跑的 benchmark (如有)

用 `docs/weekly/progress.md` 或短消息做这个交接.

## 4. 队员可以安全做的事

### Task A: 源码 review + patch 提案

适合 vLLM / Python 队员.

交付物:
- GitLab MR 或 patch
- 改动文件简要说明
- 赛题规则影响声明
- 预期性能影响
- 最小验证命令

当前安全目标:
- 小 ROCm env / 默认行为改动
- 请求解析兼容性 fix
- 启动可靠性 fix
- 结果解析 / 报告脚本

避免高风险改动 (除非队长明确批准):
- scheduler 代码
- 投机解码
- 剪枝 / skip
- 持久化权重量化
- 数据集 / 答案预处理

### Task B: log + 结果分析

适合 QA / 支持队员.

交付物:
- `result.json` 字段摘要
- 跟官方 baseline 对比
- SLA check
- 可疑 log 行
- 建议: 保留 / 重测 / revert

用已 AC 的 baseline 作参考. 不要跟不同容器的旧结果比, 除非明确标注.

### Task C: 提交材料

适合不太熟 DCU 内部的队员.

交付物:
- 更新 `reports/env-vars.md`
- 更新 `reports/optimization-plan.md`
- 更新 `reports/submission-readme.md`
- 声明必须绑实测数字

不要编加速数字. 不是官方评测或干净 bench run 的结果, 标 "diagnostic".

## 5. MR 检查清单

每个 MR 或 patch 必须答:
- 改了哪些文件?
- 是否动 LOCKED scheduler 参数或 scheduler 代码?
- 是否改模型权重 / 模型格式 / 持久化量化权重?
- 是否引入投机解码 / 剪枝 / early exit / 缓存答案?
- 最小验证命令是什么?
- 队长应该先跑哪一档 benchmark?
- 回滚 commit 是哪个?

推荐 MR 标题格式:

```text
[area] 简要改动说明
```

例子:

```text
[rocm] enable safe aiter defaults without changing attention backend
[openai] flatten text-only content arrays before multimodal parsing
[reports] record accepted baseline and env vars
```

## 6. 容器访问参考

队长在 sandbox 的 `/home/recoletas/HANDOVER.md` 维护最新可用连接笔记. 该文件本地, 不入仓, 因为含频繁变动的操作细节.

典型流程:

```bash
ssh scnet-login "squeue -u xdzs2026_c087"
ssh scnet-login "srun --overlap --jobid=<JOBID> --gres=dcu:0 --nodelist=<NODE> docker ps"
ssh scnet-login "srun --overlap --jobid=<JOBID> --gres=dcu:0 --nodelist=<NODE> docker exec <CONTAINER> bash -lc '<command>'"
```

容器重启后, 不要依赖旧的 `scnet-ctr` alias.

## 7. 比赛边界

严禁:
- 投机解码
- 剪枝 / token pruning / early exit / 跳过层
- 持久化权重量化 / 模型格式转换
- 改 LOCKED scheduler 参数
- 改 benchmark 统计 / 结果保存
- 预缓存测试答案 / 中间结果

允许:
- 非持久化 KV cache 量化
- activation 动态量化
- kernel 级低精度
- 自定义 Python 包 + custom kernel (需评测容器内可编译)
- 不改模型语义的小运行时兼容 fix

不确定 → 停手问队长, 再写代码.

## 8. 实用计划

剩余时间按这个顺序:
1. 已 AC baseline 分支保留作 rollback
2. 一次只测 1 个优化
3. 倾向低风险默认 + 运行时 fix, 而不是大改写
4. 先跑 1 档 smoke, 通常 4K-8K
5. 干净 run 后再提交, 或队长接风险后提交

当前候选分支:
- GitLab 提交仓库 `v0.18.1`
- 最新已知候选: safe ROCm AITER defaults without changing attention backend




scnet.cn ->卡的地方，我们操作得在这里完成；
course.educg.net->比赛的地方，我们交和排行榜
https://gitlab.eduxiji.net/T2026900029912415/vllm-> gitlab仓库，是vllm_cscc
https://github.com/Recoletas/Govinda 拉在wsl，agent的地方连接scnet.cn，agnet操作这个，自己在地方：作用：写agent文档，写脚本，修改，让agent传到比赛的地方和gitlab

gpu占用率->自己agent多审查然后交到gitlab分支，我到时候pr，我交评测


ssh三跳连接scnet.cn卡的地方
