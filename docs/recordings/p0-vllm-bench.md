# P0 vllm bench 跑通文字稿

**状态**: 待 队员 C 填充
**截止**: P0 末
**Owner**: 队员 C (浮动, 4 h/周 × 1.5 周)

## PENDING — assigned to 队员 C

队员 C 在本地（CPU mock 或 GPU）跑通以下命令并记录输出格式：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 &
sleep 60
vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 10 --burstiness 1.0
```

文字稿记录：
- 启动 serve 的关键日志（ready 标志）
- bench serve 输出各字段含义（TTFT / TPOT / throughput 等）
- 跑通过程中遇到的问题与解决
- 与 spec §9 评测指标的对齐情况
