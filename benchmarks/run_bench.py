#!/usr/bin/env python3
# benchmarks/run_bench.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Run vllm bench serve for a specific input-length tier."""
import argparse
import json
import subprocess
import time
from pathlib import Path

TIER_PROMPTS = {
    "4k-8k": ("--random-input-len 6000 --random-output-len 256", 6000),
    "8k-16k": ("--random-input-len 12000 --random-output-len 256", 12000),
    "16k-32k": ("--random-input-len 24000 --random-output-len 256", 24000),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--tier", choices=list(TIER_PROMPTS), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--extra-args", default="")
    args = p.parse_args()

    length_arg, _ = TIER_PROMPTS[args.tier]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.tier}-{int(time.time())}.json"

    # 启服务
    serve_log = out_dir / f"serve-{args.tier}.log"
    serve_cmd = (
        f"vllm serve {args.model} "
        f"--max-model-len 32768 --max-num-seqs 1 "
        f"--served-model-name govinda --port 8000 "
        f"{args.extra_args}"
    )
    print(f"Starting: {serve_cmd}")
    serve_proc = subprocess.Popen(serve_cmd, shell=True,
                                   stdout=open(serve_log, "w"),
                                   stderr=subprocess.STDOUT)

    try:
        # 等服务起来（最多 5 分钟）
        for _ in range(60):
            time.sleep(5)
            try:
                import requests
                if requests.get("http://localhost:8000/v1/models", timeout=2).status_code == 200:
                    break
            except Exception:
                continue
        else:
            raise RuntimeError("vllm serve did not start in 5 minutes")

        # 跑 bench
        bench_cmd = (
            f"vllm bench serve "
            f"--model govinda --backend vllm "
            f"--host localhost --port 8000 "
            f"--num-prompts {args.num_prompts} "
            f"--dataset-name random "
            f"{length_arg} "
            f"--save-result --result-filepath {out_file}"
        )
        print(f"Running: {bench_cmd}")
        subprocess.run(bench_cmd, shell=True, check=True)

        # 解析输出（vllm bench serve 默认 JSON 包含 throughput / ttft / tpot）
        data = json.loads(out_file.read_text())
        # 标准化 key 名（vllm bench 不同版本 key 名略不同）
        normalized = {
            "tier": args.tier,
            "num_prompts": args.num_prompts,
            "throughput_tokens_per_sec": data.get("total_token_throughput", 0)
                or data.get("throughput", 0),
            "ttft_p99_ms": data.get("ttft_p99", 0)
                or data.get("metrics", {}).get("ttft_p99", 0),
            "tpot_p99_ms": data.get("tpot_p99", 0)
                or data.get("metrics", {}).get("tpot_p99", 0),
            "raw": data,
        }
        out_file.write_text(json.dumps(normalized, indent=2))
        print(f"Wrote {out_file}")
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

if __name__ == "__main__":
    main()
