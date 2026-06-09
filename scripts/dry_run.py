#!/usr/bin/env python3
# scripts/dry_run.py
# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Orchestrate P5 dry run: clean build -> serve -> verify -> bench -> accuracy -> shutdown.

Steps:
  1. clean_build        : docker compose build --no-cache
  2. start_service      : docker compose up -d + poll /v1/models (max 5 min)
  3. dcu_and_testset    : verify_dcu.py + verify_testset_access.py as subprocesses
  4. bench_3tier        : run benchmarks/run_bench.py for each tier (4k-8k, 8k-16k, 16k-32k)
  5. accuracy_validation: run opencompass on 4 task types (QA / 摘要 / 检索 / 聚合)

Note: requires DCU host + docker + vllm-rocm base image. Cannot be fully executed
in WSL2 dev env without DCU hardware. This script is a runner, not a tutorial.
"""
import argparse
import datetime
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Resolve repo root from this script's location
REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker" / "compose.yml"
DEFAULT_TIERS = ["4k-8k", "8k-16k", "16k-32k"]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "weekly" / "p5-dryrun-log.md"
DEFAULT_DRYRUN_DIR = REPO_ROOT / "benchmarks" / "dryrun"
SERVICE_URL = "http://localhost:8000/v1/models"
SERVICE_POLL_TIMEOUT_SEC = 300
SERVICE_POLL_INTERVAL_SEC = 5
CLEAN_BUILD_TIMEOUT_SEC = 1800

ACCURACY_TASKS = [
    ("QA", "opencompass_qa"),
    ("摘要", "opencompass_summary"),
    ("检索", "opencompass_retrieval"),
    ("聚合", "opencompass_aggregation"),
]


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _run(cmd, cwd=None, timeout=None, log_path=None):
    """Run a subprocess, capturing output to log_path if given. Returns CompletedProcess."""
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w")
    else:
        log_fh = None
    try:
        cp = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=cwd or str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdout=log_fh if log_fh else subprocess.PIPE,
            stderr=subprocess.STDOUT if log_fh else subprocess.STDOUT,
        )
    finally:
        if log_fh:
            log_fh.close()
    return cp


def step1_clean_build(log_dir):
    """Step 1: clean build of the docker image."""
    log_path = log_dir / "step1_clean_build.log"
    start = _now_iso()
    t0 = time.time()
    try:
        cp = _run(
            f"docker compose -f {COMPOSE_FILE} build --no-cache",
            timeout=CLEAN_BUILD_TIMEOUT_SEC,
            log_path=log_path,
        )
        dur = time.time() - t0
        if cp.returncode == 0:
            return "OK", dur, None, str(log_path)
        return "FAIL", dur, f"docker build exited {cp.returncode}", str(log_path)
    except subprocess.TimeoutExpired:
        dur = time.time() - t0
        return "FAIL", dur, f"timeout after {CLEAN_BUILD_TIMEOUT_SEC}s", str(log_path)
    except Exception as e:
        dur = time.time() - t0
        return "FAIL", dur, f"{type(e).__name__}: {e}", str(log_path)


def step2_start_service(log_dir):
    """Step 2: docker compose up -d + poll /v1/models until ready (max 5 min)."""
    log_path = log_dir / "step2_start_service.log"
    up_log = log_dir / "step2_docker_up.log"
    poll_log = log_dir / "step2_poll.log"
    start = _now_iso()
    t0 = time.time()
    try:
        # Bring up detached
        up_cp = _run(
            f"docker compose -f {COMPOSE_FILE} up -d",
            timeout=300,
            log_path=up_log,
        )
        if up_cp.returncode != 0:
            dur = time.time() - t0
            return "FAIL", dur, f"docker up exited {up_cp.returncode}", str(log_path)

        # Poll /v1/models
        deadline = t0 + SERVICE_POLL_TIMEOUT_SEC
        poll_log.parent.mkdir(parents=True, exist_ok=True)
        with open(poll_log, "w") as pf:
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(SERVICE_URL, timeout=2) as r:
                        if r.status == 200:
                            pf.write(f"{_now_iso()} OK HTTP {r.status}\n")
                            dur = time.time() - t0
                            return "OK", dur, None, str(log_path)
                except Exception as e:
                    pf.write(f"{_now_iso()} {type(e).__name__}: {e}\n")
                time.sleep(SERVICE_POLL_INTERVAL_SEC)
        dur = time.time() - t0
        return "FAIL", dur, "vllm serve did not start in 5 min", str(log_path)
    except Exception as e:
        dur = time.time() - t0
        return "FAIL", dur, f"{type(e).__name__}: {e}", str(log_path)


def step3_dcu_and_testset(log_dir):
    """Step 3: run verify_dcu.py + verify_testset_access.py as subprocesses."""
    log_path = log_dir / "step3_verify.log"
    start = _now_iso()
    t0 = time.time()
    dcu_log = log_dir / "step3_verify_dcu.log"
    testset_log = log_dir / "step3_verify_testset.log"
    errors = []
    try:
        dcu_cp = _run(
            ["python", str(REPO_ROOT / "scripts" / "verify_dcu.py")],
            timeout=300,
            log_path=dcu_log,
        )
        if dcu_cp.returncode != 0:
            errors.append(f"verify_dcu exited {dcu_cp.returncode}")
        ts_cp = _run(
            ["python", str(REPO_ROOT / "scripts" / "verify_testset_access.py")],
            timeout=300,
            log_path=testset_log,
        )
        if ts_cp.returncode != 0:
            errors.append(f"verify_testset exited {ts_cp.returncode}")
        dur = time.time() - t0
        if errors:
            return "FAIL", dur, "; ".join(errors), str(log_path)
        return "OK", dur, None, str(log_path)
    except Exception as e:
        dur = time.time() - t0
        return "FAIL", dur, f"{type(e).__name__}: {e}", str(log_path)


def step4_bench_3tier(tiers, log_dir):
    """Step 4: run benchmarks/run_bench.py for each tier."""
    log_path = log_dir / "step4_bench_3tier.log"
    t0 = time.time()
    errors = []
    tier_results = []
    try:
        for tier in tiers:
            tier_out = DEFAULT_DRYRUN_DIR / tier
            tier_out.mkdir(parents=True, exist_ok=True)
            tier_log = log_path  # append all tiers to one log
            cmd = (
                f"python {REPO_ROOT / 'benchmarks' / 'run_bench.py'} "
                f"--tier {tier} --output {tier_out}"
            )
            cp = _run(cmd, timeout=3600, log_path=tier_log if tier == tiers[0] else None)
            tier_results.append((tier, cp.returncode, str(tier_out)))
            if cp.returncode != 0:
                errors.append(f"{tier} exited {cp.returncode}")
        dur = time.time() - t0
        if errors:
            return "FAIL", dur, "; ".join(errors), str(log_path)
        return "OK", dur, None, str(log_path)
    except Exception as e:
        dur = time.time() - t0
        return "FAIL", dur, f"{type(e).__name__}: {e}", str(log_path)


def step5_accuracy_validation(log_dir, skip_accuracy):
    """Step 5: run opencompass on 4 task types (placeholder)."""
    log_path = log_dir / "step5_accuracy.log"
    t0 = time.time()
    if skip_accuracy:
        return "SKIP", 0.0, "skipped via --skip-accuracy", str(log_path)
    start = _now_iso()
    if shutil.which("opencompass") is None:
        dur = time.time() - t0
        return (
            "FAIL",
            dur,
            "WARN: opencompass not installed; install via `pip install opencompass`",
            str(log_path),
        )
    # Placeholder invocation; task configs would be resolved at run time
    errors = []
    for label, cfg in ACCURACY_TASKS:
        task_log = log_dir / f"step5_opencompass_{label}.log"
        cmd = f"opencompass --config {cfg} --server-url {SERVICE_URL.replace('/v1/models','')}"
        cp = _run(cmd, timeout=3600, log_path=task_log)
        if cp.returncode != 0:
            errors.append(f"{label} exited {cp.returncode}")
    dur = time.time() - t0
    if errors:
        return "FAIL", dur, "; ".join(errors), str(log_path)
    return "OK", dur, None, str(log_path)


def shutdown_service(log_dir):
    """Tear down the docker service (best-effort)."""
    log_path = log_dir / "shutdown.log"
    try:
        cp = _run(
            f"docker compose -f {COMPOSE_FILE} down",
            timeout=180,
            log_path=log_path,
        )
        return "OK" if cp.returncode == 0 else f"FAIL(exit {cp.returncode})"
    except Exception as e:
        return f"FAIL({type(e).__name__}: {e})"


def render_markdown(run_meta, steps):
    """Render the structured run into a markdown log file."""
    lines = []
    lines.append("# P5 Dry Run Log")
    lines.append("")
    lines.append("**Phase**: P5 (演练 + 提交)")
    lines.append(f"**演练人**: {run_meta['user']}")
    lines.append(f"**演练日期**: {run_meta['start_at']}")
    lines.append(f"**总状态**: {run_meta['overall_status']}")
    lines.append(f"**总耗时**: {run_meta['total_duration_sec']:.1f}s")
    lines.append(f"**主机**: {run_meta['hostname']}")
    lines.append(f"**Tiers**: {', '.join(run_meta['tiers'])}")
    lines.append(f"**Skip accuracy**: {run_meta['skip_accuracy']}")
    lines.append("")
    for s in steps:
        lines.append(f"## Step {s['idx']}: {s['name']}")
        lines.append("")
        lines.append(f"- 状态: `{s['status']}`")
        lines.append(f"- 起: {s['start_at']}")
        lines.append(f"- 止: {s['end_at']}")
        lines.append(f"- 耗时: {s['duration_sec']:.1f}s")
        lines.append(f"- 错误: {s['error'] or '无'}")
        lines.append(f"- 输出: `{s['output_path']}`")
        lines.append("")
    lines.append("## TODO after run")
    lines.append("")
    lines.append("- [ ] 队员 C 签")
    lines.append("- [ ] 附 DCU 上跑分的截图 / nvidia-smi / rocprofv3 输出到 `docs/weekly/p5-dryrun-attachments/`")
    lines.append("- [ ] 把 step1-5 的状态从待填填入实际值")
    lines.append("- [ ] 更新 plan Task 5.1 status: completed")
    lines.append("- [ ] 若任一 step FAIL, 在 PR 描述里写明补救计划")
    lines.append("")
    lines.append("## 关联文档")
    lines.append("")
    lines.append("- spec §11 (完工标准)")
    lines.append("- spec §10 风险表 (KV 量化精度塌方回退流程)")
    lines.append("- plan Task 4.2 (精度验证 4 类任务)")
    lines.append("- plan Task 5.1 (本任务)")
    lines.append("- `scripts/dry_run.py` (orchestrator)")
    lines.append("- `scripts/verify_dcu.py`")
    lines.append("- `scripts/verify_testset_access.py`")
    lines.append("- `benchmarks/run_bench.py`")
    lines.append("- `benchmarks/analyze.py`")
    lines.append("- `docker/compose.yml`")
    lines.append("- `docker/Dockerfile`")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="P5 dry-run orchestrator")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help=f"markdown log output path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--skip-accuracy", action="store_true",
                   help="skip opencompass accuracy step")
    p.add_argument("--tier", nargs="+", default=DEFAULT_TIERS,
                   help=f"tiers to bench (default: {DEFAULT_TIERS})")
    p.add_argument("--log-dir", default=None,
                   help="directory for per-step raw logs (default: benchmarks/dryrun/logs/)")
    args = p.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else (DEFAULT_DRYRUN_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    started_at = _now_iso()
    t_run = time.time()
    hostname = socket.gethostname()
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip() or "unknown"

    print(f"=== P5 Dry Run started at {started_at} on {hostname} as {user} ===")
    print(f"log_dir={log_dir}  output={output_path}  tiers={args.tier}  skip_accuracy={args.skip_accuracy}")
    print("Note: requires DCU host + docker + vllm-rocm base image. "
          "Cannot be fully executed in WSL2 dev env without DCU hardware.")
    print("")

    steps_def = [
        ("clean_build", step1_clean_build),
        ("start_service", step2_start_service),
        ("dcu_and_testset_verify", step3_dcu_and_testset),
        ("bench_3tier", step4_bench_3tier),
        ("accuracy_validation", step5_accuracy_validation),
    ]

    results = []
    for idx, (name, fn) in enumerate(steps_def, start=1):
        start = _now_iso()
        try:
            if name == "bench_3tier":
                status, dur, err, outp = fn(args.tier, log_dir)
            elif name == "accuracy_validation":
                status, dur, err, outp = fn(log_dir, args.skip_accuracy)
            else:
                status, dur, err, outp = fn(log_dir)
        except Exception as e:
            status, dur, err, outp = "FAIL", 0.0, f"{type(e).__name__}: {e}", ""
        end = _now_iso()
        rec = {
            "idx": idx,
            "name": name,
            "status": status,
            "start_at": start,
            "end_at": end,
            "duration_sec": dur,
            "error": err,
            "output_path": outp,
        }
        results.append(rec)
        # Per-step stdout line
        print(f"[step {idx}/{len(steps_def)}] {name} -> {status} ({dur:.1f}s)"
              + (f" err={err}" if err else ""))
        # continue-on-error: do not break; collect all evidence

    total_dur = time.time() - t_run

    # Best-effort shutdown
    sd_status = shutdown_service(log_dir)
    print(f"shutdown service: {sd_status}")

    overall = "PASS" if all(s["status"] in ("OK", "SKIP") for s in results) else "FAIL"

    run_meta = {
        "user": user,
        "hostname": hostname,
        "start_at": started_at,
        "end_at": _now_iso(),
        "total_duration_sec": total_dur,
        "tiers": args.tier,
        "skip_accuracy": args.skip_accuracy,
        "overall_status": overall,
        "shutdown": sd_status,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(run_meta, results)
    output_path.write_text(md)
    # Also write a machine-readable JSON next to it
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({"meta": run_meta, "steps": results}, indent=2))
    print("")
    print(f"=== overall={overall} total={total_dur:.1f}s ===")
    print(f"markdown log: {output_path}")
    print(f"json log:     {json_path}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
