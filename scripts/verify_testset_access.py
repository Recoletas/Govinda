"""Check if LongBench / RULER test sets are downloadable."""
import sys
from pathlib import Path

CHECKS = {
    "LongBench": [
        # 公开 huggingface dataset
        ("xinrongzhang2022/longbench", "NarrativeQA"),
        ("xinrongzhang2022/longbench", "Qasper"),
    ],
    "RULER": [
        # 公开 github repo
        ("https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic.json", None),
    ],
}

def check_hf(dataset, config=None):
    try:
        from datasets import load_dataset
        if config:
            ds = load_dataset(dataset, config, split="test", streaming=True)
        else:
            ds = load_dataset(dataset, split="test", streaming=True)
        sample = next(iter(ds))
        return f"OK ({len(sample)} fields)"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"

def check_url(url):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return f"OK (HTTP {r.status})"
    except Exception as e:
        return f"FAIL: {e}"

def main():
    results = {}
    for hf_ds, cfg in CHECKS["LongBench"]:
        key = f"{hf_ds}/{cfg}" if cfg else hf_ds
        results[key] = check_hf(hf_ds, cfg)
    for url in [c[0] for c in CHECKS["RULER"]]:
        results[url] = check_url(url)

    for k, v in results.items():
        status = "OK" if v.startswith("OK") else "FAIL"
        print(f"[{status}] {k} -> {v}")

    import datetime
    output = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/verify_testset_access.py",
        "results": results,
    }
    Path("benchmarks/testset_access.json").write_text(
        __import__("json").dumps(output, indent=2) + "\n"
    )
    if any("FAIL" in v for v in results.values()):
        print("\n至少一个测试集不可下载 — 立刻询问赛方")
        sys.exit(1)

if __name__ == "__main__":
    main()
