# AI-generated, awaiting verification by <team-lead> on <YYYY-MM-DD>
"""Detect DCU SKU + FP8 support. Run on DCU host."""
import subprocess
import sys
import json
from pathlib import Path

def get_device_props():
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0)

# Currently unused — reserved for future cross-check with torch's gcnArchName.
# Plan update pending team decision (Task 0.1 review).
def get_rocminfo():
    try:
        out = subprocess.check_output(["rocminfo"], text=True)
        return out
    except FileNotFoundError:
        return None

def detect_fp8_support():
    import torch
    try:
        # FNUZ variant (CDNA3)
        x = torch.zeros(1, 1, dtype=torch.float8_e4m3fnuz, device="cuda")
        return "FNUZ"
    except (TypeError, RuntimeError):
        pass
    try:
        # Standard OCP variant
        x = torch.zeros(1, 1, dtype=torch.float8_e4m3, device="cuda")
        return "OCP"
    except (TypeError, RuntimeError):
        pass
    return "NONE"

def main():
    props = get_device_props()
    if props is None:
        print("ERROR: torch.cuda not available — not on a GPU/DCU host")
        sys.exit(1)
    gcn_arch = getattr(props, "gcnArchName", "unknown")
    device_name = props.name
    fp8 = detect_fp8_support()

    result = {
        "device_name": device_name,
        "gcn_arch": gcn_arch,
        "fp8_support": fp8,
        "total_memory_gb": props.total_memory / (1024 ** 3),
    }
    print(json.dumps(result, indent=2))
    # Resolve output path relative to this script's location (repo root)
    output_path = Path(__file__).parent.parent / "benchmarks" / "device_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    # Hard gate
    if gcn_arch == "gfx90a" and fp8 != "NONE":
        print(f"WARN: CDNA2 (gfx90a) reported FP8={fp8} — verify with AMD docs")
    if "gfx942" not in gcn_arch and "gfx90a" not in gcn_arch:
        print(f"WARN: unknown arch {gcn_arch} — check ROCm support matrix")

if __name__ == "__main__":
    main()
