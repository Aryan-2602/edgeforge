"""
profile_kernels.py — count CUDA kernels in one ResNet-50 forward pass per variant.

Same model/input setup as baseline.py (including cudnn.benchmark and warm-up,
so cuDNN has already picked its algorithms). Profiles a single forward with
torch.profiler and records every CUDA kernel name, call count and device time.
Kernels whose names look like layout conversions (nchwToNhwc, nhwcToNchw,
transpose, ...) are listed separately: if FP16 NCHW pays for tensor cores with
extra transposes, they show up here.

Writes results/kernels_<variant>_<ts>.json and a Chrome trace to logs/.

Usage:
    python -m edgeforge.profile_kernels --variants fp32:1 fp16:1 fp16cl:1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from edgeforge.baseline import PRECISIONS, build, parse_variant
from edgeforge.harness import library_versions, now_iso, nvidia_smi_snapshot, save_results

LOGS_DIR = Path(__file__).parent.parent / "logs"
LAYOUT_KERNEL_PATTERN = re.compile(r"nchw|nhwc|transpose|convert|permute", re.IGNORECASE)
N_WARMUP = 50


def profile_variant(precision: str, batch_size: int, device: torch.device) -> dict:
    """Profile one forward pass and return kernel counts for the variant."""
    model, x = build(precision, batch_size, device)
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model(x)
            torch.cuda.synchronize()

    label = f"{precision}_bs{batch_size}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = LOGS_DIR / f"trace_pytorch_eager_{label}_{now_iso()}.json"
    prof.export_chrome_trace(str(trace_path))

    kernels = sorted(
        (
            {"name": e.key, "count": e.count, "device_time_us": e.device_time_total}
            for e in prof.key_averages()
            if e.device_type == DeviceType.CUDA
        ),
        key=lambda k: -k["device_time_us"],
    )
    layout = [k for k in kernels if LAYOUT_KERNEL_PATTERN.search(k["name"])]
    total_launches = sum(k["count"] for k in kernels)
    print(
        f"{label:<12} kernels launched: {total_launches:4d} ({len(kernels)} distinct) | "
        f"layout-like: {sum(k['count'] for k in layout)} launches, "
        f"{sum(k['device_time_us'] for k in layout):.0f} us of "
        f"{sum(k['device_time_us'] for k in kernels):.0f} us device time"
    )

    del model, x
    torch.cuda.empty_cache()
    return {
        "variant": label,
        "precision": precision,
        "memory_format": str(PRECISIONS[precision][1]),
        "batch_size": batch_size,
        "total_kernel_launches": total_launches,
        "distinct_kernels": len(kernels),
        "total_device_time_us": sum(k["device_time_us"] for k in kernels),
        "layout_kernels": layout,
        "kernels": kernels,
        "chrome_trace": str(trace_path.relative_to(LOGS_DIR.parent)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", type=parse_variant, default=[parse_variant(v) for v in ("fp32:1", "fp16:1", "fp16cl:1")])
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("profile_kernels.py requires CUDA; run it on Colab.")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device("cuda")

    for precision, batch_size in args.variants:
        result = profile_variant(precision, batch_size, device)
        result["cudnn_benchmark"] = args.cudnn_benchmark
        result["gpu"] = {"name": torch.cuda.get_device_name(), "nvidia_smi": nvidia_smi_snapshot()}
        result["versions"] = library_versions()
        save_results(f"kernels_pytorch_eager_{result['variant']}", result)


if __name__ == "__main__":
    main()
