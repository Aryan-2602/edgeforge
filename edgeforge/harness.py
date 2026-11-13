"""
harness.py — GPU latency/throughput harness for edgeforge.

Ported from Inferno: now_iso() and save_results() come from inferno/utils.py,
percentile reporting from benchmarks/bench_batching.py. New here: a repeated
timing loop and environment metadata (GPU clocks, driver, library versions).

run_benchmark() takes two measurements of the same callable:

  (a) per-call — sync after every call, timed with CUDA events (perf_counter
      recorded alongside for comparison). Gives p50/p95/p99/mean latency;
      per-call throughput = batch / mean latency.
  (b) pipelined — n_iters calls back-to-back with no sync in between, one sync
      at the end. Pipelined throughput = batch * n_iters / total time. The CPU
      can enqueue call i+1 while the GPU is still running call i, so (b) shows
      how much of (a) was launch overhead / idle GPU rather than compute.

Every call writes results/<label>_<UTC timestamp>.json and never overwrites.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"

NVIDIA_SMI_FIELDS = [
    "name",
    "driver_version",
    "pstate",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "clocks.max.mem",
    "temperature.gpu",
    "power.draw",
    "power.limit",
]

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result saving (from Inferno)
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return current UTC time as a compact ISO string safe for filenames (millisecond precision)."""
    dt = datetime.now(timezone.utc)
    ms = dt.microsecond // 1000
    return dt.strftime("%Y%m%dT%H%M%S") + f"{ms:03d}Z"


def save_results(name: str, data: dict[str, Any]) -> Path:
    """Serialize data to results/<name>_<timestamp>.json and return the path. Never overwrites."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_{now_iso()}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def nvidia_smi_snapshot() -> dict[str, str]:
    """
    Query nvidia-smi for name, driver, clocks, temperature and power.

    Taken before and after timing: a T4 that throttles mid-run (clocks.sm drops,
    temperature climbs) explains a fat p99 better than any code change.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    first_gpu = out.strip().splitlines()[0]
    return dict(zip(NVIDIA_SMI_FIELDS, (v.strip() for v in first_gpu.split(","))))


def library_versions() -> dict[str, Optional[str]]:
    """Return python/torch/CUDA/cuDNN versions plus TRT and ORT if installed (None if not)."""
    versions: dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
    }
    for module in ("torchvision", "tensorrt", "onnxruntime"):
        try:
            versions[module] = importlib.import_module(module).__version__
        except ImportError:
            versions[module] = None
    return versions


def _latency_stats(ms: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
        "mean": float(ms.mean()),
        "std": float(ms.std()),
        "min": float(ms.min()),
        "max": float(ms.max()),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    fn: Callable[[], Any],
    batch_size: int,
    n_warmup: int,
    n_iters: int,
    label: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Time fn() on the current CUDA device and save results/<label>_<ts>.json.

    fn must be a zero-arg callable that does one inference on an input that is
    already on the GPU; anything fn does (including host<->device copies) is
    what gets timed. extra is merged into the JSON for run-specific context
    (model, precision, backend, ...).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_benchmark requires CUDA; no GPU found.")

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    smi_before = nvidia_smi_snapshot()

    # ---- (a) per-call latency: sync after every call ----
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    wall_ms = np.empty(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        starts[i].record()
        fn()
        ends[i].record()
        torch.cuda.synchronize()
        wall_ms[i] = (time.perf_counter() - t0) * 1000.0
    event_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])

    # ---- (b) pipelined throughput: no sync between calls ----
    torch.cuda.synchronize()
    pipe_start = torch.cuda.Event(enable_timing=True)
    pipe_end = torch.cuda.Event(enable_timing=True)
    pipe_start.record()
    for _ in range(n_iters):
        fn()
    pipe_end.record()
    torch.cuda.synchronize()
    pipelined_total_ms = pipe_start.elapsed_time(pipe_end)

    smi_after = nvidia_smi_snapshot()

    latency = _latency_stats(event_ms)
    result: dict[str, Any] = {
        "label": label,
        "timestamp_utc": now_iso(),
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        "latency_ms": latency,
        "wall_latency_ms": _latency_stats(wall_ms),
        # Raw per-call samples, so tail shape (e.g. a second mode) can be checked later.
        "latency_ms_samples": [round(float(v), 4) for v in event_ms],
        "throughput_img_s": {
            "per_call": batch_size / latency["mean"] * 1000.0,
            "pipelined": batch_size * n_iters / pipelined_total_ms * 1000.0,
        },
        "pipelined_total_ms": pipelined_total_ms,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
            "nvidia_smi_before": smi_before,
            "nvidia_smi_after": smi_after,
        },
        "versions": library_versions(),
        "extra": extra or {},
    }

    path = save_results(label, result)
    logger.info(
        "%s | p50 %.2f ms  p95 %.2f  p99 %.2f  mean %.2f | %.0f img/s per-call, %.0f img/s pipelined | %s",
        label,
        latency["p50"],
        latency["p95"],
        latency["p99"],
        latency["mean"],
        result["throughput_img_s"]["per_call"],
        result["throughput_img_s"]["pipelined"],
        path.name,
    )
    return result
