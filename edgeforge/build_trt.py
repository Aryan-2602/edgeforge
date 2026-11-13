"""
build_trt.py — build a TensorRT 10 FP16 engine from the FP32 ONNX export.

Parses models/resnet50_fp32.onnx, sets BuilderFlag.FP16 (TRT may pick FP16 or
FP32 per layer — "weak typing"), adds one optimization profile for the input
batch (min 1 / opt 8 / max 8), and builds with build_serialized_network.

TensorRT 10.x only: TRT 11 removed BuilderFlag.FP16 and weak typing. On Colab
(CUDA 13) install the CUDA 13 wheel pinned to the last 10.x release:
    pip install tensorrt-cu13==10.16.1.11

Outputs:
    models/resnet50_fp16_b1-8.engine   serialized engine (GPU + TRT version specific)
    models/trt_timing_cache.bin        timing cache, reused by later builds
    logs/trt_build_<ts>.log            full VERBOSE builder log (tactic selection, fusions)
    logs/trt_layers_<ts>.json          engine inspector output: the layers TRT actually built
    results/trt_build_<ts>.json        build time, TRT version, layer counts

Usage:
    python -m edgeforge.build_trt
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import tensorrt as trt
import torch

from edgeforge.export import INPUT_NAME, ONNX_PATH
from edgeforge.harness import library_versions, now_iso, nvidia_smi_snapshot, save_results

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
ENGINE_PATH = MODELS_DIR / "resnet50_fp16_b1-8.engine"
TIMING_CACHE_PATH = MODELS_DIR / "trt_timing_cache.bin"
IMAGE_SHAPE = (3, 224, 224)
PROFILE_BATCH = {"min": 1, "opt": 8, "max": 8}
TRT_MAJOR = 10

PRINT_SEVERITIES = {trt.ILogger.INTERNAL_ERROR, trt.ILogger.ERROR, trt.ILogger.WARNING}


def require_trt10() -> None:
    """Fail loudly on anything but TensorRT 10.x — the FP16 builder flag is gone in 11."""
    if int(trt.__version__.split(".")[0]) != TRT_MAJOR:
        raise SystemExit(f"TensorRT {trt.__version__} installed; need 10.x (pip install tensorrt-cu13==10.16.1.11)")


def engine_sha256() -> str:
    """sha256 of the serialized engine on disk, so every result names the exact engine it used."""
    return hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest()


class FileLogger(trt.ILogger):
    """TRT logger that writes every message (down to VERBOSE) to a file; warnings and errors also go to stdout."""

    def __init__(self, path: Path) -> None:
        trt.ILogger.__init__(self)
        self.path = path
        self._file = open(path, "w")

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        self._file.write(f"[{severity.name}] {msg}\n")
        if severity in PRINT_SEVERITIES:
            print(f"[TRT {severity.name}] {msg}")

    def close(self) -> None:
        self._file.close()


def main() -> None:
    require_trt10()
    if not torch.cuda.is_available():
        raise SystemExit("build_trt.py requires CUDA; run it on Colab.")
    if not ONNX_PATH.exists():
        raise SystemExit(f"{ONNX_PATH} not found; run python -m edgeforge.export first.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    ts = now_iso()
    log_path = LOGS_DIR / f"trt_build_{ts}.log"
    logger = FileLogger(log_path)

    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(ONNX_PATH)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("ONNX parse failed:\n" + "\n".join(errors))
    network_layers = network.num_layers

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    # DETAILED keeps per-layer info in the engine so the inspector can show fusions.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        (PROFILE_BATCH["min"], *IMAGE_SHAPE),
        (PROFILE_BATCH["opt"], *IMAGE_SHAPE),
        (PROFILE_BATCH["max"], *IMAGE_SHAPE),
    )
    config.add_optimization_profile(profile)

    # A warm timing cache skips most tactic timing, so build time is only
    # comparable between builds with the same cache state — recorded below.
    cache_in = TIMING_CACHE_PATH.read_bytes() if TIMING_CACHE_PATH.exists() else b""
    timing_cache = config.create_timing_cache(cache_in)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - t0
    if serialized is None:
        logger.close()
        raise SystemExit(f"build_serialized_network failed; see {log_path}")

    ENGINE_PATH.write_bytes(serialized)
    cache_out = config.get_timing_cache().serialize()
    TIMING_CACHE_PATH.write_bytes(cache_out)

    # Inspect what was actually built: fused layers and chosen tactics.
    # Keep the runtime referenced for as long as the engine is alive.
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(ENGINE_PATH.read_bytes())
    layer_info = json.loads(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    )
    layers_path = LOGS_DIR / f"trt_layers_{ts}.json"
    layers_path.write_text(json.dumps(layer_info, indent=2))
    engine_layers = len(layer_info["Layers"])
    logger.close()

    print(
        f"TensorRT {trt.__version__} | built in {build_seconds:.1f} s "
        f"(timing cache in: {len(cache_in)} B) | network layers {network_layers} -> engine layers {engine_layers}"
    )
    path = save_results(
        "trt_build",
        {
            "trt_version": trt.__version__,
            "build_seconds": build_seconds,
            "onnx_path": str(ONNX_PATH.relative_to(ROOT)),
            "engine_path": str(ENGINE_PATH.relative_to(ROOT)),
            "engine_size_mb": ENGINE_PATH.stat().st_size / 1024**2,
            "engine_sha256": engine_sha256(),
            "builder_flags": ["FP16"],
            "builder_optimization_level": config.builder_optimization_level,
            "profile": {"input": INPUT_NAME, **{k: [v, *IMAGE_SHAPE] for k, v in PROFILE_BATCH.items()}},
            "timing_cache_in_bytes": len(cache_in),
            "timing_cache_out_bytes": cache_out.nbytes,  # IHostMemory has no len()
            "network_layers": network_layers,
            "engine_layers": engine_layers,
            "build_log": str(log_path.relative_to(ROOT)),
            "layer_info": str(layers_path.relative_to(ROOT)),
            "gpu": {"name": torch.cuda.get_device_name(), "nvidia_smi": nvidia_smi_snapshot()},
            "versions": library_versions(),
        },
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
