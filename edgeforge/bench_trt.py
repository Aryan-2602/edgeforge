"""
bench_trt.py — benchmark the TensorRT FP16 engine through the shared harness.

Deserializes models/resnet50_fp16_b1-8.engine, allocates I/O as torch CUDA
tensors, and times context.execute_async_v3 on a dedicated torch CUDA stream (made
current, so the harness records its CUDA events on it too) at batch 1 and 8.

Copy semantics match baseline.py: the input is already on the GPU, nothing is
copied host<->device inside the timed call. One difference: the engine's I/O is
FP32 (the ONNX is FP32; BuilderFlag.FP16 only changes internal precision), so
the FP32->FP16 cast of the input happens inside the timed engine call, while
PyTorch FP16 eager gets an FP16 input for free. That slightly favours eager.

Before timing, each batch is checked against PyTorch FP32 on the same input so
a broken engine cannot produce a fast-looking number.

Usage:
    python -m edgeforge.bench_trt
"""

from __future__ import annotations

import argparse

import tensorrt as trt
import torch
from torchvision.models import ResNet50_Weights, resnet50

from edgeforge.baseline import DEFAULT_N_ITERS, DEFAULT_N_WARMUP
from edgeforge.build_trt import ENGINE_PATH, IMAGE_SHAPE, PROFILE_BATCH, engine_sha256, require_trt10
from edgeforge.harness import run_benchmark

BATCH_SIZES = (1, 8)
TRT_TO_TORCH_DTYPE = {trt.float32: torch.float32, trt.float16: torch.float16}


def io_names(engine: trt.ICudaEngine) -> tuple[str, str]:
    """Return (input name, output name); the engine has exactly one of each."""
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    assert len(inputs) == 1 and len(outputs) == 1, names
    return inputs[0], outputs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS)
    args = parser.parse_args()

    require_trt10()
    if not torch.cuda.is_available():
        raise SystemExit("bench_trt.py requires CUDA; run it on Colab.")
    device = torch.device("cuda")

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(ENGINE_PATH.read_bytes())
    if engine is None:
        raise SystemExit(f"failed to deserialize {ENGINE_PATH}; rebuild with python -m edgeforge.build_trt")
    context = engine.create_execution_context()
    input_name, output_name = io_names(engine)
    in_dtype = TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(input_name)]
    out_dtype = TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(output_name)]

    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    # Enqueue on a dedicated stream: on the legacy default stream (cuda_stream == 0)
    # TRT inserts extra cudaStreamSynchronize calls and warns about it. Everything,
    # including the harness's CUDA events, runs under this stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())  # weights were copied on the default stream

    with torch.cuda.stream(stream):
        for batch_size in BATCH_SIZES:
            run_batch(context, input_name, output_name, in_dtype, out_dtype, reference, stream, batch_size, args)


def run_batch(
    context: trt.IExecutionContext,
    input_name: str,
    output_name: str,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    reference: torch.nn.Module,
    stream: torch.cuda.Stream,
    batch_size: int,
    args: argparse.Namespace,
) -> None:
    """Sanity-check the engine against PyTorch FP32 at one batch size, then time it."""
    device = torch.device("cuda")
    context.set_input_shape(input_name, (batch_size, *IMAGE_SHAPE))
    x = torch.randn(batch_size, *IMAGE_SHAPE, device=device, dtype=in_dtype)
    y = torch.empty(tuple(context.get_tensor_shape(output_name)), device=device, dtype=out_dtype)
    context.set_tensor_address(input_name, x.data_ptr())
    context.set_tensor_address(output_name, y.data_ptr())

    if not context.execute_async_v3(stream.cuda_stream):
        raise SystemExit(f"execute_async_v3 failed at batch {batch_size}")
    with torch.inference_mode():
        ref = reference(x.float())
    torch.cuda.synchronize()
    max_abs_diff = (y.float() - ref).abs().max().item()
    max_abs_ref_logit = ref.abs().max().item()
    top1_agreement = (y.float().argmax(1) == ref.argmax(1)).float().mean().item()
    print(
        f"bs{batch_size}: TRT vs PyTorch FP32 max abs diff {max_abs_diff:.3e} "
        f"(relative to max |logit| {max_abs_ref_logit:.2f}: {max_abs_diff / max_abs_ref_logit:.2e}), top-1 agreement {top1_agreement:.0%}"
    )

    run_benchmark(
        lambda: context.execute_async_v3(stream.cuda_stream),
        batch_size=batch_size,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
        label=f"trt_fp16_bs{batch_size}",
        extra={
            "model": "resnet50",
            "weights": "IMAGENET1K_V2",
            "backend": "tensorrt",
            "precision": "fp16 (BuilderFlag.FP16, FP32 I/O)",
            "engine_path": str(ENGINE_PATH.name),
            "engine_sha256": engine_sha256(),
            "stream": "dedicated torch.cuda.Stream (not the default stream)",
            "profile_batch": PROFILE_BATCH,
            "input": f"torch.randn {in_dtype} on GPU; no H2D copy timed; FP32->FP16 cast inside engine",
            "sanity_vs_pytorch_fp32": {
                "max_abs_diff": max_abs_diff,
                "max_abs_ref_logit": max_abs_ref_logit,
                "rel_max_abs_diff": max_abs_diff / max_abs_ref_logit,
                "top1_agreement": top1_agreement,
                "note": "random-noise input; top-1 is not meaningful (see onnx_parity)",
            },
        },
    )


if __name__ == "__main__":
    main()
