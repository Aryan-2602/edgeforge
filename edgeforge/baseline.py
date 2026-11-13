"""
baseline.py — PyTorch eager ResNet-50 baseline for edgeforge.

torchvision ResNet-50 (IMAGENET1K_V2), eval mode, torch.inference_mode(),
FP32 and FP16 (model.half(), not autocast) at batch 1 and 8. The input is a
random tensor created on the GPU up front, so no host->device copy is timed —
the TRT/ORT runs must use the same copy semantics for a fair comparison.

A variant is <precision>:<batch>, precision one of fp32, fp16, fp16cl
(fp16 with model and input in torch.channels_last). Variants run in the order
given, so the same one can be repeated to check for run-order/thermal effects.

Usage (on a CUDA machine, from the repo root):
    python -m edgeforge.baseline                                   # fp32/fp16 x bs1/bs8
    python -m edgeforge.baseline --variants fp16:1 fp32:1 fp16:1   # run-order check
    python -m edgeforge.baseline --variants fp16cl:1               # channels_last
"""

from __future__ import annotations

import argparse

import torch
from torchvision.models import ResNet50_Weights, resnet50

from edgeforge.harness import run_benchmark

PRECISIONS = {
    "fp32": (torch.float32, torch.contiguous_format),
    "fp16": (torch.float16, torch.contiguous_format),
    "fp16cl": (torch.float16, torch.channels_last),
}
DEFAULT_VARIANTS = ["fp32:1", "fp32:8", "fp16:1", "fp16:8"]
IMAGE_SHAPE = (3, 224, 224)
DEFAULT_N_WARMUP = 50
DEFAULT_N_ITERS = 500


def parse_variant(spec: str) -> tuple[str, int]:
    """Parse '<precision>:<batch>' e.g. 'fp16cl:1'."""
    precision, batch = spec.split(":")
    if precision not in PRECISIONS:
        raise argparse.ArgumentTypeError(f"unknown precision {precision!r}; choose from {list(PRECISIONS)}")
    return precision, int(batch)


def build(precision: str, batch_size: int, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor]:
    """Return (model, input) on device in the variant's dtype and memory format."""
    dtype, memory_format = PRECISIONS[precision]
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    model = model.to(device=device, dtype=dtype, memory_format=memory_format)
    x = torch.randn(batch_size, *IMAGE_SHAPE, device=device, dtype=dtype)
    x = x.contiguous(memory_format=memory_format)
    return model, x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", type=parse_variant, default=[parse_variant(v) for v in DEFAULT_VARIANTS])
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS)
    # cudnn.benchmark is PyTorch's analogue of TRT tactic selection: it autotunes
    # conv algorithms per input shape during warm-up. On by default so the eager
    # baseline is as strong as it can be; recorded in every JSON either way.
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("baseline.py requires CUDA; run it on Colab.")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device("cuda")
    order = [f"{p}:{b}" for p, b in args.variants]

    for run_index, (precision, batch_size) in enumerate(args.variants):
        model, x = build(precision, batch_size, device)
        with torch.inference_mode():
            run_benchmark(
                lambda: model(x),
                batch_size=batch_size,
                n_warmup=args.n_warmup,
                n_iters=args.n_iters,
                label=f"pytorch_eager_{precision}_bs{batch_size}",
                extra={
                    "model": "resnet50",
                    "weights": "IMAGENET1K_V2",
                    "backend": "pytorch_eager",
                    "precision": precision,
                    "memory_format": str(PRECISIONS[precision][1]),
                    "input": "torch.randn on GPU; no H2D copy timed",
                    "cudnn_benchmark": args.cudnn_benchmark,
                    "run_order": order,
                    "run_index": run_index,
                },
            )
        del model, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
