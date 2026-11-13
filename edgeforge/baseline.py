"""
baseline.py — PyTorch eager ResNet-50 baseline for edgeforge.

torchvision ResNet-50 (IMAGENET1K_V2), eval mode, torch.inference_mode(),
FP32 and FP16 (model.half(), not autocast) at batch 1 and 8. The input is a
random tensor created on the GPU up front, so no host->device copy is timed —
the TRT/ORT runs must use the same copy semantics for a fair comparison.

Usage (on a CUDA machine, from the repo root):
    python -m edgeforge.baseline
"""

from __future__ import annotations

import argparse

import torch
from torchvision.models import ResNet50_Weights, resnet50

from edgeforge.harness import run_benchmark

BATCH_SIZES = (1, 8)
PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16}
IMAGE_SHAPE = (3, 224, 224)
DEFAULT_N_WARMUP = 50
DEFAULT_N_ITERS = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    for precision, dtype in PRECISIONS.items():
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device=device, dtype=dtype)

        for batch_size in BATCH_SIZES:
            x = torch.randn(batch_size, *IMAGE_SHAPE, device=device, dtype=dtype)
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
                        "input": "torch.randn on GPU; no H2D copy timed",
                        "cudnn_benchmark": args.cudnn_benchmark,
                    },
                )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
