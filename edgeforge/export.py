"""
export.py — ResNet-50 -> ONNX export plus onnxruntime parity check.

Exports torchvision ResNet-50 (IMAGENET1K_V2, eval, FP32 weights) with a dynamic
batch dimension, validates it with onnx.checker, then runs the same random
batch of 8 through onnxruntime and PyTorch FP32 and reports max abs logit diff
and top-1 agreement. A batch-1 run through the same ONNX file checks that the
batch axis really is dynamic.

Exporter: torch 2.14's torch.onnx.export defaults to dynamo=True (torch.export
based; default since 2.9). The dynamo path takes the dynamic batch axis via
dynamic_shapes with torch.export.Dim; dynamic_axes is the deprecated
TorchScript-exporter way, so it is not used here.

onnxruntime runs on CPUExecutionProvider: this step checks correctness, not
speed (see CLAUDE.md), and the provider actually used is recorded.

Writes models/resnet50_fp32.onnx and results/onnx_parity_<ts>.json.

Usage:
    python -m edgeforge.export
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.models import ResNet50_Weights, resnet50

from edgeforge.harness import library_versions, save_results

MODELS_DIR = Path(__file__).parent.parent / "models"
ONNX_PATH = MODELS_DIR / "resnet50_fp32.onnx"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
PARITY_BATCH = 8
SEED = 0
IMAGE_SHAPE = (3, 224, 224)
ORT_PROVIDER = "CPUExecutionProvider"


def export_onnx(model: torch.nn.Module) -> None:
    """Export with the dynamo exporter, batch dim dynamic, weights embedded in one file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    example = torch.randn(PARITY_BATCH, *IMAGE_SHAPE)
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(
        model,
        (example,),
        str(ONNX_PATH),
        dynamo=True,
        dynamic_shapes=({0: batch},),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        # ~100 MB of weights is far under the 2 GB protobuf limit; one file is
        # easier to move between Colab and the TRT builder.
        external_data=False,
    )


def input_dims(model_proto: onnx.ModelProto) -> list[str]:
    """Return the ONNX input's dims as strings: symbolic names (dim_param) or sizes."""
    dims = model_proto.graph.input[0].type.tensor_type.shape.dim
    return [d.dim_param or str(d.dim_value) for d in dims]


def parity(model: torch.nn.Module, session: ort.InferenceSession, batch_size: int, device: torch.device) -> dict:
    """Run one seeded random batch through PyTorch FP32 and ORT; compare logits and top-1."""
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(batch_size, *IMAGE_SHAPE, generator=generator)

    with torch.inference_mode():
        ref = model(x.to(device)).float().cpu().numpy()
    out = session.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]

    abs_diff = np.abs(ref - out)
    ref_top1 = ref.argmax(axis=1)
    ort_top1 = out.argmax(axis=1)
    return {
        "batch_size": batch_size,
        "ort_output_shape": list(out.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_ref_logit": float(np.abs(ref).max()),
        "top1_agreement": float((ref_top1 == ort_top1).mean()),
        "top1_pytorch": ref_top1.tolist(),
        "top1_ort": ort_top1.tolist(),
    }


def main() -> None:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

    export_onnx(model)
    model_proto = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_proto, full_check=True)
    dims = input_dims(model_proto)
    print(f"onnx.checker passed | input dims {dims} | opset {model_proto.opset_import[0].version}")
    if dims[0].isdigit():
        raise SystemExit(f"batch dim exported as fixed size {dims[0]}, expected symbolic")

    session = ort.InferenceSession(str(ONNX_PATH), providers=[ORT_PROVIDER])
    active_providers = session.get_providers()
    if active_providers[0] != ORT_PROVIDER:
        raise SystemExit(f"expected {ORT_PROVIDER}, onnxruntime is using {active_providers}")

    # Reference is PyTorch FP32 on the GPU when there is one (the baseline's path).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    runs = [parity(model, session, b, device) for b in (PARITY_BATCH, 1)]
    for r in runs:
        print(
            f"bs{r['batch_size']}: max abs diff {r['max_abs_diff']:.2e} "
            f"(max |logit| {r['max_abs_ref_logit']:.2f}) | top-1 agreement {r['top1_agreement']:.0%}"
        )

    path = save_results(
        "onnx_parity",
        {
            "model": "resnet50",
            "weights": "IMAGENET1K_V2",
            "onnx_path": str(ONNX_PATH.relative_to(MODELS_DIR.parent)),
            "onnx_size_mb": ONNX_PATH.stat().st_size / 1024**2,
            "exporter": "torch.onnx.export(dynamo=True), dynamic_shapes=({0: Dim('batch', min=1, max=64)},)",
            "opset": model_proto.opset_import[0].version,
            "onnx_input_dims": dims,
            "checker": "onnx.checker.check_model(full_check=True) passed",
            "pytorch_reference_device": str(device),
            "ort_providers": active_providers,
            "input": f"torch.randn seed {SEED}, unnormalised noise (not real images)",
            "parity": runs,
            "versions": {**library_versions(), "onnx": onnx.__version__},
        },
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
