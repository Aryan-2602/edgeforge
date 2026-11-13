"""
accuracy.py — 1000-way top-1 on 500 Imagenette val images: PyTorch FP32, PyTorch FP16, TRT FP16.

Downloads imagenette2-320 (val split only is extracted), samples 50 images per
class with a fixed seed, preprocesses with ResNet50_Weights.IMAGENET1K_V2.transforms()
(resize 232, center-crop 224, ImageNet normalisation), and runs the same batches
of 8 through all three backends.

Scoring is 1000-way: the argmax over all ImageNet classes must equal the image's
ImageNet index, not just win among the 10 Imagenette classes.

PyTorch FP16 uses model.half() + channels_last, the eager variant the latency
headline is measured against. TRT uses the engine from build_trt.py on a
dedicated stream, as in bench_trt.py.

With n=500 the standard error of a top-1 around 85% is ~1.6 points, so small
top-1 differences between backends are noise; the per-image disagreement count
vs FP32 is the sharper measure of what FP16 changed.

Usage:
    python -m edgeforge.accuracy
"""

from __future__ import annotations

import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from edgeforge.bench_trt import io_names
from edgeforge.build_trt import ENGINE_PATH, PROFILE_BATCH, engine_sha256, require_trt10
from edgeforge.harness import library_versions, save_results

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
VAL_DIR = DATA_DIR / "imagenette2-320" / "val"
PER_CLASS = 50
SEED = 0
BATCH = PROFILE_BATCH["max"]  # 8: the largest batch the TRT engine's profile accepts

# Imagenette wnid -> (ImageNet-1k index, torchvision category name). Asserted below.
IMAGENETTE_TO_IMAGENET = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}


def ensure_val_split() -> None:
    """Download imagenette2-320 and extract only its val split, if not already present."""
    if VAL_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tgz = DATA_DIR / "imagenette2-320.tgz"
    if not tgz.exists():
        print(f"downloading {IMAGENETTE_URL} (~340 MB)")
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
    with tarfile.open(tgz) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("imagenette2-320/val/")]
        tar.extractall(DATA_DIR, members=members, filter="data")


def check_mapping(categories: list[str]) -> None:
    """Assert every hardcoded ImageNet index names the expected category."""
    for wnid, (index, name) in IMAGENETTE_TO_IMAGENET.items():
        assert categories[index] == name, f"{wnid}: index {index} is {categories[index]!r}, expected {name!r}"


def check_folders() -> None:
    """Assert the hardcoded wnids are exactly the class folders in the val split."""
    folders = {p.name for p in VAL_DIR.iterdir() if p.is_dir()}
    expected = set(IMAGENETTE_TO_IMAGENET)
    assert folders == expected, f"wnid mismatch: missing {sorted(expected - folders)}, unexpected {sorted(folders - expected)}"


def sample_images() -> list[tuple[str, int]]:
    """Return 50 (relative path, ImageNet index) pairs per class, seeded, in a fixed order."""
    rng = random.Random(SEED)
    picked: list[tuple[str, int]] = []
    for wnid, (index, _name) in sorted(IMAGENETTE_TO_IMAGENET.items()):
        files = sorted(p.name for p in (VAL_DIR / wnid).iterdir() if p.suffix.lower() in {".jpeg", ".jpg"})
        picked += [(f"{wnid}/{f}", index) for f in rng.sample(files, PER_CLASS)]
    return picked


def preprocess(paths: list[str], transforms) -> torch.Tensor:
    """Load and transform every image into one FP32 CPU tensor [N, 3, 224, 224]."""
    # convert("RGB"): Imagenette has a few grayscale / CMYK JPEGs.
    return torch.stack([transforms(Image.open(VAL_DIR / p).convert("RGB")) for p in paths])


def run_torch(model: torch.nn.Module, images: torch.Tensor, dtype: torch.dtype, memory_format: torch.memory_format) -> torch.Tensor:
    """Return FP32 logits [N, 1000] on CPU from a PyTorch model, batch by batch."""
    device = torch.device("cuda")
    model = model.to(device=device, dtype=dtype, memory_format=memory_format)
    out = []
    with torch.inference_mode():
        for batch in images.split(BATCH):
            x = batch.to(device=device, dtype=dtype).contiguous(memory_format=memory_format)
            out.append(model(x).float().cpu())
    return torch.cat(out)


def run_trt(images: torch.Tensor) -> torch.Tensor:
    """Return FP32 logits [N, 1000] on CPU from the TRT FP16 engine, batch by batch."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(ENGINE_PATH.read_bytes())
    if engine is None:
        raise SystemExit(f"failed to deserialize {ENGINE_PATH}; rebuild with python -m edgeforge.build_trt")
    context = engine.create_execution_context()
    input_name, output_name = io_names(engine)

    stream = torch.cuda.Stream()
    out = []
    with torch.cuda.stream(stream):
        for batch in images.split(BATCH):
            x = batch.to("cuda").contiguous()  # engine I/O is FP32
            context.set_input_shape(input_name, tuple(x.shape))
            y = torch.empty(tuple(context.get_tensor_shape(output_name)), device="cuda", dtype=torch.float32)
            context.set_tensor_address(input_name, x.data_ptr())
            context.set_tensor_address(output_name, y.data_ptr())
            if not context.execute_async_v3(stream.cuda_stream):
                raise SystemExit("execute_async_v3 failed")
            stream.synchronize()
            out.append(y.cpu())
    return torch.cat(out)


def compare(logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor, paths: list[str], categories: list[str]) -> dict:
    """Top-1 against labels, plus per-image disagreement with the FP32 reference."""
    pred, ref_pred = logits.argmax(1), ref_logits.argmax(1)
    differs = (pred != ref_pred).nonzero().flatten().tolist()
    return {
        "top1": (pred == labels).float().mean().item(),
        "correct": int((pred == labels).sum()),
        "n_differs_from_fp32": len(differs),
        "differs_from_fp32": [
            {
                "image": paths[i],
                "label": categories[labels[i]],
                "fp32_pred": categories[ref_pred[i]],
                "pred": categories[pred[i]],
            }
            for i in differs
        ],
        "max_abs_logit_diff_vs_fp32": (logits - ref_logits).abs().max().item(),
        "max_abs_ref_logit": ref_logits.abs().max().item(),
        # Relative error makes noise-input and real-image runs comparable: real
        # images give larger, more confident logits, so absolute error grows.
        "rel_max_abs_diff_vs_fp32": (logits - ref_logits).abs().max().item() / ref_logits.abs().max().item(),
    }


def main() -> None:
    require_trt10()
    if not torch.cuda.is_available():
        raise SystemExit("accuracy.py requires CUDA; run it on Colab.")
    torch.backends.cudnn.benchmark = True

    weights = ResNet50_Weights.IMAGENET1K_V2
    categories = weights.meta["categories"]
    check_mapping(categories)

    ensure_val_split()
    check_folders()
    picked = sample_images()
    paths = [p for p, _ in picked]
    labels = torch.tensor([i for _, i in picked])
    images = preprocess(paths, weights.transforms())
    print(f"{len(paths)} images, {PER_CLASS}/class, seed {SEED}, tensor {tuple(images.shape)}")

    fp32 = run_torch(resnet50(weights=weights).eval(), images, torch.float32, torch.contiguous_format)
    fp16 = run_torch(resnet50(weights=weights).eval(), images, torch.float16, torch.channels_last)
    trt_fp16 = run_trt(images)

    results = {
        "pytorch_fp32": compare(fp32, fp32, labels, paths, categories),
        "pytorch_fp16_channels_last": compare(fp16, fp32, labels, paths, categories),
        "trt_fp16": compare(trt_fp16, fp32, labels, paths, categories),
    }
    for name, r in results.items():
        print(
            f"{name:<28} top-1 {r['top1']:.1%} ({r['correct']}/{len(paths)}) | "
            f"differs from FP32 on {r['n_differs_from_fp32']} images | max |logit diff| {r['max_abs_logit_diff_vs_fp32']:.3e} "
            f"(relative {r['rel_max_abs_diff_vs_fp32']:.2e})"
        )

    per_class = {
        categories[idx]: {
            name: float(np.mean([(lg.argmax().item() == idx) for lg, l in zip(lgs, labels) if l == idx]))
            for name, lgs in (("pytorch_fp32", fp32), ("pytorch_fp16_channels_last", fp16), ("trt_fp16", trt_fp16))
        }
        for idx, _ in IMAGENETTE_TO_IMAGENET.values()
    }

    path = save_results(
        "accuracy",
        {
            "dataset": {
                "url": IMAGENETTE_URL,
                "split": "val",
                "per_class": PER_CLASS,
                "n_images": len(paths),
                "seed": SEED,
                "sampling": "per class: sorted filenames, random.Random(seed).sample, classes in wnid order",
                "images": paths,
            },
            "preprocessing": "ResNet50_Weights.IMAGENET1K_V2.transforms()",
            "scoring": "1000-way top-1 (argmax over all ImageNet classes)",
            "mapping": {wnid: {"imagenet_index": i, "name": n} for wnid, (i, n) in IMAGENETTE_TO_IMAGENET.items()},
            "batch_size": BATCH,
            "engine_path": ENGINE_PATH.name,
            "engine_sha256": engine_sha256(),
            "results": results,
            "per_class_top1": per_class,
            "gpu": torch.cuda.get_device_name(),
            "versions": library_versions(),
        },
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
