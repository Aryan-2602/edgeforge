# edgeforge

ResNet-50 inference on a Colab Tesla T4: PyTorch eager vs TensorRT. Every number below
traces to a JSON file in `results/` (or a trace in `logs/`). Latencies are CUDA-event
timings, 50 warm-up + 500 timed calls, input already on the GPU (no host→device copy timed),
`cudnn.benchmark=True`.

**Environment** (recorded in each JSON): Tesla T4 (sm75, 70 W power limit, max SM clock 1590 MHz),
torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.28, torchvision 0.29.1, Python 3.13.

## Status

**ResNet-50 in TensorRT 10.16 FP16 runs 4.0× faster than PyTorch FP16 eager (channels_last) on a
Colab T4 at batch 1 (1.52 vs 6.04 ms p50), and its prediction differs from PyTorch FP32 on 1 of
500 Imagenette images.**

- **Batch 8:** the speedup is 2.5× (4.21 vs 10.48 ms p50).
- **One engine for everything.** The latency and accuracy runs used the same engine file,
  sha256 `c2bc78f3ca855f43…`. That hash appears in `trt_fp16_bs1_20261113T100028883Z.json`,
  `trt_fp16_bs8_20261113T100033593Z.json`, `accuracy_20261113T100119380Z.json` and
  `logs/engine_sha256.txt`. The eager runs come from the same Colab session.
- **Where the engine came from:** the 09:51 build (`trt_build_20261113T095157628Z.json`). That
  JSON predates the sha256 field, so the hash ties results to the engine file, not to a build
  record.
- **It reproduces across sessions:** an earlier session, on an engine with the same per-layer
  kernels, gave 1.53 vs 6.09 ms (4.0×) and 4.22 vs 10.44 ms (2.5×).

## PyTorch eager baseline

ResNet-50 IMAGENET1K_V2, `eval()`, `torch.inference_mode()`, FP16 = `model.half()`.

### Session 1 — all four variants

| Variant   | p50 ms | p95 ms | p99 ms | mean ms | img/s (per-call) | img/s (pipelined) | File |
|-----------|-------:|-------:|-------:|--------:|-----------------:|------------------:|------|
| FP32 bs1  |   5.65 |   7.27 |   9.43 |    5.84 |              171 |               179 | `pytorch_eager_fp32_bs1_20261113T081837426Z.json` |
| FP16 bs1  |   8.33 |  12.02 |  13.33 |    8.32 |              120 |               146 | `pytorch_eager_fp16_bs1_20261113T081911448Z.json` |
| FP32 bs8  |  23.39 |  23.79 |  24.07 |   23.40 |              342 |               338 | `pytorch_eager_fp32_bs8_20261113T081902508Z.json` |
| FP16 bs8  |  10.70 |  10.90 |  12.43 |   10.77 |              743 |               743 | `pytorch_eager_fp16_bs8_20261113T081923083Z.json` |

- **Batch 8:** FP16 is 2.2× faster than FP32 (p50 23.39 → 10.70 ms), which is what you'd expect from tensor cores.
- **Batch 1:** FP16 is *slower* than FP32 (8.33 vs 5.65 ms p50).

### Why FP16 loses at batch 1

**1. Not thermal or run order.** In one session we ran FP16 bs1, then FP32 bs1, then FP16 bs1 again:

| Order | Variant  | p50 ms | p99 ms | File |
|------:|----------|-------:|-------:|------|
| 0 | FP16 bs1 | 6.86 | 11.53 | `pytorch_eager_fp16_bs1_20261113T085039945Z.json` |
| 1 | FP32 bs1 | 5.86 | 10.83 | `pytorch_eager_fp32_bs1_20261113T085047280Z.json` |
| 2 | FP16 bs1 | 6.95 | 11.81 | `pytorch_eager_fp16_bs1_20261113T085055397Z.json` |

FP16 loses whether it runs before or after FP32. Note that FP16 bs1 measured 8.33 ms in session 1
and about 6.9 ms here. That ~20% swing between Colab sessions is why we only compare runs from the
same session.

**2. Kernel count.** We profiled one forward pass per variant with `torch.profiler`
(`results/kernels_*`, Chrome traces in `logs/`):

| Variant      | Kernel launches | Layout-transpose launches | Summed GPU kernel time |
|--------------|----------------:|--------------------------:|-----------------------:|
| FP32 bs1     | 214 | 8 (5 `nchwToNhwc` + 3 `nhwcToNchw`) | 3.90 ms |
| FP16 bs1     | 377 | 90 (53 `nchwToNhwc` + 37 `nhwcToNchw`, 0.57 ms) | 2.15 ms |
| FP16 channels_last bs1 | 296 | 0 | 2.72 ms |

- **GPU mostly idle.** FP16 needs about half the GPU time of FP32, yet takes longer end to end.
  Its kernels add up to ~2.1 ms against ~6.9 ms of latency, so the GPU sits idle most of the
  time. At batch 1, eager mode is limited by kernel launch overhead, and FP16 launches 1.76× more
  kernels than FP32.
- **Layout conversions.** cuDNN's FP16 tensor-core convolutions run in NHWC layout, so with
  NCHW tensors it wraps convolutions in `nchwToNhwc`/`nhwcToNchw` conversions.
- **Batch norm stays on PyTorch's own kernels.** In FP32 it runs as one cuDNN kernel per layer
  (`bn_fw_inf_*`). In FP16 it goes through `at::native` batch-norm kernels plus an extra copy
  (53 launches each). Our reading is that this happens because `.half()` also converts the
  batch-norm weights. We haven't verified that.

Each row is a single profiled pass. Treat the summed GPU times as rough; the launch counts are
the reliable part.

**3. channels_last.** Same session, FP32 bs1 run first:

| Variant | p50 ms | p99 ms | File |
|---------|-------:|-------:|------|
| FP32 bs1 | 5.60 | 9.34 | `pytorch_eager_fp32_bs1_20261113T085116730Z.json` |
| FP16 channels_last bs1 | 5.76 | 10.03 | `pytorch_eager_fp16cl_bs1_20261113T085124655Z.json` |

channels_last removes every layout conversion and brings FP16 bs1 from ~6.9 ms down to 5.76 ms.
That ties FP32 but **does not beat it**: 296 launches is still more than FP32's 214.

### Takeaways for the TensorRT comparison

- PyTorch eager at batch 1 on a T4 is limited by kernel launches, not by GPU compute. That's
  where TensorRT's layer fusion should matter most.
- Plain FP16 eager in NCHW layout is a weak baseline at batch 1 (it's slower than FP32).
  Comparing TensorRT against it would inflate the speedup. The FP16 channels_last run is the
  fairer FP16 eager number to compare against.

## ONNX export and parity

`edgeforge/export.py` exports with torch 2.14's default **dynamo** exporter
(`torch.onnx.export(dynamo=True)`). The batch dimension is declared dynamic with
`dynamic_shapes=({0: Dim("batch", min=1, max=64)},)`; `dynamic_axes` is only for the deprecated
TorchScript exporter. Result: opset 20, input dims `[batch, 3, 224, 224]`, 97.6 MB file, and
`onnx.checker` passed with `full_check=True`.

onnxruntime 1.31.1 (CPU provider) is compared against PyTorch FP32 on the GPU, using the same
seeded random batch (`results/onnx_parity_20261113T092423405Z.json`):

| Batch | Max abs diff | Mean abs diff | Max \|logit\| | Top-1 agreement |
|------:|-------------:|--------------:|-------------:|----------------:|
| 8 | 3.3e-6 | 3.7e-7 | 5.37 | 8/8 |
| 1 | 1.8e-6 | 2.9e-7 | 5.31 | 1/1 |

The ONNX graph matches PyTorch to within float32 rounding, and the same file runs at batch 1
and batch 8.

**Caveat:** the 8/8 top-1 agreement tells us almost nothing. Unnormalised noise input makes
every sample predict the same class (21), so the top-1 check is trivially satisfied. The max abs
diff is the real evidence. Top-1 on real images is in "Accuracy" below.

## TensorRT FP16

`edgeforge/build_trt.py` uses TensorRT 10.16.1.11 (`tensorrt-cu13`). We pinned it to 10.x because
TensorRT 11 removed `BuilderFlag.FP16`. It parses the FP32 ONNX file, sets `BuilderFlag.FP16`, and
adds one optimization profile for the input: batch 1 minimum, 8 optimal, 8 maximum.
`edgeforge/bench_trt.py` runs the engine through the same harness using `execute_async_v3` on a
dedicated `torch.cuda.Stream`. The engine's input and output stay FP32, so the FP32→FP16 cast of
the input happens inside the timed call; PyTorch FP16 eager doesn't pay that cost.

### Build

From `results/trt_build_20261113T094141828Z.json`, VERBOSE log
`logs/trt_build_20261113T094136636Z.log`, and inspector output
`logs/trt_layers_20261113T094136636Z.json`:
- **Build time: 4.35 s, with a warm timing cache** (729,596 B loaded, 1,385 entries). This
  isn't a fresh-build time, which is much longer because every tactic has to be timed. The first
  build had an empty cache, but its JSON was never written (see "What didn't work"), so no
  fresh-build time is backed by a result file.
- The engine is 49.1 MB. Its 58 layers chose exactly the same tactics (kernel implementations)
  as the first build, `logs/trt_layers_20261113T093159473Z.json`.

**What TensorRT did to the graph: 126 network layers became 58 engine layers.**
- **Fusion:** all 49 ReLUs and all 16 residual Adds were fused into convolutions. The log has
  49 `ConvReluFusion` and 16 `ConvEltwiseSumFusion` entries. Of the 53 convolutions, 33 include
  a ReLU, 16 include the residual add and the ReLU, and 4 include neither (those 4 feed an add
  that's fused into another conv).
- **Precision:** every convolution produces FP16 output and uses an FP16 tensor-core kernel
  (`sm75_xmma_*_f16`, `turing_h1688*`, `volta_h884*`). The final classifier layer, which TensorRT
  compiles separately (layer type `fusion`), outputs FP32.
- **Batch norm:** there were no BatchNormalization nodes left to fuse. The ONNX contains only
  Conv, Relu, Add, MaxPool, ReduceMean, Reshape and Gemm, because the torch dynamo exporter
  (`optimize=True`) had already folded BN into the conv weights. BN folding comes from the
  exporter, not from TensorRT.

### Latency

TensorRT and eager were measured in the same Colab session, with TensorRT on a dedicated stream,
using engine sha256 `c2bc78f3…`:

| Variant | p50 ms | p95 ms | p99 ms | img/s (per-call) | img/s (pipelined) | File |
|---|---:|---:|---:|---:|---:|---|
| TRT FP16 bs1 | 1.52 | 1.57 | 2.25 | 645 | 680 | `trt_fp16_bs1_20261113T100028883Z.json` |
| Eager FP16 channels_last bs1 | 6.04 | 7.29 | 10.30 | 160 | 135* | `pytorch_eager_fp16cl_bs1_20261113T100046663Z.json` |
| Eager FP16 bs1 | 6.94 | 9.40 | 12.07 | 139 | 143 | `pytorch_eager_fp16_bs1_20261113T100054940Z.json` |
| TRT FP16 bs8 | 4.21 | 4.29 | 4.37 | 1901 | 1899 | `trt_fp16_bs8_20261113T100033593Z.json` |
| Eager FP16 bs8 | 10.48 | 12.26 | 14.71 | 746 | 763 | `pytorch_eager_fp16_bs8_20261113T100106830Z.json` |

\* See the pipelined-throughput anomaly under "What didn't work".

The previous session (`*_20261113T0941*`/`T0942*`, engine with the same per-layer kernels but no
hash) gave TRT 1.53 / 4.22 ms and eager 6.09 / 10.44 ms p50, within 1% of these.

- **Batch 1:** TensorRT is **4.0× faster** at p50 than eager FP16 channels_last (6.04 / 1.52),
  4.6× faster than plain FP16 eager, and 4.6× faster at p99. The gain is bigger than at batch 8,
  as expected if eager batch 1 is limited by kernel launches. TensorRT runs the whole fused
  engine from one `execute_async_v3` call; eager launches 296 kernels from Python, per the
  kernel-count profile in an earlier session.
- **Batch 8:** **2.5× faster** (10.48 / 4.21) and 2.5× the throughput (1901 vs 746 img/s). Here the
  GPU is the bottleneck, so the gain comes from fusion and better kernels rather than fewer
  launches.
- **Sanity check against PyTorch FP32** (same random input): max abs logit difference 1.0e-2 at
  batch 1 and 9.0e-3 at batch 8, on max |logit| of 5.39 and 5.33. That's a relative error of
  1.9e-3 and 1.7e-3, about what FP16 should give. Top-1 isn't meaningful here, since noise input
  predicts a single class.
- The engine is tuned for batch 8 (opt = 8), so batch 1 runs on kernels chosen for batch 8.

**The batch-1 second mode is gone.** In the first TensorRT run (default stream,
`trt_fp16_bs1_20261113T093258124Z.json`), p95 and p99 were 2.91 and 2.94 ms against a p50 of
1.52 ms, about 5% of calls at roughly 2×. On the dedicated stream, 491 of 500 calls are within 10%
of p50, and the only calls above 1.5× p50 are the first two timed ones (the SM clock started at
1035 MHz). TensorRT's default-stream warning is also gone from `logs/bench_trt_stdout.txt`.
The re-run reproduces this: only call 0 is above 1.5× p50; p99 is 2.25 ms. Caveat: we never ran
default stream and dedicated stream in the same session, and the default-stream run didn't save
per-call samples. So this is consistent with the stream fix, not a controlled A/B test.

**Eager tails are periodic.** The slow eager calls come in runs of about three consecutive calls,
recurring every 140–166 calls (fp16cl bs1: calls 57–59, 223–225, 383–385; fp16 bs1: calls 65–67,
205–207, 346–348, 491–492). The re-run shows the same pattern (fp16cl bs1: calls 113–114,
277–279, 443–445). That's roughly once a second at batch 1, and it's what drives eager's
p99. The cause is unknown; Python garbage collection and Colab background activity are the
suspects. A 500-call TensorRT batch-1 run lasts only about 0.8 s, so it's too short to show a
once-a-second hiccup. Its clean tail isn't proof that TensorRT is immune.

## Accuracy (Imagenette, 500 images)

`edgeforge/accuracy.py` takes 50 images per class from the imagenette2-320 val split (seed 0;
the exact paths are in the JSON) and preprocesses them with `ResNet50_Weights.IMAGENET1K_V2.transforms()`.
Scoring is **1000-way**: the argmax over all ImageNet classes must be the right class. Each
Imagenette wnid is mapped to its ImageNet index, and every name is asserted against
`weights.meta["categories"]`. All three backends get the same batches of 8. The TRT engine has
the same sha256 as the latency runs (`c2bc78f3…`).

From `results/accuracy_20261113T100119380Z.json`. The same 500 images and the same results as the
first accuracy run, `accuracy_20261113T095543919Z.json`:

| Backend | Top-1 | Correct | Predictions differing from FP32 | Max abs logit diff vs FP32 | Relative (÷ max \|logit\| 9.56) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 86.6% | 433/500 | — | — | — |
| PyTorch FP16 eager (channels_last) | 86.8% | 434/500 | 1 | 7.4e-2 | 7.7e-3 |
| TensorRT FP16 | 86.8% | 434/500 | 1 | 7.1e-2 | 7.4e-3 |

- **This test can't tell the three apart on top-1.** 86.8% vs 86.6% is one image. With n = 500,
  the 95% confidence interval on 86.6% is about ±3 points. Neither FP16 backend "improved"
  accuracy. The claim we can make is the flip count: TRT FP16 disagrees with FP32 on 1 of 500
  images.
- **Both FP16 backends flipped the same image, the same way.** `n02102040/n02102040_791.JPEG` is
  an English springer that FP32 calls "Welsh springer spaniel" and both FP16 runs call "English
  springer". One image flipped twice the same way, not two independent flips, means that image
  sits on a decision boundary. Any small perturbation would tip it, and FP16 rounding happened
  to tip it toward the correct class.
- **Why top-1 is 86.6% and not ~95%:** this is 1000-way scoring. Per-class top-1 is
  identical across all three backends. Church (54%) and cassette player (60%) pull the average
  down; our guess is confusion with neighbouring ImageNet classes such as other buildings and
  tape players, but we haven't checked what they're predicted as. Tench, chain saw: 100%;
  everything else: 84–96%.
- **Relative error doesn't fully explain the bigger real-image differences.** Real images
  produce larger logits (max |logit| 9.56 vs ~5.4 on noise), but even relative to that, TRT's
  error is about 4× higher on real images (7.4e-3 vs 1.7–1.9e-3 on noise). Part of this is
  sample size: the real-image figure is the worst case over 500 images, the noise figure over 1
  or 8. We haven't separated that from any genuine input-dependence. Either way, only one
  prediction moved.

## What didn't work

- channels_last did not make FP16 bs1 faster than FP32; it only matched it (5.76 vs 5.60 ms p50).
- **Pipelined throughput anomaly.** For the FP16 channels_last run, pipelined throughput (133 img/s)
  came out *below* per-call throughput (169 img/s). nvidia-smi showed the SM clock falling from
  1590 to 1095 MHz across that run. Our working explanation is power or clock throttling during
  the pipelined phase, but we haven't confirmed it. Don't use that pipelined number.
  **It has now happened four times out of four channels_last runs**, and never on the other
  variants. Per-call vs pipelined: 158 vs 124 img/s (`pytorch_eager_fp16cl_bs1_20261113T093316003Z.json`),
  159 vs 128 img/s (`pytorch_eager_fp16cl_bs1_20261113T094206984Z.json`) and 160 vs 135 img/s
  (`pytorch_eager_fp16cl_bs1_20261113T100046663Z.json`). That pattern makes
  throttling look unlikely. The cause is still unknown.
- **The first TensorRT benchmark ran on the default CUDA stream.** TensorRT warned that it adds
  `cudaStreamSynchronize` calls in that case, and the batch-1 tail had a second mode at about 2×
  p50 (`trt_fp16_bs1_20261113T093258124Z.json`, `trt_fp16_bs8_20261113T093302854Z.json`). Those
  runs are superseded by the dedicated-stream runs above.
- **The first TensorRT build crashed before writing its JSON.** The cause was `len()` on the
  `IHostMemory` that `ITimingCache.serialize()` returns; `IHostMemory` has no `len()`. It's now
  `.nbytes`. The engine had already been saved, which is why the benchmark still ran. Our first
  attempted fix (guarding the logger and keeping the TensorRT runtime alive) was a guess and
  missed the real bug.

- **First accuracy run crashed on a wrong wnid.** The hardcoded wnid for parachute was
  `n03888605`, which is ImageNet's "parallel bars". The correct one is `n03888257`. The
  index-to-name assertion (701 → "parachute") passed because the index was right; only the
  folder name was wrong. `accuracy.py` now also asserts that the hardcoded wnids are exactly the
  dataset's class folders.

## Predictions vs measured

| Prediction (made before the run) | Measured p50 |
|---|---|
| bs1 ≈ 300 ms | FP32 5.65 ms, FP16 8.33 ms |
| bs8 ≈ 700 ms | FP32 23.39 ms, FP16 10.70 ms |

No predictions were recorded for the ONNX check, the TensorRT runs or the accuracy run.

## Measurement notes

- CUDA-event latency and `perf_counter` + sync latency agree to within ~0.05 ms at p50
  (`wall_latency_ms` in each JSON).
- **p99 vs p50.** At batch 1, p99 is 1.6–1.85× p50: when each call is this short, a single slow
  kernel launch shows up in the tail. At batch 8, p99 is 3–16% above p50.
- **GPU clock.** The T4 is power-capped at 70 W and its SM clock moved between 930 and 1590 MHz
  across runs. nvidia-smi before/after readings are in each JSON.

## Reproduce (Colab, T4)

```
python -m edgeforge.baseline                                    # fp32/fp16 x bs1/bs8
python -m edgeforge.baseline --variants fp16:1 fp32:1 fp16:1    # run-order check
python -m edgeforge.profile_kernels --variants fp32:1 fp16:1 fp16cl:1
python -m edgeforge.baseline --variants fp32:1 fp16cl:1
pip install onnx onnxscript onnxruntime && python -m edgeforge.export
pip install tensorrt-cu13==10.16.1.11 --extra-index-url https://pypi.nvidia.com
python -m edgeforge.build_trt
python -m edgeforge.bench_trt
python -m edgeforge.baseline --variants fp16cl:1 fp16:1 fp16:8   # same-session eager comparison
python -m edgeforge.accuracy                                        # downloads imagenette2-320 (~340 MB)
```
