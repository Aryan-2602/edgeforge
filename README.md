# edgeforge

ResNet-50 inference on a Colab Tesla T4: PyTorch eager vs a hand-built TensorRT engine. Every
number below comes from a file in [`results/`](results/) or [`logs/`](logs/). Ratios are p50 ÷ p50
(or throughput ÷ throughput) at the same batch size, rounded to 1 decimal; each one shows its
formula.

## Result

**On a Colab T4, ResNet-50 as a TensorRT FP16 engine runs 4.0× faster than PyTorch FP16 eager at
batch 1 (1.52 vs 6.04 ms p50) and 2.5× faster at batch 8 (1901 vs 746 img/s), disagreeing with
FP32 top-1 on 1 of 500 Imagenette images.**

The FP16 eager comparison is the channels_last variant. That's the fastest FP16 eager
configuration we measured at batch 1 (see "Why FP16 eager is slow at batch 1"). Against plain FP16
eager the batch-1 ratio is 4.6×.

| Ratio | Formula | Files |
|---|---|---|
| 4.0× (bs1 latency) | 6.038 / 1.519 = 3.97 | [`pytorch_eager_fp16cl_bs1_20261113T100046663Z`](results/pytorch_eager_fp16cl_bs1_20261113T100046663Z.json) / [`trt_fp16_bs1_20261113T100028883Z`](results/trt_fp16_bs1_20261113T100028883Z.json) |
| 4.6× (bs1 latency, plain NCHW FP16) | 6.936 / 1.519 = 4.57 | [`pytorch_eager_fp16_bs1_20261113T100054940Z`](results/pytorch_eager_fp16_bs1_20261113T100054940Z.json) / same TRT file |
| 2.5× (bs8 throughput, per-call) | 1901.3 / 746.2 = 2.55 | [`trt_fp16_bs8_20261113T100033593Z`](results/trt_fp16_bs8_20261113T100033593Z.json) / [`pytorch_eager_fp16_bs8_20261113T100106830Z`](results/pytorch_eager_fp16_bs8_20261113T100106830Z.json) |
| 2.5× (bs8 latency) | 10.478 / 4.212 = 2.49 | same files |
| 1 of 500 flips | `n_differs_from_fp32` = 1 | [`accuracy_20261113T100119380Z`](results/accuracy_20261113T100119380Z.json) |

**One engine for all of it.** The TRT latency and accuracy runs record the same engine sha256,
`c2bc78f3ca855f43ea14ed4111c1b4576adbb20aea488e17dbc23ff32d5d64a0`. It appears in both `trt_fp16_*`
files above, in the accuracy JSON and in [`logs/engine_sha256.txt`](logs/engine_sha256.txt). The
eager runs come from the same Colab session. The engine is the one written by the 09:51 build
([`trt_build_20261113T095157628Z`](results/trt_build_20261113T095157628Z.json)); that build JSON
predates the sha256 field, so the hash ties results to the engine file, not to the build record.

## Methodology

Checked against `edgeforge/harness.py`, `baseline.py`, `bench_trt.py` and the JSON metadata.

- **Latency:** every timed call is bracketed by CUDA events and followed by
  `torch.cuda.synchronize()`. `latency_ms` (p50/p95/p99/mean) comes from the CUDA events. A
  `perf_counter` reading around the same call is the cross-check (`wall_latency_ms`). Across all
  24 latency JSONs, wall p50 exceeds event p50 by 0.025–0.059 ms.
- **Warm-up and iterations:** 50 warm-up calls, then 500 timed calls, in every latency JSON
  (`n_warmup`, `n_iters`). Every per-call sample is saved (`latency_ms_samples`) in runs from
  09:41 onward.
- **Per-call throughput** = batch ÷ mean latency, from the timed loop above.
  **Pipelined throughput** = batch × 500 ÷ total time for 500 back-to-back calls, with no sync
  between calls and one sync at the end.
- **Input:** created on the GPU before timing; no host-to-device copy is timed for any variant.
- **Eager:** `eval()`, `torch.inference_mode()`, `cudnn.benchmark = True` (`extra.cudnn_benchmark`
  in every eager JSON). FP16 means `model.half()` with an FP16 input. Runs on the default CUDA
  stream.
- **TensorRT:** runs on a dedicated `torch.cuda.Stream` (`extra.stream`). The engine's I/O is
  **FP32 with FP16 internals** (`extra.precision`: "fp16 (BuilderFlag.FP16, FP32 I/O)"), so the
  FP32→FP16 input cast happens inside the timed call, while eager FP16 gets FP16 input.
- **Optimization profile:** batch min 1 / opt 8 / max 8 (`extra.profile_batch`, build JSON
  `profile`). Batch 1 therefore runs at a non-opt shape. An opt = 1 engine was not tested.
- **Environment** (JSON metadata): Tesla T4, compute capability 7.5, power limit 70.00 W, max SM
  clock 1590 MHz, driver 580.82.07, CUDA 13.0, torch 2.14.1+cu130, torchvision 0.29.1+cu130,
  cuDNN 92800, TensorRT 10.16.1.11, onnxruntime 1.31.1, onnx 1.23.2, Python 3.13.16.
- **SM clock:** the nvidia-smi snapshots before and after each run range from 690 MHz
  (`pytorch_eager_fp16cl_bs1_20261113T100046663Z`, before) to 1590 MHz, so clocks were not
  constant during runs. We didn't measure the cause of the batch-1 eager p99 tail; Colab's shared
  vCPU is a hypothesis, not a measurement.

## Latency

### Canonical session (TensorRT and eager, same session, engine `c2bc78f3…`)

| Variant | p50 ms | p95 ms | p99 ms | img/s per-call | img/s pipelined | File |
|---|---:|---:|---:|---:|---:|---|
| TRT FP16 bs1 | 1.52 | 1.57 | 2.25 | 645 | 680 | [`trt_fp16_bs1_20261113T100028883Z`](results/trt_fp16_bs1_20261113T100028883Z.json) |
| Eager FP16 channels_last bs1 | 6.04 | 7.29 | 10.30 | 160 | 135* | [`pytorch_eager_fp16cl_bs1_20261113T100046663Z`](results/pytorch_eager_fp16cl_bs1_20261113T100046663Z.json) |
| Eager FP16 bs1 | 6.94 | 9.40 | 12.07 | 139 | 143 | [`pytorch_eager_fp16_bs1_20261113T100054940Z`](results/pytorch_eager_fp16_bs1_20261113T100054940Z.json) |
| TRT FP16 bs8 | 4.21 | 4.29 | 4.37 | 1901 | 1899 | [`trt_fp16_bs8_20261113T100033593Z`](results/trt_fp16_bs8_20261113T100033593Z.json) |
| Eager FP16 bs8 | 10.48 | 12.26 | 14.71 | 746 | 763 | [`pytorch_eager_fp16_bs8_20261113T100106830Z`](results/pytorch_eager_fp16_bs8_20261113T100106830Z.json) |

\* See the open question about channels_last pipelined throughput.

### Reproduction in an earlier session (engine with identical per-layer tactics, no hash)

| Variant | p50 ms | p95 ms | p99 ms | File |
|---|---:|---:|---:|---|
| TRT FP16 bs1 | 1.53 | 1.57 | 1.79 | [`trt_fp16_bs1_20261113T094149430Z`](results/trt_fp16_bs1_20261113T094149430Z.json) |
| Eager FP16 channels_last bs1 | 6.09 | 7.81 | 11.23 | [`pytorch_eager_fp16cl_bs1_20261113T094206984Z`](results/pytorch_eager_fp16cl_bs1_20261113T094206984Z.json) |
| Eager FP16 bs1 | 7.00 | 8.48 | 12.35 | [`pytorch_eager_fp16_bs1_20261113T094215330Z`](results/pytorch_eager_fp16_bs1_20261113T094215330Z.json) |
| TRT FP16 bs8 | 4.22 | 4.31 | 4.47 | [`trt_fp16_bs8_20261113T094153977Z`](results/trt_fp16_bs8_20261113T094153977Z.json) |
| Eager FP16 bs8 | 10.44 | 12.62 | 14.44 | [`pytorch_eager_fp16_bs8_20261113T094227234Z`](results/pytorch_eager_fp16_bs8_20261113T094227234Z.json) |

Ratios: bs1 6.087 / 1.530 = 3.98 → 4.0×; bs8 10.444 / 4.218 = 2.48 → 2.5×. Same ratios as the
canonical session.

### p99

TRT vs eager FP16 channels_last at batch 1, p99 ratio **4.6–6.3× across two runs**:
11.229 / 1.789 = 6.28 (`…094206984Z` / `…094149430Z`) and 10.304 / 2.245 = 4.59
(`…100046663Z` / `…100028883Z`). At batch 8 vs eager FP16: 14.440 / 4.467 = 3.23 and
14.708 / 4.368 = 3.37, so 3.2–3.4×.

### FP32 context (not the headline)

| Variant | p50 ms | p99 ms | File | Note |
|---|---:|---:|---|---|
| FP32 bs1 | 5.86 | 10.83 | [`pytorch_eager_fp32_bs1_20261113T085047280Z`](results/pytorch_eager_fp32_bs1_20261113T085047280Z.json) | Between two FP16 bs1 runs |
| FP32 bs1 | 5.60 | 9.34 | [`pytorch_eager_fp32_bs1_20261113T085116730Z`](results/pytorch_eager_fp32_bs1_20261113T085116730Z.json) | Run just before FP16 channels_last |
| FP32 bs8 | 23.39 | 24.07 | [`pytorch_eager_fp32_bs8_20261113T081902508Z`](results/pytorch_eager_fp32_bs8_20261113T081902508Z.json) | First session only; **not interleaved** with anything |

FP32 bs8 exists only from the first session, next to FP16 bs8 from the same session
([`pytorch_eager_fp16_bs8_20261113T081923083Z`](results/pytorch_eager_fp16_bs8_20261113T081923083Z.json),
10.70 ms): 23.386 / 10.697 = 2.19 → FP16 is 2.2× faster than FP32 at batch 8. That run order was
fixed, not interleaved.

## Why FP16 eager is slow at batch 1

**1. FP16 eager is slower than FP32 at batch 1.** With interleaved order (FP16, FP32, FP16):

| Order | Variant | p50 ms | File |
|---:|---|---:|---|
| 0 | FP16 bs1 | 6.86 | [`pytorch_eager_fp16_bs1_20261113T085039945Z`](results/pytorch_eager_fp16_bs1_20261113T085039945Z.json) |
| 1 | FP32 bs1 | 5.86 | [`pytorch_eager_fp32_bs1_20261113T085047280Z`](results/pytorch_eager_fp32_bs1_20261113T085047280Z.json) |
| 2 | FP16 bs1 | 6.95 | [`pytorch_eager_fp16_bs1_20261113T085055397Z`](results/pytorch_eager_fp16_bs1_20261113T085055397Z.json) |

FP16 is **17–19% slower** (6.859 / 5.862 = 1.17; 6.953 / 5.862 = 1.19) on both sides of the FP32
run.

The first-session FP16 bs1 run
([`pytorch_eager_fp16_bs1_20261113T081911448Z`](results/pytorch_eager_fp16_bs1_20261113T081911448Z.json),
8.33 ms) is kept but not cited. It ran third in a fixed order (FP32 bs1, FP32 bs8, FP16 bs1,
FP16 bs8) and wasn't interleaved, so the interleaved runs supersede it. Its nvidia-smi snapshots
read 1590 MHz before and 1380 MHz after. The interleaved runs started as low as 930 MHz and were
still faster, so these point samples don't explain the 8.33 ms.

**2. The mechanism: batch-1 eager is launch-bound.** Each variant was profiled for one forward
pass with `torch.profiler`. These are single passes, so treat the device times as approximate.

| Variant | Kernel launches | Summed device time | Same-session p50 | GPU busy (derived) | File |
|---|---:|---:|---:|---:|---|
| FP32 bs1 | 214 | 3.90 ms | 5.86 ms | 3.903 / 5.862 = 67% | [`kernels_pytorch_eager_fp32_bs1_20261113T085102064Z`](results/kernels_pytorch_eager_fp32_bs1_20261113T085102064Z.json) |
| FP16 bs1 | 377 | 2.15 ms | 6.91 ms (mean of 6.86, 6.95) | 2.147 / 6.906 = 31% | [`kernels_pytorch_eager_fp16_bs1_20261113T085103521Z`](results/kernels_pytorch_eager_fp16_bs1_20261113T085103521Z.json) |
| FP16 channels_last bs1 | 296 | 2.72 ms | 5.76 ms | 2.723 / 5.763 = 47% | [`kernels_pytorch_eager_fp16cl_bs1_20261113T085104753Z`](results/kernels_pytorch_eager_fp16cl_bs1_20261113T085104753Z.json) |

"GPU busy" is **derived**: profiled device time ÷ wall p50 from separate runs in the same
session. On that basis the FP16 GPU is idle about 69% of each call. FP16 needs about half
FP32's GPU time but launches 1.76× as many kernels (377 / 214), and ends up slower.

**Where FP16's extra launches come from**, grouped by kernel name in the kernels JSONs:

| Kernel group (name match) | FP32 | FP16 | FP16 channels_last |
|---|---:|---:|---:|
| `cudnn::engines_precompiled::nchwToNhwcKernel` + `nhwcToNchwKernel` | 8 (5 + 3) | 90 (53 + 37) | 0 |
| BatchNorm: `cudnn::bn_fw_inf_*` | 53 | 0 | 0 |
| BatchNorm: `at::native::batch_norm_transform_input*` + `batch_norm_calc_invstd` | 0 | 106 (53 + 53) | 106 (53 + 53) |
| `direct_copy_kernel_cuda` | 0 | 53 | 53 |
| ReLU (`launch_clamp_scalar`) | 49 | 49 | 49 |
| Residual add (`CUDAFunctor_add`) | 16 | 16 | 16 |
| Memset | 5 | 6 | 14 |
| Conv / GEMM / other | 83 | 57 | 58 |
| **Total** | **214** | **377** | **296** |

- **Layout transforms (verified by name):** FP16 NCHW launches 53 `nchwToNhwcKernel` and 37
  `nhwcToNchwKernel`, 90 in total (474.3 + 91.9 = 566 µs of device time). FP32 launches 8. That's
  +82 of FP16's +163 extra launches. The JSON's `layout_kernels` field is **not** used as
  evidence, because its regex also matches cuDNN NHWC convolution kernels.
- **BatchNorm:** in FP32, BN runs as 53 cuDNN `bn_fw_inf_*` launches. In FP16 it runs as 106
  `at::native` batch-norm launches. FP16 also has 53 `direct_copy` launches, the same count as
  the 53 BN layers, but the files don't show what they copy. Why FP16 BN goes off cuDNN isn't
  shown by any file either.

**3. channels_last brings FP16 to roughly parity with FP32. It is not a win.** In the same
session, FP32 ran just before channels_last:

| Variant | p50 ms | p99 ms | File |
|---|---:|---:|---|
| FP32 bs1 | 5.60 | 9.34 | [`pytorch_eager_fp32_bs1_20261113T085116730Z`](results/pytorch_eager_fp32_bs1_20261113T085116730Z.json) |
| FP16 channels_last bs1 | 5.76 | 10.03 | [`pytorch_eager_fp16cl_bs1_20261113T085124655Z`](results/pytorch_eager_fp16cl_bs1_20261113T085124655Z.json) |

5.763 / 5.599 = 1.03: channels_last is still 3% slower than FP32. It removes all 90 layout
transforms (296 launches), but the 106 `at::native` BN launches and 53 copies remain, so it still
launches more kernels than FP32's 214.

## Batch 1 vs batch 8

TensorRT's gain is larger at batch 1 (4.0×) than at batch 8 (2.5×). That fits batch-1 eager
being dominated by per-call overhead, which TensorRT removes: one `execute_async_v3` call runs 58
fused engine layers, instead of 296–377 separate kernel launches from Python.

- **Eager FP16 NCHW:** going from batch 1 to batch 8 does 8× the images, yet latency rises only
  1.5× (10.478 / 6.936 = 1.51) and per-call throughput rises 5.4× (746.2 / 138.9 = 5.37). Most of
  the batch-1 time is fixed per-call cost.
- **TensorRT:** latency rises 2.8× (4.212 / 1.519 = 2.77) and throughput 2.9× (1901.3 / 644.8 = 2.95).
  TensorRT has little fixed cost left to amortize.
- **Per-call vs pipelined throughput** doesn't separate the two regimes in these files. Eager
  bs8 gives 746 vs 763 img/s (1.02), but eager bs1, which is launch-bound, also gives 139 vs
  143 (1.03). Pipelining hides the sync gap between calls, not CPU launch cost. Only TRT bs1
  gains noticeably from pipelining: 645 → 680 img/s (1.05).

## The default-stream slow mode

The first TensorRT bs1 run
([`trt_fp16_bs1_20261113T093258124Z`](results/trt_fp16_bs1_20261113T093258124Z.json)) predates
the dedicated-stream change; its JSON has no `stream` field. It had p50 1.52 ms but p95 2.91 and
p99 2.94 ms, about 2× p50. Both dedicated-stream runs have the same p50 but no such mode:

| Run | Stream | p50 | p95 | p99 | Calls > 1.5× p50 | File |
|---|---|---:|---:|---:|---|---|
| 1 | Not recorded (predates the change) | 1.52 | 2.91 | 2.94 | Samples not saved | `…093258124Z` |
| 2 | Dedicated | 1.53 | 1.57 | 1.79 | Calls 0, 1 (493 of 500 within 10% of p50) | [`…094149430Z`](results/trt_fp16_bs1_20261113T094149430Z.json) |
| 3 | Dedicated | 1.52 | 1.57 | 2.25 | Call 0 (480 of 500 within 10% of p50) | [`…100028883Z`](results/trt_fp16_bs1_20261113T100028883Z.json) |

Run 1 saved no per-call samples, so the size of the slow mode is **derived**. If the slow calls
sit near p95, the mean implies roughly (1.619 − 1.516) / (2.913 − 1.516) ≈ 7% of calls. The
expected mechanism is TensorRT adding `cudaStreamSynchronize` calls when it's enqueued on the
legacy default stream. That warning appeared in console output but wasn't saved to any file, and
default vs dedicated stream were never run in the same session. So the stream change is
consistent with the fix, not proven to be it.

## Eager tails are periodic

At batch 1, slow eager calls (> 1.5× p50) in the channels_last runs cluster in runs of 2–3
consecutive calls, 160–166 calls apart. In
[`…fp16cl_bs1_20261113T094206984Z`](results/pytorch_eager_fp16cl_bs1_20261113T094206984Z.json)
those are calls 57–59, 223–225 and 383–385. In
[`…fp16cl_bs1_20261113T100046663Z`](results/pytorch_eager_fp16cl_bs1_20261113T100046663Z.json)
they're 113–114, 277–279 and 443–445, plus an isolated call 357. In both runs, all five slowest calls (which set p99)
fall inside these clusters. At about 6.1 ms per call, 160 calls is about 1 s (derived). The
cause hasn't been measured.

A 500-call TRT bs1 run lasts about 0.78 s (500 × 1.551 ms, derived), too short to show a
once-a-second event, so its clean tail isn't evidence that TRT is immune.

## ONNX export and parity

From [`onnx_parity_20261113T092423405Z`](results/onnx_parity_20261113T092423405Z.json):
- **Export:** `torch.onnx.export(dynamo=True)` with
  `dynamic_shapes=({0: Dim('batch', min=1, max=64)},)`. Opset 20, input dims
  `[batch, 3, 224, 224]`, 97.6 MB file.
- **Validation:** `onnx.checker.check_model(full_check=True)` passed.
- **Runtime:** onnxruntime **`CPUExecutionProvider`** (recorded in `ort_providers`), compared
  against PyTorch FP32 on CUDA.

| Batch | Max abs diff | Mean abs diff | Max \|logit\| | Top-1 agreement |
|---:|---:|---:|---:|---:|
| 8 | 3.3e-6 | 3.7e-7 | 5.37 | 8/8 |
| 1 | 1.8e-6 | 2.9e-7 | 5.31 | 1/1 |

The input was unnormalised noise, and all samples predict class 21 in both runtimes, so top-1
agreement is trivially satisfied. The max abs diff is the evidence.

## TensorRT build and fusion

**What went into TensorRT.** The parser log
([`logs/trt_build_20261113T095152257Z.log`](logs/trt_build_20261113T095152257Z.log),
`Parsing node` lines) lists the ONNX ops: 53 Conv, 49 Relu, 16 Add, 1 MaxPool, 1 ReduceMean,
1 Reshape, 1 Gemm, and **0 BatchNormalization**. BN was therefore already folded into the conv
weights before TensorRT saw the graph, so BN folding is the exporter's work, not TensorRT's.
(`onnx.load(...).graph.node` was not counted directly; the ONNX file isn't on this machine.)

**What TensorRT built: 126 network layers → 58 engine layers.**

| Build | Build time | Timing cache in | Network → engine layers | Engine | File |
|---|---:|---:|---:|---:|---|
| 09:51 (benchmarked engine) | 4.30 s | 729,596 B | 126 → 58 | 49.1 MB | [`trt_build_20261113T095157628Z`](results/trt_build_20261113T095157628Z.json) |
| 09:41 | 4.35 s | 729,596 B | 126 → 58 | 49.1 MB | [`trt_build_20261113T094141828Z`](results/trt_build_20261113T094141828Z.json) |

- **Cold build:** the first build (empty timing cache) crashed before writing its JSON. Its log
  reads "Engine generation completed in 48.9972 seconds"
  ([`logs/trt_build_20261113T093159473Z.log`](logs/trt_build_20261113T093159473Z.log)). That's
  TensorRT's internal engine-generation time, not the script's wall-clock `build_seconds`.
- **Warm builds:** both load 1,385 timing-cache entries, according to their logs.
- **Same tactics:** all three builds' layer JSONs list identical per-layer tactics.

**Fusion (TensorRT's work)**, from the 09:51 log and
[`logs/trt_layers_20261113T095152257Z.json`](logs/trt_layers_20261113T095152257Z.json):
- The log has 49 `ConvReluFusion` and 16 `ConvEltwiseSumFusion` entries: every ReLU and every
  residual Add is fused into a convolution.
- Of the 53 convolution layers, 33 are conv+ReLU, 16 are conv+add+ReLU, and 4 are conv only.
  Those 4 (`node_Conv_760`, `node_Conv_782`, `node_Conv_808`, `node_Conv_846`) each feed a
  conv+add+ReLU layer.
- Example fused layer: `node_Conv_762 + node_add_95 + node_relu_3` (`CaskGemmConvolution`,
  `HasReLU: 1`, `HasResidual: 1`, tactic `volta_h884gemm_128x64_ldg8_relu_nn_v1`).
- All 53 convolution layers output `Half`. The layer `__myl_FcCast_myl56_1` (type `fusion`)
  outputs `Float`.

## Accuracy (Imagenette, 500 images)

From [`accuracy_20261113T100119380Z`](results/accuracy_20261113T100119380Z.json), engine
sha256 `c2bc78f3…`:
- **Images:** 50 per class from imagenette2-320 val, seed 0. The paths are listed in the JSON.
- **Preprocessing:** `ResNet50_Weights.IMAGENET1K_V2.transforms()`.
- **Scoring:** 1000-way top-1, in batches of 8.

| Backend | Top-1 | Correct | Differ from FP32 | Max abs logit diff vs FP32 | Relative (÷ max \|FP32 logit\| 9.56) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 86.6% | 433/500 | — | — | — |
| PyTorch FP16 channels_last | 86.8% | 434/500 | 1 | 7.4e-2 | 7.7e-3 |
| TensorRT FP16 | 86.8% | 434/500 | 1 | 7.1e-2 | 7.4e-3 |

- **No accuracy claim either way.** 86.8% vs 86.6% is one image. At n = 500 the 95% confidence
  interval on 86.6% is ±3.0 points (1.96 × √(0.866 × 0.134 / 500) = 0.030), so this test can't
  separate the three. FP16 does not "improve" accuracy. The usable number is the flip count:
  1 of 500.
- **The same image flipped in both FP16 backends.** It's `n02102040/n02102040_791.JPEG` (label
  English springer). FP32 predicts "Welsh springer spaniel"; both FP16 backends predict "English
  springer". One image flipping the same way twice, rather than two independent flips, indicates
  that image sits near a decision boundary.
- **Relative error is higher on real images.** On the noise-input sanity check
  ([`trt_fp16_bs1_…100028883Z`](results/trt_fp16_bs1_20261113T100028883Z.json),
  [`trt_fp16_bs8_…100033593Z`](results/trt_fp16_bs8_20261113T100033593Z.json)), TRT's relative
  error was 1.9e-3 and 1.7e-3 (max abs 1.0e-2 and 9.0e-3 on max |logit| 5.39 and 5.33). On real
  images it's 7.4e-3, 3.9–4.4× higher. The real-image figure is a maximum over 500 images and
  the noise figures over 1 and 8, so part of the gap may be sample size.
- **Per-class top-1 is identical for all three backends**, except English springer (FP32 84%,
  both FP16 86%: the flipped image). Lowest: church 54%, cassette player 60%. Highest: tench and
  chain saw, 100%.
- **Reproduced:** the first accuracy run
  ([`accuracy_20261113T095543919Z`](results/accuracy_20261113T095543919Z.json)) used the same
  images and got the same counts and the same flipped image.

## T4 as a Jetson Orin proxy

Nothing in this repo was measured on Orin.

- **Transfers:**
  - Tensor-core FP16 (T4: Turing, compute capability 7.5 per the JSONs; Orin: Ampere, sm_87).
  - The same ONNX → TensorRT build workflow.
  - The same batch-1, launch-bound regime for eager PyTorch.
- **Doesn't transfer:**
  - Absolute latencies, because clocks and power modes differ.
  - Host↔device copy costs: Orin's CPU and GPU share memory, and no copy was timed here anyway.
  - The DLA, which needs a separate build path and supports a limited set of layers.

## What didn't work

- **First TensorRT build crashed before writing its JSON.** It called `len()` on the
  `IHostMemory` returned by `ITimingCache.serialize()`; that's now `.nbytes`. A first attempted
  fix (a logger guard) was a guess and missed the bug.
- **First accuracy run crashed on a wrong wnid.** Parachute was hardcoded as `n03888605`; the
  correct one is `n03888257`. `accuracy.py` now asserts that the wnids match the dataset folders.
- **channels_last didn't make FP16 beat FP32 at batch 1.** It reached roughly parity.

## Open questions

- **FP16 channels_last pipelined throughput is below per-call throughput in all four
  channels_last runs:** 168.7 vs 133.4
  ([`…085124655Z`](results/pytorch_eager_fp16cl_bs1_20261113T085124655Z.json)), 157.6 vs 124.3
  ([`…093316003Z`](results/pytorch_eager_fp16cl_bs1_20261113T093316003Z.json)), 158.7 vs 128.5
  ([`…094206984Z`](results/pytorch_eager_fp16cl_bs1_20261113T094206984Z.json)) and 160.5 vs 134.7
  img/s ([`…100046663Z`](results/pytorch_eager_fp16cl_bs1_20261113T100046663Z.json)). Other
  variants show pipelined below per-call by at most 1.2% (`pytorch_eager_fp32_bs8_20261113T081902508Z`:
  337.8 vs 341.9). Unexplained.
- An opt = 1 engine vs the opt = 8 engine at batch 1.
- A `torch.compile` / CUDA Graphs eager baseline (removes launch overhead without TensorRT).
- `polygraphy run --onnxrt --trt` with deliberately chosen atol/rtol (not done).
- INT8.
- The cause of the periodic eager tail clusters.

## Reproduce (Colab, T4)

```
python -m edgeforge.baseline                                    # fp32/fp16 x bs1/bs8
python -m edgeforge.baseline --variants fp16:1 fp32:1 fp16:1    # interleaved order
python -m edgeforge.profile_kernels --variants fp32:1 fp16:1 fp16cl:1
python -m edgeforge.baseline --variants fp32:1 fp16cl:1
pip install onnx onnxscript onnxruntime && python -m edgeforge.export
pip install tensorrt-cu13==10.16.1.11 --extra-index-url https://pypi.nvidia.com
python -m edgeforge.build_trt
python -m edgeforge.bench_trt
python -m edgeforge.baseline --variants fp16cl:1 fp16:1 fp16:8   # same-session eager comparison
python -m edgeforge.accuracy                                        # downloads imagenette2-320
```
