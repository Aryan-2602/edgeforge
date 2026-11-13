# edgeforge — instructions for Claude

## Goal
One honest, one-sentence result, e.g. "ResNet-50 in TensorRT FP16 runs Nx faster than
PyTorch FP16 eager on a Colab T4 at batch 1, with top-1 within X% on 500 Imagenette images."
Timebox: ~3–4 hours. When the sentence exists and is backed by JSON, we stop.

## How to work with me (Aryan)
- Socratic, learn-as-you-build. Before writing a non-trivial piece (TRT builder, timing loop,
  optimization profile), ask me ONE question about how I think it should work, then build.
- Before each benchmark run, ask me to predict the number. Afterwards, compare against the prediction.
- Keep changes small and iterative; I'm under time pressure. No speculative abstractions.
- Be direct. If a result looks too good or a comparison is unfair, say so.

## Hard rules (carried over from Inferno)
- Every number in the README must trace to a file in `results/` (timestamped JSON). No theoretical numbers.
- Document what didn't work, honestly.
- Headline compares like-for-like: TRT FP16 vs PyTorch FP16 eager, same batch, same copy semantics.
  FP32 is context, not the headline.

## Environment facts (verify, don't assume)
- Code is edited locally; it RUNS on Colab T4 (CUDA 13, torch 2.14+cu130).
  TensorRT and onnxruntime-gpu wheels must target CUDA 13. You cannot run GPU code locally.
  Give me exact commands to run in Colab and I'll paste back output.
- TensorRT 10.x API: use `build_serialized_network`, `set_input_shape`, `set_tensor_address`,
  `execute_async_v3`, I/O tensors by name. Do NOT use TRT 8 binding-index APIs.
- onnxruntime-gpu: assert `CUDAExecutionProvider` is active; fail loudly on CPU fallback.
  Exception: if a CUDA 13 onnxruntime-gpu build is troublesome, CPU onnxruntime is fine for the
  ONNX parity check (correctness, not speed) — label it as CPU in results.
- Timing: CUDA events or explicit `torch.cuda.synchronize()`. Warm-up runs before measuring.
  Record GPU name, clocks (nvidia-smi), driver, torch/TRT/ORT versions in each JSON.

## Harness contract (`edgeforge/harness.py`)
- Inputs: callable, batch size, n_warmup, n_iters, label.
- Outputs: p50/p95/p99/mean latency (ms), throughput (img/s), metadata → `results/<label>_<UTC timestamp>.json`.

## Fallback
If the hand-built TRT path is stuck >90 min: ultralytics `export(format="engine", half=True)`,
label it "wrapped path" in results and README.

## Interview questions this repo must let me answer
FP16/tensor cores and non-uniform layer speedup; what TRT does (fusion, tactic selection) with
evidence from `logs/`; warm-up and p99 vs mean; batch 1 vs 8 latency vs throughput; Jetson Orin
differences (unified memory, power modes, DLA).