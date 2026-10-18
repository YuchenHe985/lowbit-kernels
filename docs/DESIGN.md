# lowbit: INT4 weight-only kernels for LLM decode, from the kernel to the model

Weight-only INT4 (AWQ/GPTQ-style) is how LLMs are commonly served (vLLM, SGLang, TensorRT-LLM,
llama.cpp). Decode at small batch is memory-bound: each token reads every weight once, so INT4
moves about 4x fewer bytes than FP16. This project writes that kernel path, plugs it into a real
model and reports where the theory holds, where it does not, and when to fall back.

## Scope

**A1. CUDA kernels (the core).**
* Quantization contract (lowbit/quant.py, the golden model): symmetric INT4, values -8..7, group size
  128 along K, FP16 scale per group, round-half-even, two nibbles per byte (low nibble = even k);
  K must be a multiple of the group size. Error is reported against an fp64 reference.
* W4A16 GEMV for decode (M = 1..8) as a measured ladder: v1 thread per row, v2 warp per row with
  128-bit loads and shuffle reduction, v3 offline nibble interleave + 0x6400 magic-number dequant +
  HFMA2. Each step is kept and measured on its own; regressions are reported, not hidden.
* Prefill path: dequantize kernel + cuBLAS FP16 GEMM. Crossover M where it beats the GEMV.
* Baselines on the same quantized weights and real layer shapes (Qwen2.5 0.5B/1.5B/3B and a 7B-class
  shape): FP16 cuBLAS (torch.matmul), dequant + cuBLAS, the three GEMV versions.
* Measurement: CUDA events, warm-up, N repeats, median and p10/p90; achieved GB/s against the T4
  DRAM peak; roofline position; compute-sanitizer memcheck on the kernels.

**A2. Model integration.**
* `W4Linear` drop-in that swaps the linear layers of a Hugging Face Qwen2.5 model, GEMV for decode,
  dequant + cuBLAS for prefill.
* Fixed workload (prompt length, output length, batch 1): TTFT, decode tokens/s, peak memory, weight
  bytes, WikiText-2 perplexity vs FP16, and the share of decode time spent in linear layers
  (profiler), so the end-to-end gain is attributed rather than assumed.
* Stretch, only after the above: a vLLM backend (first a small install/load test on T4; if vLLM
  does not run there, the Hugging Face path stands on its own).

**Not in this repo:** an ARM NEON backend (low new signal; NEON skill is already shown elsewhere)
and an RTL accelerator (a different hiring track; decided separately).

## Deliverables

* Reproducible: one command builds the Kaggle run; pinned versions; environment record (GPU, driver,
  CUDA, torch, commit) in every results file; raw CSV/JSON committed; README tables generated from them.
* Correctness tests: Python golden-model tests on any machine; GPU tests (all shapes, M = 1..8,
  odd N, every version) inside the Kaggle run, which fails if any check fails.
* A written report: design choices (why warp per row, why interleave, register use), the measured
  ladder, the shapes where a version is slower and why, and the fallback rule.
* Wording: "engineering-validated", never "production-grade" (there are no production users).

## Platform limits (stated in the README)

Tesla T4 (sm_75): FP16 tensor cores, no BF16/FP8; INT4 tensor-core peak is not this kernel's ceiling
(weights are unpacked to FP16). Triton support on sm_75 depends on the version, so CUDA C++ is the
main path. Kaggle sessions are time-limited: runs save results as they go.
