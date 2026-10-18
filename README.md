# lowbit: INT4 weight-only kernels for LLM decode

CUDA C++ kernels for W4A16 (4-bit weights, FP16 activations) matrix-vector and small-batch matrix
multiply, a drop-in `W4Linear` for Hugging Face Qwen2.5 models, and a measurement harness that runs
on a Tesla T4. Decode at batch 1 is memory-bound: every token reads every weight once, so 4-bit
weights move about 4x fewer bytes than FP16. The question this repo answers with measurements is how
much of that turns into real speed, at what cost in quality, and where it stops paying.

All numbers below come from [results/RESULTS.md](results/RESULTS.md), which `bench/report.py`
renders from the raw JSON of the runs (GPU, driver, CUDA, torch and commit recorded in each file).

## Results (Tesla T4, sm_75)

**Kernels (A1).** At batch 1, the INT4 GEMV (v4) beats FP16 cuBLAS on every layer shape tested,
**2.1-3.7x** (Qwen2.5 0.5B/1.5B/3B and 7B-class shapes), and reaches **up to 86%** of the
measured 276 GB/s DRAM read bandwidth. Every kernel is checked against an fp64 reference on the
same quantized weights (worst error 2.4e-3 of output RMS, below cuBLAS's own 3.4e-3) and runs clean
under compute-sanitizer memcheck.

**End to end (A2), batch 1, greedy decoding, WikiText-2 perplexity:**

| model | weights | CUDA-graph decode | peak memory | perplexity FP16 -> INT4 (clip search) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.92 -> 0.43 GiB | 106.7 -> 149.0 tok/s (1.40x) | 2.56 -> 2.04 GB | 15.79 -> 21.91 |
| Qwen2.5-1.5B | 2.88 -> 1.06 GiB | 51.4 -> 89.8 tok/s (1.75x) | 4.65 -> 2.77 GB | 10.65 -> 12.39 |
| Qwen2.5-3B | 5.75 -> 1.91 GiB | 29.1 -> 59.5 tok/s (2.05x) | 7.85 -> 3.64 GB | 9.56 -> 10.83 |

* **Kernel error is separated from quantization error.** A "fake-quant" model holds exactly the
  dequantized INT4 weights in ordinary FP16 layers. Its perplexity matches the INT4 kernels to
  within 0.001, with 99.5-99.8% next-token agreement, so the quality loss is the cost of 4-bit
  weights, not of the kernels.
* **The per-group clip search matters.** Plain absmax scales take 3B from 9.56 to 15.20 perplexity.
  Searching a clipping factor per group of 128 weights brings it to 10.83, at no runtime cost.
* **The gain grows with model size.** The larger the share of decode time spent in linear layers,
  the larger the gain: 32% of GPU time at 0.5B, 60% at 3B (profiler).

## Where it does not pay (measured, not hidden)

* **Eager decode is slower (0.84-0.88x).** Without CUDA graphs, decode at these sizes is bound by
  CPU kernel launches: the GPU is busy 8-19% of the time, so shorter kernels do not shorten the
  step. With each decode step replayed as one CUDA graph, the kernel gain shows through.
* **Prefill and time to first token.** For many rows, the kernels lose to cuBLAS tensor-core GEMMs,
  so prefill dequantizes each layer to FP16 and calls cuBLAS. That costs 1.0-2.0x the FP16 GEMM time,
  and TTFT for a 65-token prompt rises from 36-55 ms to 50-93 ms. Running long prompts through the
  tensor-core kernel in 64-row slices was measured too: it lost on every shape from 128 rows (and on
  narrow layers already at 65), so it is not used.
* **The dispatch rule comes from the sweep:**
  * 1 row: the GEMV (v6).
  * Up to 32 rows: the tensor-core kernel (v5), at most 0.78x the time of dequantize + cuBLAS on every shape.
  * 33-64 rows: v5 only on wide layers (N >= 3072); it loses on narrow and long-K layers there.
  * Beyond that: dequantize + cuBLAS.

## Kernel ladder

Each step is kept and measured on its own (`csrc/w4_gemv.cu`):

| version | idea |
|---|---|
| v1 | one thread per output row (baseline; uncoalesced) |
| v2 | one warp per row, 128-bit loads, shuffle reduction |
| v3 | offline nibble interleave, 0x6400 magic-number dequantization, HFMA2 |
| v4 | R rows per warp so each activation load feeds R rows; R picked from N |
| v5 | tensor cores (`mma.sync` m16n8k8, fp32 accumulate): weights decoded in registers as the A operand |
| v6 | v4's GEMV on v5's weight layout, so a model keeps one copy of the weights |

## Format and reproduction

Symmetric INT4 (-8..7), group size 128 along K, one FP16 scale per group, two values per byte. The
Python golden model is `lowbit/quant.py`. The GPU quantizer gives the same bits (checked in the run).

```
python kaggle/build_kernel.py full    # A1: tests, memcheck, kernel sweep -> results/a1.json
python kaggle/build_kernel.py e2e     # A2: Qwen2.5 0.5B/1.5B/3B, absmax and clip search
kaggle kernels push -p kernel
python bench/report.py results/raw/a1.json results/raw/e2e_*.json > results/RESULTS.md
```

Platform limits: the T4 has FP16 tensor cores but no BF16/FP8, and these kernels unpack weights to
FP16, so INT4 tensor-core peak is not their ceiling. Results are from one GPU model, and other GPUs
will place the crossovers differently. Design notes: [docs/DESIGN.md](docs/DESIGN.md).
