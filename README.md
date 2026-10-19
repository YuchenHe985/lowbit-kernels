# lowbit: INT4 weight-only kernels for LLM decode

CUDA C++ kernels for W4A16 (4-bit weights, FP16 activations) matrix-vector and small-batch matrix
multiply, a drop-in `W4Linear` for Hugging Face Qwen2.5 models, and a measurement harness that runs
on a Tesla T4. Decode at batch 1 is memory-bound: every token reads every weight once, so 4-bit
weights move about 4x fewer bytes than FP16. The question this repo answers with measurements is how
much of that turns into real speed, at what cost in quality, and where it stops paying.

All numbers below come from [results/RESULTS.md](results/RESULTS.md), which `bench/report.py`
renders from the raw JSON of the runs (GPU, driver, CUDA, torch, commit and a hash of the sources recorded in each file).

## Results (Tesla T4, sm_75)

**Kernels (A1).** At batch 1 on the MLP layers, which hold most of the weights, the standalone INT4
GEMV (v4) takes **0.34-0.69x the time an FP16 kernel would need just to stream its weights once** at
the measured 273 GB/s. Every call reads cold weights: operands cycle through at least 64 MiB of copies
against the T4's 4 MiB L2. v4 reaches up to **75%** of that bandwidth. The GEMV a model actually runs
(v6, on the one weight layout the model keeps) takes 0.43-0.74x. Against FP16 cuBLAS the deployed
path is 1.4-3.8x faster at batch 1, but cuBLAS itself reaches only 10-87% of bandwidth there, so the
bandwidth comparison is the fairer one. Every kernel is checked against an fp64 reference on the
same quantized weights: worst normalized errors are 2.06-2.53e-3 (cuBLAS's independent worst on the
same cases: 3.43e-3), 2.61e-3 across every boundary of the dispatch rule, and an overflow regression
is exact. compute-sanitizer finds no memory-access errors, shared-memory races or barrier misuse. Its
full memcheck pass reports one CUDA API error, an invalid handle on `cuKernelGetFunction`, which is
its only finding.

**End to end (A2), batch 1, greedy decoding, WikiText-2 perplexity:**

| model | weights | CUDA-graph decode | peak memory | perplexity FP16 -> INT4 (clip search) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.92 -> 0.43 GiB | 104.8 -> 143.6 tok/s (1.37x) | 2.56 -> 2.04 GB | 15.79 -> 21.91 |
| Qwen2.5-1.5B | 2.88 -> 1.06 GiB | 50.7 -> 85.9 tok/s (1.69x) | 4.65 -> 2.76 GB | 10.65 -> 12.39 |
| Qwen2.5-3B | 5.75 -> 1.91 GiB | 28.7 -> 57.6 tok/s (2.01x) | 7.85 -> 3.64 GB | 9.56 -> 10.83 |

Models are the Qwen2.5-Instruct checkpoints. INT4 weights total 2.1-3.0x smaller than FP16 rather than
4x, because the embeddings (tied to the output head) stay FP16 and every group of 128 weights carries
an FP16 scale. Only the decoder projections are quantized.

* **4-bit weights cost real quality, and the kernels add almost none.** With the clip search,
  perplexity rises 39% at 0.5B, 16% at 1.5B and 13% at 3B, and the INT4 model picks the same next
  token as FP16 71-80% of the time. A "fake-quant" model holds exactly the dequantized INT4 weights in
  ordinary FP16 layers, which separates the cost of the weights from the cost of the kernels. On the
  perplexity run it matches INT4 to within 0.002. Fed the same 512 tokens through the KV cache at 1, 16
  and 48 tokens per step (each decode kernel in turn), INT4 and fake-quant agree on the next token
  99.2-100% of the time, with perplexity within 0.011 and a worst logit gap of 11% of RMS.
* **The per-group clip search matters.** Plain absmax scales take 3B from 9.56 to 15.20 perplexity.
  Searching a clipping factor per group of 128 weights brings it to 10.83, at no runtime cost.
* **The gain grows with model size.** A CUDA-graph step at 0.5B spends only 30% of its GPU time in
  the INT4 linear kernels (attention, norms and elementwise work make up the rest), so faster
  kernels move the step less. At 3B the FP16 graph step is 87% as long as the eager step, which is
  consistent with a step made mostly of GPU work; the INT4 step there is half as long.
* **Graph and eager runs agree closely.** Fed the same tokens, they pick the same next token at
  99.2-100% of 128 positions, and the worst logit gap is 7.8% of RMS.

## Where it does not pay (measured, not hidden)

* **Eager decode is slower (0.84-0.85x).** Without CUDA graphs, decode at these sizes is bound by
  host-side overhead: the INT4 graph step is only 21-37% as long as the eager step, and the INT4
  eager step is longer than FP16's, even though its GPU work is shorter. With each decode step
  replayed as one CUDA graph, the kernel gain shows through.
* **Prefill and time to first token.** For many rows, the kernels lose to cuBLAS tensor-core GEMMs,
  so prefill dequantizes each layer to FP16 and calls cuBLAS. That costs up to 1.75x the FP16 GEMM
  time (about 1.1x at 1024 rows), and TTFT for a 65-token prompt rises from 35-55 ms to 48-91 ms.
  Running long prompts through the tensor-core kernel in 64-row slices was measured too. It beats
  dequantize + cuBLAS only on the widest layers at 65-96 rows, mostly by margins within timing noise.
  It loses to it on every shape at 128-512 rows and to FP16 cuBLAS everywhere, so it is not used.
* **Against FP16 cuBLAS, INT4 wins only at small batches.** Where the deployed path stops winning
  depends on the shape, roughly 8-32 rows. Beyond that, an INT4 model pays for its smaller weights
  with slower matmuls on most layers.
* **The dispatch rule comes from the sweep** (on the one weight copy a model keeps):
  * 1 row: the GEMV (v6).
  * Up to 32 rows: the tensor-core kernel (v5), at most 0.82x the time of dequantize + cuBLAS on every shape.
  * 33-64 rows: v5 only on wide layers (N >= 3072), at 0.71-0.90x (the smaller margins are within timing noise); on narrower layers it no longer wins.
  * Beyond that: dequantize + cuBLAS.
* **Timing on a shared T4 is noisy.** The GPU throttles under its 70 W cap; SM clock readings ranged
  330-1590 MHz. Each measurement starts after a warm-up load, and an order check calls the same
  kernel three ways, interleaved, before and after the sweep. After the first call, identical calls
  agree within 22% (median 7%), so smaller gaps between two cells are not resolved.
* **No DRAM counters.** The Kaggle VM denies access to GPU performance counters, so bandwidth here
  is effective bandwidth (minimum bytes moved / time), not a hardware count. `bench/ncu_dram.py`
  runs the Nsight Compute check where counters are allowed.

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

On the large layers v6 is 0-33% slower than v4: the price of keeping one weight copy instead of two.

## Format and reproduction

Symmetric INT4 (-8..7), group size 128 along K, one FP16 scale per group, two values per byte. The
Python golden model is `lowbit/quant.py`. The GPU quantizer gives the same bits (checked in the run).

```
python kaggle/build_kernel.py full    # A1: tests, sanitizers, kernel sweep -> results/a1.json
python kaggle/build_kernel.py e2e     # A2: Qwen2.5 0.5B/1.5B/3B, absmax and clip search
kaggle kernels push -p kernel
python bench/report.py results/raw/a1.json results/raw/ncu_dram.csv results/raw/e2e_*.json > results/RESULTS.md
```

Scope: the baselines are FP16 cuBLAS and the bandwidth floor, not other INT4 kernel libraries, and
speed is measured on one prompt shape (65 tokens in, 256 out) at batch 1.
Platform limits: the T4 has FP16 tensor cores but no BF16/FP8, and these kernels unpack weights to
FP16, so INT4 tensor-core peak is not their ceiling. Results are from one GPU model, and other GPUs
will place the crossovers differently. Design notes: [docs/DESIGN.md](docs/DESIGN.md).
