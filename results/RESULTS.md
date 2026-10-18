## A1: kernels

Environment: Tesla T4 (sm_75, 40 SMs), torch 2.11.0+cu128, CUDA 12.8, commit `35dc86c-dirty`, 2026-10-18T23:11:45Z

Measured DRAM bandwidth: **276 GB/s read-only**, 233 GB/s copy (spec 320 GB/s). Rooflines and `% of BW` use the read figure. Weights are read cold on every call (copies cycled past the 4 MiB L2), as in a real decode step.

Correctness: every kernel against an fp64 reference on the same quantized weights, M = 1..8, all shapes plus odd-N tails (v5 also at M = 12..64). Worst max-abs error / RMS(output): v1 2.06e-03, v2 2.06e-03, v3 2.26e-03, v4 2.26e-03, v5 2.36e-03, v6 2.19e-03, cublas 3.43e-03.

### Decode (M = 1): median latency, microseconds

| layer | K x N | FP16 cuBLAS | FP16 roofline | dequant + cuBLAS | v1 | v2 | v3 | v4 (R) | INT4 roofline | v4 vs FP16 cuBLAS | v4 vs FP16 roofline | v4 % of BW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | 896 x 896 | 9.9 | 5.8 | 11.8 | 53.9 | 4.8 | 4.4 | 4.3 (4) | 1.5 | 2.28x | 1.34x | 35% |
| qwen0.5b.kv_proj | 896 x 128 | 5.1 | 0.8 | 6.1 | 52.2 | 2.5 | 2.4 | 2.4 (1) | 0.2 | 2.12x | 0.35x | 9% |
| qwen0.5b.gate_up | 896 x 4864 | 40.2 | 31.6 | 98.3 | 78.8 | 14.0 | 14.1 | 12.7 (2) | 8.2 | 3.18x | 2.49x | 65% |
| qwen0.5b.down | 4864 x 896 | 37.9 | 31.6 | 86.2 | 358.0 | 13.5 | 13.1 | 12.5 (2) | 8.2 | 3.03x | 2.53x | 65% |
| qwen1.5b.q_proj | 1536 x 1536 | 22.7 | 17.1 | 43.3 | 116.6 | 9.5 | 9.1 | 7.8 (2) | 4.4 | 2.90x | 2.19x | 57% |
| qwen1.5b.gate_up | 1536 x 8960 | 114.8 | 99.8 | 280.4 | 306.7 | 35.3 | 52.0 | 34.2 (2) | 25.8 | 3.36x | 2.92x | 75% |
| qwen1.5b.down | 8960 x 1536 | 112.5 | 99.8 | 276.4 | 773.6 | 35.8 | 43.1 | 33.0 (2) | 25.8 | 3.41x | 3.02x | 78% |
| qwen3b.q_proj | 2048 x 2048 | 36.4 | 30.4 | 83.7 | 158.7 | 13.8 | 13.6 | 11.3 (2) | 7.9 | 3.22x | 2.69x | 70% |
| qwen3b.gate_up | 2048 x 11008 | 181.5 | 163.4 | 442.7 | 581.9 | 54.7 | 93.1 | 49.2 (2) | 42.2 | 3.69x | 3.32x | 86% |
| qwen3b.down | 11008 x 2048 | 182.0 | 163.4 | 449.3 | 870.7 | 53.9 | 99.7 | 52.7 (2) | 42.2 | 3.45x | 3.10x | 80% |
| 7b-class.gate_up | 3584 x 18944 | 558.0 | 492.1 | 1330.4 | 1710.9 | 275.8 | 247.3 | 152.2 (4) | 127.0 | 3.67x | 3.23x | 83% |
| 7b-class.down | 18944 x 3584 | 546.7 | 492.1 | 1339.3 | 2370.3 | 291.5 | 241.0 | 149.6 (4) | 127.0 | 3.65x | 3.29x | 85% |

(R) = the fastest rows-per-warp among the v4_r1/r2/r4 runs; `v4` itself uses the R picked from N.

### Batch sweep: fastest INT4 kernel vs FP16 cuBLAS, microseconds

Each cell: best INT4 time (which kernel) / FP16 cuBLAS time. v1-v4 run up to M = 8, the tensor-core v5 up to M = 64; v5 picks its row tiles (T) from N and M, the v5_tN runs fix them.

| layer | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 |
|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | **4.3** (v4) / 9.9 | **4.9** (v5_t1) / 10.7 | **5.2** (v5_t1) / 10.7 | **5.7** (v5_t2) / 10.7 | **6.9** (v5_t2) / 10.8 | **9.2** (v5_t2) / 10.9 | **17.4** (v5_t4) / 18.0 |
| qwen0.5b.kv_proj | **2.4** (v4_r1) / 5.1 | **2.7** (v4_r2) / 6.8 | **3.4** (v4_r2) / 6.9 | **3.9** (v5) / 6.9 | **4.4** (v5_t1) / 6.9 | **6.5** (v5) / 6.9 | 11.6 (v5) / 7.2 |
| qwen0.5b.gate_up | **10.9** (v4_r2) / 40.2 | **12.2** (v5_t1) / 40.4 | **12.2** (v5) / 40.7 | **15.2** (v5_t2) / 41.4 | **17.6** (v5_t4) / 42.5 | **23.5** (v5_t4) / 54.2 | 58.3 (v5) / 57.6 |
| qwen0.5b.down | **11.9** (v4_r2) / 37.9 | **13.3** (v4) / 40.4 | **17.1** (v5_t1) / 40.6 | **18.9** (v5) / 41.0 | **25.8** (v5_t2) / 42.0 | **41.0** (v5_t2) / 58.9 | 80.7 (v5_t4) / 65.5 |
| qwen1.5b.q_proj | **7.7** (v4_r2) / 22.7 | **8.5** (v5) / 23.0 | **9.2** (v5) / 23.2 | **9.9** (v5_t2) / 23.3 | **13.4** (v5_t2) / 23.8 | **19.9** (v5_t4) / 33.6 | 43.9 (v5_t4) / 36.2 |
| qwen1.5b.gate_up | **31.3** (v4_r2) / 114.8 | **34.0** (v5) / 117.7 | **35.6** (v5) / 118.4 | **43.0** (v5) / 119.7 | **52.0** (v5) / 148.0 | **71.5** (v5) / 151.3 | 213.3 (v5_t4) / 155.9 |
| qwen1.5b.down | **32.0** (v4_r2) / 112.5 | **34.6** (v4) / 118.2 | **43.4** (v5) / 118.7 | **60.8** (v5) / 119.5 | **72.2** (v5_t4) / 121.0 | **102.9** (v5_t4) / 162.8 | 262.8 (v5_t4) / 173.8 |
| qwen3b.q_proj | **11.2** (v4_r2) / 36.4 | **13.0** (v5) / 44.8 | **13.3** (v5) / 45.2 | **15.9** (v5_t2) / 45.1 | **18.8** (v5_t4) / 45.8 | **27.6** (v5_t4) / 49.1 | 69.4 (v5_t2) / 50.8 |
| qwen3b.gate_up | **49.2** (v4) / 181.5 | **53.3** (v4) / 188.0 | **63.0** (v5) / 188.9 | **77.1** (v5) / 190.7 | **78.8** (v5) / 194.7 | **140.8** (v5_t4) / 269.5 | 370.3 (v5_t4) / 276.3 |
| qwen3b.down | **52.0** (v4_r2) / 182.0 | **57.0** (v4) / 198.8 | **75.5** (v5) / 201.6 | **87.4** (v5_t2) / 206.8 | **107.6** (v5_t4) / 222.2 | **162.0** (v5_t4) / 251.4 | 422.9 (v5_t2) / 264.9 |
| 7b-class.gate_up | **152.2** (v4) / 558.0 | **204.0** (v4_r4) / 578.3 | **216.8** (v5_t4) / 581.0 | **227.0** (v5) / 586.5 | **328.0** (v5) / 596.7 | **465.8** (v5_t4) / 714.8 | 1133.7 (v5_t2) / 749.8 |
| 7b-class.down | **149.6** (v4) / 546.7 | **182.3** (v4_r4) / 575.2 | **219.5** (v5_t4) / 577.3 | **241.3** (v5_t4) / 582.5 | **340.7** (v5_t4) / 626.5 | **474.2** (v5_t4) / 750.0 | 1186.8 (v5) / 847.9 |

Bold: the INT4 kernel beats FP16 cuBLAS at that batch size.

### v4 / v6 / v5 by batch size (picked defaults), microseconds

| layer | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| qwen0.5b.q_proj | 4.3 / 4.9 / 4.7 | 5.0 / 5.7 / 4.9 | 6.5 / 8.1 / 5.2 | 9.2 / 13.2 / 5.9 |
| qwen0.5b.kv_proj | 2.4 / 2.8 / 3.2 | 2.7 / 3.5 / 3.4 | 3.5 / 4.2 / 3.6 | 4.6 / 7.6 / 3.9 |
| qwen0.5b.gate_up | 12.7 / 13.5 / 12.0 | 14.9 / 20.0 / 12.3 | 22.0 / 31.1 / 12.2 | 34.1 / 56.4 / 15.5 |
| qwen0.5b.down | 12.5 / 13.3 / 15.1 | 13.3 / 18.7 / 16.4 | 19.2 / 26.7 / 17.1 | 28.8 / 51.0 / 18.9 |
| qwen1.5b.q_proj | 7.8 / 8.4 / 8.1 | 9.1 / 11.2 / 8.5 | 12.4 / 17.7 / 9.2 | 20.3 / 30.7 / 10.8 |
| qwen1.5b.gate_up | 34.2 / 35.0 / 33.5 | 35.4 / 53.7 / 34.0 | 49.5 / 92.6 / 35.6 | 97.8 / 162.0 / 43.0 |
| qwen1.5b.down | 33.0 / 33.1 / 38.6 | 34.6 / 49.5 / 41.1 | 52.8 / 80.8 / 43.4 | 92.4 / 148.6 / 60.8 |
| qwen3b.q_proj | 11.3 / 12.5 / 12.9 | 14.1 / 18.0 / 13.0 | 19.6 / 30.0 / 13.3 | 32.6 / 54.1 / 16.6 |
| qwen3b.gate_up | 49.2 / 53.5 / 57.4 | 53.3 / 84.5 / 58.1 | 83.4 / 143.8 / 63.0 | 182.8 / 256.1 / 77.1 |
| qwen3b.down | 52.7 / 52.8 / 62.1 | 57.0 / 81.9 / 65.0 | 94.3 / 144.4 / 75.5 | 186.2 / 261.1 / 113.1 |
| 7b-class.gate_up | 152.2 / 174.6 / 206.6 | 205.9 / 260.1 / 211.6 | 298.3 / 425.2 / 282.0 | 526.6 / 755.6 / 227.0 |
| 7b-class.down | 149.6 / 167.2 / 223.7 | 186.2 / 255.3 / 239.5 | 294.0 / 423.8 / 302.0 | 521.2 / 747.5 / 275.2 |

### Prefill (M > 64): dequant + cuBLAS time / FP16 cuBLAS time

| layer | M=65 | M=96 | M=128 | M=192 | M=256 | M=384 | M=512 | M=1024 |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | 1.12x | 1.11x | 1.17x | 1.10x | 1.14x | 1.28x | 1.30x | 1.51x |
| qwen0.5b.kv_proj | 1.31x | 1.33x | 1.34x | 1.30x | 1.27x | 1.16x | 1.16x | 1.08x |
| qwen0.5b.gate_up | 1.79x | 1.77x | 1.71x | 1.69x | 1.68x | 1.41x | 1.21x | 1.10x |
| qwen0.5b.down | 1.81x | 1.73x | 1.66x | 1.72x | 1.64x | 1.32x | 1.14x | 1.06x |
| qwen1.5b.q_proj | 1.48x | 1.51x | 1.48x | 1.62x | 1.45x | 1.68x | 1.38x | 1.02x |
| qwen1.5b.gate_up | 1.99x | 1.93x | 1.83x | 1.40x | 1.36x | 1.29x | 1.18x | 1.09x |
| qwen1.5b.down | 1.75x | 1.78x | 1.68x | 1.38x | 1.33x | 1.21x | 1.16x | 1.09x |
| qwen3b.q_proj | 1.85x | 1.86x | 1.86x | 2.01x | 1.57x | 1.29x | 1.46x | 1.12x |
| qwen3b.gate_up | 1.80x | 1.65x | 1.57x | 1.38x | 1.31x | 1.19x | 1.16x | 1.11x |
| qwen3b.down | 1.64x | 1.56x | 1.47x | 1.35x | 1.30x | 1.21x | 1.13x | 1.07x |
| 7b-class.gate_up | 1.68x | 1.61x | 1.50x | 1.36x | 1.32x | 1.24x | 1.17x | 1.10x |
| 7b-class.down | 1.70x | 1.61x | 1.50x | 1.39x | 1.25x | 1.22x | 1.17x | 1.11x |

## A2: end to end (Hugging Face, batch 1, greedy)

Environment: Tesla T4 (sm_75, 40 SMs), torch 2.11.0+cu128, CUDA 12.8, commit `f2a9e7d-dirty`, 2026-10-18T01:50:15Z

Quality on WikiText-2 (40 windows x 1024 tokens). Fake-quant = FP16 layers holding exactly the dequantized INT4 weights: its gap to FP16 is the cost of 4-bit weights, its gap to INT4 is kernel error.

| model | scales | weights FP16 -> INT4 | PPL FP16 | PPL fake-quant | PPL INT4 | next-token agreement with FP16 | INT4 vs fake-quant agreement |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | absmax | 0.92 -> 0.43 GiB | 15.789 | 22.168 | 22.166 | 70.6% | 99.72% |
| Qwen2.5-0.5B-Instruct | clip search | 0.92 -> 0.43 GiB | 15.789 | 21.912 | 21.912 | 70.8% | 99.72% |
| Qwen2.5-1.5B-Instruct | absmax | 2.88 -> 1.06 GiB | 10.654 | 12.743 | 12.744 | 79.5% | 99.78% |
| Qwen2.5-1.5B-Instruct | clip search | 2.88 -> 1.06 GiB | 10.654 | 12.391 | 12.392 | 80.2% | 99.74% |
| Qwen2.5-3B-Instruct | absmax | 5.75 -> 1.91 GiB | 9.556 | 15.196 | 15.195 | 72.4% | 99.54% |
| Qwen2.5-3B-Instruct | clip search | 5.75 -> 1.91 GiB | 9.556 | 10.833 | 10.832 | 79.8% | 99.76% |

Speed at batch 1 (decode tokens/s; TTFT in ms). Eager = plain `generate`; CUDA graph = each decode step replayed as one graph over a static KV cache.

| model | scales | eager tok/s FP16 -> INT4 | CUDA graph tok/s FP16 -> INT4 | TTFT ms FP16 -> INT4 | peak mem GB |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | absmax | 32.3 -> 28.4 (0.88x) | 111.0 -> 150.0 (1.35x) | 36 -> 52 | 2.56 -> 2.04 |
| Qwen2.5-0.5B-Instruct | clip | 34.5 -> 28.9 (0.84x) | 106.7 -> 149.0 (1.40x) | 36 -> 50 | 2.56 -> 2.04 |
| Qwen2.5-1.5B-Instruct | absmax | 28.4 -> 24.2 (0.85x) | 51.4 -> 89.4 (1.74x) | 43 -> 60 | 4.65 -> 2.76 |
| Qwen2.5-1.5B-Instruct | clip | 29.2 -> 25.0 (0.86x) | 51.4 -> 89.8 (1.75x) | 42 -> 58 | 4.65 -> 2.77 |
| Qwen2.5-3B-Instruct | absmax | 22.7 -> 19.8 (0.87x) | 29.1 -> 60.0 (2.06x) | 55 -> 92 | 7.85 -> 3.64 |
| Qwen2.5-3B-Instruct | clip | 22.3 -> 19.2 (0.86x) | 29.1 -> 59.5 (2.05x) | 55 -> 93 | 7.85 -> 3.64 |

Where eager decode GPU time goes (profiler, 32 steps):

| model | run | GPU busy | INT4 kernels | other GEMM | dequant | other |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | fp16 | 13% | 0% | 49% | 0% | 51% |
| Qwen2.5-0.5B-Instruct | int4 | 12% | 32% | 2% | 0% | 67% |
| Qwen2.5-0.5B-Instruct | fp16 | 12% | 0% | 50% | 0% | 50% |
| Qwen2.5-0.5B-Instruct | int4 | 12% | 32% | 2% | 0% | 66% |
| Qwen2.5-1.5B-Instruct | fp16 | 18% | 0% | 73% | 0% | 27% |
| Qwen2.5-1.5B-Instruct | int4 | 15% | 51% | 0% | 0% | 49% |
| Qwen2.5-1.5B-Instruct | fp16 | 18% | 0% | 73% | 0% | 27% |
| Qwen2.5-1.5B-Instruct | int4 | 15% | 51% | 0% | 0% | 49% |
| Qwen2.5-3B-Instruct | fp16 | 9% | 0% | 41% | 0% | 59% |
| Qwen2.5-3B-Instruct | int4 | 19% | 60% | 0% | 0% | 39% |
| Qwen2.5-3B-Instruct | fp16 | 8% | 0% | 41% | 0% | 59% |
| Qwen2.5-3B-Instruct | int4 | 19% | 61% | 0% | 0% | 39% |
