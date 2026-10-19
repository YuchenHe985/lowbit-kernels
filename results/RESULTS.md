## A1: kernels

Environment: Tesla T4 (sm_75, 40 SMs), torch 2.11.0+cu128, CUDA 12.8, transformers 5.16.1, datasets 4.8.5, commit `647fea8-dirty`, source sha256 `3d27cbba7ceeff5f`, 2026-10-19T00:09:15Z

Measured DRAM bandwidth: **273 GB/s read-only**, 242 GB/s copy (spec 320 GB/s). Rooflines and `eff. % of BW` use the read figure.

Timing: median of five CUDA-graph replays; each replay calls the operation once per copy of its operand, cycling through at least 64 MiB of distinct copies (T4 L2: 4 MiB), so weights are not reused from L2 between calls. Spread in parentheses = range of the five replay means (p10-p90), not per-call percentiles.

Effective bandwidth = the minimum bytes an operation must move / its time. It is not a DRAM counter: an intermediate that stays in L2 (dequantize + cuBLAS at small sizes) can push it past the DRAM ceiling. The Nsight Compute DRAM-counter check (bench/ncu_dram.py) is at the end of this section, where the VM allows performance counters.

Order check: the same v6 kernel reached three ways (`gemv`, `linear()`, the raw extension call), interleaved over three rounds on four layers, before and after the sweep. The first call of each check ran 5-19% slow while the clock ramped; after it, identical calls were within 22% of each other (median 7%). Each measurement now starts after a warm-up load, but a gap of under about 20% between two cells is not resolved by this setup. The SM clock read right after each measurement ranged 330-1590 MHz (the T4 throttles under its 70 W cap), so absolute times carry that variation; compare cells within one row.

Correctness: every kernel against an fp64 reference on the same quantized weights, M = 1..8, all shapes plus odd-N tails (v5 also at M = 12..64); each kernel's worst max-abs error / RMS(output) over all cases, next to cuBLAS's worst on the same cases (independent maxima, not a per-case comparison): v1 2.06e-03, v2 2.06e-03, v3 2.48e-03, v4 2.48e-03, v5 2.36e-03, v6 2.53e-03, cublas 3.43e-03.

Overflow regression (q = 7, scale = 2^-10, x = +-4096/8192, exact output 3584 / -7168): every kernel finite and exact. Dispatch matrix (`linear()` on the mma-only copy a model keeps, G = 32/64/256, M = 1..130 across every boundary of the auto rule): worst error 2.61e-03; bad inputs refused.

### Decode (M = 1): median latency, microseconds

| layer | K x N | FP16 cuBLAS | v1 | v2 | v3 | v4 [R] | v6 [R] | `linear()` (deployed: v6) | INT4 roofline | `linear()` vs FP16 cuBLAS | v4 eff. % of BW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | 896 x 896 | 16.3 (16.2-16.4) | 73.8 | 11.1 | 8.8 | 7.5 [4] | 8.2 [4] | 8.2 (8.2-8.3) | 1.5 | 1.98x | 20% |
| qwen0.5b.kv_proj | 896 x 128 | 8.5 (8.5-8.8) | 53.4 | 5.4 | 5.6 | 5.0 [1] | 5.6 [1] | 6.2 (5.3-6.3) | 0.2 | 1.37x | 5% |
| qwen0.5b.gate_up | 896 x 4864 | 56.6 (56.6-56.7) | 115.9 | 34.0 | 30.1 | 19.7 [4] | 19.7 [4] | 19.8 (19.7-19.9) | 8.3 | 2.86x | 42% |
| qwen0.5b.down | 4864 x 896 | 49.4 (49.3-49.7) | 364.1 | 40.3 | 31.9 | 22.1 [4] | 23.7 [4] | 23.6 (22.6-23.6) | 8.3 | 2.09x | 37% |
| qwen1.5b.q_proj | 1536 x 1536 | 32.6 (32.6-32.8) | 147.1 | 26.8 | 22.4 | 14.5 [4] | 15.6 [4] | 15.4 (15.4-15.6) | 4.5 | 2.11x | 31% |
| qwen1.5b.gate_up | 1536 x 8960 | 152.3 (152.2-152.5) | 337.9 | 90.2 | 78.2 | 46.4 [4] | 54.2 [4] | 57.3 (55.4-57.6) | 26.0 | 2.66x | 56% |
| qwen1.5b.down | 8960 x 1536 | 124.1 (124.0-124.5) | 783.0 | 96.3 | 83.6 | 43.5 [4] | 56.6 [4] | 55.3 (54.0-55.4) | 26.0 | 2.24x | 60% |
| qwen3b.q_proj | 2048 x 2048 | 50.1 (50.0-50.5) | 170.8 | 38.8 | 27.9 | 16.7 [4] | 21.3 [4] | 21.8 (21.8-21.9) | 7.9 | 2.30x | 48% |
| qwen3b.gate_up | 2048 x 11008 | 293.4 (238.8-317.0) | 616.0 | 131.8 | 116.6 | 60.3 [4] | 80.5 [4] | 81.9 (78.6-84.5) | 42.6 | 3.58x | 71% |
| qwen3b.down | 11008 x 2048 | 189.9 (188.5-195.0) | 876.3 | 126.7 | 114.4 | 65.8 [4] | 83.4 [4] | 90.0 (81.0-95.5) | 42.6 | 2.11x | 65% |
| 7b-class.gate_up | 3584 x 18944 | 819.3 (628.6-828.6) | 1744.5 | 293.6 | 273.9 | 173.2 [4] | 212.2 [4] | 213.7 (213.5-221.9) | 128.3 | 3.83x | 74% |
| 7b-class.down | 18944 x 3584 | 735.2 (591.0-747.8) | 2426.8 | 282.4 | 267.3 | 170.3 [4] | 212.9 [4] | 212.3 (205.6-228.2) | 128.3 | 3.46x | 75% |

[R] = rows per warp the kernel picked from N.

### Deployed `linear()` vs FP16 cuBLAS by batch size

Each cell: `linear()` time / FP16 cuBLAS time (path: v6, v5 or dequantize + cuBLAS, as the auto rule picks it on the mma-only copy). Bold: `linear()` is faster.

| layer | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 | M=65 | M=96 | M=128 | M=192 | M=256 | M=384 | M=512 | M=1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | **0.51x** (v6) | **0.56x** (v5) | **0.59x** (v5) | **0.66x** (v5) | **0.89x** (v5) | 1.25x (v5) | 1.16x (dq) | 1.31x (dq) | 1.19x (dq) | 1.03x (dq) | **0.92x** (dq) | **0.93x** (dq) | **0.82x** (dq) | **0.81x** (dq) | **0.90x** (dq) |
| qwen0.5b.kv_proj | **0.73x** (v6) | **0.43x** (v5) | **0.55x** (v5) | **0.49x** (v5) | **0.60x** (v5) | **0.71x** (v5) | **0.86x** (dq) | **0.73x** (dq) | **0.78x** (dq) | **0.74x** (dq) | **0.71x** (dq) | **0.72x** (dq) | **0.62x** (dq) | **0.72x** (dq) | **0.64x** (dq) |
| qwen0.5b.gate_up | **0.35x** (v6) | **0.32x** (v5) | **0.37x** (v5) | **0.45x** (v5) | **0.64x** (v5) | 1.04x (v5) | 1.68x (v5) | 1.71x (dq) | 1.71x (dq) | 1.66x (dq) | 1.37x (dq) | 1.29x (dq) | 1.22x (dq) | 1.20x (dq) | 1.11x (dq) |
| qwen0.5b.down | **0.48x** (v6) | **0.55x** (v5) | **0.53x** (v5) | **0.67x** (v5) | **0.97x** (v5) | 1.39x (v5) | 1.57x (dq) | 1.53x (dq) | 1.69x (dq) | 1.42x (dq) | 1.27x (dq) | 1.23x (dq) | 1.19x (dq) | 1.19x (dq) | 1.06x (dq) |
| qwen1.5b.q_proj | **0.47x** (v6) | **0.54x** (v5) | **0.57x** (v5) | **0.73x** (v5) | **0.93x** (v5) | 1.25x (v5) | 1.15x (dq) | 1.18x (dq) | 1.13x (dq) | 1.16x (dq) | 1.22x (dq) | 1.13x (dq) | 1.18x (dq) | 1.13x (dq) | 1.06x (dq) |
| qwen1.5b.gate_up | **0.38x** (v6) | **0.32x** (v5) | **0.42x** (v5) | **0.40x** (v5) | **0.46x** (v5) | **0.72x** (v5) | 1.50x (v5) | 1.71x (dq) | 1.65x (dq) | 1.57x (dq) | 1.36x (dq) | 1.35x (dq) | 1.27x (dq) | 1.18x (dq) | 1.11x (dq) |
| qwen1.5b.down | **0.45x** (v6) | **0.45x** (v5) | **0.54x** (v5) | **0.73x** (v5) | 1.02x (v5) | 1.38x (v5) | 1.83x (dq) | 1.63x (dq) | 1.62x (dq) | 1.53x (dq) | 1.34x (dq) | 1.34x (dq) | 1.23x (dq) | 1.19x (dq) | 1.09x (dq) |
| qwen3b.q_proj | **0.44x** (v6) | **0.30x** (v5) | **0.44x** (v5) | **0.48x** (v5) | **0.84x** (v5) | 1.31x (v5) | 1.58x (dq) | 1.45x (dq) | 1.45x (dq) | 1.44x (dq) | 1.31x (dq) | 1.26x (dq) | 1.23x (dq) | 1.22x (dq) | 1.10x (dq) |
| qwen3b.gate_up | **0.28x** (v6) | **0.35x** (v5) | **0.47x** (v5) | **0.46x** (v5) | **0.54x** (v5) | **0.65x** (v5) | 1.34x (v5) | 1.65x (dq) | 1.58x (dq) | 1.49x (dq) | 1.39x (dq) | 1.35x (dq) | 1.20x (dq) | 1.18x (dq) | 1.12x (dq) |
| qwen3b.down | **0.47x** (v6) | **0.41x** (v5) | **0.52x** (v5) | **0.66x** (v5) | **0.89x** (v5) | 1.46x (v5) | 1.83x (dq) | 1.64x (dq) | 1.68x (dq) | 1.51x (dq) | 1.38x (dq) | 1.25x (dq) | 1.23x (dq) | 1.14x (dq) | 1.09x (dq) |
| 7b-class.gate_up | **0.26x** (v6) | **0.52x** (v5) | **0.58x** (v5) | **0.45x** (v5) | **0.60x** (v5) | **0.66x** (v5) | 1.33x (v5) | 1.65x (dq) | 1.58x (dq) | 1.54x (dq) | 1.36x (dq) | 1.36x (dq) | 1.26x (dq) | 1.17x (dq) | 1.11x (dq) |
| 7b-class.down | **0.29x** (v6) | **0.53x** (v5) | **0.60x** (v5) | **0.58x** (v5) | **0.73x** (v5) | **0.89x** (v5) | 1.31x (v5) | 1.66x (dq) | 1.60x (dq) | 1.53x (dq) | 1.41x (dq) | 1.29x (dq) | 1.25x (dq) | 1.18x (dq) | 1.11x (dq) |

### Every kernel variant (oracle: the fastest of all variants measured, not what `linear()` runs)

Each cell: fastest variant time (which) / FP16 cuBLAS time, microseconds. v5_tN fix the row tiles; v4_rN fix rows per warp.

| layer | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 |
|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | 7.1 (v4_r2) / 16.3 | 8.5 (v5_t1) / 15.2 | 9.2 (v5_t1) / 15.2 | 9.9 (v5_t2) / 16.2 | 12.9 (v5_t2) / 15.9 | 18.3 (v5_t2) / 17.5 | 32.9 (v5_t2) / 33.8 |
| qwen0.5b.kv_proj | 4.7 (v4_r2) / 8.5 | 5.8 (v4_r2) / 13.9 | 6.2 (v5_t1) / 14.3 | 6.9 (v5_t1) / 14.1 | 7.9 (v5_t1) / 14.3 | 9.9 (v5) / 14.6 | 14.3 (v5_t1) / 15.5 |
| qwen0.5b.gate_up | 15.4 (v5_t1) / 56.6 | 16.7 (v5_t1) / 54.4 | 19.6 (v5_t1) / 52.5 | 21.7 (v5_t4) / 54.1 | 26.2 (v5_t4) / 57.9 | 44.9 (v5_t2) / 61.2 | 101.3 (v5) / 63.8 |
| qwen0.5b.down | 19.8 (v4_r2) / 49.4 | 23.2 (v4_r4) / 53.1 | 30.0 (v5_t2) / 59.9 | 35.2 (v5_t2) / 58.8 | 49.8 (v5_t2) / 68.4 | 83.4 (v5_t2) / 74.5 | 125.4 (v5_t2) / 85.0 |
| qwen1.5b.q_proj | 13.7 (v5_t1) / 32.6 | 15.3 (v5_t2) / 28.9 | 15.9 (v5) / 29.9 | 17.9 (v5_t2) / 31.0 | 25.9 (v5_t4) / 35.8 | 42.9 (v5_t4) / 53.2 | 94.2 (v5_t4) / 69.8 |
| qwen1.5b.gate_up | 38.8 (v5) / 152.3 | 41.5 (v5_t1) / 131.6 | 48.1 (v5_t2) / 133.3 | 54.0 (v5_t4) / 136.8 | 73.6 (v5_t4) / 162.5 | 120.1 (v5_t4) / 162.3 | 254.4 (v5) / 170.1 |
| qwen1.5b.down | 43.5 (v4) / 124.1 | 56.7 (v5_t1) / 127.9 | 65.9 (v5) / 130.2 | 81.5 (v5_t4) / 138.5 | 105.0 (v5_t4) / 145.0 | 147.7 (v5_t4) / 184.9 | 305.5 (v5_t4) / 201.9 |
| qwen3b.q_proj | 16.7 (v4) / 50.1 | 20.2 (v5) / 60.1 | 22.0 (v5_t2) / 53.9 | 24.5 (v5_t2) / 63.0 | 32.4 (v5_t4) / 71.6 | 54.3 (v5_t4) / 70.8 | 110.1 (v5_t4) / 77.3 |
| qwen3b.gate_up | 58.8 (v4_r4) / 293.4 | 68.6 (v5) / 200.3 | 89.5 (v5_t2) / 206.3 | 95.0 (v5_t4) / 212.3 | 117.1 (v5) / 222.0 | 185.4 (v5_t4) / 288.2 | 411.6 (v5_t2) / 308.2 |
| qwen3b.down | 62.1 (v4_r4) / 189.9 | 83.2 (v5_t1) / 210.4 | 89.0 (v5_t2) / 213.4 | 106.7 (v5_t4) / 222.4 | 132.5 (v5_t4) / 245.0 | 192.6 (v5_t4) / 270.1 | 449.9 (v5_t2) / 309.4 |
| 7b-class.gate_up | 172.8 (v4_r4) / 819.3 | 225.1 (v5_t4) / 586.2 | 241.5 (v5_t4) / 595.8 | 289.3 (v5_t4) / 633.7 | 408.4 (v5_t4) / 706.3 | 564.0 (v5_t4) / 873.9 | 1209.8 (v5_t4) / 909.3 |
| 7b-class.down | 170.3 (v4) / 735.2 | 232.5 (v5_t2) / 585.6 | 270.7 (v5_t4) / 596.7 | 305.1 (v5_t4) / 637.8 | 420.1 (v5_t4) / 694.9 | 573.7 (v5_t4) / 858.7 | 1250.5 (v5) / 955.1 |

### Prefill (M > 64): dequantize (model's mma copy) + cuBLAS / FP16 cuBLAS; canonical-copy dequantize in parentheses; 64-row v5 slices in brackets

| layer | M=65 | M=96 | M=128 | M=192 | M=256 | M=384 | M=512 | M=1024 |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b.q_proj | 1.31x (1.28x) [1.55x] | 1.11x (1.06x) [1.55x] | 1.09x (1.06x) [1.87x] | 0.85x (0.97x) [1.91x] | 0.95x (0.95x) [2.24x] | 0.88x (0.83x) [2.12x] | 0.87x (0.89x) [2.58x] | 0.92x (0.91x) |
| qwen0.5b.kv_proj | 0.74x (0.72x) [1.06x] | 0.77x (0.78x) [1.28x] | 0.74x (0.72x) [1.43x] | 0.72x (0.72x) [1.84x] | 0.73x (0.72x) [2.27x] | 0.62x (0.62x) [2.37x] | 0.72x (0.72x) [3.44x] | 0.63x (0.63x) |
| qwen0.5b.gate_up | 1.69x (1.70x) [1.62x] | 1.69x (1.67x) [2.00x] | 1.66x (1.61x) [2.36x] | 1.37x (1.33x) [2.03x] | 1.29x (1.28x) [2.37x] | 1.22x (1.22x) [2.42x] | 1.22x (1.19x) [2.82x] | 1.10x (1.09x) |
| qwen0.5b.down | 1.50x (1.50x) [2.11x] | 1.75x (1.66x) [2.69x] | 1.42x (1.39x) [2.72x] | 1.28x (1.25x) [2.66x] | 1.26x (1.24x) [3.21x] | 1.15x (1.20x) [3.56x] | 1.18x (1.18x) [4.05x] | 1.06x (1.07x) |
| qwen1.5b.q_proj | 1.22x (1.19x) [1.45x] | 1.21x (1.18x) [1.78x] | 1.19x (1.21x) [2.09x] | 1.19x (1.19x) [2.61x] | 1.17x (1.14x) [2.56x] | 1.17x (1.17x) [3.61x] | 1.12x (1.13x) [3.36x] | 1.05x (1.05x) |
| qwen1.5b.gate_up | 1.70x (1.64x) [1.42x] | 1.63x (1.59x) [1.57x] | 1.57x (1.52x) [2.08x] | 1.36x (1.32x) [2.00x] | 1.35x (1.33x) [2.48x] | 1.29x (1.25x) [2.67x] | 1.18x (1.17x) [2.48x] | 1.11x (1.10x) |
| qwen1.5b.down | 1.62x (1.60x) [2.01x] | 1.63x (1.57x) [2.73x] | 1.53x (1.50x) [3.25x] | 1.34x (1.31x) [3.26x] | 1.34x (1.30x) [3.97x] | 1.24x (1.21x) [4.36x] | 1.19x (1.17x) [4.53x] | 1.09x (1.09x) |
| qwen3b.q_proj | 1.47x (1.46x) [1.93x] | 1.51x (1.44x) [2.32x] | 1.47x (1.43x) [2.77x] | 1.31x (1.30x) [2.85x] | 1.24x (1.23x) [3.06x] | 1.20x (1.20x) [3.61x] | 1.20x (1.21x) [4.57x] | 1.08x (1.09x) |
| qwen3b.gate_up | 1.64x (1.59x) [1.29x] | 1.58x (1.52x) [1.43x] | 1.51x (1.47x) [1.82x] | 1.40x (1.39x) [2.00x] | 1.35x (1.32x) [2.20x] | 1.21x (1.19x) [2.37x] | 1.18x (1.16x) [2.55x] | 1.11x (1.10x) |
| qwen3b.down | 1.64x (1.57x) [2.23x] | 1.67x (1.64x) [2.93x] | 1.51x (1.49x) [3.28x] | 1.39x (1.36x) [3.39x] | 1.25x (1.24x) [3.44x] | 1.23x (1.22x) [4.34x] | 1.15x (1.14x) [3.76x] | 1.09x (1.09x) |
| 7b-class.gate_up | 1.66x (1.60x) [1.37x] | 1.58x (1.54x) [1.53x] | 1.54x (1.49x) [1.98x] | 1.36x (1.34x) [1.95x] | 1.37x (1.33x) [2.47x] | 1.25x (1.23x) [2.63x] | 1.17x (1.16x) [2.44x] | 1.11x (1.11x) |
| 7b-class.down | 1.65x (1.61x) [1.37x] | 1.59x (1.54x) [1.67x] | 1.53x (1.48x) [1.93x] | 1.41x (1.38x) [2.05x] | 1.29x (1.29x) [2.03x] | 1.25x (1.23x) [2.55x] | 1.19x (1.18x) [2.66x] | 1.11x (1.10x) |

### DRAM counters (Nsight Compute)

Not available on this VM: ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.

## A2: end to end (Hugging Face, batch 1, greedy)

Environment: Tesla T4 (sm_75, 40 SMs), torch 2.11.0+cu128, CUDA 12.8, transformers 5.16.1, datasets 4.8.5, commit `5936374-dirty`, source sha256 `5a9e81dbec0b9a9f`, 2026-10-19T00:24:16Z

Models: Qwen2.5-0.5B-Instruct @ `7ae557604a`, Qwen2.5-1.5B-Instruct @ `989aa7980e`, Qwen2.5-3B-Instruct @ `aa8e725379`; WikiText-2 fingerprint `a46124b21ac53738`.

### Quality

WikiText-2 perplexity (40 windows x 1024 tokens; these long inputs run the prefill path, dequantize + cuBLAS). Fake-quant = FP16 layers holding exactly the dequantized INT4 weights: its gap to FP16 is the cost of 4-bit weights.

| model | scales | weights FP16 -> INT4 | PPL FP16 | PPL fake-quant | PPL INT4 | next-token agreement with FP16 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | absmax | 0.92 -> 0.43 GiB | 15.789 | 22.168 | 22.166 | 70.6% |
| Qwen2.5-0.5B-Instruct | clip search | 0.92 -> 0.43 GiB | 15.789 | 21.912 | 21.912 | 70.8% |
| Qwen2.5-1.5B-Instruct | absmax | 2.88 -> 1.06 GiB | 10.654 | 12.743 | 12.744 | 79.5% |
| Qwen2.5-1.5B-Instruct | clip search | 2.88 -> 1.06 GiB | 10.654 | 12.391 | 12.392 | 80.2% |
| Qwen2.5-3B-Instruct | absmax | 5.75 -> 1.91 GiB | 9.556 | 15.196 | 15.195 | 72.4% |
| Qwen2.5-3B-Instruct | clip search | 5.75 -> 1.91 GiB | 9.556 | 10.833 | 10.832 | 79.8% |

Decode kernels vs fake-quant on identical inputs: the same 512 WikiText tokens fed through the KV cache 1 token per step (INT4: v6), 16 per step (v5) and 48 per step (v5 on layers with N >= 3072, dequantize + cuBLAS on the rest). Logit difference over the first 32 positions, relative to the logits' RMS.

| model | scales | tokens/step | PPL fake-quant | PPL INT4 | argmax agreement | worst per-token NLL diff | worst logit diff / RMS |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | absmax | 1 | 17.873 | 17.877 | 100.00% | 0.1171 | 7.67e-02 |
| Qwen2.5-0.5B-Instruct | absmax | 16 | 17.876 | 17.880 | 99.41% | 0.1843 | 1.13e-01 |
| Qwen2.5-0.5B-Instruct | absmax | 48 | 17.879 | 17.877 | 99.22% | 0.0330 | 2.16e-02 |
| Qwen2.5-0.5B-Instruct | clip search | 1 | 16.159 | 16.156 | 99.41% | 0.0390 | 4.23e-02 |
| Qwen2.5-0.5B-Instruct | clip search | 16 | 16.155 | 16.152 | 99.41% | 0.0295 | 3.44e-02 |
| Qwen2.5-0.5B-Instruct | clip search | 48 | 16.147 | 16.158 | 99.61% | 0.0382 | 4.46e-02 |
| Qwen2.5-1.5B-Instruct | absmax | 1 | 10.894 | 10.894 | 99.22% | 0.0449 | 2.45e-02 |
| Qwen2.5-1.5B-Instruct | absmax | 16 | 10.891 | 10.892 | 99.22% | 0.0382 | 1.82e-02 |
| Qwen2.5-1.5B-Instruct | absmax | 48 | 10.890 | 10.891 | 100.00% | 0.0427 | 2.22e-02 |
| Qwen2.5-1.5B-Instruct | clip search | 1 | 10.647 | 10.641 | 99.61% | 0.1531 | 8.13e-02 |
| Qwen2.5-1.5B-Instruct | clip search | 16 | 10.643 | 10.642 | 99.41% | 0.1602 | 9.15e-02 |
| Qwen2.5-1.5B-Instruct | clip search | 48 | 10.639 | 10.641 | 99.41% | 0.0833 | 3.43e-02 |
| Qwen2.5-3B-Instruct | absmax | 1 | 14.059 | 14.059 | 99.80% | 0.0872 | 3.10e-02 |
| Qwen2.5-3B-Instruct | absmax | 16 | 14.057 | 14.060 | 99.41% | 0.0442 | 5.69e-02 |
| Qwen2.5-3B-Instruct | absmax | 48 | 14.067 | 14.064 | 99.80% | 0.1205 | 8.17e-02 |
| Qwen2.5-3B-Instruct | clip search | 1 | 9.811 | 9.814 | 99.61% | 0.0494 | 2.27e-02 |
| Qwen2.5-3B-Instruct | clip search | 16 | 9.809 | 9.808 | 99.80% | 0.0481 | 2.27e-02 |
| Qwen2.5-3B-Instruct | clip search | 48 | 9.809 | 9.808 | 99.80% | 0.0338 | 1.87e-02 |

### Speed

Decode tokens/s and TTFT (ms) on a 65-token chat prompt, 256 new tokens. Eager = plain `generate`; CUDA graph = each decode step replayed as one graph over a static KV cache.

| model | scales | eager tok/s FP16 -> INT4 | CUDA graph tok/s FP16 -> INT4 | TTFT ms FP16 -> INT4 | peak mem GB | graph vs eager on the same tokens: argmax agreement, worst logit diff / RMS |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | absmax | 34.3 -> 29.3 (0.85x) | 110.7 -> 146.9 (1.33x) | 35 -> 49 | 2.56 -> 2.04 | 100.0% of 128, 1.8e-02 |
| Qwen2.5-0.5B-Instruct | clip search | 35.4 -> 29.6 (0.84x) | 104.8 -> 143.6 (1.37x) | 35 -> 48 | 2.56 -> 2.04 | 99.2% of 128, 2.8e-02 |
| Qwen2.5-1.5B-Instruct | absmax | 29.9 -> 25.2 (0.84x) | 50.5 -> 85.9 (1.70x) | 41 -> 57 | 4.65 -> 2.76 | 100.0% of 128, 2.0e-02 |
| Qwen2.5-1.5B-Instruct | clip search | 29.9 -> 25.2 (0.84x) | 50.7 -> 85.9 (1.69x) | 41 -> 61 | 4.65 -> 2.76 | 100.0% of 128, 2.4e-02 |
| Qwen2.5-3B-Instruct | absmax | 23.4 -> 19.9 (0.85x) | 28.7 -> 57.5 (2.00x) | 55 -> 91 | 7.85 -> 3.64 | 100.0% of 128, 7.8e-02 |
| Qwen2.5-3B-Instruct | clip search | 23.4 -> 20.0 (0.85x) | 28.7 -> 57.6 (2.01x) | 55 -> 91 | 7.85 -> 3.64 | 99.2% of 128, 1.6e-02 |

Where eager decode time goes (decode only, after prefill). Graph / eager = CUDA-graph step time / eager step time: the graph step is about the GPU's own work, so the rest of an eager step is CPU launch time. Kernel shares come from the profiler and are shown only where its kernel time covers at least 90% of the graph step (it can drop records on long windows).

| model | scales | run | eager ms/step | graph ms/step | graph / eager | profiler coverage | INT4 kernels | dense GEMM | dequantize | other |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | absmax | fp16 | 27.7 | 9.0 | 33% | 69% | incomplete | | | |
| Qwen2.5-0.5B-Instruct | absmax | int4 | 32.8 | 6.8 | 21% | 103% | 30% | 2% | 0% | 68% |
| Qwen2.5-0.5B-Instruct | clip search | fp16 | 26.3 | 9.5 | 36% | 64% | incomplete | | | |
| Qwen2.5-0.5B-Instruct | clip search | int4 | 31.4 | 7.0 | 22% | 97% | 30% | 2% | 0% | 68% |
| Qwen2.5-1.5B-Instruct | absmax | fp16 | 31.1 | 19.8 | 64% | 54% | incomplete | | | |
| Qwen2.5-1.5B-Instruct | absmax | int4 | 36.8 | 11.6 | 32% | 62% | incomplete | | | |
| Qwen2.5-1.5B-Instruct | clip search | fp16 | 31.1 | 19.7 | 63% | 54% | incomplete | | | |
| Qwen2.5-1.5B-Instruct | clip search | int4 | 36.4 | 11.6 | 32% | 63% | incomplete | | | |
| Qwen2.5-3B-Instruct | absmax | fp16 | 39.8 | 34.9 | 87% | 16% | incomplete | | | |
| Qwen2.5-3B-Instruct | absmax | int4 | 47.2 | 17.4 | 37% | 67% | incomplete | | | |
| Qwen2.5-3B-Instruct | clip search | fp16 | 40.1 | 34.9 | 87% | 16% | incomplete | | | |
| Qwen2.5-3B-Instruct | clip search | int4 | 47.6 | 17.4 | 37% | 68% | incomplete | | | |
