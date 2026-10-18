// Weight-only INT4 GEMV/skinny-GEMM for LLM decode: y[M, N] = x[M, K] (fp16) * dequant(W)[N, K]^T, M <= 64.
// Format: lowbit/quant.py (symmetric, group size G along K, fp16 scale per group, two nibbles per byte).
//
// Six versions, kept side by side so each optimization step is measured on its own:
//   v1  one thread per output row, byte loads, fp32 math              (the "obvious" kernel)
//   v2  one warp per output row, 128-bit loads (32 weights per lane per step), fp32 math, shuffle reduction
//   v3  v2 + offline-interleaved nibbles and the 0x6400 magic-number trick: a mask and an OR turn eight
//       nibbles into four fp16 pairs, scaled to the fp16 weights the dequantize kernel writes, multiplied with
//       HFMA2 against consecutive activations
//   v4  v3 + R output rows per warp: each activation load is reused for R rows (v2/v3 re-read x from L1 for
//       every row, M times the weight bytes, which made time grow with M), and R weight loads are in flight
//       per step instead of one; R is chosen from N so the grid still fills the GPU
//   v5  tensor cores (mma.sync m16n8k8, fp32 accumulate; sm_75+): weights are the 16-row A operand and the M
//       activation rows the 8-column B operand, so a decode step wastes no tensor-core rows on padding. Each
//       lane decodes its A fragment in registers (interleave_for_mma layout, magic-number trick, x scale ->
//       the same fp16 weights the dequantize kernel writes), one weight fetch serves up to 8 tiles of 8
//       activation rows (M <= 64), and the 4 warps of a block split K and reduce through shared memory; each warp
//       takes RT 16-row tiles so one activation fragment feeds RT mma instructions
//   v6  v4's schedule reading v5's layout: a model then keeps one weight copy for both decode kernels (v4/v5
//       layouts differ, and two copies would undo half the memory saving)
// plus dequantize kernels (canonical or v5 layout -> fp16 W) for the prefill path, where cuBLAS wins.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int M>
__global__ void gemv_v1(const half* __restrict__ x, const uint8_t* __restrict__ w, const half* __restrict__ s,
                        half* __restrict__ y, int N, int K, int G) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const uint8_t* wrow = w + (size_t)n * (K / 2);
  const half* srow = s + (size_t)n * (K / G);
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int j = 0; j < K / 2; ++j) {
    uint8_t b = wrow[j];
    float sc = __half2float(srow[(2 * j) / G]);
    float w0 = (float)((int)(b & 0xF) - 8) * sc, w1 = (float)((int)(b >> 4) - 8) * sc;
#pragma unroll
    for (int m = 0; m < M; ++m)
      acc[m] += w0 * __half2float(x[(size_t)m * K + 2 * j]) + w1 * __half2float(x[(size_t)m * K + 2 * j + 1]);
  }
#pragma unroll
  for (int m = 0; m < M; ++m) y[(size_t)m * N + n] = __float2half(acc[m]);
}

template <int M>
__global__ void gemv_v2(const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ s,
                        half* __restrict__ y, int N, int K, int G) {
  int lane = threadIdx.x & 31, n = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint4* wrow = w + (size_t)n * (K / 32);           // K/2 bytes per row = K/32 uint4, 32 weights each
  const half* srow = s + (size_t)n * (K / G);
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int c = lane; c < K / 32; c += 32) {
    uint4 wv = __ldg(wrow + c);
    const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
    float sc = __half2float(srow[(c * 32) / G]);          // G is a multiple of 32: one scale per chunk
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * K + c * 32);
      float part = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {                         // word i holds k0+8i .. k0+8i+7, nibble b = k0+8i+b
        uint4 xv = __ldg(xp + i);
        const half* xh = reinterpret_cast<const half*>(&xv);
#pragma unroll
        for (int b = 0; b < 8; ++b)
          part += (float)((int)((words[i] >> (4 * b)) & 0xF) - 8) * __half2float(xh[b]);
      }
      acc[m] += part * sc;
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = warp_sum(acc[m]);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(v);
  }
}

template <int M>
__global__ void gemv_v3(const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ s,
                        half* __restrict__ y, int N, int K, int G) {
  int lane = threadIdx.x & 31, n = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint4* wrow = w + (size_t)n * (K / 32);
  const half* srow = s + (size_t)n * (K / G);
  // 0x6400 is 1024.0 in fp16 with a mantissa step of exactly 1, so (0x6400 | u) == 1024 + u for a nibble u;
  // subtracting 1032 = 1024 + 8 leaves the signed value u - 8 with no integer-to-float conversion.
  const half2 bias = __float2half2_rn(1032.f);
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int c = lane; c < K / 32; c += 32) {
    uint4 wv = __ldg(wrow + c);
    const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
    const half2 sc = __half2half2(srow[(c * 32) / G]);
    half2 wh[16];                                           // wh[4i+j] = (W[k0+8i+2j], W[k0+8i+2j+1])
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t t = ((words[i] >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
        wh[4 * i + j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&t), bias), sc);
      }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * K + c * 32);
      float part = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint4 xv = __ldg(xp + i);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        // 8 products per fp16 partial before widening to fp32. The weights are scaled first (q * scale, rounded
        // once to fp16: bit-identical to the dequantized weights), so a partial holds 8 true terms of the output
        // and overflows only where 8 terms of an fp16 GEMM would; scaling after the sum let |x| > ~2000 overflow
        // a 7 * x partial to inf on a finite output
        half2 p = __hmul2(wh[4 * i], xh[0]);
        p = __hfma2(wh[4 * i + 1], xh[1], p);
        p = __hfma2(wh[4 * i + 2], xh[2], p);
        p = __hfma2(wh[4 * i + 3], xh[3], p);
        float2 f = __half22float2(p);
        part += f.x + f.y;
      }
      acc[m] += part;
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = warp_sum(acc[m]);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(v);
  }
}

template <int M, int R>
__global__ void gemv_v4(const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ s,
                        half* __restrict__ y, int N, int K, int G) {
  int lane = threadIdx.x & 31, n0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * R;
  if (n0 >= N) return;
  const int KC = K / 32, KG = K / G;
  const half2 bias = __float2half2_rn(1032.f);
  const uint4* xv4 = reinterpret_cast<const uint4*>(x);
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int c = lane; c < KC; c += 32) {
    uint4 wv[R];
    half2 sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {                           // rows past N re-read row N-1; their sums are dropped
      int n = min(n0 + r, N - 1);
      wv[r] = __ldg(w + (size_t)n * KC + c);
      sc[r] = __half2half2(s[(size_t)n * KG + (c * 32) / G]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half2 wh[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        uint32_t word = (&wv[r].x)[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t t = ((word >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
          wh[r][j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&t), bias), sc[r]);   // scaled first, as in v3
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        uint4 xq = __ldg(xv4 + (size_t)m * (K / 8) + c * 4 + i);   // loaded once, used for all R rows
        const half2* xh = reinterpret_cast<const half2*>(&xq);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          half2 p = __hmul2(wh[r][0], xh[0]);
          p = __hfma2(wh[r][1], xh[1], p);
          p = __hfma2(wh[r][2], xh[2], p);
          p = __hfma2(wh[r][3], xh[3], p);
          float2 f = __half22float2(p);
          acc[r][m] += f.x + f.y;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = warp_sum(acc[r][m]);
      if (lane == 0 && n0 + r < N) y[(size_t)m * N + n0 + r] = __float2half(v);
    }
}

__device__ __forceinline__ void mma_16x8x8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b));
}

// y[M, N] for M <= 8 * MT. Block: 16 * RT output rows; kWarpsPerBlock warps split the 128-wide K chunks, and each
// warp decodes RT 16-row weight tiles per k-step so every activation fragment it loads feeds RT mma instructions
// (X is re-read once per block, so RT divides that traffic; it also puts 2 * RT weight loads in flight per lane).
// Fragment ownership (PTX ISA, mma.m16n8k8 .f16): lane = 4g + t; A: a0 = W[row g][k 2t, 2t+1], a1 = row g + 8;
// B: X[m = g][k 2t, 2t+1]; D: rows g / g + 8, columns m = 2t, 2t + 1.
template <int MT, int RT>
__global__ void gemm_v5(const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ s,
                        half* __restrict__ y, int M, int N, int K, int G) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int n0 = blockIdx.x * 16 * RT;
  const int KC = K / 128, KG = K / G;
  const half2 bias = __float2half2_rn(1032.f);
  int row[RT][2];
  const uint4* wr[RT][2];                                       // a row is K/32 uint4; a chunk is 4 of them
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      row[r][h] = min(n0 + 16 * r + g + 8 * h, N - 1);           // tail rows re-read row N-1, never written
      wr[r][h] = w + (size_t)row[r][h] * (K / 32) + t;
    }
  float acc[RT][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][mt][i] = 0.f;
  for (int c = warp; c < KC; c += kWarpsPerBlock) {
    uint4 q[RT][2];
    half2 sc[RT][2];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        q[r][h] = __ldg(wr[r][h] + 4 * c);
        sc[r][h] = __half2half2(s[(size_t)row[r][h] * KG + (c * 128) / G]);
      }
#pragma unroll
    for (int st = 0; st < 16; ++st) {                           // k-step: k = 128c + 8st .. +7
      uint32_t a[RT][2];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t word = (&q[r][h].x)[st >> 2];
          uint32_t tw = ((word >> (4 * (st & 3))) & 0x000F000Fu) | 0x64006400u;
          half2 hv = __hmul2(__hsub2(*reinterpret_cast<half2*>(&tw), bias), sc[r][h]);
          a[r][h] = *reinterpret_cast<uint32_t*>(&hv);
        }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        int m = 8 * mt + g;
        uint32_t b = 0;
        if (m < M) b = __ldg(reinterpret_cast<const uint32_t*>(x + (size_t)m * K + 128 * c + 8 * st + 2 * t));
#pragma unroll
        for (int r = 0; r < RT; ++r) mma_16x8x8(acc[r][mt], a[r][0], a[r][1], b);
      }
    }
  }
  __shared__ float red[kWarpsPerBlock][RT * MT][4][32];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][r * MT + mt][i][lane] = acc[r][mt][i];
  __syncthreads();
  for (int tile = warp; tile < RT * MT; tile += kWarpsPerBlock) {   // warps share the final sums and stores
    int r = tile / MT, mt = tile % MT;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = 0.f;
#pragma unroll
      for (int qw = 0; qw < kWarpsPerBlock; ++qw) v += red[qw][tile][i][lane];
      int n = n0 + 16 * r + g + (i >= 2 ? 8 : 0), m = 8 * mt + 2 * t + (i & 1);
      if (n < N && m < M) y[(size_t)m * N + n] = __float2half(v);
    }
  }
}

// four bytes (eight weights) per thread: one 32-bit load, one 128-bit store
__global__ void dequant_kernel(const uint32_t* __restrict__ w, const half* __restrict__ s, uint4* __restrict__ out,
                               int N, int K, int G) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, total = (size_t)N * (K / 8);
  if (idx >= total) return;
  int n = idx / (K / 8), k = 8 * (idx % (K / 8));
  float sc = __half2float(s[(size_t)n * (K / G) + k / G]);   // 8 | G, so all eight share a scale
  uint32_t b = w[idx];
  uint4 o;
  half2* h = reinterpret_cast<half2*>(&o);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    h[j] = __floats2half2_rn((float)((int)((b >> (8 * j)) & 0xF) - 8) * sc, (float)((int)((b >> (8 * j + 4)) & 0xF) - 8) * sc);
  out[idx] = o;
}

void check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, int64_t group) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda(), "tensors must be on the GPU");
  TORCH_CHECK(w.device() == x.device() && s.device() == x.device(), "x, w and scales must be on the same GPU");
  TORCH_CHECK(group > 0, "group size must be positive");
  TORCH_CHECK(x.scalar_type() == torch::kHalf && s.scalar_type() == torch::kHalf && w.scalar_type() == torch::kUInt8,
              "x and scales must be float16, packed weights uint8");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && s.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && s.dim() == 2, "x [M,K], w [N,K/2], scales [N,K/G]");
  int64_t K = x.size(1);
  TORCH_CHECK(group % 32 == 0 && K % group == 0, "K must be a multiple of the group size, itself a multiple of 32");
  TORCH_CHECK(w.size(1) * 2 == K && s.size(0) == w.size(0) && s.size(1) == K / group, "shape mismatch");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "x and w must be 16-byte aligned for the 128-bit loads");
}

// v6: v4 on the interleave_for_mma layout. Lane = 4cc + t reads uint4 t of 128-wide chunk c (c = cc, cc + 8, ...):
// word i, nibble pair j hold k = 128c + 32i + 8j + 2t and +1, so each pair meets one half2 of x. Needs K % 128 == 0.
template <int M, int R>
__global__ void gemv_v6(const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ s,
                        half* __restrict__ y, int N, int K, int G) {
  int lane = threadIdx.x & 31, cc = lane >> 2, t = lane & 3;
  int n0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * R;
  if (n0 >= N) return;
  const int KC = K / 128, KG = K / G;
  const half2 bias = __float2half2_rn(1032.f);
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int c = cc; c < KC; c += 8) {
    uint4 wv[R];
    half2 sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {                           // rows past N re-read row N-1; their sums are dropped
      int n = min(n0 + r, N - 1);
      wv[r] = __ldg(w + (size_t)n * (K / 32) + 4 * c + t);
      sc[r] = __half2half2(s[(size_t)n * KG + (c * 128) / G]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half2 wh[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        uint32_t word = (&wv[r].x)[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t u = ((word >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
          wh[r][j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&u), bias), sc[r]);   // scaled first, as in v3
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(x + (size_t)m * K + 128 * c + 32 * i + 2 * t);
        half2 xh[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {                       // loaded once, used for all R rows
          uint32_t u = __ldg(xr + 4 * j);
          xh[j] = *reinterpret_cast<half2*>(&u);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          half2 p = __hmul2(wh[r][0], xh[0]);
          p = __hfma2(wh[r][1], xh[1], p);
          p = __hfma2(wh[r][2], xh[2], p);
          p = __hfma2(wh[r][3], xh[3], p);
          float2 f = __half22float2(p);
          acc[r][m] += f.x + f.y;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = warp_sum(acc[r][m]);
      if (lane == 0 && n0 + r < N) y[(size_t)m * N + n0 + r] = __float2half(v);
    }
}

// v5-layout word q of a row: chunk q / 16, lane t = (q % 16) / 4, word i = q % 4 (see gemv_v6)
__global__ void dequant_mma_kernel(const uint32_t* __restrict__ w, const half* __restrict__ s, half* __restrict__ out,
                                   int N, int K, int G) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, total = (size_t)N * (K / 8);
  if (idx >= total) return;
  int n = idx / (K / 8), q = idx % (K / 8), c = q >> 4, t = (q >> 2) & 3, i = q & 3;
  half2 sc = __half2half2(s[(size_t)n * (K / G) + (c * 128) / G]);
  const half2 bias = __float2half2_rn(1032.f);
  uint32_t word = w[idx];
  half2* row = reinterpret_cast<half2*>(out + (size_t)n * K + 128 * c + 32 * i + 2 * t);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t u = ((word >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
    row[4 * j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&u), bias), sc);
  }
}

template <int M, int R>
void launch_v4(const half* x, const uint4* w, const half* s, half* y, int N, int K, int G, cudaStream_t st) {
  int rows = kWarpsPerBlock * R;
  gemv_v4<M, R><<<(N + rows - 1) / rows, 32 * kWarpsPerBlock, 0, st>>>(x, w, s, y, N, K, G);
}

template <int M, int R>
void launch_v6(const half* x, const uint4* w, const half* s, half* y, int N, int K, int G, cudaStream_t st) {
  int rows = kWarpsPerBlock * R;
  gemv_v6<M, R><<<(N + rows - 1) / rows, 32 * kWarpsPerBlock, 0, st>>>(x, w, s, y, N, K, G);
}

// measured on a T4 over Qwen2.5 0.5B-3B and 7B-class shapes, M = 1..8 (README, "choosing R"): four rows per warp
// is best or within noise everywhere except very narrow layers, where it leaves too few blocks to fill the GPU
int pick_rows(int N, int M) {
  if (N >= 512) return 4;
  return M >= 4 ? 2 : 1;
}

template <int M>
void launch(int version, int R, const half* x, const uint8_t* w, const half* s, half* y, int N, int K, int G,
            cudaStream_t st) {
  if (version == 4) {
    auto wv = reinterpret_cast<const uint4*>(w);
    if (R == 4) launch_v4<M, 4>(x, wv, s, y, N, K, G, st);
    else if (R == 2) launch_v4<M, 2>(x, wv, s, y, N, K, G, st);
    else launch_v4<M, 1>(x, wv, s, y, N, K, G, st);
  } else if (version == 6) {
    auto wv = reinterpret_cast<const uint4*>(w);
    if (R == 4) launch_v6<M, 4>(x, wv, s, y, N, K, G, st);
    else if (R == 2) launch_v6<M, 2>(x, wv, s, y, N, K, G, st);
    else launch_v6<M, 1>(x, wv, s, y, N, K, G, st);
  } else if (version == 1) {
    gemv_v1<M><<<(N + 127) / 128, 128, 0, st>>>(x, w, s, y, N, K, G);
  } else {
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock), block(32 * kWarpsPerBlock);
    auto wv = reinterpret_cast<const uint4*>(w);
    if (version == 2) gemv_v2<M><<<grid, block, 0, st>>>(x, wv, s, y, N, K, G);
    else gemv_v3<M><<<grid, block, 0, st>>>(x, wv, s, y, N, K, G);
  }
}

}  // namespace

// version 1/2 expect the canonical layout, 3/4 the interleave_for_gpu layout; rows_per_warp (v4): 0 = pick from N
torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, int64_t group, int64_t version,
                      int64_t rows_per_warp) {
  check(x, w, s, group);
  const c10::cuda::CUDAGuard guard(x.device());                  // launch on the tensors' GPU and its stream
  int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "the GEMV kernels take 1..8 rows; use the dequantize + matmul path above that");
  TORCH_CHECK((version >= 1 && version <= 4) || version == 6, "version is 1..4 or 6");
  TORCH_CHECK(version != 6 || (K % 128 == 0 && group % 128 == 0), "v6 needs K and the group size to be multiples of 128");
  TORCH_CHECK(rows_per_warp == 0 || rows_per_warp == 1 || rows_per_warp == 2 || rows_per_warp == 4,
              "rows_per_warp is 0 (auto), 1, 2 or 4");
  auto y = torch::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
  auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  auto wp = w.data_ptr<uint8_t>();
  int v = (int)version, G = (int)group, R = rows_per_warp ? (int)rows_per_warp : pick_rows(N, M);
  switch (M) {
    case 1: launch<1>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    case 2: launch<2>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    case 3: launch<3>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    case 4: launch<4>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    case 5: launch<5>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    case 6: launch<6>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    case 7: launch<7>(v, R, xp, wp, sp, yp, N, K, G, st); break;
    default: launch<8>(v, R, xp, wp, sp, yp, N, K, G, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// v5 (tensor cores): w in the interleave_for_mma layout, M <= 64
torch::Tensor w4_gemm_tc(torch::Tensor x, torch::Tensor w, torch::Tensor s, int64_t group, int64_t row_tiles) {
  check(x, w, s, group);
  const c10::cuda::CUDAGuard guard(x.device());
  int M = x.size(0), K = x.size(1), N = w.size(0), G = (int)group;
  TORCH_CHECK(M >= 1 && M <= 64, "the tensor-core kernel takes 1..64 rows");
  TORCH_CHECK(K % 128 == 0 && G % 128 == 0, "v5 needs K and the group size to be multiples of 128");
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major * 10 + prop->minor >= 75, "mma.m16n8k8 needs sm_75 or newer");
  auto y = torch::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const uint4*>(w.data_ptr<uint8_t>());
  auto sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
  auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  int MT = M <= 8 ? 1 : M <= 16 ? 2 : M <= 32 ? 4 : 8;
  // measured on a T4 (README, "v5 row tiles"): one tile while the weights dominate the traffic (M <= 4), four once
  // re-reading activations does; accumulators are 4 * RT * MT floats per lane, so MT = 8 stops at two tiles
  int RT = row_tiles ? (int)row_tiles : (M <= 4 ? 1 : 4);
  TORCH_CHECK(RT == 1 || RT == 2 || RT == 4, "row_tiles must be 1, 2 or 4");
  if (!row_tiles)                                                 // keep at least two blocks per SM
    while (RT > 1 && (N + 16 * RT - 1) / (16 * RT) < 2 * prop->multiProcessorCount) RT /= 2;
  if (MT == 8 && RT == 4) RT = 2;
  dim3 grid((N + 16 * RT - 1) / (16 * RT)), block(32 * kWarpsPerBlock);
#define V5(mt, rt) if (MT == mt && RT == rt) gemm_v5<mt, rt><<<grid, block, 0, st>>>(xp, wp, sp, yp, M, N, K, G)
  V5(1, 1); V5(1, 2); V5(1, 4); V5(2, 1); V5(2, 2); V5(2, 4); V5(4, 1); V5(4, 2); V5(4, 4); V5(8, 1); V5(8, 2);
#undef V5
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor w4_dequant(torch::Tensor w, torch::Tensor s, int64_t group, bool mma_layout) {
  TORCH_CHECK(w.is_cuda() && s.is_cuda() && w.device() == s.device(), "packed weights and scales must be on the same GPU");
  TORCH_CHECK(w.scalar_type() == torch::kUInt8 && s.scalar_type() == torch::kHalf, "packed weights uint8, scales float16");
  TORCH_CHECK(w.dim() == 2 && s.dim() == 2 && group > 0, "w [N,K/2], scales [N,K/G], positive group size");
  const c10::cuda::CUDAGuard guard(w.device());
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous() && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "packed weights must be contiguous and 16-byte aligned");
  int N = w.size(0), K = w.size(1) * 2;
  TORCH_CHECK(group % 32 == 0 && K % group == 0 && s.size(0) == N && s.size(1) == K / group, "shape mismatch");
  TORCH_CHECK(!mma_layout || (K % 128 == 0 && group % 128 == 0), "the v5 layout needs K and group % 128 == 0");
  auto out = torch::empty({N, K}, s.options());
  size_t total = (size_t)N * (K / 8);
  auto wp = reinterpret_cast<const uint32_t*>(w.data_ptr<uint8_t>());
  auto sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
  auto st = at::cuda::getCurrentCUDAStream();
  if (mma_layout)
    dequant_mma_kernel<<<(total + 255) / 256, 256, 0, st>>>(wp, sp, reinterpret_cast<half*>(out.data_ptr<at::Half>()),
                                                            N, K, (int)group);
  else
    dequant_kernel<<<(total + 255) / 256, 256, 0, st>>>(wp, sp, reinterpret_cast<uint4*>(out.data_ptr<at::Half>()),
                                                        N, K, (int)group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4_gemv", &w4_gemv, "INT4 weight-only GEMV, M <= 8 (version 1/2: canonical layout, 3/4: interleaved, 6: v5 layout)",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("s"), pybind11::arg("group"), pybind11::arg("version"),
        pybind11::arg("rows_per_warp") = 0);
  m.def("w4_gemm_tc", &w4_gemm_tc, "INT4 weight-only tensor-core GEMM, M <= 64 (interleave_for_mma layout)",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("s"), pybind11::arg("group"), pybind11::arg("row_tiles") = 0);
  m.def("w4_dequant", &w4_dequant, "INT4 canonical (or v5) layout -> fp16 weight", pybind11::arg("w"), pybind11::arg("s"),
        pybind11::arg("group"), pybind11::arg("mma_layout") = false);
}
