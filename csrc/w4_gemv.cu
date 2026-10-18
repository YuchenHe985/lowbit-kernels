// Weight-only INT4 GEMV/skinny-GEMM for LLM decode: y[M, N] = x[M, K] (fp16) * dequant(W)[N, K]^T, M <= 8.
// Format: lowbit/quant.py (symmetric, group size G along K, fp16 scale per group, two nibbles per byte).
//
// Three versions, kept side by side so each optimization step is measured on its own:
//   v1  one thread per output row, byte loads, fp32 math              (the "obvious" kernel)
//   v2  one warp per output row, 128-bit loads (32 weights per lane per step), fp32 math, shuffle reduction
//   v3  v2 + offline-interleaved nibbles and the 0x6400 magic-number trick: a mask and an OR turn eight
//       nibbles into four fp16 pairs, multiplied with HFMA2 against consecutive activations
// plus a dequantize kernel (canonical layout -> fp16 W) for the prefill path, where cuBLAS wins.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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
    float sc = __half2float(srow[(c * 32) / G]);
    half2 wh[16];                                           // wh[4i+j] = (W[k0+8i+2j], W[k0+8i+2j+1])
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t t = ((words[i] >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
        wh[4 * i + j] = __hsub2(*reinterpret_cast<half2*>(&t), bias);
      }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * K + c * 32);
      float part = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint4 xv = __ldg(xp + i);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        // 8 products per fp16 partial before widening to fp32: overflow needs |x| > ~1000, and the rounding
        // error stays at fp16 precision over 8 terms instead of 32
        half2 p = __hmul2(wh[4 * i], xh[0]);
        p = __hfma2(wh[4 * i + 1], xh[1], p);
        p = __hfma2(wh[4 * i + 2], xh[2], p);
        p = __hfma2(wh[4 * i + 3], xh[3], p);
        float2 f = __half22float2(p);
        part += f.x + f.y;
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

__global__ void dequant_kernel(const uint8_t* __restrict__ w, const half* __restrict__ s, half2* __restrict__ out,
                               int N, int K, int G) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, total = (size_t)N * (K / 2);
  if (idx >= total) return;
  int n = idx / (K / 2), k = 2 * (idx % (K / 2));
  float sc = __half2float(s[(size_t)n * (K / G) + k / G]);
  uint8_t b = w[idx];
  out[idx] = __floats2half2_rn((float)((int)(b & 0xF) - 8) * sc, (float)((int)(b >> 4) - 8) * sc);
}

void check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, int64_t group) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda(), "tensors must be on the GPU");
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

template <int M>
void launch(int version, const half* x, const uint8_t* w, const half* s, half* y, int N, int K, int G,
            cudaStream_t st) {
  if (version == 1) {
    gemv_v1<M><<<(N + 127) / 128, 128, 0, st>>>(x, w, s, y, N, K, G);
  } else {
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock), block(32 * kWarpsPerBlock);
    auto wv = reinterpret_cast<const uint4*>(w);
    if (version == 2) gemv_v2<M><<<grid, block, 0, st>>>(x, wv, s, y, N, K, G);
    else gemv_v3<M><<<grid, block, 0, st>>>(x, wv, s, y, N, K, G);
  }
}

}  // namespace

// version 1/2 expect the canonical layout, version 3 the interleave_for_gpu layout
torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, int64_t group, int64_t version) {
  check(x, w, s, group);
  int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "the GEMV kernels take 1..8 rows; use the dequantize + matmul path above that");
  TORCH_CHECK(version >= 1 && version <= 3, "version is 1, 2 or 3");
  auto y = torch::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
  auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  auto wp = w.data_ptr<uint8_t>();
  int v = (int)version, G = (int)group;
  switch (M) {
    case 1: launch<1>(v, xp, wp, sp, yp, N, K, G, st); break;
    case 2: launch<2>(v, xp, wp, sp, yp, N, K, G, st); break;
    case 3: launch<3>(v, xp, wp, sp, yp, N, K, G, st); break;
    case 4: launch<4>(v, xp, wp, sp, yp, N, K, G, st); break;
    case 5: launch<5>(v, xp, wp, sp, yp, N, K, G, st); break;
    case 6: launch<6>(v, xp, wp, sp, yp, N, K, G, st); break;
    case 7: launch<7>(v, xp, wp, sp, yp, N, K, G, st); break;
    default: launch<8>(v, xp, wp, sp, yp, N, K, G, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor w4_dequant(torch::Tensor w, torch::Tensor s, int64_t group) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && s.scalar_type() == torch::kHalf, "bad inputs");
  int N = w.size(0), K = w.size(1) * 2;
  auto out = torch::empty({N, K}, s.options());
  size_t total = (size_t)N * (K / 2);
  dequant_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      w.data_ptr<uint8_t>(), reinterpret_cast<const half*>(s.data_ptr<at::Half>()),
      reinterpret_cast<half2*>(out.data_ptr<at::Half>()), N, K, (int)group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4_gemv", &w4_gemv, "INT4 weight-only GEMV, M <= 8 (version 1/2: canonical layout, 3: interleaved)");
  m.def("w4_dequant", &w4_dequant, "INT4 canonical layout -> fp16 weight");
}
