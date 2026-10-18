"""PyTorch side of the CUDA kernels: build the extension once, hold quantized weights, dispatch decode vs prefill."""
import pathlib

import torch

from . import quant

_EXT = None
SRC = pathlib.Path(__file__).resolve().parent.parent / "csrc" / "w4_gemv.cu"


def ext():
    """Compile csrc/w4_gemv.cu for the visible GPU on first use (about a minute), then reuse it."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="lowbit_w4", sources=[str(SRC)], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


class W4Weight:
    """One linear layer's weight in the lowbit/quant.py format, on the GPU. `layouts` picks which copies to keep:
    canonical (v1, v2, dequantize), v3 (v3, v4), mma (v5). The benchmark keeps all three; a model keeps what it uses."""

    def __init__(self, packed, scales, group, layouts=("canonical", "v3", "mma")):
        self.group = group
        self.n, self.k = packed.shape[0], packed.shape[1] * 2
        self.packed = torch.from_numpy(packed).cuda() if "canonical" in layouts else None
        self.packed_v3 = torch.from_numpy(quant.interleave_for_gpu(packed)).cuda() if "v3" in layouts else None
        self.packed_mma = torch.from_numpy(quant.interleave_for_mma(packed)).cuda() if "mma" in layouts else None
        self.scales = torch.from_numpy(scales).cuda()

    @classmethod
    def from_float(cls, w, group=128, layouts=("canonical", "v3", "mma")):
        packed, scales = quant.quantize_w4(w.detach().float().cpu().numpy(), group)
        return cls(packed, scales, group, layouts)

    def nbytes(self):
        """Bytes of one copy of the weight (packed values + scales), whatever layouts are kept."""
        return self.n * self.k // 2 + self.scales.numel() * 2


def gemv(x, wq, version=4, rows_per_warp=0):
    """x [M, K] fp16 -> [M, N] fp16. v1-v4, v6: M <= 8 (rows_per_warp for v4/v6, 0 = chosen from N and M);
    v5: tensor cores, M <= 64 (rows_per_warp = 16-row tiles per warp, 1/2/4, 0 = chosen from N and M).
    v5 and v6 read the same layout (packed_mma), so a model needs only that copy."""
    if version == 5:
        return ext().w4_gemm_tc(x.contiguous(), wq.packed_mma, wq.scales, wq.group, rows_per_warp)
    w = wq.packed_mma if version == 6 else wq.packed_v3 if version >= 3 else wq.packed
    return ext().w4_gemv(x.contiguous(), w, wq.scales, wq.group, version, rows_per_warp)


def quantize_w4_gpu(w, group=128, clip_grid=None, rows_per_pass=2048):
    """Same format and, with clip_grid=None, the same bits as quant.quantize_w4, computed on the GPU (a 3B model
    takes seconds instead of a minute). clip_grid, e.g. (1.0, 0.95, ..., 0.75): per group, try scale = a * amax / 7
    for each a and keep the one with the smallest squared weight error (clipping the largest values buys finer
    steps for the rest). Returns (packed uint8 [N, K/2], scales fp16 [N, K/G]) on the GPU."""
    n, k = w.shape
    if k % group or group % 32:
        raise ValueError(f"K={k} must be a multiple of the group size {group}, itself a multiple of 32")
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
    scales = torch.empty(n, k // group, dtype=torch.float16, device="cuda")
    for r0 in range(0, n, rows_per_pass):
        g = w[r0:r0 + rows_per_pass].detach().to("cuda", torch.float32).reshape(-1, k // group, group)
        amax = g.abs().amax(-1)
        # divide by a tensor: with a Python-number divisor PyTorch's CUDA kernel multiplies by the reciprocal,
        # which can differ from numpy's correctly rounded amax / 7 in the last bit and flip an fp16 scale
        qmax = torch.full_like(amax, quant.QMAX)
        best_s, best_q, best_err = None, None, None
        for a in (clip_grid or (1.0,)):
            sc = ((amax * a if a != 1.0 else amax) / qmax).half()
            sc[sc == 0] = 1.0
            q = torch.clamp(torch.round(g / sc.float()[..., None]), quant.QMIN, quant.QMAX)
            if clip_grid is None:
                best_s, best_q = sc, q
                break
            err = (q * sc.float()[..., None] - g).pow(2).sum(-1)
            if best_err is None:
                best_s, best_q, best_err = sc, q, err
            else:
                better = err < best_err
                best_s = torch.where(better, sc, best_s)
                best_q = torch.where(better[..., None], q, best_q)
                best_err = torch.minimum(err, best_err)
        u = (best_q.reshape(g.shape[0], k) + quant.OFFSET).to(torch.uint8)
        packed[r0:r0 + rows_per_pass] = u[:, 0::2] | (u[:, 1::2] << 4)
        scales[r0:r0 + rows_per_pass] = best_s
    return packed, scales


def dequantize_w4_gpu(packed, scales, group=128):
    """fp16 [N, K]: (u - 8) * scale in fp32, rounded once to fp16 (what the dequantize kernel writes)."""
    u = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).float() - quant.OFFSET
    n, k = u.shape
    return (u.reshape(n, k // group, group) * scales.float()[..., None]).reshape(n, k).half()


def dequant(wq):
    if wq.packed is not None:
        return ext().w4_dequant(wq.packed, wq.scales, wq.group)
    return ext().w4_dequant(wq.packed_mma, wq.scales, wq.group, True)


# version 0 (auto) picks by batch size from the A1 sweep on a T4, all on the one v5 layout: the v6 GEMV for a
# single row, the tensor-core v5 up to 64 rows, then dequantize + cuBLAS
AUTO_GEMV_ROWS, AUTO_V5_ROWS = 1, 64


def linear(x, wq, max_gemv_rows=8, version=0):
    """Decode (few rows): an INT4 kernel (v4 GEMV, or v5 tensor cores up to 64 rows). Prefill (many rows):
    dequantize, then cuBLAS FP16 GEMM."""
    m = x.shape[0]
    if version == 0:
        version = 6 if m <= AUTO_GEMV_ROWS else 5
        limit = AUTO_V5_ROWS if version == 5 else max_gemv_rows
    else:
        limit = 64 if version == 5 else max_gemv_rows
    if m <= limit:
        return gemv(x, wq, version)
    return torch.nn.functional.linear(x, dequant(wq))
