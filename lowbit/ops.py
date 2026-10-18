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


LAYOUTS = ("canonical", "v3", "mma")


class W4Weight:
    """One linear layer's weight in the lowbit/quant.py format, on the GPU. `layouts` picks which copies to keep:
    canonical (v1, v2, dequantize), v3 (v3, v4), mma (v5, v6, dequantize; needs K and the group size to be
    multiples of 128). None = every layout this shape supports (the benchmark); a model keeps ("mma",). Prefill
    dequantizes, so one of canonical / mma must be kept."""

    def __init__(self, packed, scales, group, layouts=None):
        self.group = group
        self.n, self.k = packed.shape[0], packed.shape[1] * 2
        mma_ok = self.k % 128 == 0 and group % 128 == 0
        if layouts is None:
            layouts = LAYOUTS if mma_ok else ("canonical", "v3")
        if not set(layouts) <= set(LAYOUTS):
            raise ValueError(f"unknown layout in {layouts}; choose from {LAYOUTS}")
        if "mma" in layouts and not mma_ok:
            raise ValueError(f"the mma layout needs K and the group size to be multiples of 128 (K={self.k}, G={group})")
        if "canonical" not in layouts and "mma" not in layouts:
            raise ValueError("keep the canonical or the mma layout: prefill dequantizes from one of them")
        self.packed = torch.from_numpy(packed).cuda() if "canonical" in layouts else None
        self.packed_v3 = torch.from_numpy(quant.interleave_for_gpu(packed)).cuda() if "v3" in layouts else None
        self.packed_mma = torch.from_numpy(quant.interleave_for_mma(packed)).cuda() if "mma" in layouts else None
        self.scales = torch.from_numpy(scales).cuda()

    @classmethod
    def from_float(cls, w, group=128, layouts=None):
        packed, scales = quant.quantize_w4(w.detach().float().cpu().numpy(), group)
        return cls(packed, scales, group, layouts)

    def nbytes(self):
        """Bytes of one copy of the weight (packed values + scales), whatever layouts are kept."""
        return self.n * self.k // 2 + self.scales.numel() * 2


def pick_rows(n, m):
    """Rows per warp v4/v6 use when rows_per_warp = 0 (mirror of pick_rows in csrc/w4_gemv.cu, for the records)."""
    return 4 if n >= 512 else 2 if m >= 4 else 1


# the layout each kernel version reads, and the most rows it takes
NEEDS = {1: "packed", 2: "packed", 3: "packed_v3", 4: "packed_v3", 5: "packed_mma", 6: "packed_mma"}
MAX_ROWS = {1: 8, 2: 8, 3: 8, 4: 8, 5: 64, 6: 8}


def gemv(x, wq, version=4, rows_per_warp=0):
    """x [M, K] fp16 -> [M, N] fp16. v1-v4, v6: M <= 8 (rows_per_warp for v4/v6, 0 = chosen from N and M);
    v5: tensor cores, M <= 64 (rows_per_warp = 16-row tiles per warp, 1/2/4, 0 = chosen from N and M).
    v5 and v6 read the same layout (packed_mma), so a model needs only that copy."""
    if version not in NEEDS:
        raise ValueError(f"version must be one of {sorted(NEEDS)}")
    if getattr(wq, NEEDS[version]) is None:
        raise ValueError(f"v{version} reads {NEEDS[version]}, which this W4Weight does not keep")
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


def dequant(wq, layout=None):
    """fp16 W [N, K] from the canonical or the mma copy (None: canonical when kept). Both give the same bits."""
    if layout is None:
        layout = "canonical" if wq.packed is not None else "mma"
    if layout == "canonical" and wq.packed is not None:
        return ext().w4_dequant(wq.packed, wq.scales, wq.group)
    if layout == "mma" and wq.packed_mma is not None:
        return ext().w4_dequant(wq.packed_mma, wq.scales, wq.group, True)
    raise ValueError(f"cannot dequantize from the {layout} layout: not kept")


# version 0 (auto) picks by batch size and layer width from the A1 sweep on a T4 (commit 35dc86c), all on the one
# v5 layout: the v6 GEMV for a single row; v5 up to 32 rows on every layer (at most 0.78x the time of dequantize
# + cuBLAS); 33-64 rows only on wide layers, N >= 3072 (0.56-0.77x there, but 1.02-1.34x on the narrow and
# long-K ones); then dequantize + cuBLAS. Running longer inputs through v5 in 64-row slices lost on every
# shape from 128 rows.
AUTO_GEMV_ROWS, AUTO_V5_ROWS, AUTO_V5_WIDE_ROWS, AUTO_WIDE_N = 1, 32, 64, 3072


def gemm_chunked(x, wq, chunk=64):
    """Many rows through v5 in 64-row slices (no FP16 weight copy is written). Kept for the A1 sweep, which shows
    it losing to dequantize + cuBLAS on every shape from 128 rows, so linear() does not use it."""
    return torch.cat([gemv(x[i:i + chunk], wq, 5) for i in range(0, x.shape[0], chunk)])


def auto_version(m, wq):
    """The kernel linear() uses for m rows, or None for dequantize + cuBLAS. Weights without the mma layout (K or
    the group size not a multiple of 128) use v4 / v2 up to 8 rows."""
    if wq.packed_mma is not None:
        if m <= AUTO_GEMV_ROWS:
            return 6
        return 5 if m <= (AUTO_V5_WIDE_ROWS if wq.n >= AUTO_WIDE_N else AUTO_V5_ROWS) else None
    if m <= 8:
        return 4 if wq.packed_v3 is not None else 2 if wq.packed is not None else None
    return None


def linear(x, wq, max_gemv_rows=8, version=0):
    """Decode (few rows): an INT4 kernel (v6 GEMV for one row, v5 tensor cores beyond). Prefill (many rows):
    dequantize, then cuBLAS FP16 GEMM. version 0 = auto_version; a fixed version runs up to
    min(max_gemv_rows, its row limit) rows (v5: 64), and dequantize + cuBLAS above that."""
    m = x.shape[0]
    if m == 0:
        return x.new_empty(0, wq.n)
    if version == 0:
        v = auto_version(m, wq)
    elif version in MAX_ROWS:
        v = version if m <= (MAX_ROWS[5] if version == 5 else min(max_gemv_rows, MAX_ROWS[version])) else None
    else:
        raise ValueError(f"version must be 0 (auto) or one of {sorted(MAX_ROWS)}")
    if v is not None:
        return gemv(x, wq, v)
    return torch.nn.functional.linear(x, dequant(wq, "mma" if wq.packed_mma is not None else "canonical"))   # the model's copy
