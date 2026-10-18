"""Reference weight-only INT4 format (the ground truth every kernel is tested against).

Layout, for an nn.Linear weight W of shape [N, K] (out_features x in_features):
  * symmetric group-wise quantization along K, group size G (default 128):
        scale[n, g] = max|W[n, gG:(g+1)G]| / 7         stored as float16
        q[n, k]     = clamp(round(W[n, k] / scale), -8, 7)
  * each value is stored as the unsigned nibble u = q + 8 (0..15), two per byte,
    row-major along K: byte[n, j] = u[n, 2j] | (u[n, 2j+1] << 4)   ("canonical" layout).

The GPU kernel v3 reads a reordered copy (interleave_for_gpu) in which every 32-bit word holds
k0..k7 at nibble positions [k0, k2, k4, k6, k1, k3, k5, k7], so that one mask + OR turns a word into
fp16 pairs (k0,k1), (k2,k3), ... that line up with consecutive activations. Same values, same bytes.
"""
import numpy as np

QMIN, QMAX, OFFSET = -8, 7, 8


def quantize_w4(w, group=128):
    """W [N, K] float -> (packed uint8 [N, K//2], scales float16 [N, K//group])."""
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    if k % group or group % 32:
        raise ValueError(f"K={k} must be a multiple of the group size {group}, itself a multiple of 32")
    g = w.reshape(n, k // group, group)
    scale = (np.abs(g).max(axis=2) / QMAX).astype(np.float16)
    scale[scale == 0] = 1.0                                   # an all-zero group: any scale reproduces it
    q = np.clip(np.rint(g / scale.astype(np.float32)[..., None]), QMIN, QMAX).astype(np.int8)
    u = (q.reshape(n, k) + OFFSET).astype(np.uint8)
    packed = u[:, 0::2] | (u[:, 1::2] << 4)
    return packed, scale


def unpack_w4(packed):
    """packed uint8 [N, K//2] -> signed int8 values [N, K] in -8..7."""
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    out[:, 0::2] = (packed & 0x0F).astype(np.int8) - OFFSET
    out[:, 1::2] = (packed >> 4).astype(np.int8) - OFFSET
    return out


def dequantize_w4(packed, scales, group=128):
    """-> float32 [N, K], exactly what a kernel should multiply by."""
    q = unpack_w4(packed).astype(np.float32)
    n, k = q.shape
    return (q.reshape(n, k // group, group) * np.asarray(scales, np.float32)[..., None]).reshape(n, k)


# nibble slot s (0..7, bit 4s of a little-endian 32-bit word) holds value k0 + SLOT_TO_K[s]
SLOT_TO_K = np.array([0, 2, 4, 6, 1, 3, 5, 7])


def interleave_for_gpu(packed):
    """Canonical packed [N, K//2] -> the v3 layout (see module doc). Bytes per row are unchanged."""
    u = unpack_w4(packed) + OFFSET                            # back to nibbles 0..15, [N, K]
    n, k = u.shape
    words = u.reshape(n, k // 8, 8)[:, :, SLOT_TO_K]           # slot order inside every 8-value word
    words = words.reshape(n, k).astype(np.uint8)
    return words[:, 0::2] | (words[:, 1::2] << 4)


def gemv_reference(x, packed, scales, group=128):
    """y[m, n] = sum_k x[m, k] * dequant(W)[n, k], in float64 for a tight reference."""
    w = dequantize_w4(packed, scales, group).astype(np.float64)
    return np.asarray(x, np.float64) @ w.T
