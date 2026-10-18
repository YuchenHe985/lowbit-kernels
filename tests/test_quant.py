"""Golden-model tests: run anywhere (numpy only). The bit-level decoders below mirror what the CUDA kernels do."""
import numpy as np
import pytest

from lowbit import quant
from lowbit.quant import (dequantize_w4, gemv_reference, interleave_for_gpu, quantize_w4, unpack_w4)

RNG = np.random.default_rng(0)


def words_of(packed):
    """uint8 [N, K//2] -> little-endian uint32 words [N, K//8], as a GPU reads them."""
    return np.ascontiguousarray(packed).view("<u4")


def test_error_is_at_most_half_a_step():
    w = RNG.standard_normal((64, 512)).astype(np.float32)
    packed, scales = quantize_w4(w, 128)
    err = np.abs(dequantize_w4(packed, scales, 128) - w).reshape(64, 4, 128)
    step = scales.astype(np.float32)[..., None]
    assert np.all(err <= step / 2 + 1e-6)


def test_layout_two_nibbles_per_byte_low_first():
    w = np.zeros((1, 128), np.float32)
    w[0, :4] = [7, -7, 1, 0]                                   # scale = 7/7 = 1
    packed, scales = quantize_w4(w, 128)
    assert scales[0, 0] == 1.0
    assert packed[0, 0] == (7 + 8) | ((-7 + 8) << 4)
    assert packed[0, 1] == (1 + 8) | ((0 + 8) << 4)


def test_values_stay_in_range_and_zero_group_is_exact():
    w = RNG.standard_normal((8, 256)).astype(np.float32) * 50
    w[3] = 0
    packed, scales = quantize_w4(w, 128)
    q = unpack_w4(packed)
    assert q.min() >= -8 and q.max() <= 7
    assert np.all(dequantize_w4(packed, scales, 128)[3] == 0)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        quantize_w4(np.zeros((4, 100), np.float32), 128)
    with pytest.raises(ValueError):
        quantize_w4(np.zeros((4, 96), np.float32), 48)


def test_canonical_decoder_matches_v2_bit_logic():
    """v2: nibble b of word i is k = k0 + 8i + b."""
    w = RNG.standard_normal((5, 256)).astype(np.float32)
    packed, _ = quantize_w4(w, 128)
    u = unpack_w4(packed) + 8
    words = words_of(packed)
    for b in range(8):
        assert np.array_equal((words >> (4 * b)) & 0xF, u[:, b::8])


def test_interleaved_decoder_matches_v3_bit_logic():
    """v3: (word >> 4j) & 0x000F000F gives the pair (k0+2j, k0+2j+1) in the low and high 16-bit halves."""
    w = RNG.standard_normal((5, 256)).astype(np.float32)
    packed, _ = quantize_w4(w, 128)
    u = unpack_w4(packed) + 8
    words = words_of(interleave_for_gpu(packed)).astype(np.uint64)
    for j in range(4):
        t = (words >> (4 * j)) & 0x000F000F
        assert np.array_equal(t & 0xFFFF, u[:, 2 * j::8])
        assert np.array_equal(t >> 16, u[:, 2 * j + 1::8])


def test_magic_number_trick_is_exact_in_fp16():
    """0x6400 | u reads as 1024 + u in fp16, so subtracting 1032 gives u - 8 exactly."""
    u = np.arange(16, dtype=np.uint16)
    h = (u | 0x6400).view(np.float16)
    assert np.array_equal(h.astype(np.float32) - 1032.0, u.astype(np.float32) - 8)
    assert np.array_equal(h - np.float16(1032), (u.astype(np.int16) - 8).astype(np.float16))


def test_interleave_keeps_the_byte_count_and_is_a_permutation():
    w = RNG.standard_normal((3, 128)).astype(np.float32)
    packed, _ = quantize_w4(w, 128)
    inter = interleave_for_gpu(packed)
    assert inter.shape == packed.shape
    assert sorted(unpack_w4(inter)[1]) == sorted(unpack_w4(packed)[1])


def test_reference_gemv():
    w = RNG.standard_normal((16, 256)).astype(np.float32)
    x = RNG.standard_normal((3, 256)).astype(np.float32)
    packed, scales = quantize_w4(w, 128)
    y = gemv_reference(x, packed, scales, 128)
    assert y.shape == (3, 16)
    assert np.allclose(y, x @ dequantize_w4(packed, scales, 128).T, rtol=1e-5, atol=1e-4)


def test_mma_layout_feeds_each_lane_its_fragment():
    """Simulate what lane t of an mma quad decodes from the v5 layout and compare with the canonical values."""
    rng = np.random.default_rng(5)
    packed, _ = quant.quantize_w4(rng.standard_normal((3, 384)).astype(np.float32))
    u = quant.unpack_w4(packed) + quant.OFFSET
    mma = quant.interleave_for_mma(packed)
    words = mma.reshape(3, -1, 4, 4, 4).view("<u4")[..., 0]          # [row, chunk, lane t, word]
    for c in range(384 // 128):
        for t in range(4):
            for s in range(16):
                w, j = int(words[1, c, t, s >> 2]), s & 3
                k = 128 * c + 8 * s + 2 * t
                assert (w >> (4 * j)) & 0xF == u[1, k] and (w >> (4 * j + 16)) & 0xF == u[1, k + 1]


def test_mma_layout_is_a_permutation_and_rejects_bad_k():
    rng = np.random.default_rng(6)
    packed, _ = quant.quantize_w4(rng.standard_normal((2, 256)).astype(np.float32))
    a = quant.unpack_w4(packed)
    b = quant.unpack_w4(quant.interleave_for_mma(packed))
    assert sorted(a[0].tolist()) == sorted(b[0].tolist())
    with pytest.raises(ValueError):
        quant.interleave_for_mma(np.zeros((1, 32), np.uint8))
