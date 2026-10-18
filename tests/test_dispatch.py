"""linear() routing and W4Weight validation, on the CPU (no kernel runs: gemv / dequant are stubbed)."""
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lowbit import ops  # noqa: E402


def weight(n=1024, k=256, layouts=("mma",)):
    have = lambda name: object() if name in layouts else None
    return types.SimpleNamespace(n=n, k=k, group=128, packed=have("canonical"), packed_v3=have("v3"), packed_mma=have("mma"))


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(ops, "gemv", lambda x, wq, v, rpw=0: seen.append(v) or torch.zeros(x.shape[0], wq.n))
    monkeypatch.setattr(ops, "dequant", lambda wq, layout=None: seen.append("dequant") or torch.zeros(wq.n, wq.k))
    return seen


@pytest.mark.parametrize("m,n,want", [(1, 1024, 6), (2, 1024, 5), (32, 1024, 5), (33, 1024, "dequant"),
                                      (33, 4096, 5), (64, 4096, 5), (65, 4096, "dequant")])
def test_auto_rule_on_the_mma_layout(calls, m, n, want):
    ops.linear(torch.zeros(m, 256), weight(n))
    assert calls == [want]


@pytest.mark.parametrize("layouts,m,want", [(("canonical", "v3"), 1, 4), (("canonical", "v3"), 8, 4),
                                            (("canonical",), 3, 2), (("canonical", "v3"), 9, "dequant")])
def test_auto_rule_without_the_mma_layout(calls, layouts, m, want):
    ops.linear(torch.zeros(m, 256), weight(layouts=layouts))
    assert calls == [want]


def test_a_fixed_version_never_passes_its_row_limit(calls):
    ops.linear(torch.zeros(9, 256), weight(), max_gemv_rows=16, version=6)      # v6 takes at most 8 rows
    ops.linear(torch.zeros(64, 256), weight(), version=5)
    ops.linear(torch.zeros(65, 256), weight(), version=5)
    assert calls == ["dequant", 5, "dequant"]


def test_bad_versions_and_empty_input(calls):
    with pytest.raises(ValueError):
        ops.linear(torch.zeros(4, 256), weight(), version=7)
    assert ops.linear(torch.zeros(0, 256), weight()).shape == (0, 1024)
    assert calls == []


def test_gemv_refuses_a_layout_the_weight_does_not_keep():
    with pytest.raises(ValueError, match="packed_v3"):
        ops.gemv(torch.zeros(1, 256), weight(layouts=("mma",)), 4)


def test_construction_checks_layouts_before_touching_the_gpu():
    packed, scales = np.zeros((4, 48), np.uint8), np.ones((4, 3), np.float16)       # K = 96, G = 32
    with pytest.raises(ValueError, match="multiples of 128"):
        ops.W4Weight(packed, scales, 32, ("mma",))
    with pytest.raises(ValueError, match="canonical or the mma"):
        ops.W4Weight(packed, scales, 32, ("v3",))
    with pytest.raises(ValueError, match="unknown layout"):
        ops.W4Weight(packed, scales, 32, ("nope",))
