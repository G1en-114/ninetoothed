"""Launch preparation is reused across calls with fresh tensors of one structure.

Operator wrappers usually allocate a new output per call, which never hits the
identity-keyed prepared-call cache.  The structural cache must make such calls
skip the full launch preparation, while still distinguishing shapes, strides,
dtypes and memory aliasing, and must not keep call tensors alive.
"""

import gc
import weakref

import pytest
import torch

import ninetoothed
import ninetoothed.backends.materializers.triton as triton_materializer
from ninetoothed import Tensor, block_size
from tests.utils import get_available_devices

BLOCK_SIZE = block_size()


def _arrangement(x, y, out):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))


def _application(x, y, out):
    out = x + y  # noqa: F841


@pytest.fixture
def counted_prepare(monkeypatch):
    counter = {"calls": 0}
    original = triton_materializer._triton_prepare_invocation

    def counting(function, kernel):
        prepare = original(function, kernel)

        if prepare is None:
            return None

        def wrapped(*args, **kwargs):
            counter["calls"] += 1

            return prepare(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(triton_materializer, "_triton_prepare_invocation", counting)

    return counter


def _make_kernel():
    return ninetoothed.make(
        _arrangement,
        _application,
        (Tensor(1), Tensor(1), Tensor(1)),
        kernel_name="structural_launch_cache_add",
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_fresh_outputs_reuse_preparation(device, counted_prepare):
    kernel = _make_kernel()
    x = torch.randn(4096, device=device)
    y = torch.randn(4096, device=device)

    kernel(x, y, torch.empty_like(x))
    warm = counted_prepare["calls"]

    for _ in range(8):
        out = torch.empty_like(x)
        kernel(x, y, out)
        torch.testing.assert_close(out, x + y)

    assert counted_prepare["calls"] == warm


@pytest.mark.parametrize("device", get_available_devices())
def test_offset_views_reuse_preparation(device, counted_prepare):
    kernel = _make_kernel()
    storage = torch.randn(64, 1024, device=device)
    other = torch.randn(64, 1024, device=device)

    kernel(storage[0], other[0], torch.empty(1024, device=device))
    warm = counted_prepare["calls"]

    # Row views at different storage offsets, plus an unaligned offset that
    # Triton itself must specialise differently at launch.
    for row in (*range(1, 32), "unaligned"):
        if row == "unaligned":
            x, y = storage.view(-1)[1:1025], other.view(-1)[3:1027]
        else:
            x, y = storage[row], other[row]

        out = torch.empty(1024, device=device)
        kernel(x, y, out)
        torch.testing.assert_close(out, x + y)

    assert counted_prepare["calls"] == warm


@pytest.mark.parametrize("device", get_available_devices())
def test_structure_changes_are_not_reused(device):
    kernel = _make_kernel()

    for size in (4096, 1000, 1, 4096):
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        out = torch.empty_like(x)
        kernel(x, y, out)
        torch.testing.assert_close(out, x + y)

    storage = torch.randn(8192, device=device)
    y = torch.randn(4096, device=device)

    for x in (storage[::2], storage[:4096], storage[1::2]):
        out = torch.empty_like(y)
        kernel(x, y, out)
        torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("device", get_available_devices())
def test_aliasing_is_part_of_the_structure(device):
    kernel = _make_kernel()
    x = torch.randn(4096, device=device)
    y = torch.randn(4096, device=device)
    expected = x + y

    kernel(x, y, torch.empty_like(x))
    kernel(x, y, x)
    torch.testing.assert_close(x, expected)

    x = torch.randn(4096, device=device)
    out = torch.empty_like(x)
    kernel(x, y, out)
    torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("device", get_available_devices())
def test_launch_does_not_retain_call_tensors(device):
    kernel = _make_kernel()
    x = torch.randn(4096, device=device)
    y = torch.randn(4096, device=device)
    kernel(x, y, torch.empty_like(x))

    gc.collect()
    gc.disable()

    try:
        references = []

        for _ in range(4):
            out = torch.empty_like(x)
            kernel(x, y, out)
            references.append(weakref.ref(out))
            del out

        assert all(reference() is None for reference in references)
    finally:
        gc.enable()
