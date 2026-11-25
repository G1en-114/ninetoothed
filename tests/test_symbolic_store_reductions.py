"""Row reductions whose store target differs from the value only by symbol name.

``x.tile((1, -1))`` and ``out.tile((1, -1))`` keep their own row-extent
symbols, so ``out = softmax(x)`` stores a value typed with ``x``'s extent into
``out``'s extent.  That must still use the row-vector schedule (one program
per row) instead of the per-element scalar fallback, with the equality of the
two extents checked at launch.
"""

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.reductions import _symbolic_store_constraints
from tests.utils import get_available_devices


def _arrangement(x, out):
    return x.tile((1, -1)), out.tile((1, -1))


def _softmax(x, out):
    value = x.to(ntl.float32)
    numerator = ntl.exp(value - ntl.max(value, axis=1)[:, None])
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def _row_sum_broadcast(x, out):
    out = x + ntl.sum(x, axis=1)[:, None]  # noqa: F841


def _tensors():
    return (Tensor(2, other=float("-inf")), Tensor(2))


def _reduction_schedule(application):
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=application,
            tensors=_tensors(),
            backend="triton",
            kernel_name="symbolic_store_reduction",
        )
    )

    return compilation.artifact.metadata["ssa_schedule"]["reduction"]


def test_symbolic_store_constraints():
    assert _symbolic_store_constraints(("1", "a"), ("1", "b")) == (
        (("a",), ("b",)),
    )
    assert _symbolic_store_constraints(("1", "a"), ("1", "a")) == ()
    # Static extents must match exactly, and ranks must agree.
    assert _symbolic_store_constraints(("1", "a"), ("2", "a")) is None
    assert _symbolic_store_constraints(("1", "4"), ("1", "b")) is None
    assert _symbolic_store_constraints(("a",), ("1", "a")) is None


@pytest.mark.parametrize("application", (_softmax, _row_sum_broadcast))
def test_symbolic_store_uses_row_vector_schedule(application):
    reduction = _reduction_schedule(application)

    assert reduction["mode"] == "row-vector"

    extents = {
        (tuple(actual), tuple(expected))
        for actual, expected in reduction["program_constraints"]
    }

    assert any(len(actual) == 1 and actual != expected for actual, expected in extents)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", ((1, 1), (7, 3), (64, 1000), (33, 4096)))
def test_symbolic_store_softmax_matches_torch(device, shape):
    kernel = ninetoothed.make(_arrangement, _softmax, _tensors())
    x = torch.randn(shape, device=device)
    out = torch.empty_like(x)

    kernel(x, out)

    torch.testing.assert_close(out, torch.softmax(x, dim=1))


@pytest.mark.parametrize("device", get_available_devices())
def test_symbolic_store_rejects_mismatched_extents(device):
    kernel = ninetoothed.make(_arrangement, _softmax, _tensors())
    x = torch.randn((4, 8), device=device)

    kernel(x, torch.empty_like(x))

    with pytest.raises(ValueError, match="do not match"):
        kernel(x, torch.empty((4, 16), device=device))
