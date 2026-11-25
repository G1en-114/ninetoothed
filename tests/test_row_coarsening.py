"""Unit-row reductions are recompiled as multi-row block programs."""

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.row_coarsening import ENVIRONMENT_VARIABLE
from tests.utils import get_available_devices


def _rows_arrangement(x, out, width):
    return x.tile((1, width.value)), out.tile((1, width.value)), width


def _per_row_arrangement(x, out, width):
    return x.tile((1, width.value)), out.tile((1,)), width


def _full_row_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1, -1))


def _softmax(x, out, width):
    numerator = ntl.exp(x - ntl.max(x, axis=1)[:, None])
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def _row_sum(x, out, width):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _dynamic_softmax(x, out):
    numerator = ntl.exp(x - ntl.max(x, axis=1)[:, None])
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def _width(value):
    return Tensor(0, constexpr=True, value=value, name="width")


def _cases(width):
    return {
        "softmax": (_rows_arrangement, _softmax, (Tensor(2), Tensor(2), _width(width))),
        "row_sum": (
            _per_row_arrangement,
            _row_sum,
            (Tensor(2), Tensor(1), _width(width)),
        ),
    }


def _compile(arrangement, application, tensors):
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="triton",
            max_num_configs=1,
            kernel_name="row_coarsening",
        )
    )


@pytest.mark.parametrize("case", ("softmax", "row_sum"))
@pytest.mark.parametrize(("width", "rows"), ((8, 64), (16, 64), (32, 32), (64, 16)))
def test_unit_row_reductions_are_coarsened(case, width, rows):
    metadata = _compile(*_cases(width)[case]).artifact.metadata

    assert metadata.get("row_coarsening") == {"rows": rows}
    assert metadata["program_mode"]["block"]
    assert metadata["ssa_schedule"]["row_group"]["enabled"]


def test_ineligible_kernels_keep_their_schedule(monkeypatch):
    # A dynamic row extent cannot form a bounded block.
    dynamic = _compile(_full_row_arrangement, _dynamic_softmax, (Tensor(2), Tensor(2)))
    assert "row_coarsening" not in dynamic.artifact.metadata

    # Rows wider than the element budget stay one row per program.
    wide = _compile(*_cases(2048)["softmax"])
    assert "row_coarsening" not in wide.artifact.metadata

    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "0")
    disabled = _compile(*_cases(16)["softmax"])
    assert "row_coarsening" not in disabled.artifact.metadata
    assert disabled.artifact.metadata["program_mode"]["vector"]


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("rows", (1, 63, 64, 1000))
@pytest.mark.parametrize("case", ("softmax", "row_sum"))
def test_coarsened_kernels_match_torch(device, dtype, rows, case):
    width = 16
    arrangement, application, tensors = _cases(width)[case]
    kernel = ninetoothed.make(
        arrangement,
        application,
        tensors,
        max_num_configs=1,
        kernel_name=f"row_coarsening_{case}",
    )
    x = torch.randn((rows, width), device=device, dtype=dtype)

    if case == "softmax":
        out = torch.full_like(x, float("nan"))
        expected = torch.softmax(x.float(), dim=1).to(dtype)
    else:
        out = torch.full((rows,), float("nan"), device=device, dtype=dtype)
        expected = x.sum(dim=1)

    kernel(x, out, width)

    tolerance = {} if dtype == torch.float32 else {"atol": 1e-2, "rtol": 1e-2}
    torch.testing.assert_close(out, expected, **tolerance)
