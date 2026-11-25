"""Grouped block row reductions over dynamically shaped tensors.

Block-vector row groups used to require a static outer program domain, only
to prove that every tensor launches the same number of tiles.  With dynamic
shapes that equality is instead checked at launch: by the row-vector
schedule's program constraints, plus per-dimension constraints because block
programs decompose the program id with each tensor's own outer shape.
"""

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.passes import default_spec
from tests.utils import get_available_devices

GROUP_PASS = "ssa.group_reduction_rows"


def _pipeline():
    passes = default_spec("triton").passes
    insertion = passes.index("ssa.select_schedule") + 1

    return (*passes[:insertion], GROUP_PASS, *passes[insertion:])


def _arrangement(x, out, rows=16, width=8):
    return x.tile((rows, width)), out.tile((rows, width))


def _softmax(x, out):
    numerator = ntl.exp(x - ntl.max(x, axis=1)[:, None])
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def _tensors():
    return (
        Tensor(2, dtype=ninetoothed.float32),
        Tensor(2, dtype=ninetoothed.float32),
    )


def test_dynamic_outer_domain_uses_block_row_groups():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_softmax,
            tensors=_tensors(),
            backend="triton",
            pipeline=_pipeline(),
            max_num_configs=1,
            kernel_name="dynamic_row_groups",
        )
    )
    metadata = compilation.artifact.metadata
    group = metadata["ssa_schedule"]["row_group"]

    assert group["enabled"], group["reason"]
    assert metadata["program_mode"]["block"]
    # x and out own different size symbols: one equality per outer dimension.
    assert len(group["program_constraints"]) == 2


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("rows", (1, 15, 16, 1000))
def test_dynamic_row_groups_match_torch(device, rows):
    kernel = ninetoothed.make(
        _arrangement,
        _softmax,
        _tensors(),
        pipeline=_pipeline(),
        max_num_configs=1,
        kernel_name="dynamic_row_groups_runtime",
    )
    x = torch.randn((rows, 8), device=device)
    out = torch.full_like(x, float("nan"))

    kernel(x, out)

    torch.testing.assert_close(out, torch.softmax(x, dim=1))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    ("x_shape", "out_shape"),
    (
        ((100, 8), (200, 8)),
        ((100, 8), (100, 16)),
        # Same number of tiles (2 * 50 == 100 * 1), different tile mapping.
        ((32, 400), (1600, 8)),
    ),
)
def test_dynamic_row_groups_reject_mismatched_domains(device, x_shape, out_shape):
    kernel = ninetoothed.make(
        _arrangement,
        _softmax,
        _tensors(),
        pipeline=_pipeline(),
        max_num_configs=1,
        kernel_name="dynamic_row_groups_mismatch",
    )
    kernel(torch.randn((16, 8), device=device), torch.empty((16, 8), device=device))

    with pytest.raises(ValueError, match="do not match"):
        kernel(torch.randn(x_shape, device=device), torch.empty(out_shape, device=device))
