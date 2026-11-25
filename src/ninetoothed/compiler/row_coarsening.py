"""Coarsen unit-row reductions into multi-row block programs.

Row reductions are commonly arranged one row per tile, ``x.tile((1, w))``.
For a small static ``w`` that launches one tiny program per row, which wastes
most of each program on index and mask setup; on an RTX 5070 Laptop a
``(32768, 16)`` softmax runs 4-8x slower than with 64 rows per program.

When such a kernel was scheduled as a row-vector reduction (every reduction
runs along the row and every other operation is row-local), arranging the
same tensors as ``x.tile((rows, w))`` computes exactly the same values: each
row of the larger tile is processed independently and partial tiles are
masked.  This module recompiles the application under that arrangement with
the existing row-grouping pass, and keeps the result only when the backend
proves a block row-group schedule for it.  Anything else keeps the original
compilation, so the transformation never changes which kernels compile.

Set ``NINETOOTHED_ROW_COARSENING=0`` to disable it.
"""

import os
from dataclasses import replace

from ninetoothed.backends.core import Target

ENVIRONMENT_VARIABLE = "NINETOOTHED_ROW_COARSENING"
GROUP_PASS = "ssa.group_reduction_rows"

#: Elements per coarsened tile; 512-1024 was fastest for widths 8-64.
TARGET_ELEMENTS = 1024
MAX_ROWS = 64


def coarsen_row_reductions(compiler, request, compilation):
    """Return a coarsened compilation of ``request``, or ``None`` to keep it."""
    rows = coarsening_rows(request, compilation)

    if rows is None:
        return None

    from ninetoothed.compiler.driver import _compile_kernel
    from ninetoothed.compiler.passes import default_spec
    from ninetoothed.tensor import ROW_COARSENING

    token = ROW_COARSENING.set(rows)

    try:
        module = compiler.frontend(request)
    except Exception:  # noqa: BLE001 - an arrangement that cannot be widened
        return None
    finally:
        ROW_COARSENING.reset(token)

    if not _uniformly_coarsened(module, rows):
        return None

    passes = default_spec(compilation.target.backend.value).passes
    insertion = passes.index("ssa.select_schedule") + 1
    pipeline = (*passes[:insertion], GROUP_PASS, *passes[insertion:])

    try:
        coarsened = _compile_kernel(replace(request, pipeline=pipeline), module=module)
    except Exception:  # noqa: BLE001 - keep the proven original schedule
        return None

    metadata = coarsened.artifact.metadata
    group = metadata.get("ssa_schedule", {}).get("row_group", {})

    if not (group.get("enabled") and metadata.get("program_mode", {}).get("block")):
        return None

    return replace(
        coarsened,
        artifact=replace(
            coarsened.artifact,
            metadata=dict(metadata) | {"row_coarsening": {"rows": rows}},
        ),
    )


def coarsening_rows(request, compilation):
    """Return the rows per tile to coarsen ``compilation`` to, if eligible."""
    if os.environ.get(ENVIRONMENT_VARIABLE, "1") == "0":
        return None

    if (
        request.frontend is not None
        or request.arrangement is None
        or request.application is None
        or request.pipeline is not None
        or compilation.target.backend != Target.TRITON
    ):
        return None

    metadata = compilation.artifact.metadata

    if not metadata.get("program_mode", {}).get("vector"):
        return None

    reduction = metadata.get("ssa_schedule", {}).get("reduction", {})

    if reduction.get("mode") != "row-vector" or tuple(
        map(str, reduction.get("parallel_shape", ()))
    ) != ("1",):
        return None

    try:
        extent = int(str(reduction.get("extent")))
    except ValueError:
        return None

    if extent < 1 or extent & (extent - 1):
        return None

    rows = min(MAX_ROWS, TARGET_ELEMENTS // extent)

    return rows if rows >= 2 else None


def _uniformly_coarsened(module, rows):
    """Require every arranged tensor to be one widened row tile.

    Coarsening is only equivalent when all tensors move to the same
    ``rows``-row tiles together; a tensor arranged any other way (an expand,
    a permutation, a multi-level tile) would keep one-row tiles and split
    the program domains.
    """
    arranged = 0

    for tensor in module.tensors:
        attrs = tensor.attrs

        if not attrs.get("source_ndim", tensor.ndim):
            continue

        shape = tuple(map(str, attrs.get("application_shape", ())))

        if (
            tuple(attrs.get("layout_operations") or ()) != ("tile",)
            or attrs.get("custom_tiling", True)
            or len(shape) not in (1, 2)
            or shape[0] != str(rows)
        ):
            return False

        arranged += 1

    return arranged > 0
