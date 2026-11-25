"""Reusable spatial-variant PSF operator and reconstruction toolkit.

This module is the stable import surface.  Implementation is split into
``spatial_psf_core``, ``spatial_psf_operator``, ``spatial_psf_reconstruct``,
``spatial_psf_gradient``, ``spatial_psf_session`` and
``spatial_psf_workflow``.
"""
from __future__ import annotations

import json

from spatial_psf_core import (
    CalibrationMismatchError,
    CalibrationSet,
    Frame,
    FrameConflictError,
    MemoryPolicy,
    PSFError,
    RevisionConflictError,
)
from spatial_psf_gradient import (
    CalibrationGradientResult,
    calibration_gradient,
    calibration_parameter_gradient,
)
from spatial_psf_operator import FrameSet, Observability, SpatialPSFOperator
from spatial_psf_reconstruct import (
    InterpretationWindow,
    ReconstructionComparison,
    ReconstructionResult,
    compare,
    compare_reconstructions,
    laplacian_matvec,
    normalize_roi,
    normalize_roi_grouped,
    reconstruct,
    reconstruction_objective,
)
from spatial_psf_session import FrameRegistry, PredictionCall, ScanSession
from spatial_psf_workflow import (
    SAMPLE_PATH,
    assess_batch,
    assess_prediction,
    build_operator,
    build_session,
    compare_batch,
    json_summary,
    load_record,
    review_new_batch,
    run_record_workflow,
    solve_record,
    WorkflowResult,
)

# Descriptive aliases for caller conventions.
PSFOperator = SpatialPSFOperator
SpatialPSF = SpatialPSFOperator
PSF = SpatialPSFOperator
PointSpreadOperator = SpatialPSFOperator
SpatialVariantPSFOperator = SpatialPSFOperator
SpatiallyVaryingPointSpreadOperator = SpatialPSFOperator
SpatiallyVaryingPSFOperator = SpatialPSFOperator
Calibration = CalibrationSet
CalibrationGrid = CalibrationSet
CalibrationGradients = CalibrationGradientResult
FrameStore = FrameRegistry


class ReconstructionSolver:
    """Small stateful configuration wrapper around :func:`reconstruct`."""

    def __init__(self, operator=None, **defaults):
        self.operator = operator
        self.defaults = defaults

    def solve(self, problem=None, **overrides):
        options = dict(self.defaults)
        options.update(overrides)
        if self.operator is not None and problem is None:
            problem = self.operator
        return reconstruct(problem, **options)

    reconstruct = solve


class ScanReconstruction(ScanSession):
    """Descriptive alias for the frame/revision workflow object."""


Pipeline = ScanSession
ScanPipeline = ScanSession


ReconstructionEngine = ReconstructionSolver
Reconstructor = ReconstructionSolver


def solve(problem=None, **kwargs):
    """Dispatch to the full exchange workflow or a reconstruction solve."""
    workflow = kwargs.pop("workflow", None)
    looks_like_exchange = (
        workflow is not False
        and isinstance(problem, dict)
        and "part_shape" in problem
        and "frames" in problem
        and (workflow is True or "gradient_probe" in problem or "requested_roi" in problem)
    )
    if looks_like_exchange:
        return run_record_workflow(problem, **kwargs)
    return reconstruct(problem, **kwargs)


solve_problem = reconstruct


def forward(first, second=None, **kwargs):
    """Convenience forward: ``forward(op, image)`` or ``forward(image, op)``."""
    if isinstance(first, SpatialPSFOperator):
        op, image = first, second
    elif isinstance(first, dict) and "part_shape" in first:
        op, image = build_operator(first, **kwargs.pop("operator_kwargs", {})), second
    else:
        image, op = first, second
        if not isinstance(op, SpatialPSFOperator):
            op = build_operator(op, **kwargs)
    valid_keys = {"valid_only", "whitened"}
    return op.forward(image, **{k: v for k, v in kwargs.items() if k in valid_keys})


def adjoint(first, second=None, **kwargs):
    """Convenience adjoint with the same argument-order flexibility."""
    if isinstance(first, SpatialPSFOperator):
        op, upstream = first, second
    else:
        upstream, op = first, second
        if not isinstance(op, SpatialPSFOperator):
            op = build_operator(op, **{k: v for k, v in kwargs.items() if k not in {"weighted", "whitened", "apply_mask"}})
    return op.adjoint(
        upstream,
        weighted=kwargs.get("weighted", False),
        whitened=kwargs.get("whitened", False),
        apply_mask=kwargs.get("apply_mask", False),
    )


def gradient(*args, calibration=None, **kwargs):
    """Convenience alias for session-free or call-bound calibration gradients."""
    if args and isinstance(args[0], PredictionCall):
        call, upstream = args[0], args[1]
        cal = calibration or kwargs.pop("calibration_set", None)
        return call.gradient(upstream, cal, **kwargs)
    return calibration_gradient(*args, calibration=calibration, **kwargs)


create_operator = build_operator
build_psf_operator = build_operator
load_problem = load_record
load_session = build_session

__all__ = [
    "CalibrationMismatchError",
    "CalibrationSet",
    "Frame",
    "FrameConflictError",
    "FrameRegistry",
    "MemoryPolicy",
    "PSFError",
    "RevisionConflictError",
    "CalibrationGradientResult",
    "calibration_gradient",
    "calibration_parameter_gradient",
    "FrameSet",
    "Observability",
    "SpatialPSFOperator",
    "InterpretationWindow",
    "ReconstructionComparison",
    "ReconstructionResult",
    "compare",
    "compare_reconstructions",
    "laplacian_matvec",
    "normalize_roi",
    "normalize_roi_grouped",
    "reconstruct",
    "reconstruction_objective",
    "PredictionCall",
    "ScanSession",
    "SAMPLE_PATH",
    "assess_batch",
    "assess_prediction",
    "build_operator",
    "build_session",
    "compare_batch",
    "review_new_batch",
    "json_summary",
    "load_record",
    "run_record_workflow",
    "solve_record",
    "WorkflowResult",
    "solve",
    "solve_problem",
    "forward",
    "adjoint",
    "gradient",
    "create_operator",
    "build_psf_operator",
    "load_problem",
    "PSFOperator",
    "SpatialPSF",
    "PSF",
    "PointSpreadOperator",
    "SpatialVariantPSFOperator",
    "SpatiallyVaryingPointSpreadOperator",
    "SpatiallyVaryingPSFOperator",
    "Calibration",
    "CalibrationGrid",
    "CalibrationGradients",
    "FrameStore",
    "ReconstructionSolver",
    "ReconstructionEngine",
    "Reconstructor",
    "ScanReconstruction",
    "Pipeline",
    "ScanPipeline",
]


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="运行空间变 PSF 重建交换样本工作流")
    parser.add_argument("path", nargs="?", default=None, help="record.json 路径")
    parser.add_argument("--quiet", action="store_true", help="只输出精简 JSON")
    args = parser.parse_args(argv)
    workflow = run_record_workflow(path=args.path)
    summary = json_summary(workflow)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
