"""Exchange-record workflow, batch comparison and JSON-friendly summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spatial_psf_core import CalibrationSet, Frame, MemoryPolicy, PSFError
from spatial_psf_gradient import calibration_gradient
from spatial_psf_operator import SpatialPSFOperator
from spatial_psf_reconstruct import AttributeDict, compare_reconstructions, reconstruct
from spatial_psf_session import ScanSession


class WorkflowResult(AttributeDict):
    """Mapping with attribute access for the end-to-end record workflow."""

    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            current = dict.get(self, "current_result")
            if current is not None and hasattr(current, name):
                return getattr(current, name)
            raise AttributeError(name)


SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "record.json"


def load_record(path: str | Path | None = None) -> dict[str, Any]:
    path = SAMPLE_PATH if path is None else Path(path)
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    required = (
        "part_shape", "knots_y", "knots_x", "kernel_basis", "calibrations",
        "frames", "gradient_probe", "requested_roi", "smoothness",
        "relative_tolerance", "iterations",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("缺少交换字段：" + "、".join(missing))
    return record


def build_session(record: Optional[dict] = None, *, path=None, **kwargs) -> ScanSession:
    record = load_record(path) if record is None else record
    return ScanSession.from_record(record, **kwargs)


def build_operator(
    record: Any = None,
    *,
    frames: Any = None,
    which: str = "current",
    refs=None,
    **kwargs,
) -> SpatialPSFOperator:
    """Build an operator from a record, frame list, or explicit frame selection.

    ``which='current'`` selects the highest revision of each id;
    ``which='initial'`` selects the oldest archived revision of each id;
    ``which='all'`` uses all record frames without replacement semantics.
    """
    if isinstance(record, SpatialPSFOperator) and frames is None:
        return record
    cal = kwargs.pop("calibration", None)
    part_shape = kwargs.pop("part_shape", None)
    if isinstance(record, dict) and "part_shape" in record:
        cal = cal or CalibrationSet.from_record(record)
        part_shape = part_shape or record["part_shape"]
        memory = kwargs.pop("memory", None) or MemoryPolicy.from_scale(record.get("scale"))
        if frames is None:
            session = ScanSession(calibration=cal, part_shape=part_shape, memory=memory)
            for frame in record.get("frames", []):
                session.registry.add_or_replace(frame)
            if refs is not None:
                frames = session.registry.active_frames(refs)
            elif which in ("current", "latest"):
                frames = session.registry.active_frames("current")
            elif which in ("initial", "oldest", "locked"):
                frames = session.registry.active_frames("initial")
            elif which == "all":
                frames = record["frames"]
            else:
                raise PSFError(f"未知 which={which!r}")
        kwargs.setdefault("memory", memory)
        return SpatialPSFOperator(
            frames=frames, calibration=cal, part_shape=part_shape, **kwargs
        )
    if frames is None and record is not None:
        frames = record
    return SpatialPSFOperator(
        frames=frames, calibration=cal, part_shape=part_shape, **kwargs
    )


def assess_prediction(prediction, image=None, *, frame=None, pixels=None, noise_variance=None):
    """Compare a fixed prediction with measured pixels and standardised residual."""
    pred = np.asarray(getattr(prediction, "values", prediction), dtype=float)
    if pixels is None:
        if frame is not None:
            frame = frame if hasattr(frame, "valid_mask") else None
            if frame is None:
                from spatial_psf_core import Frame
                frame = Frame.coerce(frame)
            pixels = frame.pixels
            noise_variance = frame.noise_variance if noise_variance is None else noise_variance
        elif hasattr(prediction, "frame"):
            pixels = prediction.frame.pixels
            noise_variance = prediction.frame.noise_variance
    pixels = np.asarray(pixels, dtype=float)
    residual = pred - pixels
    var = np.asarray(noise_variance if noise_variance is not None else 1.0, dtype=float)
    if var.ndim == 0:
        var = np.full_like(pred, float(var))
    z = residual / np.sqrt(var)
    return {
        "prediction": pred,
        "pixels": pixels,
        "residual": residual,
        "standardized_residual": z,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "weighted_rms": float(np.sqrt(np.mean(z ** 2))),
        "max_abs_z": float(np.max(np.abs(z))),
    }


def run_record_workflow(
    record: Optional[dict] = None,
    *,
    path=None,
    smoothness: Optional[float] = None,
    tolerance: Optional[float] = None,
    relative_tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    iterations: Optional[int] = None,
    compute_gradient: bool = True,
):
    """Run the sample scenario: locked old tier, replacement, compare and gradient."""
    record = load_record(path) if record is None else record
    if tolerance is None:
        tolerance = relative_tolerance
    tol = record["relative_tolerance"] if tolerance is None else tolerance
    if max_iterations is None:
        max_iterations = iterations
    maxit = record["iterations"] if max_iterations is None else max_iterations
    lam = float(record["smoothness"] if smoothness is None else smoothness)

    session = build_session(record)
    locked_frames = session.registry.active_frames("initial")
    locked_op = session.lock("locked", [f.ref for f in locked_frames])
    locked_result = locked_op.reconstruct(
        smoothness=lam,
        tolerance=tol,
        max_iterations=maxit,
    )
    session.results["locked"] = locked_result

    current_result = session.reconstruct(
        "current",
        smoothness=lam,
        tolerance=tol,
        max_iterations=maxit,
    )
    roi = record["requested_roi"]
    comparison = compare_reconstructions(locked_result, current_result, roi=roi, inclusive=True)
    comparison_half = compare_reconstructions(locked_result, current_result, roi=roi, inclusive=False)

    probe = record.get("gradient_probe", {})
    probe_ref = probe.get("frame")
    upstream = np.asarray(probe.get("upstream"), dtype=float) if probe.get("upstream") is not None else None
    gradient_result = None
    prediction_call = None
    if compute_gradient and probe_ref is not None and upstream is not None:
        # probe_ref is [id, revision].  This resolves the archived fast frame,
        # never the current slow replacement with the same id.
        prediction_call = session.predict(
            ref=(probe_ref[0], int(probe_ref[1])),
            image=locked_result.image,
            call_id="gradient-probe-original-call",
        )
        gradient_result = session.calibration_gradient(
            prediction_call.call_id, upstream,
            expected_calibration=prediction_call.calibration_name,
        )

    replacement_id, replacement_rev = probe_ref[0], None
    current_for_probe = None
    for frame in session.registry.current.values():
        if frame.id == replacement_id:
            replacement_rev = frame.revision
            current_for_probe = assess_prediction(
                session.predict(frame.ref, image=current_result.image, store=False),
            )

    requested_window = locked_result.window(roi, inclusive=True)
    requested_window_half = locked_result.window(roi, inclusive=False)
    return WorkflowResult({
        "record": record,
        "session": session,
        "locked": locked_result,
        "current": current_result,
        "locked_operator": locked_op,
        "current_operator": session.operator,
        "locked_result": locked_result,
        "current_result": current_result,
        "comparison": comparison,
        "comparison_half_open_roi": comparison_half,
        "requested_roi": roi,
        "roi_window": requested_window,
        "roi_window_inclusive": requested_window,
        "roi_window_half_open": requested_window_half,
        "prediction_call": prediction_call,
        "gradient": gradient_result,
        "gradient_probe": probe,
        "new_batch_prediction_check": current_for_probe,
    })


solve_record = run_record_workflow


def assess_batch(
    frames,
    *,
    result=None,
    image=None,
    operator: Optional[SpatialPSFOperator] = None,
    session: Optional[ScanSession] = None,
    locked_result=None,
    gradient=None,
    roi=None,
    z_threshold: float = 3.0,
):
    """Compare predictions for a new batch with the current interpretation.

    The returned ``change_flag`` is a transparent standardized-residual/ROI
    threshold, not an automatic physical-cause conclusion.
    """
    if image is None:
        if result is None and session is not None:
            result = session.last_result
        if result is None:
            raise PSFError("assess_batch 需要 image 或 reconstruction result")
        image = result.image
    else:
        result = result

    if isinstance(frames, (str, Frame, dict)):
        frames = [frames]
    reports = []
    predictions = {}
    for item in frames:
        frame = Frame.coerce(item) if not isinstance(item, Frame) else item
        op = operator or SpatialPSFOperator(
            frames=[frame],
            calibration=session.calibration if session is not None else result.operator.calibration,
            part_shape=image.shape,
        )
        prediction = op.predict_frame(frame, image)
        check = assess_prediction(prediction, frame=frame)
        check["frame_ref"] = frame.ref
        reports.append(check)
        predictions[frame.ref] = prediction

    comparison = None
    if locked_result is not None and result is not None:
        comparison = compare_reconstructions(locked_result, result, roi=roi)
    max_z = max((report["max_abs_z"] for report in reports), default=0.0)
    roi_max = None if comparison is None or comparison.get("roi") is None else comparison["roi"]["max_abs"]
    return {
        "frame_reports": reports,
        "predictions": predictions,
        "max_abs_standardized_residual": float(max_z),
        "z_threshold": float(z_threshold),
        "change_flag": bool(max_z > z_threshold or (roi_max is not None and roi_max > z_threshold)),
        "current_result": result,
        "locked_comparison": comparison,
        "gradient": gradient,
        "roi": roi,
    }


compare_batch = assess_batch
review_new_batch = assess_batch


def json_summary(workflow: dict[str, Any]) -> dict[str, Any]:
    def result_summary(result):
        return {
            "converged": result.converged,
            "status": result.status,
            "iterations": result.iterations,
            "objective": result.objective,
            "data_cost": result.data_cost,
            "residual_norm": result.residual_norm,
            "relative_gradient_norm": result.relative_gradient_norm,
            "frame_refs": result.frame_refs,
            "directly_observed_count": int(np.count_nonzero(result.directly_observed)),
        }

    grad = workflow.get("gradient")
    return {
        "locked": result_summary(workflow["locked_result"]),
        "current": result_summary(workflow["current_result"]),
        "comparison_metrics": workflow["comparison"]["metrics"],
        "roi_metrics": workflow["comparison"]["roi"]["l2_norm"] if workflow["comparison"].get("roi") else None,
        "roi_missing_count": len(workflow["roi_window"].missing_coordinates),
        "roi_missing_coordinates": workflow["roi_window"].missing_coordinates,
        "gradient_calibration": None if grad is None else grad.calibration_name,
        "gradient_frame_ref": None if grad is None else list(grad.frame_ref),
        "gradient": None if grad is None else np.asarray(grad).tolist(),
    }
