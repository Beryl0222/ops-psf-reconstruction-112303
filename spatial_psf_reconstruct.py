"""Minimum-norm weighted least-squares reconstruction and comparison."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np

from spatial_psf_core import PSFError
from spatial_psf_operator import Observability, SpatialPSFOperator


class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class ReconstructionResult(AttributeDict):
    """Candidate or converged reconstruction.

    The object is also a mapping so ``result["converged"]`` and
    ``result.converged`` are equivalent.  ``np.asarray(result)`` returns the
    reconstructed image.
    """

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self["image"], dtype=dtype)
        return arr.copy() if copy else arr

    @property
    def image(self) -> np.ndarray:
        return self["image"]

    @property
    def candidate(self) -> np.ndarray:
        return self["image"]

    @property
    def reconstruction(self) -> np.ndarray:
        return self["image"]

    @property
    def x(self) -> np.ndarray:
        """Flat workpiece-grid vector (SciPy/optimiser convention)."""
        return self["image"].reshape(-1)

    @property
    def shape(self):
        return self["image"].shape

    def window(self, roi, *, inclusive: bool = False) -> "InterpretationWindow":
        return InterpretationWindow.from_result(self, roi, inclusive=inclusive)

    def inclusive_window(self, roi) -> "InterpretationWindow":
        return self.window(roi, inclusive=True)

    def half_open_window(self, roi) -> "InterpretationWindow":
        return self.window(roi, inclusive=False)

    def grouped_window(self, roi, *, inclusive: bool = True) -> "InterpretationWindow":
        """Crop an ROI written as ``[y0, y1, x0, x1]``."""
        op = self["operator"]
        bounds = normalize_roi_grouped(roi, op.part_shape, inclusive=inclusive)
        sy, sx = _roi_slice(bounds)
        return InterpretationWindow(
            roi=tuple(int(v) for v in roi),
            inclusive=inclusive,
            bounds_half_open=bounds,
            image=self.image[sy, sx].copy(),
            directly_observed=self["directly_observed"][sy, sx].copy(),
            precision_diag=self["observability"][sy, sx].copy(),
        )

    def interpret(self, roi, *, inclusive: bool = True) -> "InterpretationWindow":
        """Default interpretation crop for an exchange ``requested_roi``."""
        return self.window(roi, inclusive=inclusive)


@dataclass
class InterpretationWindow:
    roi: tuple[int, int, int, int]
    inclusive: bool
    bounds_half_open: tuple[int, int, int, int]
    image: np.ndarray
    directly_observed: np.ndarray
    precision_diag: np.ndarray

    @property
    def missing_direct_observation(self) -> np.ndarray:
        return ~self.directly_observed

    @property
    def shape(self):
        return self.image.shape

    @property
    def directly_observed_mask(self) -> np.ndarray:
        return self.directly_observed

    @property
    def observed(self) -> np.ndarray:
        return self.directly_observed

    @property
    def mask(self) -> np.ndarray:
        return self.directly_observed

    @property
    def missing_mask(self) -> np.ndarray:
        return ~self.directly_observed

    @property
    def missing_count(self) -> int:
        return int(np.count_nonzero(self.missing_direct_observation))

    @property
    def is_fully_observed(self) -> bool:
        return self.missing_count == 0

    @property
    def has_missing_direct_observation(self) -> bool:
        return self.missing_count > 0

    @property
    def status(self) -> str:
        return "complete_direct_observation" if self.is_fully_observed else "missing_direct_observation"

    @property
    def missing_coordinates(self) -> list[tuple[int, int]]:
        ys, xs = np.nonzero(self.missing_direct_observation)
        y0, x0, _, _ = self.bounds_half_open
        return [(int(y0 + y), int(x0 + x)) for y, x in zip(ys, xs)]

    @property
    def observed_coordinates(self) -> list[tuple[int, int]]:
        ys, xs = np.nonzero(self.directly_observed)
        y0, x0, _, _ = self.bounds_half_open
        return [(int(y0 + y), int(x0 + x)) for y, x in zip(ys, xs)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi": self.roi,
            "inclusive": self.inclusive,
            "bounds_half_open": self.bounds_half_open,
            "image": self.image,
            "directly_observed": self.directly_observed,
            "missing_direct_observation": self.missing_direct_observation,
            "precision_diag": self.precision_diag,
            "missing_coordinates": self.missing_coordinates,
            "missing_count": self.missing_count,
            "is_fully_observed": self.is_fully_observed,
            "status": self.status,
        }


def normalize_roi(
    roi: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    inclusive: bool = False,
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = (int(round(v)) for v in roi)
    by1 = y1 + 1 if inclusive else y1
    bx1 = x1 + 1 if inclusive else x1
    H, W = shape
    if not (0 <= y0 < by1 <= H and 0 <= x0 < bx1 <= W):
        raise PSFError(
            f"ROI {roi!r} (inclusive={inclusive}) converts to "
            f"({y0},{x0},{by1},{bx1}) outside part_shape {shape}"
        )
    return y0, x0, by1, bx1


def normalize_roi_grouped(
    roi: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    inclusive: bool = False,
) -> tuple[int, int, int, int]:
    """Normalise ``[y0, y1, x0, x1]`` (axis-grouped) ROI notation."""
    y0, y1, x0, x1 = (int(round(v)) for v in roi)
    corner_roi = (y0, x0, y1, x1)
    return normalize_roi(corner_roi, shape, inclusive=inclusive)


def _roi_slice(bounds):
    y0, x0, y1, x1 = bounds
    return slice(y0, y1), slice(x0, x1)


def InterpretationWindow_from_result(result, roi, *, inclusive):  # noqa: N802
    op = result["operator"]
    bounds = normalize_roi(roi, op.part_shape, inclusive=inclusive)
    sy, sx = _roi_slice(bounds)
    direct = result.get("directly_observed")
    precision = result.get("observability")
    if direct is None or precision is None:
        obs = op.observability()
        direct = obs.directly_observed if direct is None else direct
        precision = obs.precision_diag if precision is None else precision
    return InterpretationWindow(
        roi=tuple(int(v) for v in roi),
        inclusive=inclusive,
        bounds_half_open=bounds,
        image=result.image[sy, sx].copy(),
        directly_observed=direct[sy, sx].copy(),
        precision_diag=precision[sy, sx].copy(),
    )


InterpretationWindow.from_result = staticmethod(InterpretationWindow_from_result)


def first_difference_energy(image: np.ndarray, domain: np.ndarray, smoothness: float):
    horizontal = domain[:, :-1] & domain[:, 1:]
    vertical = domain[:-1, :] & domain[1:, :]
    dh = image[:, :-1] - image[:, 1:]
    dv = image[:-1, :] - image[1:, :]
    energy = float(smoothness * (np.sum(dh[horizontal] ** 2) + np.sum(dv[vertical] ** 2)))
    return energy, horizontal, vertical, dh, dv


def laplacian_matvec(x: np.ndarray, domain: np.ndarray, smoothness: float) -> np.ndarray:
    if smoothness == 0.0:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    horizontal = domain[:, :-1] & domain[:, 1:]
    vertical = domain[:-1, :] & domain[1:, :]
    dh = x[:, :-1] - x[:, 1:]
    contribution = smoothness * horizontal * dh
    out[:, :-1] += contribution
    out[:, 1:] -= contribution
    dv = x[:-1, :] - x[1:, :]
    contribution = smoothness * vertical * dv
    out[:-1, :] += contribution
    out[1:, :] -= contribution
    return out


def reconstruction_objective(op: SpatialPSFOperator, image: np.ndarray, smoothness: float):
    image = op._image(image)
    data_cost = op.data_cost(image)
    smooth_cost = first_difference_energy(image, op.domain, smoothness)[0]
    return data_cost + smooth_cost, data_cost, smooth_cost


def reconstruct(
    problem: Any = None,
    *,
    frames=None,
    record: Optional[dict] = None,
    smoothness: float = 0.0,
    tolerance: Optional[float] = None,
    relative_tolerance: Optional[float] = None,
    rtol: Optional[float] = None,
    atol: float = 0.0,
    max_iterations: Optional[int] = None,
    iterations: Optional[int] = None,
    maxiter: Optional[int] = None,
    max_iter: Optional[int] = None,
    max_iters: Optional[int] = None,
    residual_tolerance: Optional[float] = None,
    absolute_tolerance: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
    initial: Optional[np.ndarray] = None,
    callback=None,
    operator: Optional[SpatialPSFOperator] = None,
    compute_observability: bool = True,
    strict: bool = False,
    **operator_kwargs,
) -> ReconstructionResult:
    """Solve min ||A x - y||_W^2 + smoothness ||D x||_2^2 matrix-free.

    Conjugate gradients starts at zero.  For a singular positive-semidefinite
    normal matrix this range-space solution is the minimum-Euclidean-norm
    optimal reconstruction.  A reached iteration limit still returns the
    current candidate with ``converged=False``.
    """
    if operator is not None:
        op = operator
    elif isinstance(problem, SpatialPSFOperator):
        op = problem
    else:
        rec = record
        if rec is None and isinstance(problem, Mapping) and "part_shape" in problem:
            rec = problem
        if rec is not None:
            op = SpatialPSFOperator(rec, frames=frames, **operator_kwargs)
        elif frames is not None or problem is not None:
            use_frames = frames if frames is not None else problem
            op = SpatialPSFOperator(frames=use_frames, **operator_kwargs)
        else:
            raise PSFError("reconstruct 需要 operator、record、problem 或 frames")

    tol = tolerance if tolerance is not None else relative_tolerance
    tol = rtol if rtol is not None else tol
    tol = 1e-6 if tol is None else float(tol)
    limit = max_iterations
    for value in (iterations, maxiter, max_iter, max_iters):
        if value is not None:
            limit = value
    limit = 200 if limit is None else int(limit)
    if limit < 0:
        raise PSFError("最大迭代数不能为负")
    if absolute_tolerance is not None:
        atol = absolute_tolerance
    if x0 is None:
        x0 = initial
    if x0 is not None and float(np.max(np.abs(np.asarray(x0, dtype=float)))) > 0.0:
        raise PSFError("最小欧氏范数解要求从零初值开始；不接受非零 x0")
    if callback is not None and not callable(callback):
        raise PSFError("callback 必须可调用")
    smoothness = float(smoothness)
    if smoothness < 0:
        raise PSFError("smoothness 必须非负")

    H, W = op.part_shape
    domain = op.domain

    # Normal-equation RHS b = A^T W y.
    rhs = np.zeros((H, W), dtype=float)
    for frame in op.frames:
        rhs += op.weighted_adjoint_frame(frame, frame.pixels)
    rhs *= domain
    rhs_norm = float(np.linalg.norm(rhs))
    measurement_norm_sq = 0.0
    for frame in op.frames:
        measurement_norm_sq += float(np.sum(np.where(
            frame.valid_mask,
            frame.pixels * frame.pixels / frame.safe_variance,
            0.0,
        )))
    measurement_norm = float(np.sqrt(max(measurement_norm_sq, 0.0)))
    threshold = max(float(atol), tol * rhs_norm)

    def hessian(image: np.ndarray, return_data_cost: bool = False):
        out = np.zeros((H, W), dtype=float)
        data_cost = 0.0
        for frame in op.frames:
            pred = op.predict_frame(frame, image)
            out += op.weighted_adjoint_frame(frame, pred)
            if return_data_cost:
                residual = pred - frame.pixels
                chi2 = np.where(
                    frame.valid_mask,
                    residual * residual / frame.safe_variance,
                    0.0,
                )
                data_cost += float(np.sum(chi2))
        out += laplacian_matvec(image, domain, smoothness)
        out *= domain
        return (out, data_cost) if return_data_cost else out

    image = np.zeros((H, W), dtype=float)
    residual = rhs.copy()  # image0 = 0, so H image0 = 0
    direction = residual.copy()
    rs_old = float(np.dot(residual.ravel(), residual.ravel()))
    grad_norm = float(np.sqrt(rs_old))
    iteration = 0
    status = "max_iterations"
    converged = grad_norm <= threshold

    if converged:
        status = "converged"
    elif limit == 0:
        status = "max_iterations"

    while not converged and iteration < limit:
        hp = hessian(direction)
        dhp = float(np.dot(direction.ravel(), hp.ravel()))
        if dhp <= 1e-30 * max(1.0, rs_old):
            # A null direction should not occur for a nonzero range-space r.
            himage, current_data_cost = hessian(image, return_data_cost=True)
            exact = rhs - himage
            exact_norm = float(np.linalg.norm(exact))
            data_residual_norm = float(np.sqrt(max(current_data_cost, 0.0)))
            residual_tol_met = (
                residual_tolerance is not None
                and data_residual_norm <= float(residual_tolerance)
            )
            if exact_norm <= threshold or residual_tol_met:
                residual = exact
                converged = True
                status = "converged"
                grad_norm = exact_norm
            else:
                status = "numerical_breakdown"
                grad_norm = exact_norm
            break

        alpha = rs_old / dhp
        image += alpha * direction
        residual -= alpha * hp
        rs_new = float(np.dot(residual.ravel(), residual.ravel()))
        iteration += 1
        grad_norm = float(np.sqrt(max(rs_new, 0.0)))
        if callback is not None:
            callback(image)

        # Periodically remove CG recurrence drift and verify true convergence.
        verify = (
            grad_norm <= threshold
            or iteration == limit
            or iteration % 25 == 0
            or residual_tolerance is not None
        )
        if verify:
            himage, current_data_cost = hessian(image, return_data_cost=True)
            exact = rhs - himage
            exact_norm = float(np.linalg.norm(exact))
            data_residual_norm = float(np.sqrt(max(current_data_cost, 0.0)))
            residual_tol_met = (
                residual_tolerance is not None
                and data_residual_norm <= float(residual_tolerance)
            )
            if exact_norm <= threshold or residual_tol_met:
                residual = exact
                grad_norm = exact_norm
                converged = True
                status = "converged"
                break
            if grad_norm <= threshold * 100.0 or iteration % 25 == 0:
                residual = exact
                rs_new = float(np.dot(residual.ravel(), residual.ravel()))
            grad_norm = exact_norm

        beta = max(rs_new, 0.0) / max(rs_old, 1e-300)
        direction = residual + beta * direction
        rs_old = rs_new

    # Release CG work vectors before allocating observability maps.  ``image``
    # is the only grid-sized solver state needed by the result/objective.
    rhs = residual = direction = hp = himage = None

    objective, data_cost, smooth_cost = reconstruction_objective(op, image, smoothness)
    data_residual_norm = float(np.sqrt(max(data_cost, 0.0)))
    if (
        not converged
        and residual_tolerance is not None
        and data_residual_norm <= float(residual_tolerance)
    ):
        converged = True
        status = "converged"
    if not converged and iteration >= limit and status != "numerical_breakdown":
        status = "max_iterations"
    obs = op.observability() if compute_observability else None
    observed_mask = None if obs is None else obs.directly_observed
    precision_map = None if obs is None else obs.precision_diag
    message = "converged" if converged else f"stopped after {iteration}/{limit} iterations ({status})"
    result = ReconstructionResult(
        image=image,
        reconstruction=image,
        candidate=image,
        converged=bool(converged),
        success=bool(converged),
        status=status,
        message=message,
        iterations=int(iteration),
        nit=int(iteration),
        niter=int(iteration),
        max_iterations=limit,
        tolerance=tol,
        residual_tolerance=residual_tolerance,
        atol=float(atol),
        threshold=threshold,
        gradient_norm=float(grad_norm),
        relative_gradient_norm=float(grad_norm / rhs_norm) if rhs_norm > 0 else 0.0,
        normal_rhs_norm=rhs_norm,
        objective=float(objective),
        data_cost=float(data_cost),
        smoothness_cost=float(smooth_cost),
        residual_norm=data_residual_norm,
        relative_residual_norm=(
            data_residual_norm / measurement_norm if measurement_norm > 0 else 0.0
        ),
        measurement_norm=measurement_norm,
        smoothness=smoothness,
        operator=op,
        frame_refs=op.refs,
        directly_observed=observed_mask,
        direct_observed=observed_mask,
        observed=observed_mask,
        observable=observed_mask,
        observation_mask=observed_mask,
        observability_mask=observed_mask,
        coverage_mask=observed_mask,
        observability=precision_map,
        observability_weights=precision_map,
        precision_diag=precision_map,
        coverage_abs=None if obs is None else obs.coverage_abs,
    )
    if strict and not result.converged:
        raise PSFError(message + "；当前候选仍在 result.image")
    return result


# Common descriptive aliases.
solve = reconstruct
reconstruct_min_norm = reconstruct


class ReconstructionComparison(AttributeDict):
    pass


def compare_reconstructions(
    locked: ReconstructionResult,
    current: ReconstructionResult,
    roi=None,
    *,
    inclusive: bool = False,
) -> ReconstructionComparison:
    a = np.asarray(locked.image, dtype=float)
    b = np.asarray(current.image, dtype=float)
    if a.shape != b.shape:
        raise PSFError("只能比较同一工件网格上的重建")
    diff = b - a
    domain = getattr(locked.get("operator", None), "domain", np.ones(a.shape, bool))
    metrics = {
        "l2_norm": float(np.linalg.norm(diff * domain)),
        "rmse": float(np.sqrt(np.mean((diff * domain) ** 2))),
        "max_abs": float(np.max(np.abs(diff * domain))),
        "mean_abs": float(np.mean(np.abs(diff * domain))),
    }
    roi_info = None
    if roi is not None:
        lw = locked.window(roi, inclusive=inclusive)
        cw = current.window(roi, inclusive=inclusive)
        rd = cw.image - lw.image
        roi_info = {
            "locked_window": lw,
            "current_window": cw,
            "difference": rd,
            "l2_norm": float(np.linalg.norm(rd)),
            "rmse": float(np.sqrt(np.mean(rd ** 2))),
            "max_abs": float(np.max(np.abs(rd))),
        }
    locked_frames = locked.get("frame_refs", ())
    current_frames = current.get("frame_refs", ())
    removed = tuple(ref for ref in locked_frames if ref not in current_frames)
    added = tuple(ref for ref in current_frames if ref not in locked_frames)
    return ReconstructionComparison(
        locked=locked,
        current=current,
        difference=diff,
        current_minus_locked=diff,
        locked_minus_current=-diff,
        metrics=metrics,
        roi=roi_info,
        roi_comparison=roi_info,
        removed_frame_refs=removed,
        added_frame_refs=added,
        locked_converged=locked.get("converged"),
        current_converged=current.get("converged"),
    )


compare = compare_reconstructions
