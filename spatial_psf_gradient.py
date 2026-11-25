"""Calibration-node gradients at a fixed image and fixed frame call."""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from spatial_psf_core import CalibrationSet, Frame, PSFError
from spatial_psf_operator import SpatialPSFOperator


class CalibrationGradientResult(dict):
    """Dict of profile gradients; array conversion returns the active profile."""

    def __init__(self, gradients, *, calibration, frame_ref, call_id=None, masked=False, weighted=False):
        super().__init__()
        for name, value in gradients.items():
            dict.__setitem__(self, name, np.asarray(value, dtype=float))
        self.calibration_name = str(calibration)
        self.frame_ref = tuple(frame_ref)
        self.call_id = call_id
        self.masked = masked
        self.weighted = weighted

    @property
    def gradient(self) -> np.ndarray:
        return dict.__getitem__(self, self.calibration_name)

    @property
    def shape(self):
        return self.gradient.shape

    @property
    def dtype(self):
        return self.gradient.dtype

    @property
    def gradients(self) -> dict[str, np.ndarray]:
        return {k: dict.__getitem__(self, k) for k in self.keys()}

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self.gradient, dtype=dtype)
        return arr.copy() if copy else arr

    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            # Delegate ndarray helpers such as reshape/conj to active profile.
            try:
                return getattr(self.gradient, name)
            except Exception as exc:
                raise AttributeError(name) from exc


def calibration_parameter_gradient(
    calibration: CalibrationSet,
    frame: Union[Frame, dict],
    image: np.ndarray,
    upstream: np.ndarray,
    *,
    domain: Optional[np.ndarray] = None,
    apply_mask: bool = False,
    weighted: bool = False,
    profile: Optional[str] = None,
    memory=None,
) -> np.ndarray:
    """Gradient for one exposure profile's node coefficients.

    ``upstream`` is dL/d(raw_prediction), per the exchange contract.  By
    default it is contracted directly and therefore includes every supplied
    cell; pass ``apply_mask=True`` for a fitting gradient.  ``weighted=True``
    also divides by the frame noise variance.
    """
    if not isinstance(calibration, CalibrationSet):
        raise PSFError("第一个参数必须是 CalibrationSet")
    frame = Frame.coerce(frame)
    profile = profile or frame.calibration
    calibration.require_profile(profile)
    if profile != frame.calibration:
        raise PSFError(
            f"梯度档位 {profile!r} 与原调用帧档位 {frame.calibration!r} 不一致"
        )

    upstream = np.asarray(upstream, dtype=float)
    if upstream.shape != frame.shape:
        raise PSFError(f"upstream 形状应为 {frame.shape}，实际 {upstream.shape}")
    if weighted:
        upstream = np.where(frame.valid_mask, upstream / frame.safe_variance, 0.0)
    elif apply_mask:
        upstream = np.where(frame.valid_mask, upstream, 0.0)

    H, W = np.asarray(image).shape
    if domain is None:
        domain = np.ones((H, W), dtype=bool)
    domain = np.asarray(domain, dtype=bool)
    work = np.where(domain, np.asarray(image, dtype=float), 0.0)

    ny, nx, b_count = calibration.profiles[profile].shape
    grad = np.zeros((ny, nx, b_count), dtype=float)
    py, px = (k // 2 for k in calibration.kernel_shape)
    padded = np.pad(work, ((py, py), (px, px)), constant_values=0.0)
    basis = calibration.basis
    from spatial_psf_core import MemoryPolicy
    policy = memory if isinstance(memory, MemoryPolicy) else MemoryPolicy()

    for t0, t1 in policy.slices(frame.height, frame.width, b_count):
        ys = np.arange(frame.origin[0] + t0, frame.origin[0] + t1)
        xs = np.arange(frame.origin[1], frame.origin[1] + frame.width)
        iy, ix, weights = calibration.bilinear_terms(ys, xs)
        ny, nx = calibration.profiles[profile].shape[:2]
        corner_y = (iy, iy, np.minimum(iy + 1, ny - 1), np.minimum(iy + 1, ny - 1))
        corner_x = (ix, np.minimum(ix + 1, nx - 1), ix, np.minimum(ix + 1, nx - 1))
        tile_up = upstream[t0:t1]
        if weighted or apply_mask:
            # Already masked above; this keeps the semantic explicit.
            tile_up = tile_up * frame.valid_mask[t0:t1]

        # Basis-only predictions C_b at each observation in this tile.
        corr = np.zeros((t1 - t0, frame.width, b_count), dtype=float)
        for a in range(basis.shape[1]):
            for b in range(basis.shape[2]):
                patch = padded[
                    frame.origin[0] + t0 + a:frame.origin[0] + t1 + a,
                    frame.origin[1] + b:frame.origin[1] + frame.width + b,
                ]
                corr += patch[..., None] * basis[None, None, :, a, b]

        for node_weight, y_index, x_index in zip(weights, corner_y, corner_x):
            flat_y = np.broadcast_to(y_index, (t1 - t0, frame.width)).ravel()
            flat_x = np.broadcast_to(x_index, (t1 - t0, frame.width)).ravel()
            scalars = (tile_up * node_weight).ravel()
            values = corr.reshape(-1, b_count) * scalars[:, None]
            for b in range(b_count):
                np.add.at(grad[:, :, b], (flat_y, flat_x), values[:, b])
    return grad


def calibration_gradient(
    problem: Any = None,
    upstream: Optional[np.ndarray] = None,
    *args,
    calibration: Optional[CalibrationSet] = None,
    cal: Optional[CalibrationSet] = None,
    frame: Union[Frame, dict, None] = None,
    image: Optional[np.ndarray] = None,
    apply_mask: bool = False,
    weighted: bool = False,
    expected_calibration: Optional[str] = None,
    **kwargs,
) -> CalibrationGradientResult:
    """Return gradients for all profiles (zero except the called frame's tier).

    Supported positional forms include
    ``calibration_gradient(cal, frame, image, upstream)`` and
    ``calibration_gradient(operator, upstream, frame=..., image=...)``.
    """
    # Permit the fully positional calibration/frame/image/upstream form.
    if args:
        if frame is None and image is None and len(args) == 2:
            image, actual_upstream = args
            frame = upstream
            upstream = actual_upstream
        elif len(args) == 3:
            frame, image, upstream = args
        else:
            raise PSFError("无法解析 calibration_gradient 的位置参数")

    if isinstance(problem, SpatialPSFOperator):
        op = problem
        calibration = calibration or op.calibration
        if frame is None:
            if len(op.frames) != 1:
                raise PSFError("多帧算子请用 frame=(id,revision) 或先创建 PredictionCall")
            frame = op.frames[0]
        elif not isinstance(frame, Frame):
            # Allow frame key within operator.
            idx = op._frame_index_for_key(frame)
            frame = op.frames[idx]
        if image is None:
            raise PSFError("标定梯度需要固定工件图 image")
        domain = op.domain
    elif isinstance(problem, CalibrationSet):
        calibration = problem
        domain = kwargs.get("domain")
    elif calibration is None and cal is not None:
        calibration = cal
        domain = kwargs.get("domain")
    else:
        calibration = calibration
        domain = kwargs.get("domain")

    if calibration is None or frame is None or image is None or upstream is None:
        raise PSFError("用法：calibration_gradient(calibration, frame, image, upstream)")
    frame = Frame.coerce(frame)
    if expected_calibration is not None and expected_calibration != frame.calibration:
        from spatial_psf_core import CalibrationMismatchError
        raise CalibrationMismatchError(
            f"原调用标识要求 {expected_calibration!r}，但帧 {frame.ref} 使用 {frame.calibration!r}"
        )

    active = calibration_parameter_gradient(
        calibration, frame, image, upstream, domain=domain,
        apply_mask=apply_mask, weighted=weighted,
        memory=kwargs.get("memory"),
    )
    gradients = {name: np.zeros_like(calibration.profiles[name]) for name in calibration.profile_names}
    gradients[frame.calibration] = active
    return CalibrationGradientResult(
        gradients,
        calibration=frame.calibration,
        frame_ref=frame.ref,
        masked=apply_mask or weighted,
        weighted=weighted,
    )
