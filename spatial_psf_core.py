"""Memory-bounded spatially varying PSF forward/adjoint operators."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


class PSFError(ValueError):
    """Base class for protocol and numerical errors in this package."""


class FrameConflictError(PSFError):
    """A frame identity/version substitution is inconsistent."""


class RevisionConflictError(FrameConflictError):
    """A stale or non-increasing revision was submitted."""


class CalibrationMismatchError(PSFError):
    """A late gradient does not match the original prediction call."""


def _as_yx(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        y = value.get("y", value.get("row", value.get("0")))
        x = value.get("x", value.get("col", value.get("column", value.get("1"))))
        if y is None or x is None:
            raise PSFError(f"无法解析行列坐标: {value!r}")
    else:
        arr = np.asarray(value)
        if arr.shape != (2,):
            raise PSFError(f"坐标必须包含行、列两个数: {value!r}")
        y, x = arr[0], arr[1]
    yf, xf = float(y), float(x)
    if not (yf.is_integer() and xf.is_integer()):
        raise PSFError(f"采样原点必须对齐整数工件像素: {value!r}")
    return int(yf), int(xf)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _bad_mask(value: Any, shape: tuple[int, ...], *, invalid_semantics: bool) -> np.ndarray:
    """Convert coordinates or masks to a boolean *bad-pixel* mask."""
    if value is None:
        return np.zeros(shape, dtype=bool)
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value)
    if arr.size == 0:
        return np.zeros(shape, dtype=bool)

    # Exchange format: an (N, 2) array of relative [row, col] coordinates.
    if arr.ndim == 2 and arr.shape[1] == 2 and not (
        arr.dtype == bool and arr.shape == shape
    ):
        mask = np.zeros(shape, dtype=bool)
        coords = arr.astype(int, copy=False)
        if coords.size:
            if np.any(coords < 0) or np.any(coords >= np.asarray(shape)):
                raise PSFError(f"掩码坐标越界: {value!r} for shape {shape}")
            mask[coords[:, 0], coords[:, 1]] = True
        return mask

    if arr.shape != shape:
        raise PSFError(f"掩码形状应为 {shape}，实际为 {arr.shape}")
    if arr.dtype == bool:
        return arr.copy() if invalid_semantics else ~arr
    # Numeric masks: nonzero means marked/bad for invalid/saturated inputs.
    return arr != 0 if invalid_semantics else arr == 0


def _valid_mask(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    arr = np.asarray(value)
    if arr.shape != shape:
        raise PSFError(f"有效掩码形状应为 {shape}，实际为 {arr.shape}")
    return arr.astype(bool, copy=False)


@dataclass(frozen=True)
class MemoryPolicy:
    """Tile sizing used to avoid materialising kernels for every strip at once."""

    tile_rows: int = 32
    scratch_bytes: int = 32 * 1024 * 1024

    @classmethod
    def from_scale(cls, scale: Optional[Mapping[str, Any]]) -> "MemoryPolicy":
        scale = scale or {}
        return cls(
            tile_rows=int(scale.get("tile_rows", 32)),
            scratch_bytes=int(scale.get("scratch_bytes", 32 * 1024 * 1024)),
        )

    def slices(self, height: int, width: int, channels: int = 3):
        rows = max(1, int(self.tile_rows))
        # Conservative fallback if a deployment supplies a tiny scratch budget.
        per_pixel_bytes = 8 * max(8, channels + 5)
        rows = min(rows, max(1, int(self.scratch_bytes) // max(1, width * per_pixel_bytes)))
        for start in range(0, height, rows):
            end = min(height, start + rows)
            yield start, end


class CalibrationSet:
    """Bilinear calibration-grid interpolation and spatial kernels."""

    def __init__(
        self,
        kernel_basis: ArrayLike,
        calibrations: Union[Mapping[str, ArrayLike], Sequence[ArrayLike]],
        knots_y: ArrayLike,
        knots_x: ArrayLike,
        profile_names: Optional[Sequence[str]] = None,
    ):
        self.basis = np.asarray(kernel_basis, dtype=float)
        if self.basis.ndim != 3 or self.basis.shape[1] != self.basis.shape[2]:
            raise PSFError("kernel_basis 形状应为 (B, kh, kw) 且 kh == kw")
        if self.basis.shape[1] % 2 == 0:
            raise PSFError("当前实现要求奇数尺寸、中心为偏移零点的核")
        self.knots_y = np.asarray(knots_y, dtype=float).reshape(-1)
        self.knots_x = np.asarray(knots_x, dtype=float).reshape(-1)
        if np.any(np.diff(self.knots_y) <= 0) or np.any(np.diff(self.knots_x) <= 0):
            raise PSFError("标定网格节点必须严格递增")

        if isinstance(calibrations, Mapping):
            self.profiles = {str(k): np.asarray(v, dtype=float) for k, v in calibrations.items()}
        else:
            names = list(profile_names or [str(i) for i in range(len(calibrations))])
            self.profiles = {n: np.asarray(v, dtype=float) for n, v in zip(names, calibrations)}
        if not self.profiles:
            raise PSFError("至少需要一个曝光档标定")
        b_count = self.basis.shape[0]
        for name, cal in self.profiles.items():
            expected = (len(self.knots_y), len(self.knots_x), b_count)
            if cal.shape != expected:
                raise PSFError(f"标定 {name!r} 形状应为 {expected}，实际 {cal.shape}")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CalibrationSet":
        return cls(
            kernel_basis=record["kernel_basis"],
            calibrations=record["calibrations"],
            knots_y=record["knots_y"],
            knots_x=record["knots_x"],
        )

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    @property
    def kernel_shape(self) -> tuple[int, int]:
        return self.basis.shape[1:]

    def require_profile(self, name: str) -> str:
        if name not in self.profiles:
            raise PSFError(f"未知曝光档/标定 {name!r}；可用：{', '.join(self.profiles)}")
        return str(name)

    def bilinear_terms(self, y: ArrayLike, x: ArrayLike):
        ys = np.atleast_1d(np.asarray(y, dtype=float))
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        if ys.ndim != 1 or xs.ndim != 1:
            raise PSFError("y、x 坐标必须是一维数组或标量")
        ny, nx = len(self.knots_y), len(self.knots_x)
        if ny == 1:
            iy = np.zeros_like(ys, dtype=int)
            ty = np.zeros_like(ys)
        else:
            jy = np.clip(np.searchsorted(self.knots_y, ys, side="right") - 1, 0, ny - 2)
            iy = jy
            dy = self.knots_y[jy + 1] - self.knots_y[jy]
            ty = np.clip((ys - self.knots_y[jy]) / dy, 0.0, 1.0)
        if nx == 1:
            ix = np.zeros_like(xs, dtype=int)
            tx = np.zeros_like(xs)
        else:
            jx = np.clip(np.searchsorted(self.knots_x, xs, side="right") - 1, 0, nx - 2)
            ix = jx
            dx = self.knots_x[jx + 1] - self.knots_x[jx]
            tx = np.clip((xs - self.knots_x[jx]) / dx, 0.0, 1.0)
        iy2 = iy[:, None]
        ix2 = ix[None, :]
        ty = ty[:, None]
        tx = tx[None, :]
        w00 = (1.0 - ty) * (1.0 - tx)
        w01 = (1.0 - ty) * tx
        w10 = ty * (1.0 - tx)
        w11 = ty * tx
        return iy2, ix2, (w00, w01, w10, w11)

    def coefficients(self, y: ArrayLike, x: ArrayLike, profile: str) -> np.ndarray:
        profile = self.require_profile(profile)
        iy, ix, weights = self.bilinear_terms(y, x)
        # Singleton grids have zero upper-corner weights; clipping keeps the
        # indexing expression in bounds without changing the interpolation.
        iy1 = np.minimum(iy + 1, len(self.knots_y) - 1)
        ix1 = np.minimum(ix + 1, len(self.knots_x) - 1)
        cal = self.profiles[profile]
        w00, w01, w10, w11 = weights
        coef = (
            w00[..., None] * cal[iy, ix, :]
            + w01[..., None] * cal[iy, ix1, :]
            + w10[..., None] * cal[iy1, ix, :]
            + w11[..., None] * cal[iy1, ix1, :]
        )
        scalar = np.ndim(y) == 0 and np.ndim(x) == 0
        return coef[0, 0] if scalar else coef

    def kernel_at(self, y: float, x: float, profile: str) -> np.ndarray:
        coef = self.coefficients(y, x, profile)
        return np.einsum("b,bkl->kl", coef, self.basis)


class Frame:
    """One strip/frame with its own exposure profile, mask and variance."""

    def __init__(
        self,
        pixels: Optional[ArrayLike | Mapping[str, Any]] = None,
        id: Optional[str] = None,  # noqa: A002 - protocol field name
        calibration: Optional[str] = None,
        origin: ArrayLike = (0, 0),
        revision: int = 1,
        noise_variance: Optional[Union[float, ArrayLike]] = None,
        valid_mask: Optional[ArrayLike] = None,
        invalid: Optional[ArrayLike] = None,
        saturated: Optional[ArrayLike] = None,
        saturation_value: Optional[float] = None,
        mask: Optional[ArrayLike] = None,
        frame_id: Optional[str] = None,
        roi: Any = None,
        noise_std: Optional[float] = None,
        exposure: Optional[str] = None,
        profile: Optional[str] = None,
        version: Optional[int] = None,
        **metadata: Any,
    ):
        if isinstance(pixels, Mapping):
            other = Frame.from_mapping(pixels)
            self.__dict__.update(other.__dict__)
            return

        if id is None:
            id = frame_id
        if pixels is None:
            for key in ("data", "values", "image_data", "brightness", "image"):
                if key in metadata and metadata[key] is not None:
                    pixels = metadata.pop(key)
                    break
        if id is None:
            raise PSFError("帧必须包含 id")
        cal = calibration or exposure or profile
        if cal is None:
            raise PSFError(f"帧 {id!r} 必须包含 calibration/exposure")
        if pixels is None:
            raise PSFError(f"帧 {id!r} 必须包含 pixels")

        self.id = str(id)
        self.calibration = str(cal)
        self.revision = int(revision if version is None else version)
        self.pixels = np.asarray(pixels, dtype=float).copy()
        if self.pixels.ndim != 2:
            raise PSFError(f"帧 {self.id!r} 的 pixels 必须是二维数组")
        self.roi_input = roi
        if origin is None and roi is not None:
            origin = self._origin_from_roi(roi, self.pixels.shape)
        self.origin = _as_yx((0, 0) if origin is None else origin)
        if noise_variance is None:
            noise_variance = 1.0 if noise_std is None else float(noise_std) ** 2
        variance = np.asarray(noise_variance, dtype=float)
        if variance.ndim == 0:
            self.noise_variance = np.full(self.pixels.shape, float(variance))
        else:
            self.noise_variance = variance.copy()
            if self.noise_variance.shape != self.pixels.shape:
                raise PSFError("noise_variance 标量或与 pixels 同形状")

        valid = np.ones(self.pixels.shape, dtype=bool)
        if valid_mask is not None:
            valid &= _valid_mask(valid_mask, self.pixels.shape)
        if mask is not None:
            valid &= _valid_mask(mask, self.pixels.shape)
        if invalid is not None:
            valid &= ~_bad_mask(invalid, self.pixels.shape, invalid_semantics=True)
        if saturated is not None:
            valid &= ~_bad_mask(saturated, self.pixels.shape, invalid_semantics=True)
        if saturation_value is not None:
            level = float(saturation_value)
            valid &= ~(self.pixels >= level)
        self.valid_mask = valid

        bad_var = self.valid_mask & ~(self.noise_variance > 0)
        if np.any(bad_var):
            raise PSFError(f"帧 {self.id!r} 的有效像素噪声方差必须为正")
        self.safe_variance = np.where(self.valid_mask, self.noise_variance, 1.0)
        self.metadata = dict(metadata)

    @staticmethod
    def _origin_from_roi(roi: Any, shape: tuple[int, int]) -> tuple[int, int]:
        if isinstance(roi, Mapping):
            origin = _first(roi, "origin", "start", "offset", "top_left")
            if origin is not None:
                return _as_yx(origin)
            y0 = _first(roi, "y0", "top", "row_start", "ymin")
            x0 = _first(roi, "x0", "left", "col_start", "xmin")
            if y0 is not None and x0 is not None:
                return int(y0), int(x0)
        arr = np.asarray(roi)
        if arr.shape == (2,):
            return _as_yx(arr)
        if arr.shape == (4,):
            y0, x0, a, b = (int(v) for v in arr)
            h, w = shape
            # Accept both [y0,x0,y1,x1) and inclusive [y0,x0,y1,x1].
            if a - y0 == h and b - x0 == w:
                return y0, x0
            if a - y0 + 1 == h and b - x0 + 1 == w:
                return y0, x0
            # Or [y0, x0, height, width].
            if a == h and b == w:
                return y0, x0
            return y0, x0
        raise PSFError(f"无法解析 ROI: {roi!r}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Frame":
        d = dict(data)
        pixels = _first(d, "pixels", "values", "data", "image", "brightness", "frame")
        frame_id = _first(d, "id", "frame_id", "strip_id")
        calibration = _first(d, "calibration", "exposure", "profile", "tier")
        origin = _first(d, "origin", "offset", "top_left", "position", "sample_origin")
        revision = _first(d, "revision", "version", "rev", default=1)
        variance = _first(d, "noise_variance", "variance", "noise_var", "sigma2")
        noise_std = _first(d, "noise_std", "sigma")
        valid_mask = _first(d, "valid_mask", "valid", "mask_valid")
        mask = d.get("mask") if "valid_mask" not in d else None
        invalid = _first(d, "invalid", "invalid_pixels", "bad_pixels", "invalid_mask")
        saturated = _first(d, "saturated", "saturation_mask", "saturated_pixels")
        saturation_value = _first(d, "saturation_value", "saturation_level", "saturate_at")
        roi = _first(d, "roi", "region_of_interest", "sample_window", "window")
        known = {
            "pixels", "values", "data", "image", "brightness", "frame", "id", "frame_id",
            "strip_id", "calibration", "exposure", "profile", "tier", "origin", "offset",
            "top_left", "position", "sample_origin", "revision", "version", "rev",
            "noise_variance", "variance", "noise_var", "sigma2", "noise_std", "sigma",
            "valid_mask", "valid", "mask_valid", "mask", "invalid", "invalid_pixels",
            "bad_pixels", "invalid_mask", "saturated", "saturation_mask",
            "saturated_pixels", "saturation_value", "saturation_level",
            "saturate_at", "roi", "region_of_interest", "sample_window", "window",
        }
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            pixels=pixels, id=frame_id, calibration=calibration, origin=(0, 0) if origin is None else origin,
            revision=revision, noise_variance=variance, valid_mask=valid_mask, invalid=invalid,
            saturated=saturated, saturation_value=saturation_value,
            mask=mask, roi=roi, noise_std=noise_std, **extra,
        )

    @classmethod
    def coerce(cls, value: Union["Frame", Mapping[str, Any]]) -> "Frame":
        return value if isinstance(value, Frame) else cls.from_mapping(value)

    @property
    def frame_id(self) -> str:
        return self.id

    @property
    def ref(self) -> tuple[str, int]:
        return self.id, self.revision

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def weights(self) -> np.ndarray:
        return self.valid_mask / self.safe_variance

    @property
    def mask(self) -> np.ndarray:
        """Valid-pixel mask (True means measured and usable)."""
        return self.valid_mask

    @property
    def valid(self) -> np.ndarray:
        return self.valid_mask

    @property
    def invalid_mask(self) -> np.ndarray:
        return ~self.valid_mask

    @property
    def variance(self) -> np.ndarray:
        return self.noise_variance

    @property
    def roi(self) -> tuple[int, int, int, int]:
        if self.roi_input is not None:
            roi = self.roi_input
            if not isinstance(roi, Mapping) and len(np.asarray(roi).shape) == 1 and len(roi) == 4:
                return tuple(int(v) for v in roi)
        oy, ox = self.origin
        return oy, ox, oy + self.height, ox + self.width

    def same_content(self, other: "Frame") -> bool:
        other = Frame.coerce(other)
        return (
            self.id == other.id
            and self.revision == other.revision
            and self.calibration == other.calibration
            and self.origin == other.origin
            and self.pixels.shape == other.pixels.shape
            and np.array_equal(self.pixels, other.pixels)
            and np.array_equal(self.valid_mask, other.valid_mask)
            and np.array_equal(self.noise_variance, other.noise_variance)
        )

    def copy(self) -> "Frame":
        new = Frame.__new__(Frame)
        new.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})
        new.metadata = dict(self.metadata)
        return new

    def __repr__(self) -> str:
        return f"Frame(id={self.id!r}, revision={self.revision}, calibration={self.calibration!r}, shape={self.shape})"
