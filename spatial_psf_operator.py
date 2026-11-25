"""Matrix-free multi-strip spatially varying PSF operator."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from spatial_psf_core import CalibrationSet, Frame, MemoryPolicy, PSFError


@dataclass
class Observability:
    direct_observed: np.ndarray
    precision_diag: np.ndarray
    coverage_abs: np.ndarray

    @property
    def missing_direct_observation(self) -> np.ndarray:
        return ~self.direct_observed

    @property
    def directly_observed(self) -> np.ndarray:
        return self.direct_observed

    @property
    def observed(self) -> np.ndarray:
        return self.direct_observed

    @property
    def mask(self) -> np.ndarray:
        return self.direct_observed

    @property
    def missing(self) -> np.ndarray:
        return ~self.direct_observed

    # Convenience for dict-style callers.
    def __getitem__(self, key):
        if key in ("direct_observed", "observed", "mask"):
            return self.direct_observed
        if key in ("missing", "missing_direct_observation"):
            return self.missing_direct_observation
        if key in ("precision_diag", "observability", "weights"):
            return self.precision_diag
        if key == "coverage_abs":
            return self.coverage_abs
        raise KeyError(key)


class SpatialPSFOperator:
    """Forward and adjoint products over a fixed active frame set.

    The operator never forms a dense observation matrix.  Frames are processed
    one at a time and each frame is tiled; only one tile's interpolated
    calibration coefficients/temporary arrays are alive at once.
    """

    def __init__(
        self,
        first: Any = None,
        frames: Optional[Sequence[Union[Frame, Mapping[str, Any]]]] = None,
        *,
        part_shape: Optional[tuple[int, int]] = None,
        calibration: Optional[CalibrationSet] = None,
        kernel_basis=None,
        calibrations=None,
        knots_y=None,
        knots_x=None,
        domain: Optional[np.ndarray] = None,
        valid_domain: Optional[np.ndarray] = None,
        part_mask: Optional[np.ndarray] = None,
        memory: Optional[MemoryPolicy] = None,
        tile_rows: Optional[int] = None,
        scratch_bytes: Optional[int] = None,
        which: str = "current",
        **kwargs,
    ):
        record = first if isinstance(first, Mapping) and "part_shape" in first else None
        if record is not None:
            part_shape = part_shape or record["part_shape"]
            calibration = calibration or CalibrationSet.from_record(record)
            if memory is None and tile_rows is None and scratch_bytes is None:
                memory = MemoryPolicy.from_scale(record.get("scale"))
            if frames is None:
                record_frames = list(record.get("frames", []))
                if which == "all":
                    frames = record_frames
                else:
                    selected: dict[str, Frame] = {}
                    for item in record_frames:
                        candidate = Frame.coerce(item)
                        incumbent = selected.get(candidate.id)
                        if incumbent is None:
                            selected[candidate.id] = candidate
                        elif which in ("initial", "oldest"):
                            if candidate.revision < incumbent.revision:
                                selected[candidate.id] = candidate
                        elif candidate.revision >= incumbent.revision:
                            # Equal revisions are validated by FrameRegistry;
                            # low-level selection keeps the last exchange entry.
                            selected[candidate.id] = candidate
                    frames = list(selected.values())
        elif first is not None and frames is None:
            if isinstance(first, SpatialPSFOperator):
                frames = first.frames
                calibration = calibration or first.calibration
                part_shape = part_shape or first.part_shape
                domain = domain if domain is not None else first.domain
                memory = memory or first.memory
            elif isinstance(first, Frame):
                frames = [first]
            elif isinstance(first, (list, tuple)):
                frames = first
            elif isinstance(first, Mapping) and ("pixels" in first or "values" in first):
                frames = [first]
            elif isinstance(first, Mapping):
                # A bare calibration mapping is handled by keyword constructors.
                raise PSFError("无法从映射构造算子；记录请使用 build_operator(record)，帧映射会自动识别 pixels")

        if calibration is None:
            if kernel_basis is None or calibrations is None:
                raise PSFError("必须提供 CalibrationSet 或 kernel_basis/calibrations/knots")
            calibration = CalibrationSet(kernel_basis, calibrations, knots_y, knots_x)
        if part_shape is None:
            raise PSFError("必须提供 part_shape")
        self.part_shape = (int(part_shape[0]), int(part_shape[1]))
        self.calibration = calibration
        self.frames: list[Frame] = [Frame.coerce(f) for f in (frames or [])]
        for frame in self.frames:
            self.calibration.require_profile(frame.calibration)
            oy, ox = frame.origin
            h, w = frame.shape
            if oy < 0 or ox < 0 or oy + h > self.H or ox + w > self.W:
                raise PSFError(
                    f"帧 {frame.ref} 的采样窗 {frame.origin}+{frame.shape} 超出工件 {self.part_shape}"
                )

        domain_value = domain if domain is not None else valid_domain
        domain_value = domain_value if domain_value is not None else part_mask
        if domain_value is None:
            self.domain = np.ones(self.part_shape, dtype=bool)
        else:
            arr = np.asarray(domain_value)
            if arr.shape == self.part_shape:
                self.domain = arr.astype(bool, copy=False)
            elif arr.ndim == 2 and arr.shape[1] == 2:
                coords = arr.astype(int, copy=False)
                mask = np.zeros(self.part_shape, dtype=bool)
                mask[coords[:, 0], coords[:, 1]] = True
                self.domain = mask
            else:
                raise PSFError(
                    f"domain 应为 {self.part_shape} 布尔掩码或 (N,2) 坐标，实际 {arr.shape}"
                )

        if memory is not None:
            self.memory = memory
        else:
            self.memory = MemoryPolicy(
                tile_rows=32 if tile_rows is None else int(tile_rows),
                scratch_bytes=MemoryPolicy().scratch_bytes if scratch_bytes is None else int(scratch_bytes),
            )

    @property
    def H(self) -> int:
        return self.part_shape[0]

    @property
    def W(self) -> int:
        return self.part_shape[1]

    @property
    def n_pixels(self) -> int:
        return self.H * self.W

    @property
    def n_domain(self) -> int:
        return int(np.count_nonzero(self.domain))

    def pack_domain(self, image: np.ndarray) -> np.ndarray:
        """Pack only workpiece-domain pixels into a compressed vector."""
        arr = self._image(image)
        return arr[self.domain].reshape(-1)

    def unpack_domain(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size == self.n_pixels:
            return np.asarray(vector).reshape(self.part_shape) * self.domain
        if vector.size != self.n_domain:
            raise PSFError(f"域向量长度应为 {self.n_domain}，实际 {vector.size}")
        image = np.zeros(self.part_shape, dtype=float)
        image[self.domain] = vector
        return image

    def domain_forward(self, vector: np.ndarray, **kwargs):
        return self.forward(self.unpack_domain(vector), **kwargs)

    def domain_adjoint(self, upstream, **kwargs) -> np.ndarray:
        image = self.adjoint(upstream, **kwargs)
        return image[self.domain].reshape(-1)

    def domain_observed_adjoint(self, vector: np.ndarray) -> np.ndarray:
        return self.observed_adjoint(vector)[self.domain].reshape(-1)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    @property
    def refs(self) -> tuple[tuple[str, int], ...]:
        return tuple(f.ref for f in self.frames)

    def _image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=float)
        if arr.shape != self.part_shape:
            if arr.size == self.n_pixels:
                arr = arr.reshape(self.part_shape)
            else:
                raise PSFError(f"图像形状应为 {self.part_shape}，实际 {arr.shape}")
        return arr

    def _domain_image(self, image: np.ndarray) -> np.ndarray:
        return np.where(self.domain, self._image(image), 0.0)

    def _coef_tile(self, frame: Frame, t0: int, t1: int) -> np.ndarray:
        ys = np.arange(frame.origin[0] + t0, frame.origin[0] + t1)
        xs = np.arange(frame.origin[1], frame.origin[1] + frame.width)
        return self.calibration.coefficients(ys, xs, frame.calibration)

    def predict_frame(self, frame: Union[Frame, Mapping[str, Any]], image: np.ndarray) -> np.ndarray:
        """Raw prediction for every sample, including currently invalid cells."""
        frame = Frame.coerce(frame)
        work = self._domain_image(image)
        py, px = (k // 2 for k in self.calibration.kernel_shape)
        padded = np.pad(work, ((py, py), (px, px)), mode="constant", constant_values=0.0)
        pred = np.zeros(frame.shape, dtype=float)
        basis = self.calibration.basis
        for t0, t1 in self.memory.slices(frame.height, frame.width, basis.shape[0]):
            coef = self._coef_tile(frame, t0, t1)
            for a in range(basis.shape[1]):
                for b in range(basis.shape[2]):
                    tap = coef @ basis[:, a, b]
                    patch = padded[
                        frame.origin[0] + t0 + a:frame.origin[0] + t1 + a,
                        frame.origin[1] + b:frame.origin[1] + frame.width + b,
                    ]
                    pred[t0:t1, :] += tap * patch
        return pred

    forward_frame = predict_frame

    def adjoint_frame(
        self,
        frame: Union[Frame, Mapping[str, Any]],
        upstream: np.ndarray,
    ) -> np.ndarray:
        """Exact transpose of :meth:`predict_frame` for arbitrary upstream."""
        frame = Frame.coerce(frame)
        values = np.asarray(upstream, dtype=float)
        if values.shape != frame.shape:
            raise PSFError(f"伴随余量形状应为 {frame.shape}，实际 {values.shape}")
        py, px = (k // 2 for k in self.calibration.kernel_shape)
        adj = np.zeros((self.H + 2 * py, self.W + 2 * px), dtype=float)
        basis = self.calibration.basis
        for t0, t1 in self.memory.slices(frame.height, frame.width, basis.shape[0]):
            coef = self._coef_tile(frame, t0, t1)
            tile_values = values[t0:t1]
            for a in range(basis.shape[1]):
                for b in range(basis.shape[2]):
                    tap = coef @ basis[:, a, b]
                    contribution = tile_values * tap
                    ys = slice(frame.origin[0] + t0 + a, frame.origin[0] + t1 + a)
                    xs = slice(frame.origin[1] + b, frame.origin[1] + frame.width + b)
                    adj[ys, xs] += contribution
        if py == 0 and px == 0:
            out = adj[: self.H, : self.W]
        else:
            out = adj[py:-py, px:-px]
        return out * self.domain

    transpose_frame = adjoint_frame

    def weighted_adjoint_frame(
        self,
        frame: Union[Frame, Mapping[str, Any]],
        residual: np.ndarray,
    ) -> np.ndarray:
        """A^T M W r for one raw residual array (W = diag(1/noise_variance))."""
        frame = Frame.coerce(frame)
        residual = np.asarray(residual, dtype=float)
        weighted = np.where(frame.valid_mask, residual / frame.safe_variance, 0.0)
        return self.adjoint_frame(frame, weighted)

    def _frame_index_for_key(self, key: Any) -> int:
        if isinstance(key, int):
            return key
        if isinstance(key, tuple) and len(key) == 2:
            for i, frame in enumerate(self.frames):
                if frame.ref == (str(key[0]), int(key[1])):
                    return i
        key = str(key)
        if "@" in key:
            fid, rev = key.rsplit("@", 1)
            return self._frame_index_for_key((fid, int(rev)))
        matches = [i for i, f in enumerate(self.frames) if f.id == key]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"当前算子中没有帧 {key!r}")
        raise KeyError(f"帧 id {key!r} 有多个版本；请使用 (id, revision) 或 'id@revision'")

    def forward(
        self,
        image: np.ndarray,
        *,
        valid_only: bool = False,
        whitened: bool = False,
    ):
        """Raw per-frame predictions, packed valid predictions, or sqrt(W)Ax."""
        if whitened:
            return self.whitened_forward(image)
        if valid_only:
            return self.observed_forward(image)
        return [self.predict_frame(frame, image) for frame in self.frames]

    predict = forward

    def _normalise_upstream_items(self, upstream):
        if isinstance(upstream, Mapping):
            return [(self._frame_index_for_key(k), v) for k, v in upstream.items()]
        if isinstance(upstream, np.ndarray):
            if upstream.ndim == 1:
                return "valid_vector", upstream
            if len(self.frames) == 1 and upstream.shape == self.frames[0].shape:
                return [(0, upstream)]
            # Shape (n_frames, h, w), or any sequence handled below.
            return list(enumerate(upstream))
        return list(enumerate(upstream))

    def adjoint(
        self,
        upstream: Union[np.ndarray, Sequence[np.ndarray], Mapping[Any, np.ndarray]],
        *,
        weighted: bool = False,
        whitened: bool = False,
        apply_mask: bool = False,
    ) -> np.ndarray:
        """Raw transpose by default.

        ``weighted=True`` computes A^T M W r; ``whitened=True`` expects the
        packed whitened data vector and computes A^T sqrt(W) v.  Use
        :meth:`weighted_adjoint` when passing named residual arrays.
        """
        if whitened:
            if not isinstance(upstream, np.ndarray) or upstream.ndim != 1:
                raise PSFError("whitened adjoint 需要一维打包向量")
            return self.whitened_adjoint(upstream)
        if weighted:
            return self.weighted_adjoint(upstream)

        items = self._normalise_upstream_items(upstream)
        if items and items[0] == "valid_vector":
            return self.observed_adjoint(items[1])
        out = np.zeros(self.part_shape, dtype=float)
        for index, values in items:
            if apply_mask:
                frame = self.frames[index]
                values = np.where(frame.valid_mask, values, 0.0)
            out += self.adjoint_frame(self.frames[index], values)
        return out

    def masked_adjoint(self, upstream) -> np.ndarray:
        return self.adjoint(upstream, apply_mask=True)

    def weighted_adjoint(
        self,
        residuals: Union[np.ndarray, Sequence[np.ndarray], Mapping[Any, np.ndarray]],
    ) -> np.ndarray:
        if isinstance(residuals, Mapping):
            items = [(self._frame_index_for_key(k), v) for k, v in residuals.items()]
        elif isinstance(residuals, np.ndarray) and residuals.ndim == 1:
            # Packed raw residuals over valid cells; apply W cellwise.
            out = np.zeros(self.part_shape, dtype=float)
            offset = 0
            for frame in self.frames:
                count = int(np.count_nonzero(frame.valid_mask))
                values = np.zeros(frame.shape, dtype=float)
                segment = residuals[offset:offset + count]
                values[frame.valid_mask] = segment / frame.safe_variance[frame.valid_mask]
                out += self.adjoint_frame(frame, values)
                offset += count
            return out
        elif isinstance(residuals, np.ndarray) and len(self.frames) == 1 and residuals.shape == self.frames[0].shape:
            items = [(0, residuals)]
        else:
            items = list(enumerate(residuals))
        out = np.zeros(self.part_shape, dtype=float)
        for index, residual in items:
            out += self.weighted_adjoint_frame(self.frames[index], residual)
        return out

    def weighted_adjoint_valid(self, vector: np.ndarray) -> np.ndarray:
        return self.weighted_adjoint(np.asarray(vector).reshape(-1))

    def weighted_inner_product(
        self,
        a: Sequence[np.ndarray] | Mapping[Any, np.ndarray],
        b: Sequence[np.ndarray] | Mapping[Any, np.ndarray],
    ) -> float:
        """Masked Σ valid a*b/noise_variance."""
        if isinstance(a, Mapping):
            a_items = {self._frame_index_for_key(k): v for k, v in a.items()}
            a = [a_items.get(i, np.zeros_like(f.pixels)) for i, f in enumerate(self.frames)]
        if isinstance(b, Mapping):
            b_items = {self._frame_index_for_key(k): v for k, v in b.items()}
            b = [b_items.get(i, np.zeros_like(f.pixels)) for i, f in enumerate(self.frames)]
        total = 0.0
        for frame, av, bv in zip(self.frames, a, b):
            av = np.asarray(av, dtype=float)
            bv = np.asarray(bv, dtype=float)
            total += float(np.sum(frame.valid_mask * av * bv / frame.safe_variance))
        return total

    @property
    def n_observations(self) -> int:
        return int(sum(np.count_nonzero(f.valid_mask) for f in self.frames))

    def observed_forward(self, image: np.ndarray) -> np.ndarray:
        return np.concatenate([self.predict_frame(f, image)[f.valid_mask] for f in self.frames])

    forward_valid = observed_forward

    def observed_adjoint(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size != self.n_observations:
            raise PSFError(f"有效观测向量长度应为 {self.n_observations}，实际 {vector.size}")
        out = np.zeros(self.part_shape, dtype=float)
        offset = 0
        for frame in self.frames:
            count = int(np.count_nonzero(frame.valid_mask))
            values = np.zeros(frame.shape, dtype=float)
            values[frame.valid_mask] = vector[offset:offset + count]
            out += self.adjoint_frame(frame, values)
            offset += count
        return out

    adjoint_valid = observed_adjoint

    def whitened_forward(self, image: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self.predict_frame(f, image)[f.valid_mask] / np.sqrt(f.safe_variance[f.valid_mask])
            for f in self.frames
        ])

    def whitened_adjoint(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size != self.n_observations:
            raise PSFError(f"白化观测向量长度应为 {self.n_observations}")
        out = np.zeros(self.part_shape, dtype=float)
        offset = 0
        for frame in self.frames:
            count = int(np.count_nonzero(frame.valid_mask))
            values = np.zeros(frame.shape, dtype=float)
            values[frame.valid_mask] = vector[offset:offset + count] / np.sqrt(
                frame.safe_variance[frame.valid_mask]
            )
            out += self.adjoint_frame(frame, values)
            offset += count
        return out

    def residuals(self, image: np.ndarray) -> list[np.ndarray]:
        return [self.predict_frame(f, image) - f.pixels for f in self.frames]

    def data_cost(self, image: np.ndarray) -> float:
        total = 0.0
        for frame in self.frames:
            residual = self.predict_frame(frame, image) - frame.pixels
            chi2 = np.where(
                frame.valid_mask,
                residual * residual / frame.safe_variance,
                0.0,
            )
            total += float(np.sum(chi2))
        return total

    def reconstruct(self, **kwargs):
        """Matrix-free minimum-norm WLS reconstruction; see module function."""
        from spatial_psf_reconstruct import reconstruct as _reconstruct
        return _reconstruct(self, **kwargs)

    def calibration_gradient(self, frame, upstream, image=None, **kwargs):
        from spatial_psf_gradient import calibration_gradient as _gradient
        return _gradient(self, upstream, frame=frame, image=image, **kwargs)

    def observability(self, *, zero_tolerance: float = 1e-12) -> Observability:
        """Return direct-support mask and diag(A^T W A) observability weights."""
        py, px = (k // 2 for k in self.calibration.kernel_shape)
        padded_shape = (self.H + 2 * py, self.W + 2 * px)
        covered_pad = np.zeros(padded_shape, dtype=bool)
        precision_pad = np.zeros(padded_shape, dtype=float)
        abs_pad = np.zeros(padded_shape, dtype=float)
        basis = self.calibration.basis

        for frame in self.frames:
            for t0, t1 in self.memory.slices(frame.height, frame.width, basis.shape[0]):
                coef = self._coef_tile(frame, t0, t1)
                valid = frame.valid_mask[t0:t1]
                inv_var = 1.0 / frame.safe_variance[t0:t1]
                for a in range(basis.shape[1]):
                    for b in range(basis.shape[2]):
                        tap = coef @ basis[:, a, b]
                        active = valid & (np.abs(tap) > zero_tolerance)
                        weighted_sq = np.where(valid, tap * tap * inv_var, 0.0)
                        abs_weight = np.where(valid, np.abs(tap), 0.0)
                        ys = slice(frame.origin[0] + t0 + a, frame.origin[0] + t1 + a)
                        xs = slice(frame.origin[1] + b, frame.origin[1] + frame.width + b)
                        covered_pad[ys, xs] |= active
                        precision_pad[ys, xs] += weighted_sq
                        abs_pad[ys, xs] += abs_weight

        def crop(arr):
            return arr[: self.H, : self.W] if py == 0 and px == 0 else arr[py:-py, px:-px]

        covered = crop(covered_pad) & self.domain
        precision = crop(precision_pad) * self.domain
        coverage_abs = crop(abs_pad) * self.domain
        return Observability(covered, precision, coverage_abs)

    def as_linear_operator(self, *, whitened: bool = True):
        """Return a SciPy matrix-free LinearOperator (no dense matrix)."""
        from scipy.sparse.linalg import LinearOperator

        n = self.n_pixels
        m = self.n_observations

        if whitened:
            return LinearOperator(
                dtype=float,
                shape=(m, n),
                matvec=lambda x: self.whitened_forward(np.asarray(x).reshape(self.part_shape)),
                rmatvec=lambda y: self.whitened_adjoint(y).reshape(-1),
            )
        return LinearOperator(
            dtype=float,
            shape=(m, n),
            matvec=lambda x: self.observed_forward(np.asarray(x).reshape(self.part_shape)),
            rmatvec=lambda y: self.observed_adjoint(y).reshape(-1),
        )

    def dot_test(self, seed: int = 0) -> dict[str, float]:
        """Numerical adjoint identities; useful in tests and notebooks."""
        rng = np.random.default_rng(seed)
        image = rng.normal(size=self.part_shape)
        ups = [rng.normal(size=f.shape) for f in self.frames]
        preds = self.forward(image)
        raw_lhs = sum(float(np.vdot(p, u)) for p, u in zip(preds, ups))
        raw_rhs = float(np.vdot(image, self.adjoint(ups)))
        residual = [rng.normal(size=f.shape) for f in self.frames]
        weighted_lhs = self.weighted_inner_product(preds, residual)
        weighted_rhs = float(np.vdot(image, self.weighted_adjoint(residual)))
        xvec = rng.normal(size=self.part_shape)
        yvec = rng.normal(size=self.n_observations)
        vec_lhs = float(np.dot(self.observed_forward(xvec), yvec))
        vec_rhs = float(np.vdot(xvec, self.observed_adjoint(yvec)))
        zvec = rng.normal(size=self.n_observations)
        white_lhs = float(np.dot(self.whitened_forward(xvec), zvec))
        white_rhs = float(np.vdot(xvec, self.whitened_adjoint(zvec)))
        return {
            "raw_abs_error": abs(raw_lhs - raw_rhs),
            "weighted_abs_error": abs(weighted_lhs - weighted_rhs),
            "valid_vector_abs_error": abs(vec_lhs - vec_rhs),
            "whitened_abs_error": abs(white_lhs - white_rhs),
        }


# A FrameSet is the same mathematical object in this implementation.
FrameSet = SpatialPSFOperator
