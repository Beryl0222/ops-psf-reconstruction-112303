"""Frame replacement, locked snapshots, prediction calls and late gradients."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Optional, Union
import uuid

import numpy as np

from spatial_psf_core import (
    CalibrationMismatchError,
    CalibrationSet,
    Frame,
    FrameConflictError,
    MemoryPolicy,
    PSFError,
    RevisionConflictError,
)
from spatial_psf_gradient import CalibrationGradientResult, calibration_gradient
from spatial_psf_operator import SpatialPSFOperator
from spatial_psf_reconstruct import (
    ReconstructionResult,
    compare_reconstructions,
    reconstruct,
)


def parse_frame_ref(ref: Any) -> tuple[str, Optional[int]]:
    if isinstance(ref, Frame):
        return ref.id, ref.revision
    if isinstance(ref, tuple) and len(ref) == 2:
        rev = ref[1]
        return str(ref[0]), None if rev is None else int(rev)
    if isinstance(ref, (list, tuple)) and len(ref) == 2:
        return str(ref[0]), int(ref[1])
    text = str(ref)
    if "@" in text:
        fid, rev = text.rsplit("@", 1)
        return fid, int(rev)
    return text, None


class FrameRegistry:
    """Current frame plus immutable revision history needed by late gradients."""

    def __init__(self, frames=None):
        if isinstance(frames, Mapping) and "frames" in frames:
            frames = frames["frames"]
        self.current: dict[str, Frame] = {}
        self.history: dict[str, dict[int, Frame]] = {}
        if frames:
            for frame in frames:
                self.add_or_replace(frame)

    def add(self, frame: Union[Frame, dict]) -> Frame:
        frame = Frame.coerce(frame)
        if frame.id in self.current:
            raise FrameConflictError(f"帧 {frame.id!r} 已存在；替换请使用 replace_frame")
        self._store(frame)
        return frame

    def _store(self, frame: Frame) -> None:
        self.current[frame.id] = frame
        self.history.setdefault(frame.id, {})[frame.revision] = frame.copy()

    def replace(self, frame: Union[Frame, dict]) -> Frame:
        frame = Frame.coerce(frame)
        old = self.current.get(frame.id)
        if old is None:
            raise FrameConflictError(f"不能替换未知帧 {frame.id!r}；请先 add_frame")
        if frame.revision < old.revision:
            raise RevisionConflictError(
                f"帧 {frame.id!r} 新版本 {frame.revision} 低于当前版本 {old.revision}"
            )
        if frame.revision == old.revision:
            if old.same_content(frame):
                return old
            raise FrameConflictError(
                f"帧 {frame.id!r} 版本 {frame.revision} 相同但内容不同，属于输入冲突"
            )
        self._store(frame)
        return frame

    def add_or_replace(self, frame: Union[Frame, dict]) -> Frame:
        frame = Frame.coerce(frame)
        if frame.id not in self.current:
            return self.add(frame)
        return self.replace(frame)

    # Friendly aliases.
    add_frame = add
    replace_frame = replace
    submit = add_or_replace

    def get(self, frame_id: str, revision: Optional[int] = None) -> Frame:
        if revision is None:
            if frame_id not in self.current:
                raise KeyError(f"未知帧 {frame_id!r}")
            return self.current[frame_id]
        try:
            return self.history[frame_id][int(revision)]
        except KeyError as exc:
            raise KeyError(f"没有帧 {frame_id!r} 的版本 {revision}") from exc

    def resolve_ref(self, ref: Any) -> Frame:
        if isinstance(ref, Frame):
            # Prefer archived exact revision when available.
            if ref.id in self.history and ref.revision in self.history[ref.id]:
                return self.get(ref.id, ref.revision)
            return ref
        fid, rev = parse_frame_ref(ref)
        return self.get(fid, rev)

    def active_frames(self, refs: Any = None) -> list[Frame]:
        if refs is None or refs == "current" or refs == "latest":
            return list(self.current.values())
        if refs == "initial" or refs == "oldest":
            return [self.history[fid][min(revs)] for fid, revs in self.history.items()]
        if isinstance(refs, (str, tuple, list)) and not (
            isinstance(refs, list) and refs and isinstance(refs[0], (tuple, list))
        ):
            single_ref = isinstance(refs, str) or (
                isinstance(refs, (tuple, list))
                and len(refs) == 2
                and isinstance(refs[1], int)
                and not isinstance(refs[0], (tuple, list))
            )
            if single_ref:
                return [self.resolve_ref(refs)]
        return [self.resolve_ref(ref) for ref in refs]

    def refs(self, *, latest_only: bool = True):
        if latest_only:
            return tuple(f.ref for f in self.current.values())
        return tuple((fid, rev) for fid, revs in self.history.items() for rev in sorted(revs))


@dataclass
class PredictionCall:
    call_id: str
    frame_ref: tuple[str, int]
    calibration_name: str
    values: np.ndarray
    image: np.ndarray
    frame: Frame
    frame_refs: tuple[tuple[str, int], ...]
    expected_calibration: Optional[str] = None
    memory: Optional[MemoryPolicy] = None

    def gradient(
        self,
        upstream: np.ndarray,
        calibration: CalibrationSet,
        *,
        expected_calibration: Optional[str] = None,
        apply_mask: bool = False,
        weighted: bool = False,
    ) -> CalibrationGradientResult:
        expected = expected_calibration or self.expected_calibration
        if expected is not None and expected != self.calibration_name:
            raise CalibrationMismatchError(
                f"迟到梯度的档位 {expected!r} 不属于原调用 {self.call_id} "
                f"({self.frame_ref}, {self.calibration_name!r})"
            )
        return calibration_gradient(
            calibration,
            upstream,
            frame=self.frame,
            image=self.image,
            apply_mask=apply_mask,
            weighted=weighted,
            expected_calibration=expected,
            memory=self.memory,
        )


class ScanSession:
    """Owns calibration, frame revisions, locked sets and prediction calls."""

    def __init__(
        self,
        calibration: Optional[CalibrationSet] = None,
        *,
        record: Optional[dict] = None,
        part_shape: Optional[tuple[int, int]] = None,
        domain: Optional[np.ndarray] = None,
        memory: Optional[MemoryPolicy] = None,
        registry: Optional[FrameRegistry] = None,
        frames=None,
    ):
        if record is not None:
            calibration = calibration or CalibrationSet.from_record(record)
            part_shape = part_shape or record["part_shape"]
            memory = memory or MemoryPolicy.from_scale(record.get("scale"))
        if calibration is None or part_shape is None:
            raise PSFError("ScanSession 需要 calibration/part_shape 或 record")
        self.calibration = calibration
        self.part_shape = (int(part_shape[0]), int(part_shape[1]))
        self.domain = None if domain is None else np.asarray(domain, bool)
        self.memory = memory or MemoryPolicy()
        self.registry = registry or FrameRegistry(frames)
        self.locks: dict[str, SpatialPSFOperator] = {}
        self.results: dict[str, ReconstructionResult] = {}
        self.calls: dict[str, PredictionCall] = {}
        self.last_result: Optional[ReconstructionResult] = None

    @classmethod
    def from_record(cls, record: dict, **kwargs) -> "ScanSession":
        session = cls(record=record, **kwargs)
        for frame_data in record.get("frames", []):
            session.registry.add_or_replace(frame_data)
        return session

    def add_frame(self, frame) -> Frame:
        return self.registry.add(frame)

    def replace_frame(self, frame) -> Frame:
        return self.registry.replace(frame)

    def submit_frame(self, frame) -> Frame:
        return self.registry.add_or_replace(frame)

    def frame_set(self, refs: Any = None, *, name: Optional[str] = None) -> SpatialPSFOperator:
        frames = self.registry.active_frames(refs)
        op = SpatialPSFOperator(
            frames=frames,
            calibration=self.calibration,
            part_shape=self.part_shape,
            domain=self.domain,
            memory=self.memory,
        )
        if name is not None:
            self.locks[name] = op
        return op

    def lock(self, name: str = "locked", refs: Any = None) -> SpatialPSFOperator:
        if refs is None:
            refs = tuple(f.ref for f in self.registry.current.values())
        op = self.frame_set(refs, name=name)
        return op

    @property
    def operator(self) -> SpatialPSFOperator:
        return self.frame_set("current")

    def reconstruct(
        self,
        refs_or_name: Any = "current",
        *,
        lock_name: Optional[str] = None,
        **kwargs,
    ) -> ReconstructionResult:
        if isinstance(refs_or_name, str) and refs_or_name in self.locks and kwargs.pop("use_lock", True):
            op = self.locks[refs_or_name]
        elif isinstance(refs_or_name, SpatialPSFOperator):
            op = refs_or_name
        else:
            op = self.frame_set(refs_or_name)
        result = op.reconstruct(**kwargs)
        self.last_result = result
        if lock_name:
            self.results[lock_name] = result
        if isinstance(refs_or_name, str):
            self.results[refs_or_name] = result
        return result

    def compare_locked(
        self,
        locked: Union[str, ReconstructionResult, SpatialPSFOperator] = "locked",
        current: Optional[ReconstructionResult] = None,
        *,
        reconstruct_missing: bool = True,
        **compare_kwargs,
    ):
        if isinstance(locked, ReconstructionResult):
            locked_result = locked
        elif isinstance(locked, SpatialPSFOperator):
            locked_result = locked.reconstruct(**compare_kwargs.pop("reconstruct_kwargs", {}))
        elif locked in self.results:
            locked_result = self.results[locked]
        elif locked in self.locks:
            locked_result = self.locks[locked].reconstruct()
            self.results[locked] = locked_result
        else:
            raise KeyError(f"没有锁定集合/结果 {locked!r}")

        if current is None:
            current = self.last_result if self.last_result is not None else self.reconstruct("current")
        return compare_reconstructions(locked_result, current, **compare_kwargs)

    def predict(
        self,
        frame: Any = None,
        *,
        frame_id: Optional[str] = None,
        revision: Optional[int] = None,
        ref: Any = None,
        image: Optional[np.ndarray] = None,
        result: Optional[ReconstructionResult] = None,
        expected_calibration: Optional[str] = None,
        call_id: Optional[str] = None,
        store: bool = True,
    ) -> PredictionCall:
        if isinstance(frame, PredictionCall):
            return frame
        if ref is not None:
            resolved = self.registry.resolve_ref(ref)
        elif frame is not None:
            resolved = self.registry.resolve_ref(frame)
        elif frame_id is not None:
            resolved = self.registry.get(frame_id, revision)
        else:
            raise PSFError("predict 需要 frame/ref/frame_id")

        if expected_calibration is not None and expected_calibration != resolved.calibration:
            raise CalibrationMismatchError(
                f"期望档位 {expected_calibration!r}，但原调用帧 {resolved.ref} 是 {resolved.calibration!r}"
            )
        if image is None:
            result = result or self.last_result
            if result is None:
                raise PSFError("predict 需要固定 image 或已有 reconstruction result")
            image = result.image
        image = np.asarray(image, dtype=float).copy()
        one_op = SpatialPSFOperator(
            frames=[resolved], calibration=self.calibration, part_shape=self.part_shape,
            domain=self.domain, memory=self.memory,
        )
        values = one_op.predict_frame(resolved, image)
        call_id = call_id or f"call-{uuid.uuid4().hex[:12]}"
        call = PredictionCall(
            call_id=call_id,
            frame_ref=resolved.ref,
            calibration_name=resolved.calibration,
            values=values,
            image=image,
            frame=resolved.copy(),
            frame_refs=tuple(f.ref for f in self.registry.current.values()),
            expected_calibration=expected_calibration,
            memory=self.memory,
        )
        if store:
            self.calls[call_id] = call
        return call

    def calibration_gradient(
        self,
        call_or_id: Any = None,
        upstream: Optional[np.ndarray] = None,
        *,
        expected_calibration: Optional[str] = None,
        frame_id: Optional[str] = None,
        revision: Optional[int] = None,
        image: Optional[np.ndarray] = None,
        result: Optional[ReconstructionResult] = None,
        apply_mask: bool = False,
        weighted: bool = False,
        **kwargs,
    ) -> CalibrationGradientResult:
        # Stored original prediction call: strongest identity guarantee.
        if isinstance(call_or_id, PredictionCall):
            call = call_or_id
            if upstream is None:
                upstream = kwargs.get("upstream")
            return call.gradient(
                upstream, self.calibration,
                expected_calibration=expected_calibration,
                apply_mask=apply_mask, weighted=weighted,
            )
        if isinstance(call_or_id, str) and call_or_id in self.calls:
            if upstream is None:
                raise PSFError("缺少 upstream 余量")
            return self.calls[call_or_id].gradient(
                upstream, self.calibration,
                expected_calibration=expected_calibration,
                apply_mask=apply_mask, weighted=weighted,
            )

        # Late explicit frame ref.  Revision is mandatory for identity safety
        # unless a registered call id was supplied above.
        if upstream is None and not isinstance(call_or_id, (np.ndarray, list, tuple)):
            raise PSFError("缺少 upstream 余量")
        if upstream is None:
            upstream = call_or_id
        if frame_id is None and not isinstance(call_or_id, (str, tuple, list, Frame, dict)):
            raise PSFError("必须通过 call_id 或 frame_id+revision 提交梯度")
        if frame_id is None:
            ref = call_or_id
            fid, rev = parse_frame_ref(ref)
        else:
            fid, rev = frame_id, revision
        if rev is None:
            raise CalibrationMismatchError(
                "迟到梯度必须给出原调用 revision/call_id；不能自动归属到当前曝光档"
            )
        frame = self.registry.get(fid, rev)
        if expected_calibration is not None and expected_calibration != frame.calibration:
            raise CalibrationMismatchError(
                f"梯度声明档位 {expected_calibration!r} 与原调用 {frame.ref} 的 {frame.calibration!r} 不符"
            )
        if image is None:
            result = result or self.last_result
            if result is None:
                raise PSFError("梯度需要那次调用对应的固定 image/result")
            image = result.image
        return calibration_gradient(
            self.calibration, frame, image, upstream,
            domain=self.domain, apply_mask=apply_mask, weighted=weighted,
            expected_calibration=expected_calibration,
        )

    submit_gradient = calibration_gradient

    # Descriptive method aliases.
    lock_frames = lock
    lock_frame_set = lock
    replace = replace_frame
    add_or_replace_frame = submit_frame

    def get_frame(self, frame_id: str, revision: Optional[int] = None) -> Frame:
        return self.registry.get(frame_id, revision)
