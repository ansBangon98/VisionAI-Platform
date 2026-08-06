from __future__ import annotations

import time
from collections.abc import Mapping

from core.results.frame_result import BoundingBox, Detection, FrameResult


def parse_batch_meta(
    batch_meta: object,
    source_ids: Mapping[int, str] | None = None,
    labels: Mapping[int, str] | None = None,
) -> list[FrameResult]:
    pyds = _require_pyds()
    results: list[FrameResult] = []
    source_ids = source_ids or {}
    labels = labels or {}

    frame_list = getattr(batch_meta, "frame_meta_list", None)
    while frame_list is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
        except StopIteration:
            break

        pad_index = int(getattr(frame_meta, "pad_index", 0) or 0)
        source_id = source_ids.get(pad_index, str(pad_index))
        results.append(
            FrameResult(
                source_id=source_id,
                frame_number=int(getattr(frame_meta, "frame_num", 0) or 0),
                timestamp=time.time(),
                detections=parse_object_meta_list(
                    getattr(frame_meta, "obj_meta_list", None),
                    labels=labels,
                ),
            )
        )
        frame_list = _next_list_node(frame_list)

    return results


def parse_object_meta_list(
    object_meta_list: object,
    labels: Mapping[int, str] | None = None,
) -> list[Detection]:
    pyds = _require_pyds()
    labels = labels or {}
    detections: list[Detection] = []
    object_list = object_meta_list

    while object_list is not None:
        try:
            object_meta = pyds.NvDsObjectMeta.cast(object_list.data)
        except StopIteration:
            break

        class_id = int(getattr(object_meta, "class_id", 0) or 0)
        label = _object_label(object_meta, labels)
        track_id = _object_track_id(object_meta, pyds)
        detections.append(
            Detection(
                class_id=class_id,
                label=label or f"class_{class_id}",
                confidence=float(getattr(object_meta, "confidence", 0.0) or 0.0),
                bbox=_object_bbox(object_meta),
                track_id=track_id,
            )
        )
        object_list = _next_list_node(object_list)

    return detections


def _object_bbox(object_meta: object) -> BoundingBox:
    rect = object_meta.rect_params
    return BoundingBox(
        x=float(getattr(rect, "left", 0.0) or 0.0),
        y=float(getattr(rect, "top", 0.0) or 0.0),
        width=float(getattr(rect, "width", 0.0) or 0.0),
        height=float(getattr(rect, "height", 0.0) or 0.0),
    )


def _object_label(object_meta: object, labels: Mapping[int, str]) -> str:
    raw_label = getattr(object_meta, "obj_label", "")
    if raw_label:
        return str(raw_label)

    class_id = int(getattr(object_meta, "class_id", 0) or 0)
    return str(labels.get(class_id, ""))


def _object_track_id(object_meta: object, pyds: object) -> int | None:
    object_id = int(getattr(object_meta, "object_id", -1) or -1)
    untracked_id = int(getattr(pyds, "UNTRACKED_OBJECT_ID", -1))
    if object_id < 0 or object_id == untracked_id:
        return None
    return object_id


def _next_list_node(node: object):
    try:
        return node.next
    except StopIteration:
        return None


def _require_pyds():
    try:
        import pyds
    except ImportError as error:
        raise RuntimeError(
            "DeepStream Python bindings are not available. Install pyds from "
            "the NVIDIA DeepStream SDK."
        ) from error
    return pyds
