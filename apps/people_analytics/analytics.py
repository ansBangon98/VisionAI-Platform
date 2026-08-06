from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.results.frame_result import FrameResult


class PeopleAnalytics:
    """People-specific analytics rules.

    The first version only reports the current tracked person count. Gender,
    emotion, dwell time, staff/customer labels, and other domain-specific
    metrics can be added here without changing the detection pipeline contract.
    """

    def update(self, tracks: Sequence[object] | FrameResult) -> dict[str, int]:
        if isinstance(tracks, FrameResult):
            return {
                "current_people": sum(
                    1 for detection in tracks.detections if _is_person(detection)
                ),
            }

        return {
            "current_people": sum(1 for track in tracks if _is_person(track)),
        }


def _is_person(track: object) -> bool:
    label = getattr(track, "label", None)
    class_id = getattr(track, "class_id", None)
    if label is not None or class_id is not None:
        return str(label or "").lower() in {"person", "people"} or class_id == 0

    if not isinstance(track, Mapping):
        return True

    item_type = str(track.get("type", track.get("label", "person"))).lower()
    return item_type in {"person", "people"}
