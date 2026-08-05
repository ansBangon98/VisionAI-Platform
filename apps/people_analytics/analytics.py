from __future__ import annotations

from collections.abc import Mapping, Sequence


class PeopleAnalytics:
    """People-specific analytics rules.

    The first version only reports the current tracked person count. Gender,
    emotion, dwell time, staff/customer labels, and other domain-specific
    metrics can be added here without changing the detection pipeline contract.
    """

    def update(self, tracks: Sequence[object]) -> dict[str, int]:
        return {
            "current_people": sum(1 for track in tracks if _is_person(track)),
        }


def _is_person(track: object) -> bool:
    if not isinstance(track, Mapping):
        return True

    item_type = str(track.get("type", track.get("label", "person"))).lower()
    return item_type in {"person", "people"}
