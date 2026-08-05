from __future__ import annotations

from collections.abc import Sequence


class PeopleAnalytics:
    """People-specific analytics rules.

    The first version only reports the current tracked person count. Gender,
    emotion, dwell time, staff/customer labels, and other domain-specific
    metrics can be added here without changing the detection pipeline contract.
    """

    def update(self, tracks: Sequence[object]) -> dict[str, int]:
        return {
            "current_people": len(tracks),
        }
