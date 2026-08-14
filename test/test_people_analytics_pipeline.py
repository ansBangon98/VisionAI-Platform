from __future__ import annotations

import unittest

import numpy as np

from apps.people_analytics.pipeline import (
    ByteTrackPeopleTracker,
    PeopleAnalyticsPipeline,
    TrackResult,
)
from core.models.base_detector import Detection


class FakeDetector:
    def __init__(self, detections: list[Detection]):
        self.detections = detections

    def predict(self, frame: object) -> list[Detection]:
        return list(self.detections)

    def model_profile(self) -> dict[str, object]:
        return {}

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        del confidence_threshold


class FakeTracker:
    def __init__(self, tracks: list[TrackResult]):
        self.tracks = tracks

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackResult]:
        del detections, frame_shape
        return list(self.tracks)


class PeopleAnalyticsPipelineTest(unittest.TestCase):
    def test_draws_bytetrack_bbox_for_tracked_person(self):
        detector_bbox = (10.0, 10.0, 50.0, 90.0)
        track_bbox = (20, 15, 62, 96)
        pipeline = PeopleAnalyticsPipeline(
            detector=FakeDetector(
                [Detection(bbox=detector_bbox, score=0.92, class_id=0)]
            ),
            tracker=FakeTracker([TrackResult(track_id=7, bbox=track_bbox)]),
            people_class_ids=[0],
            class_labels={0: "person"},
        )

        result = pipeline.process_frame_result(_empty_frame())

        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.track_id, 7)
        self.assertEqual(detection.label, "person")
        self.assertAlmostEqual(detection.confidence, 0.92)
        self.assertEqual(_xyxy(detection.bbox), track_bbox)

    def test_bytetrack_adapter_returns_latest_detection_bbox(self):
        tracker = ByteTrackPeopleTracker(
            track_thresh=0.2,
            match_thresh=0.9,
            track_buffer=30,
            frame_rate=30,
        )
        first_bbox = (10.0, 10.0, 50.0, 90.0)
        second_bbox = (14.0, 10.0, 54.0, 90.0)

        first_tracks = tracker.update(
            [Detection(bbox=first_bbox, score=0.95, class_id=0)],
            frame_shape=(120, 100),
        )
        second_tracks = tracker.update(
            [Detection(bbox=second_bbox, score=0.95, class_id=0)],
            frame_shape=(120, 100),
        )

        self.assertEqual(len(first_tracks), 1)
        self.assertEqual(first_tracks[0].bbox, first_bbox)
        self.assertEqual(len(second_tracks), 1)
        self.assertEqual(second_tracks[0].bbox, second_bbox)

    def test_uses_track_bbox_when_current_detection_does_not_match(self):
        track_bbox = (70, 80, 95, 115)
        pipeline = PeopleAnalyticsPipeline(
            detector=FakeDetector(
                [Detection(bbox=(0.0, 0.0, 10.0, 10.0), score=0.80, class_id=0)]
            ),
            tracker=FakeTracker([TrackResult(track_id=9, bbox=track_bbox)]),
            people_class_ids=[0],
            class_labels={0: "person"},
        )

        result = pipeline.process_frame_result(_empty_frame())

        detection = result.detections[0]
        self.assertEqual(detection.track_id, 9)
        self.assertEqual(detection.confidence, 1.0)
        self.assertEqual(_xyxy(detection.bbox), track_bbox)


def _empty_frame() -> np.ndarray:
    return np.zeros((120, 100, 3), dtype=np.uint8)


def _xyxy(bbox) -> tuple[float, float, float, float]:
    return (
        bbox.x,
        bbox.y,
        bbox.x + bbox.width,
        bbox.y + bbox.height,
    )


if __name__ == "__main__":
    unittest.main()
