from __future__ import annotations

import unittest

import numpy as np

from core.models.detection.postprocess import create_yolo_postprocessor
from core.models.yolo_detector import YoloDetector


class YoloDetectorPostprocessTest(unittest.TestCase):
    def test_yolo26_end_to_end_output_skips_nms(self):
        detector = _detector(end2end=True)
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [10.0, 20.0, 110.0, 220.0, 0.90, 0.0]
        output[0, 1] = [12.0, 22.0, 112.0, 222.0, 0.80, 0.0]

        detections = detector._postprocess(
            output,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            frame_width=640,
            frame_height=640,
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].bbox, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(detections[1].bbox, (12.0, 22.0, 112.0, 222.0))
        self.assertEqual([detection.class_id for detection in detections], [0, 0])

    def test_traditional_yolo_output_applies_classwise_nms(self):
        detector = _detector(end2end=False, bbox_format="xywh_center")
        output = np.zeros((1, 6, 8400), dtype=np.float32)
        output[0, :, 0] = [60.0, 70.0, 100.0, 100.0, 0.90, 0.05]
        output[0, :, 1] = [62.0, 72.0, 100.0, 100.0, 0.80, 0.05]

        detections = detector._postprocess(
            output,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            frame_width=640,
            frame_height=640,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (10.0, 20.0, 110.0, 120.0))
        self.assertAlmostEqual(detections[0].score, 0.90, places=5)
        self.assertEqual(detections[0].class_id, 0)

    def test_prefers_end_to_end_output_when_dual_head_outputs_are_present(self):
        detector = _detector(end2end=True)
        traditional_output = np.zeros((1, 6, 8400), dtype=np.float32)
        end_to_end_output = np.zeros((1, 300, 6), dtype=np.float32)

        selected = detector._select_output(
            {
                "one_to_many": traditional_output,
                "one_to_one": end_to_end_output,
            }
        )

        self.assertIs(selected, end_to_end_output)

    def test_prefers_traditional_output_when_end_to_end_is_disabled(self):
        detector = _detector(end2end=False)
        traditional_output = np.zeros((1, 6, 8400), dtype=np.float32)
        end_to_end_output = np.zeros((1, 300, 6), dtype=np.float32)

        selected = detector._select_output(
            {
                "one_to_one": end_to_end_output,
                "one_to_many": traditional_output,
            }
        )

        self.assertIs(selected, traditional_output)


def _detector(
    *,
    end2end: bool | None,
    bbox_format: str = "auto",
    output_format: str = "auto",
) -> YoloDetector:
    detector = YoloDetector.__new__(YoloDetector)
    detector.confidence_threshold = 0.25
    detector.nms_iou_threshold = 0.45
    detector.class_ids = set()
    detector.output_format = output_format
    detector.bbox_format = bbox_format
    detector.end2end = end2end
    detector.postprocessor = create_yolo_postprocessor(
        output_format=output_format,
        bbox_format=bbox_format,
        end2end=end2end,
        confidence_threshold=detector.confidence_threshold,
        nms_iou_threshold=detector.nms_iou_threshold,
        class_ids=detector.class_ids,
    )
    return detector


if __name__ == "__main__":
    unittest.main()
