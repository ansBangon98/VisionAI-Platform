from __future__ import annotations

import unittest

import numpy as np

from apps.segmentation.pipeline import SegmentationPipeline
from core.postprocess.segmentation import semantic_mask_from_logits
from core.results import SegmentationResult


class FakeSegmentor:
    class_labels = {
        0: "background",
        1: "drivable_fallback",
        2: "road",
    }

    def predict(self, frame: object) -> np.ndarray:
        image = np.asarray(frame)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[:, image.shape[1] // 2 :] = 2
        return mask

    def model_profile(self) -> dict[str, object]:
        return {"base_model": "SegFormer"}


class FakeRenderer:
    def render(self, original_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        visualization = np.zeros_like(original_frame)
        visualization[mask == 2] = (34, 197, 94)
        return visualization


class SegmentationPipelineTest(unittest.TestCase):
    def test_process_frame_result_returns_segmentation_result(self):
        pipeline = SegmentationPipeline(
            segmentor=FakeSegmentor(),
            renderer=FakeRenderer(),
            source_id="road_camera",
        )

        result = pipeline.process_frame_result(np.zeros((6, 8, 3), dtype=np.uint8))

        self.assertIsInstance(result, SegmentationResult)
        self.assertEqual(result.source_id, "road_camera")
        self.assertEqual(result.frame_number, 1)
        self.assertEqual(result.mask.shape, (6, 8))
        self.assertEqual(result.visualization_frame.shape, (6, 8, 3))
        self.assertEqual(result.class_labels[2], "road")
        self.assertGreaterEqual(result.inference_ms, 0.0)

    def test_semantic_mask_from_nchw_logits(self):
        logits = np.zeros((1, 3, 4, 5), dtype=np.float32)
        logits[:, 2, :, :] = 1.0

        mask = semantic_mask_from_logits(logits, class_count=3)

        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(mask == 2))


if __name__ == "__main__":
    unittest.main()
