from __future__ import annotations

import unittest

from core.camera.camera_config import CameraConfig
from core.pipelines.cpu.gstreamer_pipeline import GStreamerCamera


class GStreamerPipelineTest(unittest.TestCase):
    def test_inference_caps_force_square_pixels_when_only_width_is_configured(self):
        camera = GStreamerCamera(
            CameraConfig(
                inference_width=640,
                inference_height=0,
                inference_fps=30,
            )
        )

        caps = camera._inference_caps()

        self.assertIn("width=640", caps)
        self.assertNotIn("height=0", caps)
        self.assertIn("pixel-aspect-ratio=1/1", caps)


if __name__ == "__main__":
    unittest.main()
