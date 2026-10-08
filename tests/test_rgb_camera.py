"""Offline checks for the ordinary USB RGB camera adapter."""

import unittest
from unittest.mock import patch

import numpy as np

from airbot_camera import USBCamera


class _FakeCapture:
    def __init__(self, *_args):
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, *_args):
        return True

    def get(self, prop):
        # Match the configured 720p profile.
        return 1280 if prop == 3 else 720

    def read(self):
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.opened = False


class RgbCameraTest(unittest.TestCase):
    @patch("airbot_camera.cv2.VideoCapture", _FakeCapture)
    def test_usb_camera_accepts_calibration_and_releases(self):
        camera = USBCamera()
        bgr, depth, depth_map = camera.get_frame(["bgr", "depth", "depth_map"])
        self.assertEqual(bgr.shape, (720, 1280, 3))
        self.assertEqual(depth.shape, (720, 1280))
        self.assertEqual(depth_map.shape, (720, 1280, 3))

        # Intersecting rays with the configured table plane provides a valid
        # camera-space cloud without requiring a depth sensor.
        cloud = camera.create_point_cloud(
            depth, end_pose=([0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])
        )
        self.assertEqual(cloud.shape, (720, 1280, 3))
        self.assertTrue(np.isfinite(cloud).all())
        self.assertTrue(camera.deinit())
        self.assertIsNone(camera.cap)


if __name__ == "__main__":
    unittest.main()
