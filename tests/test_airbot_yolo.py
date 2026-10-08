"""Offline tests for deterministic block color classification."""

import unittest

import cv2
import numpy as np

from airbot_yolo import (Detection, DetectionStabilizer, TargetResolutionError,
                         classify_block_color, resolve_unique_target)


def solid_hsv(hue, saturation=220, value=220, size=(100, 120)):
    hsv = np.full((*size, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detection(bbox=(10, 10, 50, 50), color="blue", confidence=0.9):
    return Detection(bbox, "block", confidence, color, 0.8,
                     {"blue": 0.8 if color == "blue" else 0.0,
                      "green": 0.8 if color == "green" else 0.0})


class ColorClassificationTest(unittest.TestCase):
    def test_blue(self):
        color, coverage, _ = classify_block_color(solid_hsv(110), (0, 0, 120, 100))
        self.assertEqual(color, "blue")
        self.assertGreater(coverage, 0.99)

    def test_green(self):
        color, coverage, _ = classify_block_color(solid_hsv(60), (0, 0, 120, 100))
        self.assertEqual(color, "green")
        self.assertGreater(coverage, 0.99)

    def test_unknown_low_saturation(self):
        color, coverage, coverages = classify_block_color(
            solid_hsv(110, saturation=20), (0, 0, 120, 100))
        self.assertIsNone(color)
        self.assertEqual(coverage, 0.0)
        self.assertEqual(coverages, {"blue": 0.0, "green": 0.0})

    def test_low_color_coverage_is_unknown(self):
        image = solid_hsv(10)
        image[20:30, 20:30] = solid_hsv(110, size=(10, 10))
        color, _, _ = classify_block_color(image, (0, 0, 120, 100))
        self.assertIsNone(color)

    def test_small_blue_patch_on_gray_object_is_unknown(self):
        image = np.full((100, 120, 3), 150, dtype=np.uint8)
        image[20:30, 20:30] = solid_hsv(110, size=(10, 10))
        color, coverage, _ = classify_block_color(image, (0, 0, 120, 100))
        self.assertIsNone(color)
        self.assertLess(coverage, 0.02)

    def test_edge_box_is_clamped(self):
        color, _, _ = classify_block_color(solid_hsv(60), (-50, -40, 200, 160))
        self.assertEqual(color, "green")

    def test_empty_box_is_unknown(self):
        color, coverage, _ = classify_block_color(solid_hsv(60), (20, 20, 20, 20))
        self.assertIsNone(color)
        self.assertEqual(coverage, 0.0)

    def test_detection_display_label(self):
        detection = Detection((1, 2, 3, 4), "block", 0.9, "blue", 0.8)
        self.assertEqual(detection.display_label, "蓝色积木")

    def test_unique_target(self):
        blue = Detection((1, 2, 3, 4), "block", 0.9, "blue", 0.8)
        green = Detection((5, 6, 7, 8), "cube", 0.8, "green", 0.7)
        self.assertIs(resolve_unique_target([blue, green], "blue"), blue)

    def test_missing_target_rejected(self):
        with self.assertRaisesRegex(TargetResolutionError, "未检测到"):
            resolve_unique_target([], "blue")

    def test_multiple_targets_rejected(self):
        detections = [
            Detection((1, 2, 3, 4), "block", 0.9, "blue", 0.8),
            Detection((5, 6, 7, 8), "cube", 0.8, "blue", 0.7),
        ]
        with self.assertRaisesRegex(TargetResolutionError, "目标不唯一"):
            resolve_unique_target(detections, "blue")


class DetectionStabilizerTest(unittest.TestCase):
    def stabilizer(self, **overrides):
        config = {
            "tracking_confirm_frames": 3,
            "tracking_max_missed_frames": 3,
            "tracking_smoothing_alpha": 0.5,
            "tracking_color_window": 5,
            "tracking_color_confirm_frames": 2,
        }
        config.update(overrides)
        return DetectionStabilizer(config)

    def test_single_frame_false_positive_is_not_published(self):
        stabilizer = self.stabilizer()
        self.assertEqual(stabilizer.update([detection()]), [])
        self.assertEqual(stabilizer.update([]), [])

    def test_target_requires_three_consecutive_frames(self):
        stabilizer = self.stabilizer()
        self.assertEqual(stabilizer.update([detection()]), [])
        self.assertEqual(stabilizer.update([detection((11, 10, 51, 50))]), [])
        result = stabilizer.update([detection((12, 10, 52, 50))])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].color, "blue")
        self.assertGreater(result[0].bbox[0], 10)
        self.assertLess(result[0].bbox[0], 12)

    def test_confirmed_target_survives_short_detection_dropout(self):
        stabilizer = self.stabilizer()
        for _ in range(3):
            result = stabilizer.update([detection()])
        self.assertEqual(len(result), 1)
        for _ in range(3):
            held = stabilizer.update([])
            self.assertEqual(len(held), 1)
            self.assertFalse(held[0].observed)
            with self.assertRaisesRegex(TargetResolutionError, "重新确认"):
                resolve_unique_target(held, "blue")
        self.assertEqual(stabilizer.update([]), [])

    def test_one_wrong_color_frame_does_not_flip_stable_color(self):
        stabilizer = self.stabilizer()
        stabilizer.update([detection(color="blue")])
        stabilizer.update([detection(color="blue")])
        result = stabilizer.update([detection(color="blue")])
        self.assertEqual(result[0].color, "blue")
        result = stabilizer.update([detection(color="green")])
        self.assertEqual(result[0].color, "blue")


if __name__ == "__main__":
    unittest.main()

class DuplicateDetectionTest(unittest.TestCase):
    def test_overlapping_and_contained_boxes_removed(self):
        from airbot_yolo import suppress_duplicate_detections
        first = detection()
        duplicates = [detection((12,12,49,49)), detection((20,20,40,40))]
        self.assertEqual(suppress_duplicate_detections([first] + duplicates), [first])

    def test_separate_same_color_blocks_remain_ambiguous(self):
        from airbot_yolo import suppress_duplicate_detections
        result = suppress_duplicate_detections([detection(), detection((45,10,85,50))])
        self.assertEqual(len(result), 2)
        with self.assertRaises(TargetResolutionError):
            resolve_unique_target(result, 'blue')

    def test_different_colors_are_not_merged(self):
        from airbot_yolo import suppress_duplicate_detections
        self.assertEqual(len(suppress_duplicate_detections([detection(), detection(color='green')])), 2)
