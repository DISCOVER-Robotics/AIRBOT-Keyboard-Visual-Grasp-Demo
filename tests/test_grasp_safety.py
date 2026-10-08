"""Offline tests for the grasp-height safety floor."""

import math
import unittest

from airbot_grasp_simple import clamp_grasp_z


class GraspHeightSafetyTest(unittest.TestCase):
    def test_low_prediction_is_raised_to_safety_floor(self):
        self.assertEqual(clamp_grasp_z(0.003, 0.01338870424854599),
                         0.01338870424854599)

    def test_higher_prediction_is_preserved(self):
        self.assertEqual(clamp_grasp_z(0.12, 0.01338870424854599), 0.12)

    def test_invalid_limits_are_rejected(self):
        for value in (-0.01, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clamp_grasp_z(0.1, value)


if __name__ == "__main__":
    unittest.main()
