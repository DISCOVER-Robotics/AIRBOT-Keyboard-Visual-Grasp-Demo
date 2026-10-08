"""Offline validation for the configured common placement path."""

import math
import unittest

import yaml


class PlacePoseConfigTest(unittest.TestCase):
    def test_pre_place_is_ten_centimeters_above_common_place(self):
        with open("configs/sam_simplegrasp.yaml", encoding="utf-8") as file:
            grasp = yaml.safe_load(file)["AirbotGrasp"]
        place = grasp["place_pose"]
        pre_place = grasp["pre_place_pose"]

        self.assertEqual(pre_place[0][:2], place[0][:2])
        self.assertAlmostEqual(pre_place[0][2] - place[0][2], 0.1)
        self.assertEqual(pre_place[1], place[1])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in place[1])), 1.0)


if __name__ == "__main__":
    unittest.main()
