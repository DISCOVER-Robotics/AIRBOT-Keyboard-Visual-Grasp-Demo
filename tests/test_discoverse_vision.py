"""Vision guards plus opt-in real model/RGB-D/physics integration checks."""
from dataclasses import replace
import os
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from airbot_yolo import DEFAULT_DETECTION_CONFIG, Detection
from discoverse_vision import CameraFrame, SimulationVision, SimulatedRGBDCamera, camera_to_end


def blue_detection():
    return Detection((30, 30, 70, 70), 'block', .9, 'blue', 1.)


def synthetic_frame():
    bgr = np.zeros((100, 100, 3), np.uint8)
    bgr[30:70, 30:70] = (255, 0, 0)
    # Top-down camera: 40 image pixels cover a 0.04 m cube at z=0.04.
    focal = 460.
    fovy = np.rad2deg(2 * np.arctan(50 / focal))
    return CameraFrame(bgr, np.full((100, 100), .46), np.array([.32, -.1, .5]),
                       np.eye(3), fovy, time.monotonic(), np.array([.2, 0., .3]), np.eye(3))


class VisionGuardsTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimpleNamespace(busy=False)
        self.detector = SimpleNamespace(model=SimpleNamespace(names={0: 'block'}),
                                        detection_config=dict(DEFAULT_DETECTION_CONFIG))
        self.vision = SimulationVision(self.sim, detector=self.detector)
        self.addCleanup(self.vision.close)

    def confirm(self, detections):
        for _ in range(3):
            self.vision._accept(synthetic_frame(), detections)

    def test_depth_backprojection_and_hand_eye_transform(self):
        frame = synthetic_frame()
        camera = SimulatedRGBDCamera(frame)
        cloud = camera.create_point_cloud(frame.depth)
        transform = camera_to_end(frame)
        point_end = transform[:3, :3] @ cloud[50, 50] + transform[:3, 3]
        point_world = frame.end_rotation @ point_end + frame.end_position
        np.testing.assert_allclose(point_world, [.3205, -.1005, .04], atol=1e-6)

    def test_empty_and_unconfirmed_results_reject_grasp(self):
        for ds in ([], [blue_detection()]):
            self.vision.invalidate()
            self.vision._accept(synthetic_frame(), ds)
            with self.assertRaises(ValueError):
                self.vision.resolve('blue')

    def test_stale_and_new_duplicate_results_reject_grasp(self):
        self.confirm([blue_detection()])
        self.assertEqual(self.vision.resolve('blue').color, 'blue')
        self.vision.frame.captured -= 5
        with self.assertRaisesRegex(ValueError, '过期'):
            self.vision.resolve('blue')
        self.confirm([blue_detection()])
        duplicate = replace(blue_detection(), bbox=(75, 30, 99, 70))
        self.vision._accept(synthetic_frame(), [blue_detection(), duplicate])
        with self.assertRaisesRegex(ValueError, '不唯一'):
            self.vision.resolve('blue')

    def test_invalidate_discards_old_inflight_result(self):
        from concurrent.futures import Future
        self.vision.future = Future()
        self.vision.pending_generation = self.vision.generation
        self.vision.pending_frame = synthetic_frame()
        self.vision.future.set_result([blue_detection()])
        self.vision.invalidate()
        self.sim.busy = True  # Do not enqueue a new frame in this unit test.
        self.vision.poll()
        self.assertIsNone(self.vision.frame)
        self.assertEqual(self.vision.detections, [])


@unittest.skipUnless(os.getenv('GRASP_TEST_YOLO_SIM') == '1',
                     'Set GRASP_TEST_YOLO_SIM=1 MUJOCO_GL=egl for real model/render tests')
class RealYoloSimulationTest(unittest.TestCase):
    def test_user_checkpoint_rgbd_to_physical_pick_place(self):
        from discoverse_sim import DiscoverseSimulation
        from discoverse_voice import SimulationCommands
        from voice_commands import execute_command
        sim = DiscoverseSimulation()
        self.addCleanup(sim.close)
        vision = SimulationVision(sim)
        sim.vision = vision
        target = SimulationCommands()
        target.sim = sim
        for color, command in [('blue', '抓取蓝色积木'), ('green', '抓取绿色积木')]:
            with self.subTest(color=color):
                sim.reset()
                vision.warmup()
                self.assertTrue(str(vision.detector.checkpoint).endswith('yolo_blocks_v1.pt'))
                self.assertEqual(sim.model.camera('eye_arm').name, 'eye_arm')
                sim.select(color)
                # The full segmentation/prediction worker must not read object poses.
                with patch.object(sim, 'position', side_effect=AssertionError('ground truth used')):
                    sim.predict()
                    sim.run_until_idle()
                self.assertIsNotNone(vision.segment)
                self.assertIsNotNone(vision.grasp_predictor)
                self.assertGreater(np.count_nonzero(sim.prepared_preview != vision.frame.bgr), 100)
                np.testing.assert_allclose(sim.predicted[:2], sim.position(color)[:2], atol=.008)
                self.assertGreaterEqual(sim.predicted[2], sim.sim_config['min_grasp_z'])
                vision.warmup()
                execute_command(target, command)
                sim.run_until_idle()
                self.assertIn('成功', sim.message)
                self.assertTrue(any('接触已确认' in event for event in sim.events))
                self.assertTrue(any('步骤10' in event for event in sim.events))
                np.testing.assert_allclose(sim.position(color)[:2], sim.place[:2], atol=.045)
        sim.reset()
        vision._accept(vision.capture(), [])
        with self.assertRaises(ValueError):
            execute_command(target, '抓取蓝色积木')
        self.assertFalse(sim.busy)


if __name__ == '__main__':
    unittest.main()
