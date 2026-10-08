"""Failure-recovery tests; no camera, service, or robot is accessed."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from airbot_interface import RobotGraspThread


class GraspRecoveryTest(unittest.TestCase):
    def thread(self):
        robot = Mock()
        interface = SimpleNamespace(
            robot=robot,
            gripper_width=0.072,
            gripper_opened=False,
            observe_pose=[[0.2, 0.0, 0.3], [0, 0, 0, 1]],
        )
        thread = RobotGraspThread(interface)
        thread.log = Mock()
        return thread, robot, interface

    def test_failed_unconfirmed_grasp_opens_retreats_and_observes(self):
        thread, robot, interface = self.thread()
        retreat = [[0.3, 0.0, 0.12], [0, 0, 0, 1]]

        errors = thread.recover_to_observe("grasp", retreat, False)

        self.assertEqual(errors, [])
        robot.move_gripper.assert_called_once_with(0.072)
        robot.move_end_pose_linear.assert_called_once_with(retreat)
        robot.move_end_pose.assert_called_once_with(interface.observe_pose)
        self.assertTrue(interface.gripper_opened)

    def test_failed_place_releases_before_retreat(self):
        thread, robot, interface = self.thread()
        retreat = [[0.25, 0.2, 0.18], [0, 0, 0, 1]]

        thread.recover_to_observe("place", retreat, True)

        robot.move_gripper.assert_called_once_with(0.072)
        robot.move_end_pose_linear.assert_called_once_with(retreat)
        robot.move_end_pose.assert_called_once_with(interface.observe_pose)

    def test_observe_is_still_attempted_when_retreat_fails(self):
        thread, robot, interface = self.thread()
        robot.move_end_pose_linear.side_effect = RuntimeError("linear failed")

        errors = thread.recover_to_observe(
            "grasp", [[0.3, 0.0, 0.12], [0, 0, 0, 1]], False)

        self.assertEqual(len(errors), 1)
        robot.move_end_pose.assert_called_once_with(interface.observe_pose)


if __name__ == "__main__":
    unittest.main()
