import unittest
from grasp_clearance import require_open_gripper


class GraspClearanceTest(unittest.TestCase):
    def test_opening_is_measured_not_assumed_from_command(self):
        good = dict(position=.055, velocity=0., error_id=0)
        require_open_gripper(good, .055)
        for changes in [dict(position=.035), dict(position=float('nan')),
                        dict(velocity=.004), dict(error_id=0x0E)]:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                require_open_gripper({**good, **changes}, .055)

    def test_g2_enabled_status_is_normal_but_faults_still_stop(self):
        for status in (0, 1):
            require_open_gripper(dict(position=.055, velocity=0., error_id=status), .055)
        for status in (0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0xFF):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                require_open_gripper(dict(position=.055, velocity=0., error_id=status), .055)
        with self.assertRaisesRegex(RuntimeError, '未稳定张开'):
            require_open_gripper(dict(position=.035, velocity=0., error_id=1), .055)

    def test_incomplete_open_stops_before_any_arm_movement(self):
        from types import SimpleNamespace
        from threading import Lock
        from unittest.mock import Mock, patch, mock_open
        from airbot_interface import RobotGraspThread
        robot = Mock()
        robot.clear_gripper_error.return_value = 0
        robot.get_gripper_diagnostics.return_value = dict(position=.035, velocity=0., error_id=0)
        interface = SimpleNamespace(robot=robot, robot_operation_lock=Lock(),
            gripper_open_clearance=.030, gripper_min_open_width=.055, gripper_width=.072,
            gripper_grasp_compression=.003, gripper_feedback_poll_interval=.05)
        thread = RobotGraspThread(interface)
        thread.log = Mock()
        thread.set_predicted_info(dict(trans=[.3,0,.02], orient=[0,0,0,1], o_height=.025, o_width=.025))
        with patch('builtins.open', mock_open()), patch('airbot_interface.time.sleep'):
            thread.run()
        robot.move_gripper.assert_called_once_with(.055)
        robot.move_end_pose.assert_not_called()
        robot.move_end_pose_linear.assert_not_called()
