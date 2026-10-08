"""arm-sdk 5.2.2 adapter tests; no service, CAN or robot is accessed."""

from types import SimpleNamespace
import unittest
import time
from threading import Lock

from arm_sdk import CartesianPose, Controller
from airbot_arm import (AirbotArm, AirbotMonitor, GripperContactTimeout,
                        SpeedProfile,
                        describe_eef_status, gripper_contact_detected)


class FakeClient:
    acquire_result = True

    def __init__(self, host="localhost", port=50051):
        self.host = host
        self.port = port
        self.calls = []
        self.closed = False
        self._lease_mu = Lock()
        self._lease_id = None
        self._lease_expire_unix_ms = None

    def acquire_control(self):
        self.calls.append(("acquire_control",))
        if self.acquire_result:
            self._lease_id = 1
            self._lease_expire_unix_ms = time.time() * 1000 + 15000
        return self.acquire_result

    def release_control(self):
        self.calls.append(("release_control",))

    def close(self):
        self.calls.append(("close",))
        self.closed = True

    def set_arm_speed(self, speed):
        self.calls.append(("set_arm_speed", speed))
        return True

    def switch_controller(self, controller):
        self.calls.append(("switch_controller", controller))
        return True

    def move_end_pose(self, pose, options, timeout_ms):
        self.calls.append(("move_end_pose", pose, options, timeout_ms))
        return True

    def move_end_pose_linear(self, start, target, options, timeout_ms):
        self.calls.append(
            ("move_end_pose_linear", start, target, options, timeout_ms))
        return True

    def move_end_pose_waypoints(self, poses, options, timeout_ms):
        self.calls.append(("move_end_pose_waypoints", poses, options, timeout_ms))
        return True

    def move_eef(self, width, options, timeout_ms):
        self.calls.append(("move_eef", width, options, timeout_ms))
        return True

    def clear_eef_motor_err(self):
        self.calls.append(("clear_eef_motor_err",))
        return True

    def enter_gravity_compensation_mode(self):
        self.calls.append(("enter_gravity_compensation_mode",))
        return True

    def get_end_pose(self):
        return CartesianPose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))

    def get_arm_joint_state(self):
        return SimpleNamespace(angles=(1, 2, 3, 4, 5, 6))

    def get_eef_joint_state(self):
        return SimpleNamespace(eef_pos=0.031, eef_vel=0.0, eef_eff=2.5)

    def get_eef_motor_state(self):
        return SimpleNamespace(
            eef_motor_temp=35.0, eef_mos_temp=37.0, eef_error_id=0)


class FailedLeaseClient(FakeClient):
    acquire_result = False


class FailedMotionClient(FakeClient):
    def move_end_pose(self, pose, options, timeout_ms):
        return False


class ArmAdapterTest(unittest.TestCase):
    def arm(self, factory=FakeClient):
        return AirbotArm(
            port=50051,
            motion_timeout_ms=90000,
            arm_joint_efforts=[7.0] * 6,
            eef_effort=6.0,
            client_factory=factory,
        )

    def test_acquires_single_control_lease(self):
        arm = self.arm()
        self.assertEqual(arm.client.calls[0], ("acquire_control",))
        self.assertEqual(sum(call[0] == "acquire_control" for call in arm.client.calls), 1)

    def test_lease_failure_closes_client(self):
        client = None

        def factory(**kwargs):
            nonlocal client
            client = FailedLeaseClient(**kwargs)
            return client

        with self.assertRaisesRegex(RuntimeError, "控制权"):
            self.arm(factory)
        self.assertTrue(client.closed)

    def test_close_releases_control_before_closing_client(self):
        arm = self.arm()
        client = arm.client

        arm.close()

        self.assertEqual(client.calls[-2:], [("release_control",), ("close",)])
        self.assertIsNone(arm.client)

    def test_speed_setup_failure_closes_client(self):
        class SpeedFailureClient(FakeClient):
            def set_arm_speed(self, speed):
                self.calls.append(("set_arm_speed", speed))
                return False

        client = None

        def factory(**kwargs):
            nonlocal client
            client = SpeedFailureClient(**kwargs)
            return client

        with self.assertRaisesRegex(RuntimeError, "set_arm_speed"):
            self.arm(factory)
        self.assertTrue(client.closed)

    def test_pose_motion_uses_planning_controller_and_blocking(self):
        arm = self.arm()
        arm.move_end_pose([[0.1, 0.2, 0.3], [0, 0, 0, 1]])
        self.assertIn(("switch_controller", Controller.planning_control), arm.client.calls)
        call = next(item for item in arm.client.calls if item[0] == "move_end_pose")
        self.assertIsInstance(call[1], CartesianPose)
        self.assertTrue(call[2].blocking)
        self.assertEqual(call[2].eff, [7.0] * 6)
        self.assertEqual(call[3], 90000)

    def test_waypoints_are_blocking_and_typed(self):
        arm = self.arm()
        arm.move_end_pose_waypoints([
            [[0.1, 0.2, 0.3], [0, 0, 0, 1]],
            [[0.1, 0.2, 0.4], [0, 0, 0, 1]],
        ])
        call = next(item for item in arm.client.calls if item[0] == "move_end_pose_waypoints")
        self.assertTrue(all(isinstance(pose, CartesianPose) for pose in call[1]))
        self.assertTrue(call[2].blocking)

    def test_linear_motion_uses_measured_start_and_blocking(self):
        arm = self.arm()
        arm.move_end_pose_linear([[0.1, 0.2, 0.15], [0, 0, 0, 1]])
        call = next(item for item in arm.client.calls
                    if item[0] == "move_end_pose_linear")
        self.assertEqual(list(call[1].position), [0.1, 0.2, 0.3])
        self.assertEqual(list(call[2].position), [0.1, 0.2, 0.15])
        self.assertTrue(call[3].blocking)
        self.assertEqual(call[4], 90000)

    def test_descent_rejects_offset_and_rotation_before_motion(self):
        from airbot_arm import DescentStartMismatch
        for expected in ([[.104,.2,.3],[0,0,0,1]],
                         [[.1,.2,.3],[0,0,.03,.99955]]):
            arm = self.arm()
            with self.assertRaises(DescentStartMismatch):
                arm.move_end_pose_linear([[.1,.2,.15],[0,0,0,1]], expected_start=expected)
            self.assertFalse(any(c[0] == 'move_end_pose_linear' for c in arm.client.calls))

    def test_aligned_descent_caps_speed_and_accepts_quaternion_sign(self):
        arm = self.arm()
        arm.move_end_pose_linear([[.1,.2,.15],[0,0,0,1]],
                                expected_start=[[.1,.2,.3],[0,0,0,-1]], speed_limit=.10)
        call = next(c for c in arm.client.calls if c[0] == 'move_end_pose_linear')
        self.assertLessEqual(call[3].velocity_scaling_factor, .10)
        self.assertLessEqual(call[3].acceleration_scaling_factor, .10)

    def test_gripper_uses_real_width_and_direct_controller(self):
        arm = self.arm()
        arm.move_gripper(0.072)
        self.assertIn(("switch_controller", Controller.direct_control), arm.client.calls)
        call = next(item for item in arm.client.calls if item[0] == "move_eef")
        self.assertEqual(call[1], 0.072)
        self.assertEqual(call[2].eef_eff, 6.0)
        self.assertTrue(call[2].blocking)
        self.assertEqual(call[3], 5000)

    def test_grasp_gripper_can_be_nonblocking_with_short_timeout(self):
        arm = self.arm()
        arm.move_gripper(0.027, blocking=False, timeout_ms=1500, effort=5.0)
        call = next(item for item in arm.client.calls if item[0] == "move_eef")
        self.assertEqual(call[1], 0.027)
        self.assertFalse(call[2].blocking)
        self.assertEqual(call[2].eef_eff, 5.0)
        self.assertEqual(call[3], 1500)

    def test_gripper_diagnostics_include_fault_and_temperature(self):
        arm = self.arm()
        state = arm.get_gripper_diagnostics()
        self.assertEqual(state["position"], 0.031)
        self.assertEqual(state["effort"], 2.5)
        self.assertEqual(state["error_id"], 0)
        self.assertEqual(state["motor_temperature"], 35.0)

    def test_g2_status_one_is_enabled_not_a_fault(self):
        self.assertEqual(describe_eef_status(1), "使能（正常）")
        state = {
            "position": 0.031, "velocity": 0.0, "effort": 2.5,
            "error_id": 1,
        }
        self.assertTrue(gripper_contact_detected(
            state, 0.027, 1.0, 0.002, 0.001))

    def test_g2_error_eight_is_over_voltage(self):
        self.assertEqual(describe_eef_status(8), "过压")

    def test_enabled_gripper_does_not_trigger_error_clear(self):
        arm = self.arm()
        arm.client.get_eef_motor_state = lambda: SimpleNamespace(
            eef_motor_temp=35.0, eef_mos_temp=37.0, eef_error_id=1)
        self.assertEqual(
            arm.clear_gripper_error(timeout_ms=50, poll_interval=0.001), 0)
        self.assertNotIn(("clear_eef_motor_err",), arm.client.calls)

    def test_clear_gripper_error_recovers_latched_fault(self):
        class LatchedFaultClient(FakeClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.error_id = 8

            def get_eef_motor_state(self):
                return SimpleNamespace(
                    eef_motor_temp=35.0, eef_mos_temp=37.0,
                    eef_error_id=self.error_id)

            def clear_eef_motor_err(self):
                self.calls.append(("clear_eef_motor_err",))
                self.error_id = 1
                return True

        arm = self.arm(LatchedFaultClient)
        self.assertEqual(
            arm.clear_gripper_error(timeout_ms=50, poll_interval=0.001), 8)
        self.assertIn(("clear_eef_motor_err",), arm.client.calls)
        self.assertEqual(arm.get_gripper_diagnostics()["error_id"], 1)

    def test_clear_gripper_error_rejects_persistent_fault(self):
        class PersistentFaultClient(FakeClient):
            def get_eef_motor_state(self):
                return SimpleNamespace(
                    eef_motor_temp=35.0, eef_mos_temp=37.0,
                    eef_error_id=8)

        arm = self.arm(PersistentFaultClient)
        with self.assertRaisesRegex(RuntimeError, "过压.*无法清除"):
            arm.clear_gripper_error(timeout_ms=5, poll_interval=0.001)

    def test_contact_requires_gap_low_velocity_effort_and_no_fault(self):
        state = {
            "position": 0.031, "velocity": 0.0, "effort": 2.5,
            "error_id": 0,
        }
        self.assertTrue(gripper_contact_detected(
            state, 0.027, 1.0, 0.002, 0.001))
        for key, value in (("position", 0.027), ("velocity", 0.01),
                           ("effort", 0.2), ("error_id", 3)):
            changed = dict(state, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(gripper_contact_detected(
                    changed, 0.027, 1.0, 0.002, 0.001))

    def test_wait_for_contact_uses_consecutive_feedback(self):
        arm = self.arm()
        state = arm.wait_for_gripper_contact(
            0.027, timeout_ms=100, effort_threshold=1.0,
            strong_effort_threshold=8.0,
            velocity_threshold=0.002, position_tolerance=0.001,
            confirm_samples=2, poll_interval=0.001)
        self.assertEqual(state["position"], 0.031)
        self.assertEqual(state["contact_mode"], "开度受阻")

    def test_contact_timeout_exposes_last_feedback(self):
        arm = self.arm()
        arm.client.get_eef_joint_state = lambda: SimpleNamespace(
            eef_pos=0.0268, eef_vel=0.0002, eef_eff=1.72)
        with self.assertRaises(GripperContactTimeout) as raised:
            arm.wait_for_gripper_contact(
                0.0268, timeout_ms=5, effort_threshold=1.0,
                strong_effort_threshold=8.0,
                velocity_threshold=0.002, position_tolerance=0.001,
                confirm_samples=3, poll_interval=0.001)
        self.assertAlmostEqual(raised.exception.target_width, 0.0268)
        self.assertAlmostEqual(
            raised.exception.last_state["effort"], 1.72)

    def test_strong_effort_confirms_contact_near_target(self):
        state = {
            "position": 0.0251, "velocity": 0.0002, "effort": 15.0,
            "error_id": 1,
        }
        self.assertTrue(gripper_contact_detected(
            state, 0.0252, 1.0, 0.002, 0.001, 8.0))

    def test_low_effort_near_target_is_not_contact(self):
        state = {
            "position": 0.0251, "velocity": 0.0002, "effort": 2.0,
            "error_id": 1,
        }
        self.assertFalse(gripper_contact_detected(
            state, 0.0252, 1.0, 0.002, 0.001, 8.0))

    def test_speed_profile_calls_new_sdk_api(self):
        arm = self.arm()
        arm.set_speed_profile(SpeedProfile.SLOW)
        calls = [item for item in arm.client.calls if item[0] == "set_arm_speed"]
        self.assertEqual(len(calls[-1][1]), 6)
        self.assertTrue(all(speed > 0 for speed in calls[-1][1]))

    def test_gravity_mode_transitions(self):
        arm = self.arm()
        arm.enter_gravity_compensation()
        self.assertTrue(arm.gravity_enabled)
        arm.leave_gravity_compensation()
        self.assertFalse(arm.gravity_enabled)
        self.assertEqual(arm.client.calls[-1], ("switch_controller", Controller.planning_control))

    def test_failed_motion_raises(self):
        arm = self.arm(FailedMotionClient)
        with self.assertRaisesRegex(RuntimeError, "move_end_pose"):
            arm.move_end_pose([[0.1, 0.2, 0.3], [0, 0, 0, 1]])

    def test_lost_lease_blocks_motion_without_reacquiring(self):
        arm = self.arm()
        arm.client._lease_id = None
        before = len(arm.client.calls)
        with self.assertRaisesRegex(RuntimeError, '租约已失效'):
            arm.move_gripper(.04)
        self.assertEqual(len(arm.client.calls), before)

    def test_expired_lease_blocks_arm_motion(self):
        arm = self.arm()
        arm.client._lease_expire_unix_ms = 0
        before = len(arm.client.calls)
        with self.assertRaisesRegex(RuntimeError, '租约已失效'):
            arm.move_end_pose([[.3, 0, .2], [0, 0, 0, 1]])
        self.assertEqual(len(arm.client.calls), before)

    def test_explicit_restore_creates_new_client_without_motion(self):
        arm = self.arm()
        old = arm.client
        old._lease_id = None
        arm.restore_control()
        self.assertTrue(old.closed)
        self.assertIsNot(old, arm.client)
        self.assertEqual([c[0] for c in arm.client.calls], ['acquire_control', 'set_arm_speed'])
        arm.move_gripper(.04)
        self.assertEqual(arm.client.calls[-1][0], 'move_eef')

    def test_failed_restore_does_not_keep_dead_client(self):
        arm = self.arm()
        arm._client_factory = FailedLeaseClient
        with self.assertRaisesRegex(RuntimeError, '无法恢复'):
            arm.restore_control()
        self.assertIsNone(arm.client)
        with self.assertRaisesRegex(RuntimeError, '控制连接已关闭'):
            arm.move_gripper(.04)

    def test_monitor_never_acquires_control(self):
        monitor = AirbotMonitor(client_factory=FakeClient)
        state = monitor.read_state()
        self.assertFalse(any(call[0] == "acquire_control" for call in monitor.client.calls))
        self.assertEqual(state["trans"], [0.1, 0.2, 0.3])
        self.assertEqual(state["joints"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(state["eef"], [0.031])


if __name__ == "__main__":
    unittest.main()
