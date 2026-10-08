"""AIRBOT arm-sdk 5.2.2 integration used by the grasp demo.

The control client owns the single lease used by the GUI.  The monitor client is
read-only, so the camera/state thread cannot compete for motion authority.
"""

from __future__ import annotations

from enum import Enum
import math
import numpy as np
from scipy.spatial.transform import Rotation
from threading import RLock
import time

from arm_sdk import (
    AirbotClient,
    ArmControlOptions,
    CartesianPose,
    Controller,
    WaypointSegmentOptions,
    WaypointsControlOptions,
    __version__ as ARM_SDK_VERSION,
)

REQUIRED_ARM_SDK_VERSION = "5.2.2"
if ARM_SDK_VERSION != REQUIRED_ARM_SDK_VERSION:
    raise ImportError(
        f"graspdemo 需要 arm-sdk {REQUIRED_ARM_SDK_VERSION}，当前为 {ARM_SDK_VERSION}"
    )


class SpeedProfile(Enum):
    SLOW = "SLOW"
    DEFAULT = "DEFAULT"
    MEDIUM = "MEDIUM"
    FAST = "FAST"


DEFAULT_SPEED_PROFILES = {
    SpeedProfile.SLOW: (math.pi / 6, 0.10, 0.10),
    SpeedProfile.DEFAULT: (math.pi / 3, 0.30, 0.30),
    SpeedProfile.MEDIUM: (math.pi / 2, 0.50, 0.50),
    SpeedProfile.FAST: (math.pi, 0.70, 0.70),
}

# G2 uses a DM J4310 motor.  Its status values differ from the OD motor table:
# 0 and 1 are disabled/enabled states, not faults.
EEF_NON_FAULT_CODES = frozenset((0x00, 0x01))
EEF_STATUS_DESCRIPTIONS = {
    0x00: "失能（非故障）",
    0x01: "使能（正常）",
    0x08: "过压",
    0x09: "欠压",
    0x0A: "过电流",
    0x0B: "MOS 过温",
    0x0C: "电机绕圈过温",
    0x0D: "通信丢失",
    0x0E: "过载",
}


class ControlLeaseLost(RuntimeError):
    """Control must be restored explicitly; never replay an interrupted motion."""


class DescentStartMismatch(RuntimeError):
    """No descent was sent; keep the robot at its current pre-grasp pose."""


class GripperContactTimeout(RuntimeError):
    """The gripper reached no confirmed contact before the deadline."""

    def __init__(self, message, *, target_width, last_state):
        super().__init__(message)
        self.target_width = float(target_width)
        self.last_state = last_state


def describe_eef_status(error_id):
    error_id = int(error_id)
    return EEF_STATUS_DESCRIPTIONS.get(error_id, "未知故障")


def _require(result, operation):
    if not result:
        raise RuntimeError(f"arm-sdk 5.2.2 调用失败：{operation}")


def _pose(value):
    if len(value) != 2 or len(value[0]) != 3 or len(value[1]) != 4:
        raise ValueError("末端位姿必须为 [[x, y, z], [qx, qy, qz, qw]]")
    return CartesianPose(
        position=tuple(float(item) for item in value[0]),
        orientation=tuple(float(item) for item in value[1]),
    )


def gripper_contact_detected(
    state,
    target_width,
    effort_threshold,
    velocity_threshold,
    position_tolerance,
    strong_effort_threshold=None,
):
    """Return true when stable feedback indicates physical contact.

    Vision and gripper width estimates can differ by about a millimetre.  A
    large, sustained effort therefore also confirms contact near the target,
    while the normal path still requires the gripper to stop above it.
    """
    position = float(state["position"])
    target_width = float(target_width)
    effort = abs(float(state["effort"]))
    position_tolerance = float(position_tolerance)
    blocked_above_target = (
        position > target_width + position_tolerance
        and effort >= float(effort_threshold)
    )
    strong_contact_near_target = (
        strong_effort_threshold is not None
        and position >= target_width - position_tolerance
        and effort >= float(strong_effort_threshold)
    )
    return (
        (blocked_above_target or strong_contact_near_target)
        and abs(float(state["velocity"])) <= float(velocity_threshold)
        and state["error_id"] in EEF_NON_FAULT_CODES
    )


class AirbotMonitor:
    """Read-only state connection; it intentionally never acquires a lease."""

    def __init__(self, host="localhost", port=50051, client_factory=AirbotClient):
        self.host = host
        self.port = int(port)
        self._client_factory = client_factory
        self.client = client_factory(host=host, port=self.port)

    def reconnect(self):
        self.close()
        self.client = self._client_factory(host=self.host, port=self.port)

    def read_state(self):
        pose = self.client.get_end_pose()
        arm = self.client.get_arm_joint_state()
        eef = self.client.get_eef_joint_state()
        if pose is None or arm is None or eef is None:
            raise RuntimeError("arm-sdk 5.2.2 未返回完整机械臂状态")
        return {
            "trans": list(pose.position),
            "orient": list(pose.orientation),
            "joints": list(arm.angles),
            "eef": [eef.eef_pos],
            "eeff": [eef.eef_eff],
        }

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None


class AirbotArm:
    """Single lease-owning arm-sdk 5.2.2 client for all motion commands."""

    def __init__(
        self,
        host="localhost",
        port=50051,
        motion_timeout_ms=120000,
        gripper_motion_timeout_ms=5000,
        arm_joint_efforts=None,
        eef_effort=8.0,
        client_factory=AirbotClient,
    ):
        self.host = host
        self.port = int(port)
        self.motion_timeout_ms = int(motion_timeout_ms)
        self.gripper_motion_timeout_ms = int(gripper_motion_timeout_ms)
        self.arm_joint_efforts = list(arm_joint_efforts or [8.0] * 6)
        self.eef_effort = float(eef_effort)
        self.speed_profile = SpeedProfile.DEFAULT
        self.gravity_enabled = False
        self._lock = RLock()
        self._client_factory = client_factory
        self.client = client_factory(host=host, port=self.port)
        if not self.client.acquire_control():
            self.client.close()
            raise RuntimeError("无法取得机械臂控制权；可能已有其他客户端持有 lease")
        try:
            self.set_speed_profile(self.speed_profile)
        except Exception:
            # Do not leave the lease-renewal thread alive when initialization
            # fails after control has already been acquired.
            self.client.close()
            self.client = None
            raise

    @classmethod
    def from_config(cls, config, client_factory=AirbotClient):
        return cls(
            host=config.get("host", "localhost"),
            port=config["port"],
            motion_timeout_ms=config.get("motion_timeout_ms", 120000),
            gripper_motion_timeout_ms=config.get(
                "gripper_motion_timeout_ms", 5000),
            arm_joint_efforts=config.get("arm_joint_efforts", [8.0] * 6),
            eef_effort=config.get("eef_effort", 8.0),
            client_factory=client_factory,
        )

    def _ensure_control(self):
        if self.client is None:
            raise ControlLeaseLost('控制连接已关闭，请点击“恢复控制连接”后重新确认指令。')
        # SDK 5.2.2 has no public lease-status accessor. Read its bookkeeping
        # under the SDK lock; never modify it or start a second renewal thread.
        with self.client._lease_mu:
            lease = self.client._lease_id
            expiry = self.client._lease_expire_unix_ms
        if lease is None or expiry is None or expiry <= time.time() * 1000:
            raise ControlLeaseLost(
                '机械臂控制租约已失效，本次动作已取消。请检查 airbot-arm 服务及 '
                'renew_control failed 日志，点击“恢复控制连接”，重新检测后再确认指令。')

    def restore_control(self):
        """Explicitly restore a client, without switching controllers or moving."""
        with self._lock:
            self.close()
            client = self._client_factory(host=self.host, port=self.port)
            try:
                if not client.acquire_control():
                    raise RuntimeError('无法恢复控制权：请检查服务及其他控制客户端；未执行任何运动。')
                self.client = client
                self._ensure_control()
                self.set_speed_profile(self.speed_profile)
                self.gravity_enabled = False
            except Exception:
                self.client = None
                client.close()
                raise

    def _motion_options(self):
        _, velocity, acceleration = DEFAULT_SPEED_PROFILES[self.speed_profile]
        return ArmControlOptions(
            eff=self.arm_joint_efforts.copy(),
            eef_eff=self.eef_effort,
            velocity_scaling_factor=velocity,
            acceleration_scaling_factor=acceleration,
            allow_planning_time=5.0,
            blocking=True,
        )

    def set_speed_profile(self, profile):
        if isinstance(profile, str):
            profile = SpeedProfile[profile]
        max_speed, _, _ = DEFAULT_SPEED_PROFILES[profile]
        with self._lock:
            _require(self.client.set_arm_speed([max_speed] * 6), "set_arm_speed")
            self.speed_profile = profile

    def get_end_pose(self):
        pose = self.client.get_end_pose()
        if pose is None:
            raise RuntimeError("arm-sdk 5.2.2 未返回末端位姿")
        return [list(pose.position), list(pose.orientation)]

    def move_end_pose(self, pose):
        with self._lock:
            self._ensure_control()
            _require(
                self.client.switch_controller(Controller.planning_control),
                "switch_controller(planning_control)",
            )
            self.gravity_enabled = False
            _require(
                self.client.move_end_pose(
                    _pose(pose), self._motion_options(), self.motion_timeout_ms
                ),
                "move_end_pose",
            )

    def move_end_pose_linear(self, pose, *, expected_start=None, speed_limit=None):
        """Move linearly from the measured current pose to ``pose``."""
        with self._lock:
            self._ensure_control()
            _require(
                self.client.switch_controller(Controller.planning_control),
                "switch_controller(planning_control)",
            )
            self.gravity_enabled = False
            start = self.client.get_end_pose()
            if start is None:
                raise RuntimeError("arm-sdk 5.2.2 未返回直线运动起点位姿")
            if expected_start is not None:
                position_error = np.linalg.norm(np.asarray(start.position) - expected_start[0])
                rotation_error = (Rotation.from_quat(start.orientation).inv()
                                  * Rotation.from_quat(expected_start[1])).magnitude()
                if (not np.isfinite(position_error) or not np.isfinite(rotation_error)
                        or position_error > .001 or rotation_error > math.radians(1.)):
                    raise DescentStartMismatch(
                        f'预抓取起点未到位：位置误差 {position_error*1000:.4f} mm，'
                        f'姿态误差 {math.degrees(rotation_error):.2f}°；'
                        f'XYZ 偏差（反馈−目标，mm）{((np.asarray(start.position) - expected_start[0])*1000).round(4).tolist()}；'
                        '要求不超过 1 mm / 1°，'
                        '未发送下降指令，停留在当前位置')
            options = self._motion_options()
            if speed_limit is not None:
                if not math.isfinite(speed_limit) or not 0 < speed_limit <= 1:
                    raise ValueError('直线运动速度比例必须在 (0, 1] 内')
                options.velocity_scaling_factor = min(options.velocity_scaling_factor, speed_limit)
                options.acceleration_scaling_factor = min(options.acceleration_scaling_factor, speed_limit)
            _require(
                self.client.move_end_pose_linear(
                    start, _pose(pose), options,
                    self.motion_timeout_ms,
                ),
                "move_end_pose_linear",
            )

    def move_end_pose_waypoints(self, waypoints):
        poses = [_pose(item) for item in waypoints]
        if len(poses) < 2:
            raise ValueError("笛卡尔路点至少需要两个位姿")
        _, velocity, acceleration = DEFAULT_SPEED_PROFILES[self.speed_profile]
        segment = WaypointSegmentOptions(
            motion_type="ptp",
            velocity_scaling_factor=velocity,
            acceleration_scaling_factor=acceleration,
            allow_planning_time=5.0,
        )
        options = WaypointsControlOptions(default_segment=segment, blocking=True)
        with self._lock:
            self._ensure_control()
            _require(
                self.client.switch_controller(Controller.planning_control),
                "switch_controller(planning_control)",
            )
            self.gravity_enabled = False
            _require(
                self.client.move_end_pose_waypoints(
                    poses, options, self.motion_timeout_ms
                ),
                "move_end_pose_waypoints",
            )

    def move_gripper(
        self, width, *, blocking=True, timeout_ms=None, effort=None
    ):
        timeout_ms = (self.gripper_motion_timeout_ms
                      if timeout_ms is None else int(timeout_ms))
        if timeout_ms <= 0:
            raise ValueError("夹爪运动超时必须大于 0")
        with self._lock:
            self._ensure_control()
            _require(
                self.client.switch_controller(Controller.direct_control),
                "switch_controller(direct_control)",
            )
            self.gravity_enabled = False
            options = self._motion_options()
            options.blocking = bool(blocking)
            if effort is not None:
                options.eef_eff = float(effort)
            _require(
                self.client.move_eef(
                    float(width), options, timeout_ms
                ),
                "move_eef",
            )

    def get_gripper_diagnostics(self):
        with self._lock:
            joint = self.client.get_eef_joint_state()
            motor = self.client.get_eef_motor_state()
        if joint is None or motor is None:
            raise RuntimeError("arm-sdk 5.2.2 未返回完整夹爪状态")
        return {
            "position": float(joint.eef_pos),
            "velocity": float(joint.eef_vel),
            "effort": float(joint.eef_eff),
            "motor_temperature": float(motor.eef_motor_temp),
            "mos_temperature": float(motor.eef_mos_temp),
            "error_id": int(motor.eef_error_id),
        }

    def clear_gripper_error(self, *, timeout_ms=500, poll_interval=0.05):
        """Clear a latched EEF fault and verify that the motor is ready.

        Returns the old error code so callers can report that recovery took
        place.  A fault which cannot be cleared remains a hard stop.
        """
        timeout_ms = int(timeout_ms)
        poll_interval = float(poll_interval)
        if timeout_ms <= 0 or poll_interval <= 0:
            raise ValueError("夹爪错误清除参数必须大于 0")

        with self._lock:
            self._ensure_control()
        initial = self.get_gripper_diagnostics()
        old_error = initial["error_id"]
        if old_error in EEF_NON_FAULT_CODES:
            return 0

        with self._lock:
            _require(self.client.clear_eef_motor_err(), "clear_eef_motor_err")

        deadline = time.monotonic() + timeout_ms / 1000.0
        last_state = initial
        while time.monotonic() < deadline:
            last_state = self.get_gripper_diagnostics()
            if last_state["error_id"] in EEF_NON_FAULT_CODES:
                return old_error
            time.sleep(poll_interval)

        error_id = last_state["error_id"]
        raise RuntimeError(
            f"夹爪历史错误码 {error_id} (0x{error_id:02X}, "
            f"{describe_eef_status(error_id)}) 无法清除，停止抓取")

    def wait_for_gripper_contact(
        self,
        target_width,
        *,
        timeout_ms,
        effort_threshold,
        strong_effort_threshold,
        velocity_threshold,
        position_tolerance,
        confirm_samples=3,
        poll_interval=0.05,
    ):
        """Wait until consecutive feedback samples prove physical contact."""
        timeout_ms = int(timeout_ms)
        confirm_samples = int(confirm_samples)
        poll_interval = float(poll_interval)
        effort_threshold = float(effort_threshold)
        strong_effort_threshold = float(strong_effort_threshold)
        if (timeout_ms <= 0 or confirm_samples <= 0 or poll_interval <= 0
                or effort_threshold <= 0
                or strong_effort_threshold < effort_threshold):
            raise ValueError("夹爪接触检测参数必须大于 0")
        deadline = time.monotonic() + timeout_ms / 1000.0
        consecutive = 0
        last_state = None
        while time.monotonic() < deadline:
            last_state = self.get_gripper_diagnostics()
            if last_state["error_id"] not in EEF_NON_FAULT_CODES:
                error_id = last_state["error_id"]
                raise RuntimeError(
                    f"夹爪电机错误码 {error_id} (0x{error_id:02X}, "
                    f"{describe_eef_status(error_id)})，停止抓取；"
                    f"实际开度 {last_state['position']:.4f} m，"
                    f"速度 {last_state['velocity']:.4f}，"
                    f"作用力 {last_state['effort']:.2f}，"
                    f"电机温度 {last_state['motor_temperature']:.1f} C，"
                    f"MOS 温度 {last_state['mos_temperature']:.1f} C")
            if gripper_contact_detected(
                last_state, target_width, effort_threshold,
                velocity_threshold, position_tolerance,
                strong_effort_threshold,
            ):
                consecutive += 1
                if consecutive >= confirm_samples:
                    normal_gap = (
                        last_state["position"]
                        > float(target_width) + float(position_tolerance)
                    )
                    last_state["contact_mode"] = (
                        "开度受阻" if normal_gap else "强作用力"
                    )
                    return last_state
            else:
                consecutive = 0
            time.sleep(poll_interval)
        detail = "无反馈" if last_state is None else (
            f"目标开度 {float(target_width):.4f} m，"
            f"实际开度 {last_state['position']:.4f} m，"
            f"开度差 {last_state['position'] - float(target_width):+.4f} m，"
            f"速度 {last_state['velocity']:.4f}，"
            f"作用力 {last_state['effort']:.2f}，"
            f"强接触阈值 {strong_effort_threshold:.2f}")
        raise GripperContactTimeout(
            f"夹持超时，未确认夹住物体（{detail}）",
            target_width=target_width,
            last_state=last_state,
        )

    def enter_gravity_compensation(self):
        with self._lock:
            self._ensure_control()
            _require(
                self.client.enter_gravity_compensation_mode(),
                "enter_gravity_compensation_mode",
            )
            self.gravity_enabled = True

    def leave_gravity_compensation(self):
        with self._lock:
            self._ensure_control()
            _require(
                self.client.switch_controller(Controller.planning_control),
                "switch_controller(planning_control)",
            )
            self.gravity_enabled = False

    def close(self):
        with self._lock:
            if self.client is not None:
                client = self.client
                self.client = None
                try:
                    client.release_control()
                finally:
                    client.close()
