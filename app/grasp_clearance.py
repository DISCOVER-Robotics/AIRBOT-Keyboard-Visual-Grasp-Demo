"""Validate measured jaw opening before approaching an object (metres)."""
import math
from airbot_arm import EEF_NON_FAULT_CODES, describe_eef_status


def require_open_gripper(feedback, requested_width, tolerance=.001):
    actual = float(feedback['position'])
    velocity = float(feedback['velocity'])
    if (not all(math.isfinite(v) for v in (actual, velocity, requested_width, tolerance))
            or actual < 0 or requested_width <= 0 or tolerance < 0):
        raise RuntimeError('夹爪开度反馈无效，不允许下降')
    status = feedback["error_id"]
    if status not in EEF_NON_FAULT_CODES:
        raise RuntimeError(
            f"夹爪错误码 {status}（{describe_eef_status(status)}），不允许下降")
    if actual < requested_width - tolerance or abs(velocity) > .002:
        raise RuntimeError(
            f'夹爪未稳定张开到位：要求 {requested_width*1000:.1f} mm，'
            f'反馈 {actual*1000:.1f} mm，速度 {velocity*1000:.1f} mm/s；不允许下降')
