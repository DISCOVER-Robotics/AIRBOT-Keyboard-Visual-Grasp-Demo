"""Calibrated pixel rays for aligned RGB-D, without opening hardware."""
from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def position_camera_offset(profile, image_x_fraction):
    """Interpolate total camera offsets; clamp outside the two reference points."""
    anchors = np.asarray(profile['image_x_fractions'], dtype=float)
    offsets = np.asarray(profile['offsets_camera_xy_m'], dtype=float)
    if (anchors.shape != (2,) or offsets.shape != (2, 2)
            or not np.isfinite(anchors).all() or not np.isfinite(offsets).all()
            or not 0 <= anchors[0] < anchors[1] <= 1):
        raise ValueError('位置补偿需要两个递增的画面横坐标比例和两组有限 XY 补偿')
    if not np.isfinite(image_x_fraction) or not 0 <= image_x_fraction <= 1:
        raise ValueError('位置补偿的目标横坐标比例无效')
    weight = float(np.clip((image_x_fraction - anchors[0]) / (anchors[1] - anchors[0]), 0, 1))
    return offsets[0] * (1 - weight) + offsets[1] * weight


def camera_xy_offset_in_base(offset, end_pose, camera_to_end):
    """Camera right/down metres, constrained to a constant base Z plane."""
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError('相机方向补偿必须为两个有限数值（米）')
    if not np.any(offset):
        return np.zeros(2)
    rotation = Rotation.from_quat(end_pose[1]).as_matrix() @ np.asarray(camera_to_end)[:3, :3]
    mapping = rotation.T[:2, :2]
    if not np.isfinite(mapping).all() or np.linalg.cond(mapping) > 10:
        raise ValueError('相机视角过于倾斜，无法可靠换算水平抓取补偿')
    return np.linalg.solve(mapping, offset)


@lru_cache(maxsize=4)
def _rays(height, width, matrix, coefficients):
    intrinsic = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    y, x = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, intrinsic, np.asarray(coefficients, dtype=np.float64))
    rays = np.concatenate((xy.reshape(height, width, 2),
                           np.ones((height, width, 1))), axis=-1)
    rays.setflags(write=False)
    return rays


def calibrated_point_cloud(depth, intrinsic, distortion, depth_factor, profile):
    """Keep pixel indexing unchanged; invert OpenCV lens distortion in rays."""
    depth = np.asarray(depth)
    if depth.shape != (profile[1], profile[0]):
        raise ValueError('深度图尺寸与标定分辨率不一致，不能计算抓取点云')
    matrix = np.asarray(intrinsic, dtype=float)
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError('相机内参无效')
    if coefficients.size not in (4, 5, 8, 12, 14) or not np.isfinite(coefficients).all():
        raise ValueError('需要有效的 OpenCV 相机畸变系数')
    if not np.isfinite(depth_factor) or depth_factor <= 0:
        raise ValueError('深度单位比例无效')
    rays = _rays(*depth.shape, tuple(matrix.ravel()), tuple(coefficients))
    return rays * (depth.astype(float) / depth_factor)[..., None]
