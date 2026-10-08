"""Read-only RGB-D scene measurements for the live robot's simulation mirror.

Never imports a robot controller, camera driver, Qt or MuJoCo.
"""
from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MirrorCalibration:
    intrinsic: np.ndarray
    camera_to_end: np.ndarray
    block_size: np.ndarray
    freshness: float
    colors: dict
    min_points: int = 40
    top_band: float = .003
    size_tolerance: float = .012

    @classmethod
    def from_config(cls, config):
        from airbot_yolo import DEFAULT_DETECTION_CONFIG
        name = config.get('Camera', {}).get('calibration_type', config['AirbotGrasp']['camera_type'])
        resolution = config[name]['resolution']
        source = config[name][resolution]
        mirror = config.get('SimulationMirror', {})
        intrinsic = np.asarray(source['intrinsic'], dtype=float)
        transform = np.asarray(source['extrinsic'], dtype=float)
        size = np.asarray(mirror.get('block_size_m', [.04, .04, .04]), dtype=float)
        if (intrinsic.shape != (3, 3) or transform.shape != (4, 4) or size.shape != (3,)
                or not np.isfinite(intrinsic).all() or not np.isfinite(transform).all()
                or not np.isfinite(size).all() or np.any(size <= 0)
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
            raise ValueError('仿真镜像的内参、手眼外参或积木尺寸无效')
        if (not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.005)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=.005)):
            raise ValueError('仿真镜像手眼外参不是有效刚体变换')
        return cls(intrinsic.copy(), transform.copy(), size.copy(),
                   float(mirror.get('freshness_seconds', 1.)),
                   config.get('RealtimeDetection', {}).get('colors', DEFAULT_DETECTION_CONFIG['colors']))


def measure_blocks(snapshot, detections, calibration, depth_factor, now=None):
    """Estimate visible, axis-aligned cube centers, never the grasp-tool position.

    Top-surface coverage and expected dimensions are checked so side-only/occluded
    detections cannot silently become displaced cube centers. Rotation/stacking
    and arbitrary object meshes require a separate pose estimator.
    """
    now = time.monotonic() if now is None else now
    if not 0 <= now - snapshot.timestamp <= calibration.freshness:
        raise ValueError('真实检测帧已过期')
    trans = np.asarray(snapshot.state['trans'], dtype=float)
    quat = np.asarray(snapshot.state['orient'], dtype=float)
    if (trans.shape != (3,) or quat.shape != (4,) or not np.isfinite(trans).all()
            or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6):
        raise ValueError('该帧的真实末端位姿无效')
    if not np.isfinite(depth_factor) or depth_factor <= 0:
        raise ValueError('真实深度单位无效')
    depth = np.asarray(snapshot.depth, dtype=float) / depth_factor
    if depth.ndim != 2 or snapshot.color.shape != (*depth.shape, 3):
        raise ValueError('真实 RGB 与深度图未对齐')
    end_to_base = np.eye(4)
    end_to_base[:3, :3] = Rotation.from_quat(quat).as_matrix()
    end_to_base[:3, 3] = trans
    camera_to_base = end_to_base @ calibration.camera_to_end
    hsv = cv2.cvtColor(snapshot.color, cv2.COLOR_BGR2HSV)
    height, width = depth.shape
    blocks, rejected = [], []
    for index, detection in enumerate(detections):
        if not detection.observed or detection.color not in calibration.colors:
            continue
        color = detection.color
        x1, y1, x2, y2 = detection.bbox
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bounds = calibration.colors[color]
        mask = cv2.inRange(hsv[y1:y2, x1:x2], np.array(bounds['lower'], np.uint8),
                           np.array(bounds['upper'], np.uint8)).astype(bool)
        rows, cols = np.nonzero(mask)
        rows, cols = rows + y1, cols + x1
        z = depth[rows, cols]
        valid = np.isfinite(z) & (z > .05) & (z < 2.)
        rows, cols, z = rows[valid], cols[valid], z[valid]
        if len(z) < calibration.min_points:
            rejected.append(f'{color}：有效深度点不足')
            continue
        k = calibration.intrinsic
        points = np.column_stack(((cols - k[0, 2]) * z / k[0, 0],
                                  (rows - k[1, 2]) * z / k[1, 1], z))
        base = points @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
        top_height = np.percentile(base[:, 2], 90)
        top = base[np.abs(base[:, 2] - top_height) < calibration.top_band]
        if len(top) < calibration.min_points:
            rejected.append(f'{color}：顶面不可见')
            continue
        lower, upper = np.percentile(top[:, :2], [2, 98], axis=0)
        if np.any(np.abs((upper - lower) - calibration.block_size[:2]) > calibration.size_tolerance):
            rejected.append(f'{color}：顶面尺寸不符、旋转或遮挡')
            continue
        center = np.r_[(lower + upper) / 2, np.median(top[:, 2]) - calibration.block_size[2] / 2]
        blocks.append({'color': color, 'position': center.tolist(),
                       'size': calibration.block_size.tolist(), 'confidence': float(detection.confidence)})
    return {'event': 'blocks', 'timestamp': float(snapshot.timestamp),
            'blocks': blocks, 'rejected': rejected}
