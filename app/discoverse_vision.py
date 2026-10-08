"""User YOLO on simulated RGB, with metric depth backprojection for grasping."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from copy import deepcopy
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from pathlib import Path
import time

import cv2
import numpy as np

from airbot_yolo import (AirbotYolo, DetectionStabilizer, draw_detections,
                         resolve_unique_target)


@dataclass
class CameraFrame:
    bgr: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    fovy: float
    captured: float
    end_position: np.ndarray | None = None
    end_rotation: np.ndarray | None = None


class SimulatedRGBDCamera:
    """RealSense-compatible CV optical frame: X right, Y down, Z forward."""
    has_hardware_depth = True
    depth_factor = 1.0  # MuJoCo depth is already in metres.
    distortion = np.zeros(5)

    def __init__(self, frame):
        height, width = frame.depth.shape
        focal = height / (2 * np.tan(np.deg2rad(frame.fovy) / 2))
        self.intrinsic = np.array([[focal, 0., width / 2 - .5],
                                   [0., focal, height / 2 - .5], [0., 0., 1.]])

    def create_point_cloud(self, depth, organized=True, end_pose=None):
        rows, cols = np.indices(depth.shape)
        cloud = np.stack(((cols - self.intrinsic[0, 2]) * depth / self.intrinsic[0, 0],
                          (rows - self.intrinsic[1, 2]) * depth / self.intrinsic[1, 1],
                          depth), axis=-1)
        return cloud if organized else cloud.reshape(-1, 3)


def camera_to_end(frame):
    transform = np.eye(4)
    transform[:3, :3] = frame.end_rotation.T @ frame.rotation @ np.diag([1., -1., -1.])
    transform[:3, 3] = frame.end_rotation.T @ (frame.position - frame.end_position)
    return transform


class SimulationVision:
    """One inference in flight; rendering/physics stay on the calling thread."""
    def __init__(self, sim, checkpoint=None, detector=None):
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(f'找不到 YOLO 权重：{checkpoint}')
        self.sim = sim
        self.detector = detector or AirbotYolo(checkpoint=checkpoint)
        labels = set(self.detector.model.names.values())
        if not labels.intersection(self.detector.detection_config['candidate_labels']):
            raise ValueError(f'模型类别 {sorted(labels)} 与 candidate_labels 不匹配。')
        self.settings = self.detector.detection_config
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sim-yolo')
        self.future = None
        self.segment = None
        self.grasp_predictor = None
        self.generation = 0
        self.interval = 1 / float(self.settings.get('inference_hz', 10))
        self.freshness = float(self.settings.get('freshness_seconds', 1.))
        self.submitted = 0.
        self.invalidate()

    def invalidate(self):
        self.generation += 1
        self.stabilizer = DetectionStabilizer(self.settings)
        self.detections = []
        self.raw_detections = []
        self.frame = None
        self.error = None
        self.status = '等待 YOLO 连续确认目标…'

    def capture(self):
        bgr = cv2.cvtColor(self.sim.render('eye_arm'), cv2.COLOR_RGB2BGR)
        depth = self.sim.render('eye_arm', depth=True)
        camera = self.sim.data.camera('eye_arm')
        return CameraFrame(bgr, depth, camera.xpos.copy(), camera.xmat.reshape(3, 3).copy(),
                           float(self.sim.model.camera('eye_arm').fovy[0]), time.monotonic(),
                           self.sim.data.body('link6').xpos.copy(),
                           self.sim.data.body('link6').xmat.reshape(3, 3).copy())

    def _accept(self, frame, detections):
        self.frame = frame
        self.raw_detections = detections
        self.detections = self.stabilizer.update(detections)
        self.error = None
        self.status = 'YOLO：' + ('；'.join(
            f'{d.display_label} {d.confidence:.0%}' for d in self.detections)
            or '未确认积木（不会执行抓取）')

    def poll(self):
        """Call from Qt's timer. Never touch MuJoCo from the inference thread."""
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            if self.pending_generation == self.generation and not self.sim.busy:
                try:
                    self._accept(self.pending_frame, future.result())
                except Exception as exc:
                    self.invalidate()
                    self.error = str(exc)
                    self.status = 'YOLO 检测失败：' + self.error
            else:
                # Observe exceptions even for invalidated results.
                try:
                    future.result()
                except Exception:
                    pass
        if self.future is None and not self.sim.busy and time.monotonic() - self.submitted >= self.interval:
            self.pending_frame = self.capture()
            self.pending_generation = self.generation
            self.future = self.executor.submit(self.detector.detect_candidates, self.pending_frame.bgr)
            self.submitted = time.monotonic()

    def warmup(self):
        """Synchronous path for command-line checks, never used by Qt's timer."""
        self.invalidate()
        for _ in range(max(self.stabilizer.confirm_frames, self.stabilizer.color_confirm)):
            frame = self.capture()
            self._accept(frame, self.detector.detect_candidates(frame.bgr))

    def resolve(self, color):
        if self.error:
            raise ValueError('YOLO 检测失败：' + self.error)
        if self.frame is None or time.monotonic() - self.frame.captured > self.freshness:
            raise ValueError('YOLO 结果尚未就绪或已过期，请等检测框稳定后重试。')
        resolve_unique_target(self.detections, color)
        # Reject newly appearing duplicates too; depth must use the unsmoothed box.
        return resolve_unique_target(self.raw_detections, color)

    def prepare(self, color):
        detection = self.resolve(color)
        frame = self.frame
        detections = tuple(self.detections)
        self.status = 'MobileSAM 分割 → 深度点云 → SimpleGrasp 位姿计算…'
        return self.executor.submit(self._prepare_frame, frame, detection, detections)

    def _prepare_frame(self, frame, detection, detections):
        # This worker reads ONLY the captured frame and static config, never MjData.
        from airbot_segment import AirbotSegment
        from airbot_grasp_simple import SimpleGrasp
        from grasp_preparation import prepare_color_grasp
        if self.segment is None:
            self.segment = AirbotSegment()
        if self.grasp_predictor is None:
            config = deepcopy(self.sim.real_config)
            config['RunTime']['dump'] = False
            config['AirbotGrasp']['grasp_offset_base_xy_m'] = [0., 0.]
            config['AirbotGrasp']['grasp_offset_camera_xy_m'] = [0., 0.]
            config['AirbotGrasp']['position_offset'] = {'enabled': False}
            config['AirbotGrasp']['block_size_m'] = .04 # standalone scene cubes
            config['ArmParams']['gripper_length'] = self.sim.sim_config['gripper_length']
            config['AirbotGrasp']['min_grasp_z'] = self.sim.sim_config['min_grasp_z']
            self.grasp_predictor = SimpleGrasp(config=config)
        self.grasp_predictor.cam2end = camera_to_end(frame)
        camera = SimulatedRGBDCamera(frame)
        depth = frame.depth.copy()
        depth[~np.isfinite(depth) | (depth <= 0) | (depth > 2.)] = 0.
        snapshot = SimpleNamespace(
            color=frame.bgr, depth=depth, timestamp=frame.captured,
            state={'trans': frame.end_position.copy(),
                   'orient': Rotation.from_matrix(frame.end_rotation).as_quat()})
        return prepare_color_grasp(snapshot, detection, detections, self.segment, camera,
                                   self.grasp_predictor, self.settings, save_cloud=False)

    def overlay(self):
        if self.frame is None:
            return None
        detections = self.detections if time.monotonic() - self.frame.captured <= self.freshness else []
        return cv2.cvtColor(draw_detections(self.frame.bgr, detections), cv2.COLOR_BGR2RGB)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
