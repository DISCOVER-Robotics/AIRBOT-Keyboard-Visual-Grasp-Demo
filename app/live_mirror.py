"""Read-only real robot mirror. Never sends a command to a robot or simulation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def load_settings(path='configs/live_mirror.yaml'):
    with open(path, encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    transform = np.asarray(cfg['world_from_base'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.)):
        raise ValueError('world_from_base 必须是有效刚体变换')
    for key in ('joint_signs', 'joint_offsets'):
        arr = np.asarray(cfg[key], dtype=float)
        if arr.shape != (6,) or not np.isfinite(arr).all():
            raise ValueError(f'{key} 必须有六个有限数值')
    if not np.all(np.isin(cfg['joint_signs'], [-1, 1])):
        raise ValueError('joint_signs 只允许 ±1')
    size = np.asarray(cfg['block_size'], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('block_size 必须是三个正数')
    for key in ('state_timeout', 'object_timeout', 'publish_hz'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} 必须为正数')
    return cfg


def measured_blocks(batch, camera, cam2end, detection_settings, settings, *, diagnostics=None):
    """Estimate visible cube centers from real RGB-D in the REAL base frame.

    No simulator poses enter this function. Duplicates/occlusion/invalid depth
    yield no target. Fixed-size cube geometry is an explicit approximation.
    """
    if not camera.has_hardware_depth:
        raise ValueError('同步积木需要真实 RGB-D 深度；USB 平面模式不提供可靠三维位置')
    diagnostics = diagnostics if diagnostics is not None else {}
    snapshot = batch.snapshot
    pose = snapshot.state
    translation = np.asarray(pose['trans'], dtype=float)
    quat = np.asarray(pose['orient'], dtype=float)
    if translation.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[translation, quat]).all():
        raise ValueError('缺少与图像对应的有效末端位姿')
    cloud = camera.create_point_cloud(snapshot.depth, end_pose=[translation, quat])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quat).as_matrix()
    transform[:3, 3] = translation
    transform = transform @ np.asarray(cam2end)
    hsv = cv2.cvtColor(snapshot.color, cv2.COLOR_BGR2HSV)
    objects = {}
    for color in ('blue', 'green'):
        candidates = [d for d in batch.detections if d.color == color]
        if len(candidates) != 1:
            diagnostics[color] = '未检测到该颜色' if not candidates else '同色目标不唯一'
            continue
        if not candidates[0].observed:
            diagnostics[color] = '检测暂时丢失，等待重新确认'
            continue
        d = candidates[0]
        x1, y1, x2, y2 = d.bbox
        height, width = snapshot.depth.shape
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        bounds = detection_settings['colors'][color]
        mask = cv2.inRange(hsv, np.array(bounds['lower'], np.uint8),
                           np.array(bounds['upper'], np.uint8)).astype(bool)
        inside = np.zeros_like(mask)
        inside[y1:y2, x1:x2] = True
        mask &= inside & np.isfinite(snapshot.depth) & (snapshot.depth > 0)
        points = cloud[mask]
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] < 2.)]
        if len(points) < 20:
            diagnostics[color] = f'有效彩色深度点不足（{len(points)}）'
            continue
        # Reject isolated depth returns before estimating the top surface.
        distance = np.median(points[:, 2])
        deviation = np.median(np.abs(points[:, 2] - distance))
        points = points[np.abs(points[:, 2] - distance) <= max(.015, 4 * deviation)]
        if len(points) < 20:
            diagnostics[color] = '深度离群点过多'
            continue
        points = points @ transform[:3, :3].T + transform[:3, 3]
        # Noisy oblique views need not expose the entire calibrated top face.
        top = np.percentile(points[:, 2], 85)
        surface = points[np.abs(points[:, 2] - top) < .008]
        size = np.asarray(settings['block_size'])
        complete = False
        if len(surface) >= 30:
            low, high = np.percentile(surface[:, :2], [2, 98], axis=0)
            complete = bool(np.all(high - low >= size[:2] * .55)
                            and np.all(high - low <= size[:2] * 1.8))
        if complete:
            center = np.r_[(low + high) / 2, np.median(surface[:, 2]) - size[2] / 2]
            quality = 'top_face'
            diagnostics[color] = '顶面位置已同步'
        else:
            # Visualization only: visible surfaces can bias XY. Never use this
            # approximation as a robot grasp target or claim exact alignment.
            center = np.r_[np.median(points[:, :2], axis=0), top - size[2] / 2]
            quality = 'estimated'
            diagnostics[color] = '顶面不完整：显示可见点云估计位置（半透明）'
        if not np.isfinite(center).all() or np.linalg.norm(center) > 2.:
            diagnostics[color] = '基座坐标异常，请检查手眼标定和深度单位'
            continue
        objects[color] = {'position': center.tolist(), 'timestamp': snapshot.timestamp,
                          'confidence': float(d.confidence), 'quality': quality}
    return objects


def measure_packet(*args):
    diagnostics = {}
    objects = measured_blocks(*args, diagnostics=diagnostics)
    return objects, diagnostics




class MirrorPublisher:
    """Latest-only atomic file transport; a slow render process cannot block control."""
    def __init__(self, settings):
        self.settings = settings
        self.directory = tempfile.TemporaryDirectory(prefix='grasp-live-mirror-')
        self.path = Path(self.directory.name) / 'snapshot.json'
        self.state = None
        self.objects = {}
        self.diagnostics = {}
        self.status = '等待真实相机与机械臂反馈'
        self.command = ''
        self.sequence = 0

    def observe(self, snapshot):
        joints = np.asarray(snapshot.state.get('joints'), dtype=float)
        eef = np.asarray(snapshot.state.get('eef'), dtype=float)
        if joints.shape != (6,) or eef.shape != (1,) or not np.isfinite(np.r_[joints, eef]).all():
            self.state = None
            self.status = '真机反馈无效；停止更新'
            return
        self.state = {'joints': joints.tolist(), 'opening': float(eef[0]),
                      'timestamp': snapshot.timestamp}

    def publish(self):
        self.sequence += 1
        payload = {'sequence': self.sequence, 'state': self.state, 'objects': self.objects,
                   'status': self.status, 'command': self.command, 'diagnostics': self.diagnostics}
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, self.path)

    def close(self):
        self.directory.cleanup()


def start_mirror(interface):
    """Called explicitly by --with-sim. All expensive rendering is in a child process."""
    from concurrent.futures import ThreadPoolExecutor
    from PyQt6.QtCore import QProcess, QTimer

    class LiveMirror:
        def __init__(self):
            self.settings = load_settings()
            self.publisher = MirrorPublisher(self.settings)
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mirror-points')
            self.future = None
            self.last_sequence = -1
            self.process = QProcess(interface)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
            self.process.errorOccurred.connect(
                lambda error: interface.log('仿真镜像进程启动/运行异常：' + self.process.errorString()))
            self.process.finished.connect(
                lambda code, status: interface.log(f'仿真镜像窗口已退出（{code}）；真机继续独立运行'))
            self.process.start(os.sys.executable, ['-u', str(Path(__file__).with_name('live_mirror_window.py')),
                                                  '--snapshot', str(self.publisher.path)])
            self.timer = QTimer(interface)
            self.timer.timeout.connect(self.poll)
            self.timer.start(max(20, round(1000 / self.settings['publish_hz'])))

        def observe(self, snapshot):
            self.publisher.observe(snapshot)

        def detect(self, batch):
            if self.future is not None or batch.snapshot.sequence <= self.last_sequence:
                return
            self.last_sequence = batch.snapshot.sequence
            age = time.monotonic() - batch.snapshot.timestamp
            if age > self.settings['object_timeout']:
                self.publisher.status = f'检测帧已过期（{age:.1f} 秒），等待新帧'
                return
            self.future = self.pool.submit(measure_packet, batch, interface.realsense,
                                           np.array(interface.airbot_grasp.cam2end).copy(),
                                           interface.airbot_yolo.detection_config, self.settings)

        def poll(self):
            try:
                if self.future is not None and self.future.done():
                    future, self.future = self.future, None
                    try:
                        objects, diagnostics = future.result()
                        self.publisher.objects = objects
                        self.publisher.diagnostics = diagnostics
                        self.publisher.status = '积木来自真实 RGB-D；半透明表示不完整顶面的估计位置'
                    except Exception as exc:
                        self.publisher.objects = {}
                        self.publisher.diagnostics = {}
                        self.publisher.status = '位置同步不可用：' + str(exc)
                self.publisher.publish()
            except Exception as exc:
                self.timer.stop()
                interface.log('仿真镜像已停用：' + str(exc))

        def close(self):
            self.timer.stop()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1000)
            self.publisher.close()

    return LiveMirror()
