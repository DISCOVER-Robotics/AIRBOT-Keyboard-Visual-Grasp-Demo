"""Separate MuJoCo window replaying measured real states, never integrating physics."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from live_mirror import load_settings


def apply_snapshot(sim, packet, settings, now=None):
    now = time.monotonic() if now is None else now
    state = packet.get('state')
    valid = state is not None and 0 <= now - state['timestamp'] <= settings['state_timeout']
    transform = np.asarray(settings['world_from_base'])
    if valid:
        joints = np.asarray(state['joints']) * settings['joint_signs'] + settings['joint_offsets']
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('无效关节反馈')
        sim.data.qpos[:6] = joints
        opening = float(state['opening'])
        if not np.isfinite(opening) or not 0 <= opening <= .08:
            raise ValueError('夹爪反馈超出模型范围')
        sim.data.joint('endleft').qpos[0] = opening / 2
        sim.data.joint('endright').qpos[0] = -opening / 2
    visible = []
    for color in sim.colors:
        obj = packet.get('objects', {}).get(color)
        fresh = valid and obj is not None and 0 <= now - obj['timestamp'] <= settings['object_timeout']
        sim.model.geom(f'{color}_cube').rgba[3] = (
            .55 if fresh and obj.get('quality') == 'estimated' else 1. if fresh else 0.)
        if fresh:
            position = np.asarray(obj['position'], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('无效积木位置')
            sim.data.joint(f'{color}_free').qpos[:3] = transform[:3, :3] @ position + transform[:3, 3]
            sim.data.joint(f'{color}_free').qpos[3:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
            visible.append(color)
    sim.data.qvel[:] = 0
    sim.mujoco.mj_forward(sim.model, sim.data)
    return valid, visible


def object_status(packet, settings, now=None):
    now = time.monotonic() if now is None else now
    lines = []
    for color, label in [('blue', '蓝色'), ('green', '绿色')]:
        obj = packet.get('objects', {}).get(color)
        if obj is None:
            detail = packet.get('diagnostics', {}).get(color, '等待位置测量')
        else:
            age = now - obj['timestamp']
            point = obj['position']
            detail = f"基座坐标 [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}] m；{age:.1f} 秒前"
            if age > settings['object_timeout'] or age < 0:
                detail += '（已过期，隐藏）'
            elif obj.get('quality') == 'estimated':
                detail += '（估计位置，半透明）'
            if point[2] < 0:
                detail += '；位于基座平面下方，请核对桌面高度/手眼标定'
        lines.append(label + '：' + detail)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    from discoverse_sim import DiscoverseSimulation
    application = QApplication([])
    settings = load_settings()
    sim = DiscoverseSimulation()
    transform = np.asarray(settings['world_from_base'])
    sim.model.body('arm_base').pos[:] = transform[:3, 3]
    sim.model.body('arm_base').quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    for color in sim.colors:
        sim.model.geom(f'{color}_cube').size[:] = np.asarray(settings['block_size']) / 2
        sim.model.geom(f'{color}_cube').rgba[3] = 0.
    sim.model.site('place_zone').rgba[3] = 0.
    window = QWidget()
    window.setWindowTitle('真机同步仿真 · 只读反馈镜像')
    layout = QVBoxLayout(window)
    label = QLabel()
    label.setFixedSize(640, 480)
    layout.addWidget(label)
    status = QLabel('等待真机反馈，尚未同步')
    status.setWordWrap(True)
    status.setMaximumWidth(640)
    layout.addWidget(status)
    layout.addWidget(QLabel('实色：顶面位置；半透明：不完整点云估计。数据过期时隐藏。\n语音在真机主窗口输入；此窗口不发送运动指令。'))
    latest = {}

    def refresh():
        nonlocal latest
        try:
            source = Path(args.snapshot)
            if source.exists():
                latest = json.loads(source.read_text(encoding='utf-8'))
            valid, visible = apply_snapshot(sim, latest, settings)
            if valid:
                rgb = sim.render('overview')
                image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
                label.setPixmap(QPixmap.fromImage(image))
            else:
                label.clear()
                label.setText('真机反馈未就绪或已过期，暂停显示')
            status.setText(('同步中' if valid else '未同步') + '；可见积木：' + ', '.join(visible)
                           + '\n' + latest.get('status', '') + '\n' + object_status(latest, settings)
                           + '\n' + latest.get('command', ''))
        except Exception as exc:
            label.clear()
            status.setText('同步失败：' + str(exc))

    timer = QTimer()
    timer.timeout.connect(refresh)
    timer.start(round(1000 / settings['publish_hz']))
    window.show()
    try:
        return application.exec()
    finally:
        timer.stop()
        sim.close()


if __name__ == '__main__':
    raise SystemExit(main())
