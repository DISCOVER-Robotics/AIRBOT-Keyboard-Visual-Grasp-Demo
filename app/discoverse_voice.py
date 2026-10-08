"""Voice/text UI and headless runner for the DISCOVERSE simulation."""
from __future__ import annotations

import argparse
import os
import sys

from voice_commands import execute_command


class SimulationCommands:
    """Callback contract used by the keyboard command panel."""
    def voice_is_busy(self):
        return self.sim.busy

    def voice_has_selection(self):
        return self.sim.selected is not None

    def capture_frame(self):
        self.sim.capture()

    def predict_grasp_pose(self):
        self.sim.predict()

    def predict_and_grasp(self):
        if self.sim.selected is None:
            raise ValueError('请先选择积木。')
        self.sim.grasp(self.sim.selected)

    def grasp_color(self, color):
        self.sim.grasp(color)

    def set_gripper_open(self, opened):
        self.sim.grip(opened)

    def move_to_observe(self):
        self.sim.go_observe()


def make_window(sim):
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                                QPushButton, QComboBox, QTextEdit, QLineEdit)

    class SimulationWindow(QWidget, SimulationCommands):
        def __init__(self):
            super().__init__()
            self.sim = sim
            self.setWindowTitle('DISCOVERSE · AIRBOT 键盘抓取仿真')
            layout = QHBoxLayout(self)
            left, right = QVBoxLayout(), QVBoxLayout()
            layout.addLayout(left)
            layout.addLayout(right)
            right.setContentsMargins(8, 0, 0, 0)
            views = QVBoxLayout()
            left.addLayout(views)
            overview_layout, camera_layout = QVBoxLayout(), QVBoxLayout()
            views.addLayout(overview_layout)
            views.addLayout(camera_layout)
            overview_layout.addWidget(QLabel('第三视角 · 机械臂与工作区'))
            camera_layout.addWidget(QLabel('第一视角 · 机械臂末端相机'))
            self.overview_image = QLabel()
            self.image = QLabel()
            for view_layout, label in ((overview_layout, self.overview_image),
                                       (camera_layout, self.image)):
                label.setFixedSize(480, 360)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet('background: #181818;')
                view_layout.addWidget(label)
            left.addStretch()
            self.vision_status = QLabel('')
            self.vision_status.setWordWrap(True)
            left.addWidget(self.vision_status)
            self.state = QLabel('')
            self.state.setWordWrap(True)
            left.addWidget(self.state)
            right.addWidget(QLabel('蓝色 / 绿色积木 → 橙色放置区\n每次完成后可重置场景再试另一颜色。'))
            self.targets = QComboBox()
            self.targets.addItem('手动选择目标', None)
            self.targets.addItem('蓝色积木', 'blue')
            self.targets.addItem('绿色积木', 'green')
            self.targets.currentIndexChanged.connect(self.choose)
            right.addWidget(self.targets)
            self.reset_button = QPushButton('重置场景')
            self.reset_button.clicked.connect(self.reset_scene)
            right.addWidget(self.reset_button)
            self.logs = QTextEdit()
            self.logs.setReadOnly(True)
            self.logs.document().setMaximumBlockCount(300)
            if sim.vision:
                self.log('YOLO 权重：' + str(sim.vision.detector.checkpoint))
            else:
                self.log('物理调试模式：使用场景真值定位，未启用 YOLO。')
            self.text = QLineEdit()
            self.text.setPlaceholderText('输入：抓取蓝色积木')
            self.execute_button = QPushButton('执行文字指令')
            self.execute_button.clicked.connect(self.execute_text)
            self.text.returnPressed.connect(self.execute_text)
            right.addWidget(self.text)
            right.addWidget(self.execute_button)
            right.addWidget(QLabel('最近一次分割 / 位姿计算快照'))
            self.preparation_image = QLabel()
            self.preparation_image.setFixedSize(240, 180)
            self.preparation_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right.addWidget(self.preparation_image)
            self.prepared_text = QLabel()
            self.prepared_text.setWordWrap(True)
            self.prepared_text.setMaximumWidth(300)
            right.addWidget(self.prepared_text)
            right.addWidget(self.logs)
            self.last_message = ''
            self.event_count = 0
            self.frames = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_simulation)
            self.timer.start(10)
            self.update_simulation()

        def log(self, message):
            self.logs.append(message)

        def choose(self):
            color = self.targets.currentData()
            if color:
                try:
                    self.sim.select(color)
                except ValueError as exc:
                    self.log(str(exc))

        def reset_scene(self):
            try:
                self.sim.reset()
                self.targets.setCurrentIndex(0)
                self.sim.message = '场景已重置。'
                self.event_count = 0
            except ValueError as exc:
                self.log(str(exc))

        def execute_text(self):
            try:
                execute_command(self, self.text.text())
            except Exception as exc:
                self.log(str(exc))

        def update_simulation(self):
            try:
                self.sim.tick()
                if self.event_count > len(self.sim.events):
                    self.event_count = 0
                for message in self.sim.events[self.event_count:]:
                    self.log(message)
                self.event_count = len(self.sim.events)
                self.frames += 1
                if self.sim.vision:
                    self.sim.vision.poll()
                    self.vision_status.setText(
                        'MobileSAM 分割和位姿计算中…' if self.sim.preparation is not None else
                        '运动中，完成后重新检测。' if self.sim.busy else
                        '末端相机 · ' + self.sim.vision.status)
                if self.frames % 3 == 0:
                    overview = self.sim.render('overview')
                    camera = self.sim.vision.overlay() if self.sim.vision and not self.sim.busy else None
                    if camera is None:
                        camera = self.sim.render('eye_arm')
                    for label, rgb in ((self.overview_image, overview), (self.image, camera)):
                        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                       QImage.Format.Format_RGB888).copy()
                        label.setPixmap(QPixmap.fromImage(image).scaled(
                            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation))
                if self.sim.prepared_preview is not None:
                    rgb_preview = self.sim.prepared_preview[:, :, ::-1].copy()
                    preview_image = QImage(rgb_preview.data, rgb_preview.shape[1], rgb_preview.shape[0],
                                           rgb_preview.strides[0], QImage.Format.Format_RGB888).copy()
                    self.preparation_image.setPixmap(QPixmap.fromImage(preview_image).scaled(
                        self.preparation_image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    info = self.sim.prepared_info
                    self.prepared_text.setText(
                        f"位置：{[round(v, 3) for v in info['trans']]} m\n"
                        f"宽度：{info['o_width'] * 1000:.1f} mm；角度：{info['o_angle']:.1f}°")
                else:
                    self.preparation_image.clear()
                    self.prepared_text.clear()
                self.state.setText(('执行中：' if self.sim.busy else '空闲：') + self.sim.message)
                if self.last_message != self.sim.message:
                    if not self.sim.events or self.sim.events[-1] != self.sim.message:
                        self.log(self.sim.message)
                    self.last_message = self.sim.message
                self.targets.setEnabled(not self.sim.busy)
                self.reset_button.setEnabled(not self.sim.busy)
                if self.sim.selected is None and self.targets.currentIndex() != 0:
                    self.targets.setCurrentIndex(0)
            except Exception as exc:
                self.timer.stop()
                self.log('仿真停止：' + str(exc))
                self.targets.setEnabled(False)
                self.reset_button.setEnabled(False)
                self.execute_button.setEnabled(False)

        def closeEvent(self, event):
            self.timer.stop()
            self.sim.close()
            event.accept()

    return SimulationWindow()


def main():
    parser = argparse.ArgumentParser(description='DISCOVERSE 视觉抓取仿真，不连接真实机械臂')
    parser.add_argument('--headless', action='store_true', help='无窗口运行，不加载语音模型')
    parser.add_argument('--text', help='启动后执行一条白名单口令')
    parser.add_argument('--yolo-checkpoint', help='自定义 YOLO .pt 权重，默认沿用现有配置')
    parser.add_argument('--ground-truth', action='store_true', help='仅物理调试：显式禁用 YOLO，使用场景真值定位')
    parser.add_argument('--device', help='麦克风编号或名称，默认系统输入设备')
    args = parser.parse_args()
    if args.headless and not args.text:
        parser.error('--headless 需要 --text')
    if args.ground_truth and args.yolo_checkpoint:
        parser.error('--ground-truth 不能与 --yolo-checkpoint 同时使用')
    if args.headless and not args.ground_truth:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    if args.device:
        os.environ['GRASP_VOICE_DEVICE'] = args.device
    # DISCOVERSE imports matplotlib; it does not own our Qt application.
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from discoverse_sim import DiscoverseSimulation
    sim = None
    try:
        if args.headless:
            sim = DiscoverseSimulation()
            if not args.ground_truth:
                from discoverse_vision import SimulationVision
                sim.vision = SimulationVision(sim, args.yolo_checkpoint)
                sim.vision.warmup()
            target = SimulationCommands()
            target.sim = sim
            execute_command(target, args.text)
            sim.run_until_idle()
            print(sim.message)
            return 0
        from PyQt6.QtWidgets import QApplication
        application = QApplication(sys.argv[:1])
        sim = DiscoverseSimulation()
        if not args.ground_truth:
            from discoverse_vision import SimulationVision
            sim.vision = SimulationVision(sim, args.yolo_checkpoint)
        window = make_window(sim)
        window.show()
        if args.text:
            # A startup grasp must wait for confirmed detections, not bypass YOLO.
            if sim.vision:
                sim.vision.warmup()
            execute_command(window, args.text)
        return application.exec()
    except Exception as exc:
        print(f'仿真运行失败：{exc}\n依赖安装：./install_sim.sh', file=sys.stderr)
        return 1
    finally:
        if sim is not None:
            sim.close()


if __name__ == '__main__':
    raise SystemExit(main())
