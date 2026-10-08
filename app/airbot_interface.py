# -*- coding: utf-8 -*-
import os
import sys
import cv2

# The GUI uses PyQt6. Linux OpenCV wheels inject their bundled Qt5 plugin
# directory during import, which prevents PyQt6 from loading its own xcb plugin.
for qt_env_name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    qt_env_value = os.environ.get(qt_env_name, "")
    if "/cv2/qt/" in qt_env_value:
        os.environ.pop(qt_env_name, None)

import yaml
import time
from dataclasses import dataclass
from threading import Condition, Lock, RLock
import inspect
import traceback
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
                            QGroupBox, QGridLayout, QComboBox, QSplitter,
                            QTabWidget, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QFont, QFontDatabase, QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import torch
from functools import wraps
from voice_panel import VoiceCommandPanel
from grasp_preparation import prepare_color_grasp
from grasp_clearance import require_open_gripper

try:
    from airbot_arm import (AirbotArm, AirbotMonitor, GripperContactTimeout,
                            SpeedProfile, describe_eef_status, ControlLeaseLost, DescentStartMismatch)
    from airbot_camera import (RealsenseCamera, create_camera, CameraConnectionError,
                               check_camera_connection)
    from airbot_segment import AirbotSegment, SegmentMode
    from airbot_grasp_simple import SimpleGrasp
    from airbot_yolo import (AirbotYolo, COLOR_LABELS, DetectionStabilizer,
                             draw_detections, resolve_unique_target)
except ImportError as e:
    print(f"Failed to import airbot modules, {e}")
    exit()
    
# temporary delay caused by moveit early return and gripper no blocking
arm_delay = 0.3 # second
gripper_delay = 0.3

def safe_func():
    def decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())[1:]
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                param_cnt = len(param_names)
                if param_cnt == 0:
                    return func(self)
                else:
                    return func(self, *args, **kwargs)
            except Exception as e:
                detail = traceback.format_exc()
                error_info = f"{func.__name__} with params: [{param_names}] error"
                self.log(f"{error_info}: {e}\n{detail}")
        return wrapper
    return decorator

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int, int)  # x, y, button
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        button = 0
        if event.button() == Qt.MouseButton.LeftButton:
            button = 0
        elif event.button() == Qt.MouseButton.RightButton:
            button = 1
        elif event.button() == Qt.MouseButton.MiddleButton:
            button = 2
            
        pos = event.position().toPoint()  # 转为 QPoint（整数）
        self.clicked.emit(pos.x(), pos.y(), button)


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    timestamp: float
    color: np.ndarray
    depth: np.ndarray
    depth_map: np.ndarray
    state: dict


@dataclass(frozen=True)
class DetectionBatch:
    snapshot: FrameSnapshot
    detections: tuple
    completed_at: float


class DetectionThread(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, detector, inference_hz=10, stabilizer=None):
        super().__init__()
        self.detector = detector
        self.stabilizer = stabilizer
        self.minimum_interval = 1.0 / max(0.1, float(inference_hz))
        self._condition = Condition()
        self._pending = None
        self._stopping = False

    def submit(self, snapshot):
        with self._condition:
            self._pending = snapshot
            self._condition.notify()

    def run(self):
        last_started = 0.0
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                snapshot, self._pending = self._pending, None
            remaining = self.minimum_interval - (time.monotonic() - last_started)
            if remaining > 0:
                self.msleep(max(1, int(remaining * 1000)))
                with self._condition:
                    if self._pending is not None:
                        snapshot, self._pending = self._pending, None
            try:
                last_started = time.monotonic()
                detections = self.detector.detect_candidates(snapshot.color)
                if self.stabilizer is not None:
                    detections = self.stabilizer.update(detections)
                detections = tuple(detections)
                self.result_ready.emit(DetectionBatch(
                    snapshot=snapshot, detections=detections,
                    completed_at=time.monotonic()))
            except Exception as exc:
                self.failed.emit(str(exc))

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self.wait(3000)


class ColorGraspPreparationThread(QThread):
    prepared = pyqtSignal(dict, np.ndarray)
    failed = pyqtSignal(str)

    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.task = None

    def set_task(self, batch, detection, *, execute=True):
        if self.isRunning():
            raise RuntimeError("目标位姿正在计算，请稍候")
        self.task = (batch, detection, execute)

    def run(self):
        batch, detection, execute = self.task
        snapshot = batch.snapshot
        try:
            if time.monotonic() - snapshot.timestamp > self.interface.detection_freshness:
                raise ValueError("目标画面已过期，请等待新的检测结果")
            info, preview = prepare_color_grasp(
                snapshot, detection, batch.detections,
                self.interface.airbot_segment, self.interface.realsense,
                self.interface.airbot_grasp, self.interface.detection_config,
                execute_motion=execute)
            info['execute_motion'] = execute
            self.prepared.emit(info, preview)
        except Exception as exc:
            self.failed.emit(str(exc))
        
class ObserveThread(QThread):
    update_frame = pyqtSignal(object)
    
    def __init__(self, interface, robot_port, camera):
        super().__init__()
        self.interface = interface
        self.robot = AirbotMonitor(host=interface.robot_host, port=robot_port)
        self.camera = camera
        self.sequence = 0
        
    def run(self):
        while not QThread.currentThread().isInterruptionRequested():
            frame_timestamp = time.monotonic()
            color_frame = None
            depth_frame = None
            depth_map = None
            try:
                color_frame, depth_frame, depth_map = self.camera.get_frame(frame_type=["bgr", "depth", "depth_map"], align=True)
            except Exception as e:
                if "Device disconnected" in str(e):
                    print("Reconnecting to the camera...")
                    del self.interface.realsense
                    self.interface.realsense = create_camera()
                    self.camera = self.interface.realsense
           
            pose = [[],[]]
            joints = []
            eef = []
            eeff = []
            state = {"trans": [], "orient": [], "joints": [], "eef": [], "eeff": []}
            
            try:
                state = self.robot.read_state()
                pose = [state["trans"], state["orient"]]
                joints = state["joints"]
                eef = state["eef"]
                eeff = state["eeff"]
            except Exception as e:
                print(f"读取机械臂状态失败，尝试重连 arm-sdk 5.2.2: {e}")
                try:
                    self.robot.reconnect()
                except Exception as reconnect_error:
                    print(f"重连机械臂状态客户端失败: {reconnect_error}")
            
            if color_frame is None or depth_frame is None or depth_map is None:
                color_frame = np.zeros(
                    (self.camera.HEIGHT, self.camera.WIDTH, 3), dtype=np.uint8)
                depth_frame = np.zeros(
                    (self.camera.HEIGHT, self.camera.WIDTH), dtype=np.uint16)
                depth_map = np.zeros(
                    (self.camera.HEIGHT, self.camera.WIDTH), dtype=np.uint8)
                
            with self.interface.observe_frame_lock:
                self.interface.color_frame = color_frame.copy()
                self.interface.depth_frame = depth_frame.copy()
                self.interface.depth_map = depth_map.copy()
                self.interface.current_state["trans"] = pose[0]
                self.interface.current_state["orient"] = pose[1]
                self.interface.current_state["joints"] = joints
                self.interface.current_state["eef"] = eef
                self.interface.current_state["eeff"] = eeff
            self.sequence += 1
            self.update_frame.emit(FrameSnapshot(
                sequence=self.sequence, timestamp=frame_timestamp,
                color=color_frame.copy(), depth=depth_frame.copy(),
                depth_map=depth_map.copy(),
                state={"trans": pose[0], "orient": pose[1], "joints": joints,
                       "eef": eef, "eeff": eeff}))
    
    def stop(self):
        self.requestInterruption()
        self.wait()
        self.robot.close()
        self.camera.deinit()

class RobotGraspThread(QThread):
    log = pyqtSignal(str)
    capture_signal = pyqtSignal()
    
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.robot = self.interface.robot
        self.running_lock = Lock()
        self.predicted_info = None
    
    def set_predicted_info(self, predicted_info):
        with self.running_lock:
            self.predicted_info = predicted_info

    def recover_to_observe(self, zone, retreat_pose, object_held):
        """Best-effort recovery after motion has left the observation pose."""
        errors = []
        release_object = ((zone == "grasp" and not object_held)
                          or (zone == "place" and object_held))
        if release_object:
            try:
                self.log.emit("失败恢复: 在当前位置打开夹爪")
                self.robot.move_gripper(self.interface.gripper_width)
                self.interface.gripper_opened = True
            except Exception as exc:
                errors.append(f"打开夹爪失败：{exc}")

        if retreat_pose is not None:
            try:
                self.log.emit(f"失败恢复: 直线上抬到安全位 {retreat_pose[0]}")
                self.robot.move_end_pose_linear(retreat_pose)
            except Exception as exc:
                errors.append(f"上抬到安全位失败：{exc}")

        try:
            self.log.emit("失败恢复: 返回初始观察位")
            self.robot.move_end_pose(self.interface.observe_pose)
            if object_held and not release_object:
                self.log.emit("已返回初始观察位；夹爪仍保持夹持，请人工处理物体")
            else:
                self.log.emit("抓取失败后已返回初始观察位")
        except Exception as exc:
            errors.append(f"返回观察位失败：{exc}")

        if errors:
            self.log.emit("失败恢复未完全完成：" + "；".join(errors))
        return errors
    
    def run(self):
        with self.running_lock, self.interface.robot_operation_lock:
            trans = self.predicted_info["trans"]
            orient = self.predicted_info["orient"]
            o_height = self.predicted_info["o_height"]
            o_width = self.predicted_info["o_width"]
            stage = "初始化抓取任务"
            arm_motion_started = False
            recovery_zone = None
            recovery_retreat = None
            object_held = False
            try:
                os.makedirs("runtime", exist_ok=True)
                with open("runtime/waypoints.txt", "a") as f:
                    pre_grasp = [[trans[0], trans[1], trans[2] + 0.1], orient]
                    grasp = [trans, orient]

                    if not np.isfinite(o_width) or o_width <= 0:
                        raise ValueError(f"视觉估算的积木宽度无效：{o_width}")
                    open_width = min(
                        max(o_width + self.interface.gripper_open_clearance,
                            o_width * 1.2,
                            getattr(self.interface, "gripper_min_open_width", 0.0)),
                        self.interface.gripper_width,
                    )
                    grasp_width = max(
                        0.0,
                        min(o_width - self.interface.gripper_grasp_compression,
                            self.interface.gripper_width),
                    )

                    stage = "夹爪健康检查"
                    old_error = self.robot.clear_gripper_error(
                        timeout_ms=500,
                        poll_interval=self.interface.gripper_feedback_poll_interval,
                    )
                    if old_error:
                        self.log.emit(
                            f"抓取前已清除夹爪历史错误码 {old_error} "
                            f"(0x{old_error:02X})")

                    stage = "打开夹爪"
                    self.log.emit("步骤1: 打开夹爪")
                    self.robot.move_gripper(open_width)
                    time.sleep(0.3)
                    self.interface.gripper_opened = False
                    opening_feedback = self.robot.get_gripper_diagnostics()
                    require_open_gripper(opening_feedback, open_width)
                    self.interface.gripper_opened = True
                    self.log.emit(
                        f"抓取诊断：估算宽度 {o_width*1000:.1f} mm；"
                        f"张开指令 {open_width*1000:.1f} mm；"
                        f"反馈开度 {opening_feedback['position']*1000:.1f} mm")

                    stage = "移动到预抓取位"
                    self.log.emit(f"步骤2: 移动到预抓取位 {pre_grasp[0]}")
                    f.write(str(pre_grasp) + ",\n")
                    arm_motion_started = True
                    recovery_zone = "grasp"
                    recovery_retreat = pre_grasp
                    self.robot.move_end_pose(pre_grasp)
                    time.sleep(arm_delay)

                    measured_pre_grasp = self.robot.get_end_pose()
                    self.log.emit(
                        f"预抓取诊断：目标 {pre_grasp}；SDK 反馈 {measured_pre_grasp}；"
                        "SDK 反馈不能替代实际指尖中心测量")
                    stage = "下降前夹爪开度复核"
                    require_open_gripper(self.robot.get_gripper_diagnostics(), open_width)
                    stage = "直线下降到抓取位"
                    self.log.emit(f"步骤3: 直线下降到抓取位 {grasp[0]}")
                    f.write(str(grasp) + ",\n")
                    self.robot.move_end_pose_linear(
                        grasp, expected_start=pre_grasp, speed_limit=.10)
                    time.sleep(arm_delay)

                    stage = "关闭夹爪"
                    self.log.emit(
                        f"步骤4: 夹持积木，估算宽度 {o_width:.4f} m，"
                        f"目标开度 {grasp_width:.4f} m")
                    target_widths = []
                    for attempt in range(
                            self.interface.gripper_grasp_retry_count + 1):
                        target = max(
                            self.interface.gripper_grasp_min_width,
                            grasp_width - (
                                attempt
                                * self.interface.gripper_grasp_retry_step),
                        )
                        if not target_widths or target < target_widths[-1] - 1e-6:
                            target_widths.append(target)

                    gripper = None
                    for attempt, target_width in enumerate(target_widths, 1):
                        if attempt > 1:
                            self.log.emit(
                                f"未检测到接触，第 {attempt - 1} 次收紧夹爪："
                                f"目标开度 {target_width:.4f} m")
                        self.robot.move_gripper(
                            target_width, blocking=False,
                            timeout_ms=self.interface.gripper_grasp_timeout_ms,
                            effort=self.interface.gripper_grasp_effort)
                        try:
                            gripper = self.robot.wait_for_gripper_contact(
                                target_width,
                                timeout_ms=(
                                    self.interface.gripper_grasp_timeout_ms),
                                effort_threshold=(
                                    self.interface.gripper_contact_effort),
                                strong_effort_threshold=(
                                    self.interface.gripper_contact_strong_effort),
                                velocity_threshold=(
                                    self.interface.gripper_contact_velocity),
                                position_tolerance=(
                                    self.interface
                                    .gripper_contact_position_tolerance),
                                confirm_samples=(
                                    self.interface
                                    .gripper_contact_confirm_samples),
                                poll_interval=(
                                    self.interface
                                    .gripper_feedback_poll_interval),
                            )
                            break
                        except GripperContactTimeout:
                            if attempt == len(target_widths):
                                raise
                    self.log.emit(
                        "已识别夹住物体："
                        f"判定方式 {gripper['contact_mode']}，"
                        f"实际开度 {gripper['position']:.4f} m，"
                        f"速度 {gripper['velocity']:.4f}，"
                        f"作用力 {gripper['effort']:.2f}，"
                        f"电机温度 {gripper['motor_temperature']:.1f} C，"
                        f"末端状态 {gripper['error_id']}"
                        f"（{describe_eef_status(gripper['error_id'])}）")
                    self.interface.gripper_opened = False
                    object_held = True

                    stage = "直线抬升物体"
                    self.log.emit("步骤5: 直线抬升物体")
                    self.robot.move_end_pose_linear(pre_grasp)
                    time.sleep(arm_delay)
                    recovery_zone = None
                    recovery_retreat = None

                    stage = "移动到预放置位"
                    self.log.emit(
                        f"步骤6: 移动到公共放置位上方 {self.interface.pre_place_pose[0]}")
                    f.write(str(self.interface.pre_place_pose) + ",\n")
                    self.robot.move_end_pose(self.interface.pre_place_pose)
                    time.sleep(arm_delay)
                    recovery_zone = "place"
                    recovery_retreat = self.interface.pre_place_pose

                    stage = "直线下降到公共放置位"
                    self.log.emit(
                        f"步骤7: 直线下降到公共放置位 {self.interface.place_pose[0]}")
                    f.write(str(self.interface.place_pose) + ",\n")
                    self.robot.move_end_pose_linear(self.interface.place_pose)
                    time.sleep(arm_delay)

                    stage = "松开夹爪"
                    self.log.emit("步骤8: 松开夹爪释放物体")
                    self.robot.move_gripper(self.interface.gripper_width)
                    time.sleep(gripper_delay)
                    self.interface.gripper_opened = True
                    object_held = False

                    stage = "离开放置位置"
                    self.log.emit("步骤9: 直线抬升离开放置位置")
                    self.robot.move_end_pose_linear(self.interface.pre_place_pose)
                    time.sleep(arm_delay)
                    recovery_zone = None
                    recovery_retreat = None

                    stage = "返回观察位置"
                    self.log.emit("步骤10: 返回观察位置")
                    f.write(str(self.interface.observe_pose) + ",\n")
                    self.robot.move_end_pose(self.interface.observe_pose)
                    time.sleep(arm_delay)

                    self.log.emit("抓取执行完成")
                    self.capture_signal.emit()

            except Exception as e:
                detail = traceback.format_exc()
                print(f"Failed to perform grasp: {e}\n{detail}")
                self.interface.selected_detection_bbox = None
                self.log.emit(
                    f"抓取运动已停止（阶段：{stage}）：{e}；未继续执行后续路径。"
                    f"目标位姿 position={list(trans)}, orientation={list(orient)}"
                )
                if arm_motion_started and not isinstance(e, (ControlLeaseLost, DescentStartMismatch)):
                    self.recover_to_observe(
                        recovery_zone, recovery_retreat, object_held)

class AutoGraspThread(QThread):
    finish_signal = pyqtSignal()
    capture_signal = pyqtSignal()
    update_info = pyqtSignal(dict)
    update_image = pyqtSignal(np.ndarray)
    log = pyqtSignal(str)
    
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.grasping_object = None
        self.grasping_mask = None
        self.airbot_segment = self.interface.airbot_segment
        self.airbot_yolo = self.interface.airbot_yolo
        self.realsense = self.interface.realsense
        self.airbot_grasp = self.interface.airbot_grasp
        self.finish_cnt = 0

    def run(self):
        self.finish_cnt = 0
        with self.interface.predict_lock:
            self.interface.predicted_info = None
        self.log.emit("开始自动抓取")
        # 捕获图像     
        self.capture_signal.emit()
        time.sleep(0.1)
        while not QThread.currentThread().isInterruptionRequested():
            try:                                              
                with self.interface.captured_frame_lock:
                    captured_color = self.interface.captured_color
                    captured_depth = self.interface.captured_depth
                    captured_pose = self.interface.captured_pose
                
                
                # 获取边界框和标签
                object_bbox, object_label, object_conf = self.airbot_yolo.get_max_conf_bbox_and_label(captured_color, label_filter=[self.grasping_object, "gripper"])
                if object_bbox is None or len(object_bbox) == 0:
                    self.log.emit("未检测到物体，跳过当前循环")
                    self.capture_signal.emit()
                    if self.interface.robot_grasp_thread.isRunning():
                        self.interface.robot_grasp_thread.requestInterruption()
                        self.interface.robot_grasp_thread.wait()
                    time.sleep(1)
                    self.grasping_mask = None
                    self.grasping_object = None
                    self.finish_cnt += 1
                    if self.finish_cnt == self.interface.auto_stop_cnt:
                        self.finish_cnt = 0
                        self.log.emit("自动抓取结束，切换到手动选择模式")
                        self.finish_signal.emit()
                        return

                    continue
                
                mask = self.airbot_segment.inference_bbox(captured_color, object_bbox)
                
                # 创建掩码预览
                masked_color = captured_color.copy()
                masked_color[mask] = [191, 214, 238]  # 预览掩码
                # cv2.imwrite("mask_color.png", masked_color)
                if self.grasping_mask is not None:
                    masked_color[self.grasping_mask] = [0, 255, 0]
                
                x1, y1, x2, y2 = object_bbox

                cv2.rectangle(masked_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    masked_color,
                    f"{object_label} {object_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                self.update_image.emit(masked_color)
                
                self.grasping_mask = mask
                self.grasping_object = object_label
                
                cloud = self.realsense.create_point_cloud(captured_depth, end_pose=captured_pose)
                if cloud.shape[0] == 0:
                    self.log.emit("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.capture_signal.emit()
                    if self.interface.robot_grasp_thread.isRunning():
                        self.interface.robot_grasp_thread.requestInterruption()
                        self.interface.robot_grasp_thread.wait()
                    time.sleep(1)
                    self.grasping_mask = None
                    self.grasping_object = None
                    self.finish_cnt += 1
                    if self.finish_cnt == self.interface.auto_stop_cnt:
                        self.finish_cnt = 0
                        self.log.emit("自动抓取结束，切换到手动选择模式")
                        self.finish_signal.emit()
                        return
                    continue
                trans, orient, cloud_base = self.airbot_grasp.inference(
                    color_image=captured_color, 
                    depth_image=captured_depth, 
                    end_pose=captured_pose, 
                    cloud_cam_raw=cloud,
                    mask=mask
                )
                if trans is None or orient is None or cloud_base is None:
                    self.log.emit("未检测到物体，跳过当前循环")
                    self.capture_signal.emit()
                    if self.interface.robot_grasp_thread.isRunning():
                        self.interface.robot_grasp_thread.requestInterruption()
                        self.interface.robot_grasp_thread.wait()
                    time.sleep(1)
                    self.grasping_mask = None
                    self.grasping_object = None
                    self.finish_cnt += 1
                    if self.finish_cnt == self.interface.auto_stop_cnt:
                        self.finish_cnt = 0
                        self.log.emit("自动抓取结束，切换到手动选择模式")
                        self.finish_signal.emit()
                        return
                    continue
                
                object_height = np.max(cloud_base[:,2])
                object_width = float(self.airbot_grasp.width_m)
                pbject_angle = self.airbot_grasp.angle
                
                predicted_info = {
                    "trans": trans.tolist(),
                    "orient": orient.tolist(),
                    "o_height": object_height,
                    "o_width": object_width,
                    "o_angle": pbject_angle,
                    "o_label": object_label,
                    "o_bbox": object_bbox
                }
                
                with self.interface.predict_lock:
                    self.interface.predicted_info = predicted_info.copy()
                
                self.update_info.emit(predicted_info)
                
                if self.interface.dump:
                    import json
                    with self.interface.captured_frame_lock:
                        dump_info = {
                            "observe": self.interface.capture_state,
                            "predicted": predicted_info
                        }
                    file_name = time.strftime("%Y%m%d%H%M%S", time.localtime()) + "_info.json"
                    file_path = os.path.join(self.interface.dump_path, file_name)
                    with open(file_path, "w") as f:
                        json.dump(dump_info, f, indent=4)

                # 等待完成上一次抓取
                self.interface.robot_grasp_thread.wait()            
                self.interface.robot_grasp_thread.set_predicted_info(predicted_info)
                self.interface.robot_grasp_thread.start() 
                
            except Exception as e:
                error_info = traceback.format_exc()
                self.log.emit(f"自动抓取异常: {str(e)}\n详细信息:\n{error_info}")
                self.capture_signal.emit()
                time.sleep(1)
        
        self.capture_signal.emit()
    
    def stop(self):
        self.wait()

class AirbotControlInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        config = yaml.safe_load(open(config_path, "r"))
        
        # 初始化变量
        self.color_frame = None
        self.depth_frame = None
        self.depth_map = None
        self.current_state = {"trans": None, "orient": None, "joints": None, "eef": None, "eeff": None}
        self.observe_frame_lock = Lock()
        
        self.captured_color = None
        self.captured_depth = None
        self.captured_depth_map = None
        self.captured_pose = {"trans": None, "orient": None}
        self.captured_state = {"trans": None, "orient": None, "joints": None, "eef": None, "eeff": None}
        self.mask_map = None
        self.captured_frame_lock = Lock()
        
        self.predicted_info = {"trans": None, "orient": None, "o_height": None, "o_width": None, "o_angle": None, "o_label": None, "o_bbox": None}
        self.predict_lock = Lock()
        self.detection_lock = Lock()
        self.latest_detection_batch = None
        self.selected_detection_bbox = None
        
        # 初始化相机和分割模块
        self.realsense = create_camera()
        self.frame_width = self.realsense.WIDTH
        self.frame_height = self.realsense.HEIGHT
        self.airbot_segment = AirbotSegment()
        self.airbot_segment.clear_prompt()
        self.airbot_grasp = SimpleGrasp()
        self.airbot_yolo = AirbotYolo()
        detection_config = config.get("RealtimeDetection", {})
        self.detection_config = detection_config
        self.detection_freshness = float(detection_config.get("freshness_seconds", 1.0))
        self.preparation_timeout = float(
            detection_config.get("preparation_timeout_seconds", 8.0))
        
        # 视频流标志
        self.running = True
        
        # 初始化GUI
        if getattr(self.realsense, "resolution", "") == "480p":
            factor = 1
        else:
            factor = 2
        self.video_width = int(self.frame_width/factor)
        self.video_height = int(self.frame_height/factor)
        self.init_ui()

        self.detection_thread = DetectionThread(
            self.airbot_yolo, detection_config.get("inference_hz", 10),
            DetectionStabilizer(detection_config))
        self.detection_thread.result_ready.connect(self.update_detection_result)
        self.detection_thread.failed.connect(
            lambda message: self.log("实时检测失败：" + message))
        self.detection_thread.start()
        
        # 启动相机线程
        self.gripper_opened = True
        self.robot_speed = SpeedProfile.DEFAULT
        arm_config = config["ArmParams"]
        self.robot_host = arm_config.get("host", "localhost")
        self.robot_port = arm_config["port"]
        self.gripper_width = arm_config["gripper_width"]
        self.gripper_grasp_timeout_ms = int(
            arm_config.get("gripper_grasp_timeout_ms", 1500))
        self.gripper_grasp_compression = float(
            arm_config.get("gripper_grasp_compression", 0.003))
        self.gripper_grasp_effort = float(
            arm_config.get("gripper_grasp_effort", 6.0))
        self.gripper_grasp_retry_step = float(
            arm_config.get("gripper_grasp_retry_step", 0.003))
        self.gripper_grasp_retry_count = int(
            arm_config.get("gripper_grasp_retry_count", 2))
        self.gripper_grasp_min_width = float(
            arm_config.get("gripper_grasp_min_width", 0.005))
        if (self.gripper_grasp_retry_step <= 0
                or self.gripper_grasp_retry_count < 0
                or not 0 <= self.gripper_grasp_min_width <= self.gripper_width):
            raise ValueError("夹爪分级闭合配置无效")
        self.gripper_open_clearance = float(
            arm_config.get("gripper_open_clearance", 0.015))
        self.gripper_min_open_width = float(
            arm_config.get("gripper_min_open_width", 0.0))
        if not 0 <= self.gripper_min_open_width <= self.gripper_width:
            raise ValueError("最小张开宽度必须在夹爪行程范围内")
        self.gripper_contact_effort = float(
            arm_config.get("gripper_contact_effort", 1.0))
        self.gripper_contact_strong_effort = float(
            arm_config.get("gripper_contact_strong_effort", 8.0))
        self.gripper_contact_velocity = float(
            arm_config.get("gripper_contact_velocity", 0.002))
        self.gripper_contact_position_tolerance = float(
            arm_config.get("gripper_contact_position_tolerance", 0.001))
        self.gripper_contact_confirm_samples = int(
            arm_config.get("gripper_contact_confirm_samples", 3))
        self.gripper_feedback_poll_interval = float(
            arm_config.get("gripper_feedback_poll_interval", 0.05))
        self.robot_operation_lock = RLock()
        self.robot = AirbotArm.from_config(arm_config)
        self.observe_thread = ObserveThread(self, self.robot_port, self.realsense)
        self.observe_thread.update_frame.connect(self.update_observe_frame)
        self.observe_thread.start()
        
        # 初始化抓取线程和自动抓取线程
        self.auto_mode = False
        self.robot_grasp_thread = RobotGraspThread(self)
        self.robot_grasp_thread.log.connect(self.log)
        self.robot_grasp_thread.capture_signal.connect(self.capture_frame)
        self.color_grasp_thread = ColorGraspPreparationThread(self)
        self.color_grasp_thread.prepared.connect(self.start_prepared_color_grasp)
        self.color_grasp_thread.failed.connect(self.color_grasp_failed)
        self.auto_grasp_thread = AutoGraspThread(self)
        self.auto_grasp_thread.finish_signal.connect(self.change_auto_mode)
        self.auto_grasp_thread.capture_signal.connect(self.capture_frame)
        self.auto_grasp_thread.update_image.connect(lambda image: self.update_capture_view(image))
        self.auto_grasp_thread.update_info.connect(self.update_info)
        self.auto_grasp_thread.log.connect(self.log)
        
        # 加载运行时参数
        self.observe_pose = config["AirbotGrasp"]["observe_pose"]
        self.place_pose = config["AirbotGrasp"]["place_pose"]
        self.pre_place_pose = config["AirbotGrasp"]["pre_place_pose"]
        
        # 日志
        self.dump = config["RunTime"]["dump"]
        self.dump_path = config["RunTime"]["dump_path"]
        self.auto_stop_cnt = config["RunTime"]["auto_stop_cnt"]
        os.makedirs(self.dump_path, exist_ok=True)
        
        self.live_mirror = None
        if os.getenv('GRASP_LIVE_MIRROR') == '1':
            try:
                from live_mirror import start_mirror
                self.live_mirror = start_mirror(self)
            except Exception as exc:
                self.log('无法启动仿真镜像：' + str(exc))
        # 初始化robot
        self.move_to_observe()
    
    def init_ui(self):
        self.setWindowTitle("AIRBOT 实时语音抓取")
        self.setMinimumSize(1280, 720)

        self.button_style = """
            QPushButton { background: #176b5b; color: white; border: 0;
                border-radius: 5px; padding: 8px 12px; font-size: 13px; }
            QPushButton:hover { background: #21806e; }
            QPushButton:pressed { background: #105247; }
            QPushButton:disabled { background: #a9b5b2; color: #eef1f0; }
        """
        self.button_style_red = """
            QPushButton { background: #b8433f; color: white; border: 0;
                border-radius: 5px; padding: 8px 12px; font-size: 13px; }
            QPushButton:hover { background: #ce5550; }
        """
        self.button_style_grey = """
            QPushButton { background: #737d83; color: white; border: 0;
                border-radius: 5px; padding: 8px 12px; font-size: 13px; }
        """

        self.camera_view_group = QGroupBox("实时目标检测")
        camera_view_layout = QVBoxLayout(self.camera_view_group)
        live_header = QHBoxLayout()
        live_title = QLabel("积木位置、颜色、置信度与颜色覆盖率")
        live_title.setObjectName("mutedLabel")
        live_header.addWidget(live_title)
        live_header.addStretch(1)
        self.live_indicator = QLabel("● 实时")
        self.live_indicator.setObjectName("liveIndicator")
        live_header.addWidget(self.live_indicator)
        camera_view_layout.addLayout(live_header)
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 420)
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        # Keep the 16:9 RealSense image aspect ratio; stretching it into the
        # 640x420 minimum label makes the scene visibly distorted.
        self.camera_label.setScaledContents(False)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background: #101415; border-radius: 4px;")
        camera_view_layout.addWidget(self.camera_label)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(420)

        task_tab = QWidget()
        task_layout = QVBoxLayout(task_tab)
        target_group = QGroupBox("目标状态")
        target_group.setMinimumHeight(200)
        target_layout = QVBoxLayout(target_group)
        self.target_status = QLabel("等待实时检测…")
        self.target_status.setWordWrap(True)
        self.target_status.setFixedHeight(58)
        self.target_status.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.target_status.setObjectName("targetStatus")
        target_layout.addWidget(self.target_status)
        swatches = QHBoxLayout()
        self.blue_grasp_btn = QPushButton("抓取蓝色积木")
        self.blue_grasp_btn.setObjectName("blueAction")
        self.green_grasp_btn = QPushButton("抓取绿色积木")
        self.green_grasp_btn.setObjectName("greenAction")
        self.blue_grasp_btn.clicked.connect(lambda: self.request_color_grasp("blue"))
        self.green_grasp_btn.clicked.connect(lambda: self.request_color_grasp("green"))
        swatches.addWidget(self.blue_grasp_btn)
        swatches.addWidget(self.green_grasp_btn)
        target_layout.addLayout(swatches)
        predictions = QHBoxLayout()
        self.blue_predict_btn = QPushButton("仅预测蓝色（不运动）")
        self.green_predict_btn = QPushButton("仅预测绿色（不运动）")
        self.blue_predict_btn.clicked.connect(lambda: self.request_color_prediction("blue"))
        self.green_predict_btn.clicked.connect(lambda: self.request_color_prediction("green"))
        predictions.addWidget(self.blue_predict_btn)
        predictions.addWidget(self.green_predict_btn)
        target_layout.addLayout(predictions)
        task_layout.addWidget(target_group)
        self.voice_panel = VoiceCommandPanel(self)
        self.voice_panel.listening_changed.connect(self.set_voice_recording)
        task_layout.addWidget(self.voice_panel)
        safety = QLabel("请求的颜色必须只有一个有效目标；多个同色积木时不会自动抓取。\n"
                        "语音停止不是急停，紧急情况请使用硬件急停。")
        safety.setObjectName("safetyNote")
        safety.setWordWrap(True)
        task_layout.addWidget(safety)
        task_layout.addStretch(1)
        self.tabs.addTab(task_tab, "语音抓取")

        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        self.captured_view_group = QGroupBox("手动选择与分割")
        captured_view_layout = QVBoxLayout(self.captured_view_group)
        self.capture_instruction = QLabel("备用流程：拍照后点击目标")
        self.capture_instruction.setWordWrap(True)
        self.captured_label = ClickableLabel()
        self.captured_label.setMinimumSize(360, 270)
        self.captured_label.setScaledContents(False)
        self.captured_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captured_label.setStyleSheet("background: #101415; border-radius: 4px;")
        self.captured_label.clicked.connect(self.handle_segmentation_click)
        captured_view_layout.addWidget(self.capture_instruction)
        captured_view_layout.addWidget(self.captured_label)
        manual_layout.addWidget(self.captured_view_group)

        self.button_group = QGroupBox("手动控制")
        button_layout = QGridLayout(self.button_group)
        self.capture_btn = QPushButton("拍照")
        self.predict_btn = QPushButton("预测位姿")
        self.grasp_btn = QPushButton("抓取并放置")
        self.auto_btn = QPushButton("旧版自动模式")
        self.gravity_btn = QPushButton("重力补偿")
        self.restore_control_btn = QPushButton("恢复控制连接")
        self.restore_control_btn.clicked.connect(self.restore_control_connection)
        button_layout.addWidget(self.restore_control_btn, 5, 0, 1, 2)
        self.observe_btn = QPushButton("回到观察位")
        self.gripper_btn = QPushButton("切换夹爪")
        self.set_place_btn = QPushButton("设置放置位")
        buttons = (self.capture_btn, self.predict_btn, self.grasp_btn, self.auto_btn,
                   self.gravity_btn, self.observe_btn, self.gripper_btn, self.set_place_btn)
        for index, button in enumerate(buttons):
            button.setStyleSheet(self.button_style)
            button.setMinimumHeight(38)
            button_layout.addWidget(button, index // 2, index % 2)
        self.capture_btn.clicked.connect(self.capture_frame)
        self.gravity_btn.clicked.connect(self.trigger_gravity)
        self.observe_btn.clicked.connect(self.move_to_observe)
        self.predict_btn.clicked.connect(self.request_manual_prediction)
        self.grasp_btn.clicked.connect(self.predict_and_grasp)
        self.gripper_btn.clicked.connect(self.trigger_gripper)
        self.set_place_btn.clicked.connect(self.set_place_pose)
        self.auto_btn.clicked.connect(self.change_auto_mode)
        manual_layout.addWidget(self.button_group)
        self.tabs.addTab(manual_tab, "手动备用")

        robot_tab = QWidget()
        robot_layout = QVBoxLayout(robot_tab)
        self.info_group = QGroupBox("机械臂与预测信息")
        info_layout = QVBoxLayout(self.info_group)
        info_layout.addWidget(QLabel("当前状态"))
        self.current_state_text = QTextEdit()
        self.current_state_text.setReadOnly(True)
        self.current_state_text.setMinimumHeight(90)
        info_layout.addWidget(self.current_state_text)
        info_layout.addWidget(QLabel("预测位姿"))
        self.predicted_info_text = QTextEdit()
        self.predicted_info_text.setReadOnly(True)
        self.predicted_info_text.setFixedHeight(70)
        info_layout.addWidget(self.predicted_info_text)
        info_layout.addWidget(QLabel("目标信息"))
        self.object_info_text = QTextEdit()
        self.object_info_text.setReadOnly(True)
        self.object_info_text.setMinimumHeight(100)
        info_layout.addWidget(self.object_info_text)
        robot_layout.addWidget(self.info_group)
        self.select_group = QGroupBox("运动设置")
        select_layout = QVBoxLayout(self.select_group)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("机械臂速度"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["SLOW", "DEFAULT", "MEDIUM", "FAST"])
        self.speed_combo.setCurrentText("DEFAULT")
        self.speed_combo.currentTextChanged.connect(self.change_speed_profile)
        speed_layout.addWidget(self.speed_combo)
        select_layout.addLayout(speed_layout)
        robot_layout.addWidget(self.select_group)
        robot_layout.addStretch(1)
        self.tabs.addTab(robot_tab, "机械臂")

        diagnostics_tab = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_tab)
        self.log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(self.log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        diagnostics_layout.addWidget(self.log_group)
        self.tabs.addTab(diagnostics_tab, "诊断日志")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.camera_view_group)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 460])
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

        self.setStyleSheet("""
            QMainWindow { background: #eef1f0; }
            QGroupBox {
                font-weight: 600; border: 1px solid #c8d0cd;
                border-radius: 5px;
                margin-top: 10px; padding-top: 12px; color: #202624;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
            }
            QLabel { font-size: 13px; color: #202624; }
            QLabel#mutedLabel { color: #66716d; }
            QLabel#liveIndicator { color: #16836f; font-weight: 600; }
            QLabel#targetStatus { font-size: 15px; font-weight: 600; padding: 6px 0; }
            QLabel#safetyNote { color: #685334; background: #fff6df;
                border: 1px solid #ead5a6; border-radius: 4px; padding: 8px; }
            QTextEdit {
                font-size: 12px; border: 1px solid #c8d0cd;
                border-radius: 4px; background: #fbfcfc; color: #202624;
            }
            QTabWidget::pane { border: 1px solid #c8d0cd; background: #f7f9f8; }
            QTabBar::tab { padding: 8px 12px; background: #dfe5e2; }
            QTabBar::tab:selected { background: #ffffff; color: #176b5b; }
            QPushButton#blueAction { background: #246fbd; color: white; min-height: 40px; }
            QPushButton#greenAction { background: #278457; color: white; min-height: 40px; }
            QComboBox { min-height: 30px; padding: 2px 8px; }
        """)
        self.log("AIRBOT 实时语音抓取界面已初始化")
    
    @pyqtSlot(str)
    @safe_func()
    def log(self, message):
        """将消息添加到日志输出区域"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        if getattr(self, "live_mirror", None) is not None:
            self.live_mirror.publisher.command = str(message)
        self.log_output.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")
    
    def qpixmap_from_image(self, frame):
        resized_frame = cv2.resize(frame, (self.video_width, self.video_height))
        h, w, ch = resized_frame.shape
        qt_image = QImage(resized_frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        return pixmap

    def voice_is_busy(self):
        return (self.auto_mode or self.robot_grasp_thread.isRunning()
                or self.auto_grasp_thread.isRunning()
                or (hasattr(self, "color_grasp_thread")
                    and self.color_grasp_thread.isRunning()))

    @safe_func()
    def restore_control_connection(self):
        if self.voice_is_busy():
            raise ValueError('任务进行中，不能重建控制连接。')
        self.invalidate_selection(discard_frame=True)
        with self.detection_lock:
            self.latest_detection_batch = None
        with self.robot_operation_lock:
            self.robot.restore_control()
        self.log('控制连接已恢复，未执行运动。请等待新的检测结果，再确认一条新指令。')

    def voice_has_selection(self):
        with self.captured_frame_lock:
            return (self.captured_color is not None and self.mask_map is not None
                    and bool(np.any(self.mask_map)))

    def set_voice_recording(self, recording):
        # 录音/识别期间固定目标和模式，避免识别结果操作另一个目标。
        self.button_group.setEnabled(not recording)
        self.select_group.setEnabled(not recording)
        self.captured_label.setEnabled(not recording)
        self.blue_grasp_btn.setEnabled(not recording)
        self.blue_predict_btn.setEnabled(not recording)
        self.green_predict_btn.setEnabled(not recording)
        self.green_grasp_btn.setEnabled(not recording)

    def invalidate_selection(self, discard_frame=False):
        self.selected_detection_bbox = None
        with self.captured_frame_lock:
            self.mask_map = None
            if discard_frame:
                self.captured_color = None
                self.captured_label.clear()
                self.capture_instruction.setText("目标已失效，请重新拍照并选择目标")
        with self.predict_lock:
            self.predicted_info = dict.fromkeys(self.predicted_info)
        self.predicted_info_text.clear()
        self.object_info_text.clear()
    @pyqtSlot(object)
    @safe_func()
    def update_observe_frame(self, snapshot):
        """更新观测"""
        pose = snapshot.state
        if getattr(self, 'live_mirror', None) is not None:
            self.live_mirror.observe(snapshot)
        self.camera_label.setPixmap(self.qpixmap_from_image(snapshot.color))
        if hasattr(self, "detection_thread"):
            self.detection_thread.submit(snapshot)
        trans_str = ", ".join([f"{v:.6f}" for v in pose["trans"]])
        orient_str = ", ".join([f"{v:.6f}" for v in pose["orient"]])
        joints_str = ", ".join([f"{v:.6f}" for v in pose["joints"]])
        
        eef_pos_str = ", ".join([f"{v:.2f}" for v in pose["eef"]])
        eef_eff_str = ", ".join([f"{v:.2f}" for v in pose["eeff"]])
        
        
        refer_pose = [[0.188852, -0.005085, 0.259439], [-0.001318, 0.554186, -0.015383, 0.832249]]
        # error = np.linalg.norm(np.array(pose["trans"]) - np.array(refer_pose[0])) + np.linalg.norm(np.array(pose["orient"] - np.array(refer_pose[1])))
        # print(error)
        self.current_state_text.setText(f"平移: [{trans_str}]\n旋转: [{orient_str}]\n关节角: [{joints_str}]\n末端执行器: [{eef_pos_str}] m  -- [{eef_eff_str}] N")

    @pyqtSlot(object)
    @safe_func()
    def update_detection_result(self, batch):
        if getattr(self, 'live_mirror', None) is not None:
            self.live_mirror.detect(batch)
        with self.detection_lock:
            if (self.latest_detection_batch is not None
                    and batch.snapshot.sequence < self.latest_detection_batch.snapshot.sequence):
                return
            self.latest_detection_batch = batch
        self.camera_label.setPixmap(self.qpixmap_from_image(
            draw_detections(batch.snapshot.color, batch.detections,
                            self.selected_detection_bbox)))
        blue = sum(item.color == "blue" for item in batch.detections)
        green = sum(item.color == "green" for item in batch.detections)
        unknown = sum(item.color is None for item in batch.detections)
        summary = f"稳定目标：蓝色 {blue} | 绿色 {green} | 未知 {unknown}"
        details = [
            f"{item.display_label}{'' if item.observed else '（重确认）'} "
            f"{item.confidence:.0%}/{item.color_coverage:.0%}"
            for item in batch.detections[:3]
        ]
        message = summary + "\n" + ("；".join(details) if details else "正在确认目标…")
        if hasattr(self, "target_status"):
            self.target_status.setText(message)

    def grasp_color(self, color, *, execute=True):
        if color not in ("blue", "green"):
            raise ValueError("仅支持蓝色或绿色积木")
        if self.voice_is_busy():
            raise ValueError("机械臂或抓取位姿计算正在进行")
        with self.detection_lock:
            batch = self.latest_detection_batch
        if batch is None:
            raise ValueError("尚无实时检测结果")
        age = time.monotonic() - batch.snapshot.timestamp
        if age > self.detection_freshness:
            raise ValueError(f"实时检测结果已过期（{age:.1f} 秒）")
        detection = resolve_unique_target(batch.detections, color)
        self.selected_detection_bbox = detection.bbox
        if hasattr(self, "target_status"):
            self.target_status.setText(
                f"已选择{COLOR_LABELS[color]}，正在计算抓取位姿…")
        self.log(f"已锁定{COLOR_LABELS[color]}，检测置信度 "
                 f"{detection.confidence:.2f}，颜色覆盖率 "
                 f"{detection.color_coverage:.0%}")
        self.color_grasp_thread.set_task(batch, detection, execute=execute)
        self.color_grasp_thread.start()

    @safe_func()
    def request_color_grasp(self, color):
        self.grasp_color(color)

    @safe_func()
    def request_color_prediction(self, color):
        self.grasp_color(color, execute=False)

    @pyqtSlot(dict, np.ndarray)
    @safe_func()
    def start_prepared_color_grasp(self, info, preview):
        if time.monotonic() - info["snapshot_timestamp"] > self.preparation_timeout:
            self.color_grasp_failed("目标位姿计算完成时已过期")
            return
        execute_motion = info.get('execute_motion', True)
        info = {key: value for key, value in info.items()
                if key not in ("snapshot_timestamp", "execute_motion")}
        with self.predict_lock:
            self.predicted_info = info.copy()
        self.update_info(info)
        self.update_capture_view(preview)
        if 'target_mask_clearance_mm' in info:
            self.log(f"目标投影距分割边缘余量：{info['target_mask_clearance_mm']:.1f} mm"
                     "（图像估算，不代表实测夹爪定位精度）")
        if not execute_motion:
            self.tabs.setCurrentIndex(1)
            self.log("仅预测完成，未发送机械臂或夹爪指令；标记显示在手动备用页。"
                     "青色 RAW 为拟合顶面中心，品红 TARGET 为补偿后 XY 在同一顶面的投影；"
                     "不是实测指尖位置，也不能独立验证手眼标定。")
            return
        self.robot_grasp_thread.set_predicted_info(info.copy())
        self.robot_grasp_thread.start()
        self.log("实时颜色目标位姿已验证，开始抓取")

    @pyqtSlot(str)
    def color_grasp_failed(self, message):
        self.selected_detection_bbox = None
        if hasattr(self, "target_status"):
            self.target_status.setText("未执行：" + message)
        self.log("实时颜色抓取已取消：" + message)
    
    @pyqtSlot(np.ndarray)
    @safe_func()
    def update_capture_view(self, frame):
        """更捕获视图"""
        self.captured_label.setPixmap(self.qpixmap_from_image(frame))
    
    @pyqtSlot(dict)
    @safe_func()
    def update_info(self, info):
        """更新信息"""
        trans_str = ", ".join([f"{v:.6f}" for v in info["trans"]])
        orient_str = ", ".join([f"{v:.6f}" for v in info["orient"]])
        object_height = info["o_height"]
        object_width = info["o_width"]
        object_angle = info["o_angle"]
        object_label = info["o_label"]
        object_bbox = info["o_bbox"]
        
        self.predicted_info_text.setText(f"平移: [{trans_str}]\n旋转: [{orient_str}]")
        self.object_info_text.setText(f"高度: {object_height:.4f}\n宽度: {object_width:.4f}\n角度: {object_angle:.4f}\n标签: {object_label}\n边界框: {object_bbox}")
    
    @pyqtSlot(int, int, int)
    @safe_func()
    def handle_segmentation_click(self, x, y, button):
        print(f"handle_segmentation_click: {x}, {y}, {button}")
        """处理分割视图上的点击"""
        if self.airbot_segment.mode != SegmentMode.POINT or self.captured_color is None:
            self.log("处于非手动选择模式或未捕获图像")
            return
        self.invalidate_selection()
        with self.captured_frame_lock:
            _captured_color = self.captured_color
            
        # 调整坐标到原始图像大小
        # The image is centered without stretching. Account for the letterbox
        # margins before mapping a click back to the original RGB frame.
        pixmap = self.captured_label.pixmap()
        display_width = pixmap.width() if pixmap is not None else self.captured_label.width()
        display_height = pixmap.height() if pixmap is not None else self.captured_label.height()
        offset_x = max(0, (self.captured_label.width() - display_width) // 2)
        offset_y = max(0, (self.captured_label.height() - display_height) // 2)
        if x < offset_x or y < offset_y or x >= offset_x + display_width or y >= offset_y + display_height:
            self.log("点击位于图像边缘空白区域，请点击积木图像")
            return
        orig_x = int((x - offset_x) * _captured_color.shape[1] / max(1, display_width))
        orig_y = int((y - offset_y) * _captured_color.shape[0] / max(1, display_height))

        if button == 0:  # 左键-前景
            self.airbot_segment.add_point(orig_x, orig_y, True)
            self.log(f"({orig_x}, {orig_y}) : 左键点击，添加为前景")
        elif button == 1:  # 右键-背景
            self.airbot_segment.add_point(orig_x, orig_y, False)
            self.log(f"({orig_x}, {orig_y}) : 右键点击，添加为背景")
        elif button == 2:  # 中键-清除
            self.airbot_segment.clear_prompt()
            self.log("中键点击，清除所有点")
            self.update_capture_view(_captured_color)
            return
        
        # 执行分割
        mask = self.airbot_segment.inference(_captured_color)
        if mask is None:
            self.log("分割失败，请重试")
            return
        
        # cv2.imwrite("mask.png", mask.astype(np.uint8) * 255)
        print("mask shape:", mask.shape)
        
        # 应用分割结果到图像
        masked_color = _captured_color.copy()
        print("masked_color shape:", masked_color.shape)
        masked_color[mask] = [191, 214, 238]  # 预览掩码
        # cv2.imwrite("mask_color.png", masked_color)
        with self.captured_frame_lock:
            self.mask_map = mask
        
        # 更新预览
        self.update_capture_view(masked_color)
        
        # if self.dump:
        #     # 保存结果
        #     cv2.imwrite("mask.png", mask.astype(np.uint8) * 255)
        #     cv2.imwrite("masked_color.png", masked_color)
    
    @pyqtSlot()
    @safe_func()
    def capture_frame(self):
        """捕获当前图像"""
        self.invalidate_selection(discard_frame=True)
        with self.observe_frame_lock, self.captured_frame_lock:
            if self.color_frame is None:
                self.log("错误：无法捕获图像，相机未连接或未初始化")
                return
            
            self.captured_color = self.color_frame.copy()
            self.captured_depth = self.depth_frame.copy()
            self.captured_depth_map = self.depth_map.copy()
            self.capture_state = self.current_state.copy()
            self.captured_pose = [self.current_state["trans"], self.current_state["orient"]]
                  
        # 更新捕获视图
        with self.captured_frame_lock:
            self.update_capture_view(self.captured_color)
            self.airbot_segment.clear_prompt()
                    
        # 更新提示
        if not self.auto_mode:
            self.capture_instruction.setText("点击图片添加前景(左键)或背景(右键)点，中键清除所有点")
        else:
            self.capture_instruction.setText("自动模式, 持续捕获图像帧")

        self.log("图像已捕获")

    @safe_func()
    def move_to_observe(self):
        if self.voice_is_busy():
            self.log("机械臂忙碌，不能移动到观察位")
            return
        self.invalidate_selection(discard_frame=True)
        self.airbot_segment.clear_prompt()
        self.gravity_btn.setStyleSheet(self.button_style)
        """移动机械臂到观察姿态"""
        with self.robot_operation_lock:
            self.robot.set_speed_profile(self.robot_speed)
            self.robot.move_end_pose(self.observe_pose)
            time.sleep(arm_delay)
            self.log("已移动到观察姿态")

    @safe_func()
    def predict_grasp_pose(self):
        """预测抓取位姿"""
        with self.captured_frame_lock:
            if (self.captured_color is None or self.mask_map is None
                    or not np.any(self.mask_map)):
                self.log("请先捕获图像并进行分割")
                return False
            mask_map = self.mask_map.copy()
            color = self.captured_color.copy()
            depth = self.captured_depth.copy()
            pose = self.captured_pose.copy()

        # 计算抓取位姿
        cloud = self.realsense.create_point_cloud(depth, end_pose=pose)
        trans, orient, cloud_base = self.airbot_grasp.inference(
            color_image=color, 
            depth_image=depth, 
            end_pose=pose, 
            cloud_cam_raw=cloud,
            mask=mask_map
        )
        
        print(trans, orient)
        
        object_height = np.max(cloud_base[:,2])
        object_width = float(self.airbot_grasp.width_m)
        object_angle = self.airbot_grasp.angle
        
        masked_color = np.ones_like(color)
        masked_color[mask_map] = color[mask_map]

        bbox, label, conf = self.airbot_yolo.get_max_conf_bbox_and_label(masked_color)
        
        predict_info = {
            "trans": trans,
            "orient": orient,
            "o_height": object_height,
            "o_width": object_width,
            "o_angle": object_angle,
            "o_label": label,
            "o_bbox": bbox
        }
        
        # 更新预测位姿
        with self.predict_lock:
            self.predicted_info = predict_info.copy() 
        
        # 更新UI
        self.update_info(predict_info)
        
        return True
    
    @safe_func()
    def request_manual_prediction(self):
        if self.voice_is_busy():
            self.log('任务进行中，暂不能预测')
            return
        if self.predict_grasp_pose():
            self.log('仅预测完成，未发送机械臂或夹爪指令')

    @safe_func()
    def move_to_predicted(self):
        self.gravity_btn.setStyleSheet(self.button_style)
        """移动到预测位姿"""
        if self.voice_is_busy() or not self.predict_grasp_pose():
            return
        with self.predict_lock:
            trans = self.predicted_info["trans"]
            orient = self.predicted_info["orient"]
        self.invalidate_selection(discard_frame=True)
        self.airbot_segment.clear_prompt()
        with self.robot_operation_lock:
            self.robot.move_gripper(self.gripper_width)
            time.sleep(gripper_delay)
            self.gripper_opened = True
            self.robot.move_end_pose([
                [trans[0], trans[1], trans[2] + 0.2], 
                orient
            ])
            time.sleep(arm_delay)
            self.robot.move_gripper(0)
            time.sleep(gripper_delay)
            self.log("已移动到预测位姿")
    
    @safe_func()
    def predict_and_grasp(self):
        """执行抓取并移动到放置位置"""
        if self.voice_is_busy():
            self.log("机械臂忙碌，不能重复开始抓取")
            return
        if not self.predict_grasp_pose():
            self.log("预测失败，本次不执行抓取")
            return
        with self.predict_lock:
            self.robot_grasp_thread.set_predicted_info(self.predicted_info.copy())
        self.invalidate_selection(discard_frame=True)
        self.airbot_segment.clear_prompt()
        self.robot_grasp_thread.start()
        
    @safe_func()
    def trigger_gripper(self):
        """控制夹爪开合"""
        self.set_gripper_open(not self.gripper_opened)

    @safe_func()
    def set_gripper_open(self, opened):
        """明确设置开合状态，重复“打开”不会变成关闭。"""
        if self.voice_is_busy():
            self.log("机械臂忙碌，不能单独控制夹爪")
            return
        self.invalidate_selection(discard_frame=True)
        self.airbot_segment.clear_prompt()
        with self.robot_operation_lock:
            self.robot.move_gripper(self.gripper_width if opened else 0)
            time.sleep(gripper_delay)
            self.gripper_opened = opened
            self.log("夹爪已打开" if opened else "夹爪已关闭")
    
    @safe_func()
    def set_place_pose(self):
        """设置放置姿态"""
        with self.robot_operation_lock:
            pose = self.robot.get_end_pose()
            self.place_pose = pose
            self.pre_place_pose = [
                [pose[0][0], pose[0][1], pose[0][2] + 0.1], pose[1]
            ]
            self.log("已更新放置姿态")
    
    @safe_func()
    def change_speed_profile(self, speed_text):
        """根据下拉菜单选择更改机器人速度配置"""
        self.robot_speed = SpeedProfile[speed_text]

        with self.robot_operation_lock:
            self.robot.set_speed_profile(self.robot_speed)
            
        self.log(f"机器人速度配置已更改为: {speed_text}")
    
    @safe_func()
    def trigger_gravity(self):
        """控制重力开关"""
        self.invalidate_selection(discard_frame=True)
        self.airbot_segment.clear_prompt()
        with self.robot_operation_lock:
            if not self.robot.gravity_enabled:
                self.robot.enter_gravity_compensation()
                self.gravity_btn.setStyleSheet(self.button_style_red)
                self.log("切换到重力补偿模式")
            else:
                self.robot.leave_gravity_compensation()
                self.gravity_btn.setStyleSheet(self.button_style)
                self.log("切换到PLANNING模式")
    
    def change_auto_mode(self):
        self.gravity_btn.setStyleSheet(self.button_style)
        """修改抓取模式"""
        if not self.auto_mode:
            try:
                self.auto_btn.setEnabled(False)
                self.auto_btn.setText("Waiting...") 
                self.auto_btn.setStyleSheet(self.button_style_grey)

                self.auto_mode = True
                self.airbot_segment.mode = SegmentMode.BBOX
                self.airbot_segment.clear_prompt()
                self.auto_grasp_thread.start()
                
                self.auto_btn.setText("Manual Select Mode")
                self.auto_btn.setStyleSheet(self.button_style_red)
                self.auto_btn.setEnabled(True)
            except Exception as e:
                details = traceback.format_exc()
                self.log(f"更改抓取模式失败: {e}\n{details}")
        else:
            try:
                self.auto_btn.setEnabled(False)
                self.auto_btn.setText("Waiting...") 
                self.auto_btn.setStyleSheet(self.button_style_grey)
                
                if hasattr(self, 'robot_grasp_thread') and self.robot_grasp_thread.isRunning():
                    self.robot_grasp_thread.requestInterruption()
                    self.robot_grasp_thread.wait()
                
                if hasattr(self, 'auto_grasp_thread') and self.auto_grasp_thread.isRunning():
                    self.auto_grasp_thread.requestInterruption()
                    self.auto_grasp_thread.wait()
                    
                self.auto_mode = False
                self.airbot_segment.mode = SegmentMode.POINT
                self.airbot_segment.clear_prompt()
                
                self.auto_btn.setText("Auto Grasp Mode")
                self.auto_btn.setStyleSheet(self.button_style)
                self.auto_btn.setEnabled(True)
            except Exception as e:
                details = traceback.format_exc()
                self.log(f"更改抓取模式失败: {e}\n{details}")
                
        if not self.auto_mode:
            self.invalidate_selection(discard_frame=True)
        self.log(f"抓取模式已更改为: {'自动抓取' if self.auto_mode else '手动选择'}")
    
    @safe_func()
    def closeEvent(self, event):
        """关闭窗口时的处理"""
        self.voice_panel.shutdown()
        if getattr(self, 'live_mirror', None) is not None:
            self.live_mirror.close()
            self.live_mirror = None
        # 停止所有线程
        self.running = False

        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        if hasattr(self, 'color_grasp_thread') and self.color_grasp_thread.isRunning():
            self.color_grasp_thread.requestInterruption()
            self.color_grasp_thread.wait()
        
        if hasattr(self, 'robot_grasp_thread') and self.robot_grasp_thread.isRunning():
            self.robot_grasp_thread.requestInterruption()
            self.robot_grasp_thread.wait()
        
        if hasattr(self, 'auto_grasp_thread') and self.auto_grasp_thread.isRunning():
            self.auto_grasp_thread.requestInterruption()
            self.auto_grasp_thread.wait()

        if hasattr(self, 'observe_thread') and self.observe_thread.isRunning():
            self.observe_thread.stop()

        if hasattr(self, 'robot'):
            self.robot.close()
            
        self.log("正在关闭应用...")
        super().closeEvent(event)
    
def main():
    try:
        check_camera_connection()
    except CameraConnectionError as exc:
        print('启动取消：' + str(exc), file=sys.stderr)
        sys.exit(1)
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 尝试加载自定义字体
    try:
        font_id = QFontDatabase.addApplicationFont("fonts/MapleMonoBold.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            custom_font = QFont(font_family, 12)
            app.setFont(custom_font)
    except:
        custom_font = QFont("Monospace", 12)
        app.setFont(custom_font)
    
    # 创建并显示主窗口
    try:
        window = AirbotControlInterface()
    except CameraConnectionError as exc:
        print('启动取消：' + str(exc), file=sys.stderr)
        sys.exit(1)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
