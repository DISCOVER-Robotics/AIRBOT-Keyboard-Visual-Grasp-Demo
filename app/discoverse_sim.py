"""DISCOVERSE AIRBOT Play physics backend; no hardware SDK or camera imports."""
from __future__ import annotations

from pathlib import Path
import tempfile
import time
import yaml
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def write_scene(assets, destination):
    """Reference the upstream robot, keeping all generated files in our runtime."""
    assets = Path(assets).resolve()
    robot = assets / 'mjcf/manipulator/airbot_play'
    if not (robot / 'airbot_play.xml').is_file():
        raise FileNotFoundError(f'缺少 DISCOVERSE 模型目录：{robot}')
    root = ET.Element('mujoco', model='voice_grasp')
    ET.SubElement(root, 'compiler', angle='radian', meshdir=str(assets / 'meshes'), autolimits='true')
    ET.SubElement(root, 'option', timestep='0.002', integrator='implicitfast')
    for part in ('dependencies', 'appendix'):
        ET.SubElement(root, 'include', file=str(robot / f'airbot_play_{part}.xml'))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0.3 -0.4 1.5', dir='0 0 -1', ambient='.42 .42 .42', diffuse='.7 .7 .7')
    ET.SubElement(world, 'geom', name='table', type='plane', size='1 1 .05', rgba='.7 .72 .75 1', friction='1 .005 .0001')
    ET.SubElement(world, 'camera', name='overview', pos='.85 -.85 .75', xyaxes='.807 .590 0 -.30 .411 .860', fovy='45')
    ET.SubElement(world, 'include', file=str(robot / 'airbot_play.xml'))
    for color, y, rgba in [('blue', -.10, '.05 .2 .9 1'), ('green', .10, '.05 .75 .15 1')]:
        body = ET.SubElement(world, 'body', name=f'{color}_block', pos=f'.32 {y} .02')
        ET.SubElement(body, 'freejoint', name=f'{color}_free')
        ET.SubElement(body, 'geom', name=f'{color}_cube', type='box', size='.02 .02 .02', mass='.035', rgba=rgba, friction='1.5 .01 .001', condim='4')
    # A visual marker only, so the block lands on the table, not on a hidden weld.
    ET.SubElement(world, 'site', name='place_zone', group='0', type='box', pos='.25 .25 .001', size='.055 .055 .001', rgba='1 .65 .1 .5')
    for tag, suffix in [('actuator', 'control'), ('sensor', 'sensor')]:
        ET.SubElement(ET.SubElement(root, tag), 'include', file=str(robot / f'airbot_play_{suffix}.xml'))
    ET.ElementTree(root).write(destination, encoding='unicode')


class DiscoverseSimulation:
    """Single-threaded, incremental motion; Qt and headless runners use the same API."""
    colors = ('blue', 'green')
    place = np.array([.25, .25, .022])
    observe = np.array([.28, 0., .18])
    orientation = Rotation.from_euler('xyz', [0, np.pi / 2, 0]).as_matrix()

    def __init__(self):
        import mujoco
        from discoverse import DISCOVERSE_ASSETS_DIR
        from discoverse.robots import AirbotPlayIK
        from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg

        with open('configs/config_file.yaml', encoding='utf-8') as file:
            config_path = yaml.safe_load(file)['Path']
        with open(config_path, encoding='utf-8') as file:
            self.real_config = yaml.safe_load(file)
        with open('configs/discoverse_sim.yaml', encoding='utf-8') as file:
            self.sim_config = yaml.safe_load(file)
        self.observe = np.array(self.sim_config['observe_position'])
        self.observe_orientation = Rotation.from_euler(
            'xyz', self.sim_config['observe_euler_xyz_degrees'], degrees=True).as_matrix()
        self.place = np.array(self.sim_config['place_position'])
        self.approach = float(self.sim_config['approach_height'])
        self.preparation = None
        self.preparation_execute = False
        self.prepared_info = None
        self.prepared_preview = None
        self.events = []
        self.mujoco = mujoco
        self.ik = AirbotPlayIK()
        self._directory = tempfile.TemporaryDirectory(prefix='graspdemo-sim-')
        scene = Path(self._directory.name) / 'scene.xml'
        write_scene(DISCOVERSE_ASSETS_DIR, scene)
        cfg = AirbotPlayCfg()
        cfg.mjcf_file_path = str(scene)
        cfg.enable_render = False  # Qt owns the camera renderer; headless needs no GL.
        cfg.headless = True
        cfg.sync = False
        cfg.timestep = .002
        cfg.decimation = 5
        cfg.init_qpos = np.r_[self.ik.properIK(self.observe, self.observe_orientation, np.zeros(6)), .04]
        self.env = AirbotPlayBase(cfg)
        self.model, self.data = self.env.mj_model, self.env.mj_data
        self.renderer = None
        self.vision = None
        self.motion = None
        self.selected = None
        self.predicted = None
        self.message = '仿真已就绪：可说“抓取蓝色积木”或“抓取绿色积木”。'
        self.last_error = None
        self.reset()

    @property
    def busy(self):
        return self.motion is not None or self.preparation is not None

    def reset(self):
        if self.busy:
            raise ValueError('任务进行中，不能重置场景。')
        self.env.reset()
        if self.vision is not None:
            self.vision.invalidate()
        # The upstream robot has two coupled fingers but seven controls.
        self.data.joint('endright').qpos[0] = -.04
        self.mujoco.mj_forward(self.model, self.data)
        self.control = self.env.init_joint_ctrl.copy()
        self.selected = self.predicted = None
        self.last_error = None
        self.prepared_info = self.prepared_preview = None
        self.events.clear()
        self._set_gripper_servo(False)
        for _ in range(50):
            self.env.step(self.control)

    def position(self, color):
        if color not in self.colors:
            raise ValueError('只支持蓝色或绿色积木。')
        return self.data.body(f'{color}_block').xpos.copy()

    def select(self, color):
        if self.busy:
            raise ValueError('任务进行中，不能更换目标。')
        self.position(color)
        self.selected, self.predicted = color, None

    def capture(self):
        if self.busy:
            raise ValueError('任务进行中，不能拍照。')
        self.selected = self.predicted = None
        self.prepared_info = self.prepared_preview = None
        self.message = '已拍照并清除目标；请在目标列表中重新选择积木。'

    def predict(self):
        if self.selected is None:
            raise ValueError('请先选择积木。')
        if self.busy:
            raise ValueError('目标位姿正在计算或机械臂正在运动。')
        if self.vision is not None:
            self._prepare(self.selected, execute=False)
            return
        self.predicted = self.position(self.selected)
        self.message = f'真值调试预测：{self.predicted}'

    def _prepare(self, color, execute):
        self.preparation = self.vision.prepare(color)
        self.preparation_color = color
        self.preparation_execute = execute
        self.prepared_info = self.prepared_preview = None
        self.last_error = None
        self.message = '已锁定目标，正在进行 MobileSAM 分割和 SimpleGrasp 位姿计算…'

    def _finish_preparation(self):
        future, self.preparation = self.preparation, None
        try:
            info, preview = future.result()
            timeout = float(self.vision.settings.get('preparation_timeout_seconds', 8.))
            if time.monotonic() - info['snapshot_timestamp'] > timeout:
                raise ValueError('目标位姿计算完成时已过期')
            self.prepared_info, self.prepared_preview = info, preview
            self.predicted = np.asarray(info['trans'])
            self.message = (f"位姿已预测：{np.round(info['trans'], 3)} m；"
                            f"宽度 {info['o_width'] * 1000:.1f} mm，角度 {info['o_angle']:.1f}°")
            if self.preparation_execute:
                self._start_prepared_grasp(self.preparation_color, info)
        except Exception as exc:
            self.last_error = str(exc)
            self.predicted = self.prepared_info = self.prepared_preview = None
            self.message = '目标准备失败：' + str(exc)

    def _start(self, motion):
        if self.busy:
            raise ValueError('仿真任务正在进行。')
        self.selected = self.predicted = None
        self.last_error = None
        self.motion = motion
        if self.vision is not None:
            self.vision.invalidate()

    def grip(self, opened):
        self.prepared_info = self.prepared_preview = None
        self._start(self._gripper(.04 if opened else 0.))
        self.message = '正在打开夹爪…' if opened else '正在关闭夹爪…'

    def go_observe(self):
        self.prepared_info = self.prepared_preview = None
        self._start(self._move(self.observe, orientation=self.observe_orientation))
        self.message = '正在返回观察位…'

    def grasp(self, color):
        if self.busy:
            raise ValueError('仿真任务正在进行。')
        self.select(color)
        if self.vision is not None:
            self._prepare(color, execute=True)
        else:
            self._start_prepared_grasp(color, {
                'trans': np.r_[self.position(color)[:2], self.sim_config['min_grasp_z']],
                'orient': Rotation.from_matrix(self.orientation).as_quat(),
                'o_width': .04,
            })

    def _start_prepared_grasp(self, color, info):
        position = np.asarray(info['trans'], dtype=float)
        orientation = Rotation.from_quat(info['orient']).as_matrix()
        if not np.all(np.isfinite(position)) or not .01 <= position[2] <= .08:
            raise ValueError('抓取位姿不在当前桌面安全高度范围。')
        if not np.isfinite(info['o_width']) or not 0 < info['o_width'] <= self.real_config['ArmParams']['gripper_width']:
            raise ValueError('视觉估算的物体宽度无效或超过夹爪开度。')
        # Ground truth is used only for evaluation/place occupancy, never prediction.
        for other in self.colors:
            if other != color and np.linalg.norm(self.position(other)[:2] - self.place[:2]) < .065:
                raise ValueError('放置区已有积木，请先重置场景。')
        if np.linalg.norm(position[:2] - self.place[:2]) < .065:
            raise ValueError('该积木已经在放置区。')
        for target in (position, position + [0, 0, self.approach]):
            self.ik.properIK(target, orientation, self.control[:6])
        for target in (self.place, self.place + [0, 0, self.approach]):
            self.ik.properIK(target, self.orientation, self.control[:6])
        self._start(self._pick_place(color, position, orientation, float(info['o_width'])))

    def _hold(self, seconds):
        for _ in range(round(seconds / self.env.delta_t)):
            yield

    def _set_gripper_servo(self, grasping):
        kp = float(self.sim_config['gripper_grasp_kp']) if grasping else 30.
        self.model.actuator_gainprm[6, 0] = kp
        self.model.actuator_biasprm[6, 1] = -kp
        effort = float(self.real_config['ArmParams']['gripper_grasp_effort']) if grasping else 15.
        self.model.actuator_forcerange[6] = [-effort, effort]

    def _gripper(self, value):
        self._set_gripper_servo(False)
        self.control[6] = value
        yield from self._hold(1.)

    def _move(self, target, duration=1.8, linear=False, orientation=None):
        orientation = self.orientation if orientation is None else orientation
        start = self.control[:6].copy()
        goal = np.asarray(self.ik.properIK(np.array(target), orientation, start))
        initial_position = self.data.body('link6').xpos.copy()
        count = round(duration / self.env.delta_t)
        # Precompute the entire segment, so IK failure cannot leave half a path.
        path = []
        ref = start
        for i in range(1, count + 1):
            ratio = i / count
            ratio = ratio * ratio * (3 - 2 * ratio)
            if linear:
                point = initial_position + ratio * (target - initial_position)
                ref = np.asarray(self.ik.properIK(point, orientation, ref))
            else:
                ref = start + ratio * (goal - start)
            path.append(ref.copy())
        for joints in path:
            self.control[:6] = joints
            yield
        yield from self._hold(.35)
        if np.max(np.abs(self.data.qpos[:6] - goal)) > .09:
            raise RuntimeError('关节未到达目标位置，已停止后续动作。')

    def _log_stage(self, message):
        self.message = message
        self.events.append(message)

    def _contact_feedback(self):
        pad_ids = [self.model.geom(name).id for name in ('left_finger_pad', 'right_finger_pad')]
        touching = [set(), set()]
        for contact in self.data.contact:
            for i, pad in enumerate(pad_ids):
                if pad in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == pad else contact.geom1
                    body = self.model.geom_bodyid[other]
                    joint = self.model.body_jntadr[body]
                    if joint >= 0 and self.model.jnt_type[joint] == self.mujoco.mjtJoint.mjJNT_FREE:
                        touching[i].add(body)
        return {
            'bilateral': bool(touching[0] & touching[1]),
            'position': float(self.data.joint('endleft').qpos[0] - self.data.joint('endright').qpos[0]),
            'velocity': float(abs(self.data.joint('endleft').qvel[0] - self.data.joint('endright').qvel[0])),
            'effort': float(abs(self.data.actuator_force[6])),
        }

    def _close_on_object(self, width):
        params = self.real_config['ArmParams']
        target = max(0., width - params['gripper_grasp_compression'])
        self._set_gripper_servo(True)
        samples = max(1, round(params['gripper_feedback_poll_interval'] / self.env.delta_t))
        for attempt in range(params['gripper_grasp_retry_count'] + 1):
            target_width = max(params['gripper_grasp_min_width'], target - attempt * params['gripper_grasp_retry_step'])
            self.control[6] = target_width / 2  # MuJoCo tendon is half the physical opening.
            confirmed = 0
            for step in range(round(params['gripper_grasp_timeout_ms'] / 1000 / self.env.delta_t)):
                yield
                if step % samples:
                    continue
                feedback = self._contact_feedback()
                if (feedback['bilateral'] and feedback['velocity'] <= params['gripper_contact_velocity']
                        and feedback['effort'] >= params['gripper_contact_effort']):
                    confirmed += 1
                else:
                    confirmed = 0
                if confirmed >= params['gripper_contact_confirm_samples']:
                    self._log_stage(f"夹持接触已确认：开度 {feedback['position']:.4f} m，作用力 {feedback['effort']:.2f} N")
                    return
            self._log_stage(f'接触未确认，夹爪收紧重试 {attempt + 1}')
        raise RuntimeError('夹持超时：未确认双侧接触，不执行抬升。')

    def _pick_place(self, color, position, orientation, width):
        params = self.real_config['ArmParams']
        pre_grasp = position + [0, 0, self.approach]
        pre_place = self.place + [0, 0, self.approach]
        zone, retreat, held, moved = None, None, False, False
        stage = '打开夹爪'
        try:
            self._log_stage('步骤1：打开夹爪')
            opening = min(max(width + params['gripper_open_clearance'], width * 1.2, params.get('gripper_min_open_width', 0.0)), params['gripper_width'])
            yield from self._gripper(opening / 2)
            stage = '移动到预抓取位'
            self._log_stage('步骤2：' + stage)
            zone, retreat, moved = 'grasp', pre_grasp, True
            yield from self._move(pre_grasp, orientation=orientation)
            stage = '直线下降到抓取位'
            self._log_stage('步骤3：' + stage)
            yield from self._move(position, linear=True, orientation=orientation)
            stage = '夹持接触确认'
            self._log_stage('步骤4：' + stage)
            yield from self._close_on_object(width)
            held = True
            stage = '直线抬升物体'
            self._log_stage('步骤5：' + stage)
            yield from self._move(pre_grasp, linear=True, orientation=orientation)
            if self.position(color)[2] < position[2] + self.approach * .6:
                raise RuntimeError('抓取失败：积木没有被实际抬起。')
            zone, retreat = None, None
            stage = '移动到预放置位'
            self._log_stage('步骤6：' + stage)
            yield from self._move(pre_place)
            zone, retreat = 'place', pre_place
            stage = '直线下降到公共放置位'
            self._log_stage('步骤7：' + stage)
            yield from self._move(self.place, linear=True)
            stage = '松开夹爪'
            self._log_stage('步骤8：' + stage)
            yield from self._gripper(params['gripper_width'] / 2)
            held = False
            stage = '直线抬升离开放置位'
            self._log_stage('步骤9：' + stage)
            yield from self._move(pre_place, linear=True)
            zone, retreat = None, None
            stage = '返回观察位'
            self._log_stage('步骤10：' + stage)
            yield from self._move(self.observe, orientation=self.observe_orientation)
            landed = self.position(color)
            if np.linalg.norm(landed[:2] - self.place[:2]) > .045 or not .012 < landed[2] < .035:
                raise RuntimeError('放置失败：积木没有稳定落入放置区。')
            self._log_stage(f'{color} 积木抓取并放置成功。')
        except Exception as exc:
            original = f'阶段 {stage}：{exc}'
            self._log_stage('任务失败：' + original)
            if moved:
                release = (zone == 'grasp' and not held) or (zone == 'place' and held)
                if release:
                    try:
                        self._log_stage('失败恢复：打开夹爪')
                        yield from self._gripper(params['gripper_width'] / 2)
                    except Exception as recovery_error:
                        self._log_stage('恢复松爪失败：' + str(recovery_error))
                if retreat is not None:
                    try:
                        self._log_stage('失败恢复：直线上抬')
                        yield from self._move(retreat, linear=True,
                                              orientation=orientation if zone == 'grasp' else self.orientation)
                    except Exception as recovery_error:
                        self._log_stage('恢复上抬失败：' + str(recovery_error))
                try:
                    self._log_stage('失败恢复：返回观察位' + ('（保持夹持）' if held and not release else ''))
                    yield from self._move(self.observe, orientation=self.observe_orientation)
                except Exception as recovery_error:
                    self._log_stage('恢复观察位失败：' + str(recovery_error))
            raise RuntimeError(original) from exc

    def tick(self):
        if self.preparation is not None and self.preparation.done():
            self._finish_preparation()
        if self.motion is not None:
            try:
                next(self.motion)
            except StopIteration:
                self.motion = None
                if self.message.startswith('正在'):
                    self.message = '动作已完成。'
            except Exception as exc:
                self.motion = None
                self.last_error = str(exc)
                self.message = '任务失败：' + str(exc)
                # Hold the measured arm pose; keep grip rather than drop a load.
                self.control[:6] = self.data.qpos[:6]
        self.env.step(self.control)

    def run_until_idle(self, timeout=60.):
        deadline = time.monotonic() + timeout
        steps = 0
        while time.monotonic() < deadline and steps < int(timeout / self.env.delta_t):
            preparing = self.preparation is not None
            self.tick()
            if not self.busy:
                if self.last_error:
                    raise RuntimeError(self.last_error)
                return
            if preparing:
                time.sleep(.002)  # Yield to MobileSAM's worker; no GUI uses this loop.
            else:
                steps += 1
        self.motion = None
        if self.preparation is not None:
            self.preparation.cancel()
            self.preparation = None
        self.control[:6] = self.data.qpos[:6]
        raise TimeoutError('仿真任务超时。')

    def render(self, camera="overview", depth=False):
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=480, width=640)
        option = self.mujoco.MjvOption()
        option.geomgroup[3] = 0  # Hide collision proxies, retain the robot visuals.
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        if depth:
            self.renderer.enable_depth_rendering()
        try:
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

    def close(self):
        if self.vision is not None:
            self.vision.close()
            self.vision = None
        self.motion = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._directory.cleanup()
