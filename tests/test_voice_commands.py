"""离线回归测试，不导入 SDK，不初始化相机，不连接机械臂。"""

import ast
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

from voice_commands import PHRASES, execute_command, match_command, parse_command
from airbot_yolo import COLOR_LABELS, Detection, resolve_unique_target


def interface_method(name, namespace=None):
    # 直接测试现有 GUI 方法体，跳过模块顶层的硬件和深度学习依赖。
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app" /
                      "airbot_interface.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "AirbotControlInterface")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {} if namespace is None else namespace
    exec(compile(module, "airbot_interface.py", "exec"), namespace)
    return namespace[name]


class CommandsTest(unittest.TestCase):
    def target(self):
        target = Mock()
        target.voice_is_busy.return_value = False
        target.voice_has_selection.return_value = True
        return target

    def test_all_phrases(self):
        for action, phrases in PHRASES.items():
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    self.assertEqual(parse_command(phrase), action)

    def test_polite_asr_punctuation(self):
        self.assertEqual(parse_command(" 请帮我 打开夹爪吧。 "), "open_gripper")

    def test_curated_asr_confusions(self):
        cases = {
            "打开假爪": "open_gripper",
            "打 开 假 找": "open_gripper",
            "请帮我抓取篮色鸡木吧": "grasp_blue",
            "抓去率色积木": "grasp_green",
        }
        for text, action in cases.items():
            with self.subTest(text=text):
                match = match_command(text)
                self.assertEqual(match.action, action)
                self.assertEqual(match.quality, "curated")
                self.assertTrue(match.auto_executable)

    def test_weak_fuzzy_requires_confirmation(self):
        match = match_command("打开夹爪子")
        self.assertEqual(match.action, "open_gripper")
        self.assertEqual(match.quality, "fuzzy")
        self.assertFalse(match.auto_executable)

    def test_extracts_one_command_from_unrelated_speech_for_confirmation(self):
        match = match_command("嗯现场有点吵请打开假爪谢谢")
        self.assertEqual(match.action, "open_gripper")
        self.assertEqual(match.normalized, "打开夹爪")
        self.assertEqual(match.quality, "extracted")
        self.assertFalse(match.auto_executable)

    def test_benign_words_inside_color_command(self):
        self.assertEqual(parse_command("请帮我抓一下蓝色积木吧"), "grasp_blue")

    def test_reported_embedded_command_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_command("刚才有人说打开夹爪")

    def test_reject_ambiguous_or_unsafe_text(self):
        for text in ("", "不要打开夹爪", "请不要开始抓取", "打开夹爪然后关闭夹爪",
                     "打开夹爪，关闭夹爪", "打开夹爪？", "停止", "抓红色杯子",
                     "打开夹爪或者拍照", "print('打开夹爪')", "打开夹爪\n日志",
                     "开始抓取吗", "关闭夹爪不对打开夹爪", "他说打开夹爪",
                     "电视里说抓取蓝色积木"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_command(text)

    def test_dispatch(self):
        methods = {"capture": "capture_frame", "predict": "predict_grasp_pose",
                   "grasp": "predict_and_grasp", "observe": "move_to_observe"}
        for action, method in methods.items():
            target = self.target()
            self.assertEqual(execute_command(target, PHRASES[action][0]), action)
            getattr(target, method).assert_called_once_with()

    def test_color_grasp_dispatch(self):
        target = self.target()
        execute_command(target, "抓取蓝色积木")
        target.grasp_color.assert_called_once_with("blue")
        target = self.target()
        execute_command(target, "抓取绿色积木")
        target.grasp_color.assert_called_once_with("green")

    def test_explicit_gripper_not_toggle(self):
        target = self.target()
        execute_command(target, "打开夹爪")
        execute_command(target, "打开夹爪")
        self.assertEqual(target.set_gripper_open.call_count, 2)
        target.set_gripper_open.assert_called_with(True)
        execute_command(target, "关闭夹爪")
        target.set_gripper_open.assert_called_with(False)
        target.trigger_gripper.assert_not_called()

    def test_busy_rejects_every_action(self):
        for phrases in PHRASES.values():
            target = self.target()
            target.voice_is_busy.return_value = True
            with self.assertRaises(ValueError):
                execute_command(target, phrases[0])
            self.assertEqual(len(target.mock_calls), 1)

    def test_selection_required(self):
        for text in ("预测抓取", "开始抓取"):
            target = self.target()
            target.voice_has_selection.return_value = False
            with self.assertRaises(ValueError):
                execute_command(target, text)
            target.predict_and_grasp.assert_not_called()
            target.predict_grasp_pose.assert_not_called()

    def test_failed_prediction_never_starts_grasp(self):
        for result in (False, None):
            target = self.target()
            target.predict_grasp_pose.return_value = result
            interface_method("predict_and_grasp")(target)
            target.robot_grasp_thread.start.assert_not_called()
            target.robot_grasp_thread.set_predicted_info.assert_not_called()

    def test_busy_never_predicts_or_starts_grasp(self):
        target = self.target()
        target.voice_is_busy.return_value = True
        interface_method("predict_and_grasp")(target)
        target.predict_grasp_pose.assert_not_called()
        target.robot_grasp_thread.start.assert_not_called()

    def test_successful_grasp_copies_prediction_and_invalidates_target(self):
        target = self.target()
        target.predict_lock = Lock()
        target.predicted_info = {"trans": [1, 2, 3], "orient": [0, 0, 0, 1]}
        target.predict_grasp_pose.return_value = True
        interface_method("predict_and_grasp")(target)
        passed = target.robot_grasp_thread.set_predicted_info.call_args.args[0]
        self.assertEqual(passed, target.predicted_info)
        self.assertIsNot(passed, target.predicted_info)
        target.invalidate_selection.assert_called_once_with(discard_frame=True)
        target.robot_grasp_thread.start.assert_called_once_with()

    def test_invalidation_clears_mask_and_prediction(self):
        target = self.target()
        target.captured_frame_lock = Lock()
        target.predict_lock = Lock()
        target.predicted_info = {"trans": [1, 2, 3], "o_width": 0.02}
        interface_method("invalidate_selection")(target, discard_frame=True)
        self.assertIsNone(target.mask_map)
        self.assertIsNone(target.captured_color)
        self.assertEqual(target.predicted_info, {"trans": None, "o_width": None})

    def test_failed_capture_invalidates_old_target(self):
        target = self.target()
        target.observe_frame_lock = Lock()
        target.captured_frame_lock = Lock()
        target.color_frame = None
        interface_method("capture_frame")(target)
        target.invalidate_selection.assert_called_once_with(discard_frame=True)
        target.update_capture_view.assert_not_called()

    def test_prediction_without_selection_fails_before_inference(self):
        target = self.target()
        target.captured_frame_lock = Lock()
        target.mask_map = None
        self.assertFalse(interface_method("predict_grasp_pose")(target))
        target.airbot_grasp.inference.assert_not_called()

    def test_gripper_sdk_receives_explicit_position(self):
        target = self.target()
        target.gripper_width = 0.072
        target.robot_operation_lock = Lock()
        method = interface_method("set_gripper_open", {"time": Mock(), "gripper_delay": 0})
        method(target, True)
        target.robot.move_gripper.assert_called_with(0.072)
        self.assertTrue(target.gripper_opened)
        method(target, False)
        target.robot.move_gripper.assert_called_with(0)
        self.assertFalse(target.gripper_opened)

    def test_gripper_busy_never_connects_sdk(self):
        target = self.target()
        target.voice_is_busy.return_value = True
        interface_method("set_gripper_open")(target, True)
        target.robot.move_gripper.assert_not_called()

    def test_color_prediction_never_starts_motion(self):
        prepared=interface_method('start_prepared_color_grasp', {'time': time})
        for execute in (False, True):
            target=self.target()
            target.predict_lock=Lock()
            target.preparation_timeout=8.
            info={'snapshot_timestamp':time.monotonic(), 'execute_motion':execute,
                  'trans':[.3,0,.02]}
            prepared(target,info,object())
            target.update_capture_view.assert_called_once()
            if execute:
                target.robot_grasp_thread.start.assert_called_once()
            else:
                target.robot_grasp_thread.start.assert_not_called()
                target.robot_grasp_thread.set_predicted_info.assert_not_called()
                self.assertEqual(target.robot.mock_calls, [])
                target.tabs.setCurrentIndex.assert_called_once_with(1)

    def test_manual_prediction_button_does_not_move_robot(self):
        method=interface_method('request_manual_prediction')
        for busy in (False, True):
            target=self.target()
            target.voice_is_busy.return_value=busy
            target.predict_grasp_pose.return_value=True
            method(target)
            if busy:
                target.predict_grasp_pose.assert_not_called()
            else:
                target.predict_grasp_pose.assert_called_once()
            target.move_to_predicted.assert_not_called()
            self.assertEqual(target.robot.mock_calls, [])
        source=(Path(__file__).resolve().parents[1]/'app'/'airbot_interface.py').read_text()
        self.assertIn('self.predict_btn.clicked.connect(self.request_manual_prediction)', source)
        self.assertNotIn('self.predict_btn.clicked.connect(self.move_to_predicted)', source)

    def test_expired_color_prediction_cannot_start_motion(self):
        prepared=interface_method('start_prepared_color_grasp', {'time': time})
        target=self.target(); target.preparation_timeout=1.
        prepared(target,{'snapshot_timestamp':time.monotonic()-3, 'execute_motion':False},object())
        target.color_grasp_failed.assert_called_once()
        target.robot_grasp_thread.start.assert_not_called()
        target.update_capture_view.assert_not_called()

    def test_color_grasp_rejections_never_start_preparation(self):
        method = interface_method("grasp_color", {
            "time": time, "resolve_unique_target": resolve_unique_target,
            "COLOR_LABELS": COLOR_LABELS,
        })
        cases = [
            ([], 0.0),
            ([Detection((0, 0, 20, 20), "block", 0.9, "blue", 0.8)], 5.0),
            ([Detection((0, 0, 20, 20), "block", 0.9, "blue", 0.8),
              Detection((30, 0, 50, 20), "cube", 0.8, "blue", 0.7)], 0.0),
        ]
        for detections, age in cases:
            target = self.target()
            target.detection_lock = Lock()
            target.detection_freshness = 1.0
            target.latest_detection_batch = SimpleNamespace(
                detections=detections,
                snapshot=SimpleNamespace(timestamp=time.monotonic() - age))
            target.color_grasp_thread = Mock()
            target.color_grasp_thread.isRunning.return_value = False
            with self.subTest(count=len(detections), age=age), self.assertRaises(ValueError):
                method(target, "blue")
            target.color_grasp_thread.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
