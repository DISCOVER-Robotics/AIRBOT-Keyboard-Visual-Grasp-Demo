"""Qt panel tests with a structured persistent worker; no audio or hardware."""

import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch


if "--asr-device" in sys.argv:
    print(json.dumps({"event": "ready"}), flush=True)
    slow = False
    for line in sys.stdin:
        message = json.loads(line)
        if message["command"] == "shutdown":
            sys.exit(0)
        if message["command"] == "cancel":
            slow = False
            print(json.dumps({"event": "cancelled"}), flush=True)
            continue
        if message["command"] != "record":
            print(json.dumps({"event": "error", "message": "bad command"}), flush=True)
            continue
        mode = message.get("device") or "ok"
        print(json.dumps({"event": "listening"}), flush=True)
        if mode == "slow":
            slow = True
            continue
        if mode == "error":
            print(json.dumps({"event": "error", "message": "模拟麦克风不可用"}), flush=True)
            continue
        text = "不要打开夹爪" if mode == "negative" else (
            "打开夹爪子" if mode == "fuzzy" else (
                "现场声音很杂请打开假爪谢谢" if mode == "ambient" else "打开夹爪"))
        print(json.dumps({"event": "transcribing"}), flush=True)
        print(json.dumps({"event": "transcript", "text": text}, ensure_ascii=False), flush=True)
    sys.exit(0)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PyQt6.QtWidgets import QApplication, QWidget
    from voice_panel import VoiceCommandPanel
    HAS_QT = True
except ImportError:
    HAS_QT = False


@unittest.skipUnless(HAS_QT, "需要 PyQt6")
class VoicePanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.env = patch.dict(os.environ, {
            "GRASP_VOICE_PYTHON": sys.executable,
            "GRASP_VOICE_WORKER": str(Path(__file__).resolve()),
            "GRASP_VOICE_DEVICE": "ok",
        })
        self.env.start()
        self.target = QWidget()
        for method in ("log", "capture_frame", "predict_grasp_pose", "predict_and_grasp",
                       "set_gripper_open", "move_to_observe", "grasp_color"):
            setattr(self.target, method, Mock())
        self.target.voice_is_busy = Mock(return_value=False)
        self.target.voice_has_selection = Mock(return_value=True)
        self.panel = VoiceCommandPanel(self.target)
        self.listening_events = []
        self.panel.listening_changed.connect(self.listening_events.append)
        self.wait_until(lambda: self.panel.worker_ready)

    def tearDown(self):
        self.panel.shutdown()
        self.panel.deleteLater()
        self.target.deleteLater()
        self.app.processEvents()
        self.env.stop()

    def wait_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), "异步操作未按时完成")

    def record_and_wait(self):
        self.panel.start_recording()
        self.wait_until(lambda: not self.panel.listening)

    def test_worker_stays_loaded_between_recordings(self):
        process = self.panel.process
        self.record_and_wait()
        self.assertIs(self.panel.process, process)
        self.record_and_wait()
        self.assertIs(self.panel.process, process)

    def test_default_requires_execute_click(self):
        self.record_and_wait()
        self.assertEqual(self.panel.text.text(), "打开夹爪")
        self.target.set_gripper_open.assert_not_called()
        self.panel.execute.click()
        self.target.set_gripper_open.assert_called_once_with(True)

    def test_opt_in_automatic_executes_once(self):
        self.panel.automatic.setChecked(True)
        self.record_and_wait()
        self.target.set_gripper_open.assert_called_once_with(True)

    def test_weak_fuzzy_never_auto_executes(self):
        self.panel.automatic.setChecked(True)
        self.panel.device.setText("fuzzy")
        self.record_and_wait()
        self.target.set_gripper_open.assert_not_called()
        self.assertIn("手动确认", self.panel.status.text())

    def test_embedded_background_command_never_auto_executes(self):
        self.panel.automatic.setChecked(True)
        self.panel.device.setText("ambient")
        self.record_and_wait()
        self.target.set_gripper_open.assert_not_called()
        self.assertIn("手动确认", self.panel.status.text())

    def test_negative_transcript_never_executes(self):
        self.panel.automatic.setChecked(True)
        self.panel.device.setText("negative")
        self.record_and_wait()
        self.target.set_gripper_open.assert_not_called()
        self.assertIn("未执行", self.panel.status.text())

    def test_error_never_executes(self):
        self.panel.device.setText("error")
        self.record_and_wait()
        self.assertIn("模拟麦克风不可用", self.panel.status.text())

    def test_cancel_discards_transcript(self):
        self.panel.device.setText("slow")
        self.panel.start_recording()
        self.panel.cancel_recording()
        self.wait_until(lambda: not self.panel.listening)
        self.target.set_gripper_open.assert_not_called()
        self.assertIn("已取消", self.panel.status.text())

    def test_busy_rejects_recording(self):
        self.target.voice_is_busy.return_value = True
        self.panel.start_recording()
        self.assertFalse(self.panel.listening)

    def test_shutdown_terminates_worker(self):
        process = self.panel.process
        self.panel.shutdown()
        self.assertIsNone(self.panel.process)
        self.assertEqual(process.state(), process.ProcessState.NotRunning)


if __name__ == "__main__":
    unittest.main()
