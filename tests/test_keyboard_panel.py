"""Exercise keyboard dispatch without connecting real hardware."""
import unittest
from unittest.mock import Mock
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget
from keyboard_panel import KeyboardCommandPanel

class KeyboardPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.target = QWidget()
        for name in ('log', 'capture_frame', 'predict_grasp_pose', 'predict_and_grasp',
                     'grasp_color', 'set_gripper_open', 'move_to_observe'):
            setattr(self.target, name, Mock())
        self.target.voice_is_busy = Mock(return_value=False)
        self.target.voice_has_selection = Mock(return_value=False)
        self.panel = KeyboardCommandPanel(self.target)

    def tearDown(self):
        self.panel.shutdown()
        self.target.close()

    def test_enter_dispatches_once(self):
        self.panel.text.setText('抓取绿色积木')
        QTest.keyClick(self.panel.text, Qt.Key.Key_Return)
        self.target.grasp_color.assert_called_once_with('green')
        self.assertEqual(self.panel.text.text(), '')
        QTest.keyClick(self.panel.text, Qt.Key.Key_Return)
        self.target.grasp_color.assert_called_once()

    def test_button_opens_gripper(self):
        self.panel.text.setText('打开夹爪')
        self.panel.execute.click()
        self.target.set_gripper_open.assert_called_once_with(True)

    def test_busy_rejects_command(self):
        self.target.voice_is_busy.return_value = True
        self.panel.text.setText('抓取蓝色积木')
        self.panel.execute.click()
        self.target.grasp_color.assert_not_called()
        self.assertIn('正在进行', self.panel.status.text())

    def test_selection_required(self):
        self.panel.text.setText('开始抓取')
        self.panel.execute.click()
        self.target.predict_and_grasp.assert_not_called()

    def test_invalid_command(self):
        self.panel.text.setText('跳舞')
        self.panel.execute.click()
        self.target.grasp_color.assert_not_called()
        self.target.set_gripper_open.assert_not_called()

if __name__ == '__main__':
    unittest.main()
