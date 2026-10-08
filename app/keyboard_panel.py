"""Keyboard command panel used by the keyboard-only real robot demo."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from voice_commands import LABELS, execute_command, match_command


class KeyboardCommandPanel(QGroupBox):
    """Drop-in replacement for VoiceCommandPanel without ASR or microphone state."""

    listening_changed = pyqtSignal(bool)

    def __init__(self, target):
        super().__init__("键盘指令", target)
        self.target = target
        self.listening = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("输入一条指令后按回车或点击执行。"))
        self.text = QLineEdit()
        self.text.setPlaceholderText("抓取蓝色积木 / 抓取绿色积木 / 打开夹爪")
        self.text.returnPressed.connect(self.execute_text)
        layout.addWidget(self.text)

        buttons = QHBoxLayout()
        self.execute = QPushButton("确认并执行")
        self.execute.clicked.connect(self.execute_text)
        self.clear = QPushButton("清空")
        self.clear.clicked.connect(self.text.clear)
        buttons.addWidget(self.execute, 2)
        buttons.addWidget(self.clear, 1)
        layout.addLayout(buttons)

        self.status = QLabel("键盘控制已就绪；未启动语音识别或麦克风。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def report(self, message):
        self.status.setText(message)
        self.target.log(message)

    def set_listening(self, listening):
        self.listening = bool(listening)
        self.text.setEnabled(not self.listening)
        self.execute.setEnabled(not self.listening)
        self.clear.setEnabled(not self.listening)
        self.listening_changed.emit(self.listening)

    def execute_text(self):
        if self.listening:
            return
        command = self.text.text().strip()
        if not command:
            self.report("请输入一条指令。")
            return
        try:
            match = match_command(command)
            if match.quality not in ("exact", "curated"):
                self.report(f"未执行：{command}\n弱模糊或背景提取结果必须人工确认后重输精确指令。")
                return
            execute_command(self.target, command)
        except Exception as exc:
            self.report(str(exc))
            return
        self.report("已调用指令入口：" + LABELS[match.action] + "；执行结果见日志。")
        self.text.clear()

    def shutdown(self):
        """Keep the same lifecycle API as VoiceCommandPanel."""
        self.set_listening(False)
