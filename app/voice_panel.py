"""Qt panel for a persistent, local FunASR worker and safe command dispatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from PyQt6.QtCore import QProcess, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QSpinBox, QStyle,
                            QVBoxLayout)

from voice_commands import LABELS, execute_command, match_command


class VoiceCommandPanel(QGroupBox):
    listening_changed = pyqtSignal(bool)

    def __init__(self, target):
        super().__init__("语音指令", target)
        self.target = target
        self.process = None
        self.worker_ready = False
        self.listening = False
        self.cancelled = False
        self.stdout_buffer = bytearray()
        self.auto_execute_this_recording = False

        layout = QVBoxLayout(self)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("麦克风："))
        self.device = QLineEdit(os.getenv("GRASP_VOICE_DEVICE", ""))
        self.device.setPlaceholderText("默认输入")
        settings.addWidget(self.device, 1)
        settings.addWidget(QLabel("最长："))
        self.seconds = QSpinBox()
        self.seconds.setRange(2, 10)
        self.seconds.setValue(4)
        self.seconds.setSuffix(" 秒")
        settings.addWidget(self.seconds)
        layout.addLayout(settings)

        buttons = QHBoxLayout()
        self.listen = QPushButton("开始录音")
        self.listen.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.listen.setToolTip("开始按键说话，说完后会根据静音自动结束")
        self.cancel = QPushButton("取消")
        self.cancel.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogCancelButton))
        self.cancel.setToolTip("取消当前录音，不执行部分语音")
        self.listen.setEnabled(False)
        self.cancel.setEnabled(False)
        buttons.addWidget(self.listen, 2)
        buttons.addWidget(self.cancel, 1)
        layout.addLayout(buttons)

        self.text = QLineEdit()
        self.text.setPlaceholderText("可说：抓取蓝色积木 / 抓取绿色积木 / 打开夹爪")
        layout.addWidget(self.text)
        self.automatic = QCheckBox("高可信识别后直接执行")
        layout.addWidget(self.automatic)
        self.execute = QPushButton("确认并执行文字指令")
        self.execute.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogApplyButton))
        layout.addWidget(self.execute)
        self.status = QLabel("正在加载本地 FunASR 模型…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.listen.clicked.connect(self.start_recording)
        self.cancel.clicked.connect(self.cancel_recording)
        self.execute.clicked.connect(self.execute_text)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(self.on_timeout)
        QTimer.singleShot(0, self.start_worker)

    def report(self, message):
        self.status.setText(message)
        self.target.log(message)

    def start_worker(self):
        if self.process is not None:
            return
        python = Path(os.getenv("GRASP_VOICE_PYTHON", sys.executable)).expanduser()
        worker = Path(os.getenv("GRASP_VOICE_WORKER", str(
            Path(__file__).resolve().with_name("voice_asr_worker.py")))).expanduser()
        if not python.is_file() or not worker.is_file():
            self.report("找不到语音 Python 或常驻工作脚本。")
            return
        process = QProcess(self)
        self.process = process
        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.errorOccurred.connect(self.process_error)
        process.finished.connect(self.worker_finished)
        process.start(str(python), ["-u", str(worker.resolve()), "--asr-device", "cpu"])

    def send(self, payload):
        if self.process is None or self.process.state() != QProcess.ProcessState.Running:
            raise RuntimeError("语音工作进程未运行")
        self.process.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

    def set_listening(self, listening):
        self.listening = listening
        self.listen.setEnabled(self.worker_ready and not listening)
        self.cancel.setEnabled(listening)
        self.device.setEnabled(not listening)
        self.seconds.setEnabled(not listening)
        self.text.setEnabled(not listening)
        self.execute.setEnabled(not listening)
        self.automatic.setEnabled(not listening)
        self.listening_changed.emit(listening)

    def start_recording(self):
        if self.listening:
            return
        if not self.worker_ready:
            self.report("语音模型尚未就绪。")
            return
        if self.target.voice_is_busy():
            self.report("机械臂或抓取计算正在进行，请稍候。")
            return
        self.text.clear()
        self.cancelled = False
        self.auto_execute_this_recording = self.automatic.isChecked()
        self.set_listening(True)
        self.report("请说话…说完后短暂停顿即可。")
        payload = {"command": "record", "device": self.device.text().strip() or None,
                   "max_seconds": self.seconds.value(), "silence_seconds": 0.8}
        try:
            self.send(payload)
            self.timeout.start((self.seconds.value() + 60) * 1000)
        except Exception as exc:
            self.set_listening(False)
            self.report(str(exc))

    def read_stdout(self):
        if self.process is None:
            return
        self.stdout_buffer.extend(bytes(self.process.readAllStandardOutput()))
        while b"\n" in self.stdout_buffer:
            raw, _, remainder = self.stdout_buffer.partition(b"\n")
            self.stdout_buffer = bytearray(remainder)
            if not raw.strip():
                continue
            try:
                message = json.loads(raw.decode("utf-8"))
                if not isinstance(message, dict) or not isinstance(message.get("event"), str):
                    raise ValueError
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self.report("语音工作进程返回了无效协议消息，本次不执行。")
                continue
            self.handle_message(message)

    def read_stderr(self):
        if self.process is not None:
            # Model diagnostics stay out of the command protocol. They remain
            # available in the terminal without becoming executable text.
            text = bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
            if text.strip():
                print(text, end="", file=os.sys.stderr)

    def handle_message(self, message):
        event = message["event"]
        if event == "ready":
            self.worker_ready = True
            self.listen.setEnabled(True)
            self.status.setText("语音模型已就绪")
        elif event == "listening":
            self.status.setText("正在录音，请说指令…")
        elif event == "transcribing":
            self.status.setText("录音已结束，正在识别…")
        elif event == "transcript":
            self.finish_transcript(str(message.get("text", "")))
        elif event == "no_speech":
            self.finish_recording()
            self.report("未检测到有效说话，本次未执行。")
        elif event == "cancelled":
            self.finish_recording()
            self.report("已取消本次语音，不会执行。")
        elif event in ("error", "fatal"):
            self.finish_recording()
            if event == "fatal":
                self.worker_ready = False
                self.listen.setEnabled(False)
            self.report("语音识别失败：" + str(message.get("message", "未知错误")))
        else:
            self.report("语音工作进程返回未知事件，本次不执行。")

    def finish_recording(self):
        self.timeout.stop()
        self.set_listening(False)

    def finish_transcript(self, text):
        self.finish_recording()
        if self.cancelled:
            return
        self.text.setText(text)
        try:
            match = match_command(text)
        except ValueError as exc:
            self.report(f"识别到：{text}\n{exc}")
            return
        quality = {"exact": "精确", "curated": "已校正常见错词",
                   "fuzzy": "弱模糊，需确认",
                   "extracted": "已从背景语音中提取，需确认"}[match.quality]
        self.report(f"识别到：{text} → {LABELS[match.action]}（{quality}）")
        if self.auto_execute_this_recording and match.auto_executable:
            self.execute_text()
        elif self.auto_execute_this_recording and not match.auto_executable:
            self.report("非精确匹配不会自动执行，请核对文字后手动确认。")

    def execute_text(self):
        if self.listening:
            return
        try:
            match = match_command(self.text.text())
            execute_command(self.target, self.text.text())
        except Exception as exc:
            self.report(str(exc))
            return
        self.report("已调用指令入口：" + LABELS[match.action] + "；执行结果见日志。")
        self.text.clear()

    def cancel_recording(self):
        if self.listening:
            self.cancelled = True
            try:
                self.send({"command": "cancel"})
            except Exception as exc:
                self.finish_recording()
                self.report(str(exc))

    def on_timeout(self):
        self.cancel_recording()
        self.report("语音输入超时，本次未执行。")

    def process_error(self, _error):
        if self.process is not None:
            self.report("语音工作进程启动失败：" + self.process.errorString())
        self.worker_ready = False
        self.finish_recording()

    def worker_finished(self, exit_code, _exit_status):
        expected = self.process is None
        self.worker_ready = False
        self.finish_recording()
        if not expected and exit_code != 0:
            self.report(f"语音工作进程已退出（{exit_code}）。")

    def shutdown(self):
        self.timeout.stop()
        process, self.process = self.process, None
        if process is None:
            return
        if process.state() == QProcess.ProcessState.Running:
            process.write(b'{"command":"shutdown"}\n')
            process.waitForBytesWritten(500)
            if not process.waitForFinished(3500):
                process.terminate()
                if not process.waitForFinished(1000):
                    process.kill()
                    process.waitForFinished(1000)
        process.deleteLater()
