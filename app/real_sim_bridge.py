"""Qt bridge: single latest-frame worker and isolated, read-only mirror process."""
import json
import os
from pathlib import Path
import sys
from threading import Condition
import time

from PyQt6.QtCore import QObject, QProcess, QThread, pyqtSignal
from real_sim_sync import MirrorCalibration, measure_blocks


class MeasurementWorker(QThread):
    result = pyqtSignal(object)

    def __init__(self, calibration, depth_factor):
        super().__init__()
        self.calibration, self.depth_factor = calibration, depth_factor
        self.condition = Condition()
        self.pending = None
        self.stopping = False

    def submit(self, batch):
        with self.condition:
            self.pending = batch
            self.condition.notify()

    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                batch, self.pending = self.pending, None
            try:
                payload = measure_blocks(batch.snapshot, batch.detections, self.calibration, self.depth_factor)
            except Exception as exc:
                payload = {'event': 'blocks', 'timestamp': batch.snapshot.timestamp,
                           'blocks': [], 'rejected': [str(exc)]}
            self.result.emit(payload)

    def stop(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify()
        self.wait()


class RealSimulationBridge(QObject):
    def __init__(self, parent, config, camera, log):
        super().__init__(parent)
        if not camera.has_hardware_depth:
            raise ValueError('真机仿真镜像需要真实 RGB-D 深度；USB RGB 平面推算不能保证积木中心一致')
        self.log = log
        self.last_state = 0.
        self.last_batch = 0.
        self.closing = False
        self.buffer = bytearray()
        self.worker = MeasurementWorker(MirrorCalibration.from_config(config), camera.depth_factor)
        self.worker.result.connect(self.send)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.errorOccurred.connect(lambda _: self.log('仿真镜像进程错误：' + self.process.errorString()))
        self.process.finished.connect(self.finished)
        self.process.started.connect(lambda: self.send({'event': 'config', 'settings': config.get('SimulationMirror', {})}))
        self.worker.start()
        path = Path(__file__).with_name('real_sim_window.py').resolve()
        self.process.start(sys.executable, ['-u', str(path)])

    def send(self, payload):
        if self.closing or self.process.state() != QProcess.ProcessState.Running:
            return
        # A stalled renderer cannot accumulate an unbounded command queue or block the robot GUI.
        if self.process.bytesToWrite() > 128 * 1024:
            return
        try:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n'
        except (TypeError, ValueError):
            return
        self.process.write(data.encode('utf-8'))

    def submit_state(self, snapshot):
        now = time.monotonic()
        if now - self.last_state < .05:
            return
        self.last_state = now
        self.send({'event': 'state', 'timestamp': snapshot.timestamp,
                   'joints': snapshot.state.get('joints'), 'eef': snapshot.state.get('eef')})

    def submit_batch(self, batch):
        now = time.monotonic()
        if now - self.last_batch < .2 or self.closing:
            return
        self.last_batch = now
        self.worker.submit(batch)

    def read_output(self):
        self.buffer.extend(bytes(self.process.readAllStandardOutput()))
        while b'\n' in self.buffer:
            line, _, rest = self.buffer.partition(b'\n')
            self.buffer = bytearray(rest)
            message = line.decode('utf-8', 'replace').strip()
            if message:
                self.log('仿真镜像：' + message)

    def finished(self, code, _status):
        if not self.closing:
            self.log(f'仿真镜像窗口已关闭（{code}）；真机界面继续运行')

    def shutdown(self):
        if self.closing:
            return
        self.worker.stop()
        self.send({'event': 'shutdown'})
        self.closing = True
        if not self.process.waitForFinished(1500):
            self.process.terminate()
            if not self.process.waitForFinished(1000):
                self.process.kill()
                self.process.waitForFinished(1000)
