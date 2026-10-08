"""Qt worker tests; no camera, model, ASR, or robot connection."""

import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication

from airbot_interface import DetectionThread, FrameSnapshot


class FakeDetector:
    def __init__(self):
        self.sequences = []

    def detect_candidates(self, image):
        self.sequences.append(int(image[0, 0, 0]))
        return []


def snapshot(sequence):
    return FrameSnapshot(
        sequence=sequence, timestamp=time.monotonic(),
        color=np.full((2, 2, 3), sequence, dtype=np.uint8),
        depth=np.zeros((2, 2), dtype=np.uint16),
        depth_map=np.zeros((2, 2), dtype=np.uint8),
        state={"trans": [0, 0, 0], "orient": [0, 0, 0, 1]},
    )


class DetectionThreadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_pending_frame_is_replaced_by_latest(self):
        detector = FakeDetector()
        worker = DetectionThread(detector, inference_hz=100)
        batches = []
        worker.result_ready.connect(batches.append)
        worker.submit(snapshot(1))
        worker.submit(snapshot(2))
        worker.start()
        deadline = time.monotonic() + 2
        while not batches and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        worker.stop()
        self.assertTrue(batches)
        self.assertEqual(detector.sequences[0], 2)
        self.assertEqual(batches[0].snapshot.sequence, 2)


if __name__ == "__main__":
    unittest.main()
