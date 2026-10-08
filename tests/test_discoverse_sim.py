"""Optional integration tests against real DISCOVERSE/MuJoCo physics (no GL)."""
import importlib.util
import unittest

import numpy as np

from discoverse_voice import SimulationCommands
from voice_commands import execute_command

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('discoverse', 'mujoco'))


@unittest.skipUnless(AVAILABLE, 'run ./install_sim.sh to install simulation dependencies')
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from discoverse_sim import DiscoverseSimulation
        cls.sim = DiscoverseSimulation()
        cls.target = SimulationCommands()
        cls.target.sim = cls.sim

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset()

    def test_both_colors_are_physically_lifted_and_placed(self):
        for color, phrase in [('blue', '抓取蓝色积木'), ('green', '抓取绿色积木')]:
            with self.subTest(color=color):
                self.sim.reset()
                other = 'green' if color == 'blue' else 'blue'
                original_other = self.sim.position(other)
                execute_command(self.target, phrase)
                maximum_z = 0.
                for _ in range(4000):
                    self.sim.tick()
                    maximum_z = max(maximum_z, self.sim.position(color)[2])
                    if not self.sim.busy:
                        break
                self.assertFalse(self.sim.busy)
                self.assertIsNone(self.sim.last_error)
                self.assertGreater(maximum_z, .09)
                np.testing.assert_allclose(self.sim.position(color)[:2], self.sim.place[:2], atol=.045)
                self.assertLess(self.sim.position(color)[2], .035)
                np.testing.assert_allclose(self.sim.position(other), original_other, atol=.005)
                with self.assertRaisesRegex(ValueError, '放置区'):
                    self.sim.grasp(other)

    def test_input_guards_and_selection_invalidation(self):
        with self.assertRaises(ValueError):
            execute_command(self.target, '不要抓取蓝色积木')
        with self.assertRaises(ValueError):
            execute_command(self.target, '开始抓取')
        self.sim.select('blue')
        execute_command(self.target, '预测抓取')
        self.assertFalse(self.sim.busy)
        self.assertIsNotNone(self.sim.predicted)
        execute_command(self.target, '拍照')
        self.assertIsNone(self.sim.selected)
        self.assertIsNone(self.sim.predicted)
        execute_command(self.target, '打开夹爪')
        with self.assertRaises(ValueError):
            execute_command(self.target, '关闭夹爪')
        with self.assertRaises(ValueError):
            self.sim.reset()
        self.sim.run_until_idle()
        self.assertGreater(self.sim.data.joint('endleft').qpos[0], .035)
        execute_command(self.target, '关闭夹爪')
        self.sim.run_until_idle()
        self.assertLess(self.sim.data.joint('endleft').qpos[0], .005)
        execute_command(self.target, '回到观察位')
        self.sim.run_until_idle()
        np.testing.assert_allclose(self.sim.data.body('link6').xpos, self.sim.observe, atol=.02)

    def test_expired_preparation_never_starts_motion(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        import time
        self.sim.vision = SimpleNamespace(settings={'preparation_timeout_seconds': 8.})
        future = Future()
        future.set_result(({'snapshot_timestamp': time.monotonic() - 20}, None))
        self.sim.preparation = future
        self.sim.preparation_execute = True
        try:
            self.sim.tick()
            self.assertIsNone(self.sim.motion)
            self.assertIn('过期', self.sim.last_error)
            self.assertIsNone(self.sim.prepared_info)
        finally:
            self.sim.vision = None

    def test_missing_grasp_does_not_report_success(self):
        execute_command(self.target, '抓取蓝色积木')
        # Move the free body away AFTER target capture; no object to grip there.
        self.sim.data.joint('blue_free').qpos[:3] = [.6, -.3, .02]
        self.sim.mujoco.mj_forward(self.sim.model, self.sim.data)
        with self.assertRaisesRegex(RuntimeError, '夹持超时'):
            self.sim.run_until_idle()
        self.assertFalse(self.sim.busy)
        np.testing.assert_allclose(self.sim.data.body('link6').xpos, self.sim.observe, atol=.02)
        self.assertTrue(any('失败恢复' in event for event in self.sim.events))
        self.assertIn('任务失败', self.sim.message)
        self.assertGreater(np.linalg.norm(self.sim.position('blue')[:2] - self.sim.place[:2]), .2)


if __name__ == '__main__':
    unittest.main()
