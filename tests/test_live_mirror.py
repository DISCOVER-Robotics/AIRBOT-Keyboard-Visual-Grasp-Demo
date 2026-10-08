import json
import time
import unittest
from types import SimpleNamespace
import numpy as np
from unittest.mock import Mock
from live_mirror import load_settings, MirrorPublisher, measured_blocks
from live_mirror_window import apply_snapshot
from airbot_yolo import Detection, DEFAULT_DETECTION_CONFIG


class LiveMirrorTest(unittest.TestCase):
    def test_publisher_latest_state_and_invalid_feedback(self):
        publisher = MirrorPublisher(load_settings())
        self.addCleanup(publisher.close)
        snap = SimpleNamespace(timestamp=time.monotonic(), state={'joints': [0]*6, 'eef': [.04]})
        publisher.observe(snap)
        publisher.publish()
        self.assertEqual(json.loads(publisher.path.read_text())['state']['opening'], .04)
        snap.state['joints'] = []
        publisher.observe(snap)
        publisher.publish()
        self.assertIsNone(json.loads(publisher.path.read_text())['state'])

    def test_real_depth_and_extrinsic_produce_base_center(self):
        # Visible top face on an ideal aligned camera, 4 cm cube.
        rows, cols = np.indices((40, 40))
        cloud = np.stack((.3 + (cols-19.5)*.001, -.1 + (rows-19.5)*.001,
                          np.full((40, 40), .04)), axis=-1)
        camera = SimpleNamespace(has_hardware_depth=True, create_point_cloud=lambda *a, **kw: cloud)
        snap = SimpleNamespace(color=np.full((40,40,3), [255,0,0], np.uint8),
                               depth=np.ones((40,40)), timestamp=time.monotonic(),
                               state={'trans': [0,0,0], 'orient':[0,0,0,1]})
        d = Detection((0,0,40,40), 'block', .9, 'blue', 1.)
        batch = SimpleNamespace(snapshot=snap, detections=[d])
        result = measured_blocks(batch, camera, np.eye(4), DEFAULT_DETECTION_CONFIG, {**load_settings(), "block_size": [.04]*3})
        np.testing.assert_allclose(result['blue']['position'], [.3,-.1,.02], atol=.001)
        batch.detections = [d,d]
        self.assertEqual(measured_blocks(batch,camera,np.eye(4),DEFAULT_DETECTION_CONFIG,{**load_settings(), "block_size": [.04]*3}), {})
        batch.detections = [d]
        snap.depth[:] = 0
        self.assertEqual(measured_blocks(batch,camera,np.eye(4),DEFAULT_DETECTION_CONFIG,{**load_settings(), "block_size": [.04]*3}), {})

    def test_replay_does_not_step_physics_or_guess_occluded_objects(self):
        settings = load_settings()
        joints = {name: SimpleNamespace(qpos=np.zeros(n)) for name,n in
                  [('endleft',1),('endright',1),('blue_free',7),('green_free',7)]}
        geoms = {color+'_cube': SimpleNamespace(rgba=np.ones(4)) for color in ('blue','green')}
        sim = SimpleNamespace(colors=('blue','green'), model=SimpleNamespace(geom=geoms.__getitem__),
                              data=SimpleNamespace(qpos=np.zeros(8),qvel=np.zeros(8),joint=joints.__getitem__),
                              mujoco=Mock())
        packet = {'state': {'joints':[.1]*6,'opening':.04,'timestamp':10.},
                  'objects':{'blue':{'position':[.3,-.1,.02],'timestamp':10.}}}
        valid, visible = apply_snapshot(sim,packet,settings,now=10.1)
        self.assertTrue(valid)
        self.assertEqual(visible,['blue'])
        np.testing.assert_allclose(sim.data.qpos[:6],[.1]*6)
        self.assertEqual(joints['endleft'].qpos[0],.02)
        self.assertEqual(geoms['green_cube'].rgba[3],0)
        sim.mujoco.mj_step.assert_not_called()
        valid, visible = apply_snapshot(sim,packet,settings,now=12.)
        self.assertFalse(valid)
        self.assertEqual(visible,[])
        self.assertEqual(geoms['blue_cube'].rgba[3],0)

    def test_oblique_small_top_is_visible_as_estimate_not_silently_dropped(self):
        rows, cols = np.indices((40, 40))
        cloud = np.stack((.3 + (cols-19.5)*.0003, -.1 + (rows-19.5)*.001,
                          np.full((40, 40), .04)), axis=-1)
        camera = SimpleNamespace(has_hardware_depth=True, create_point_cloud=lambda *a, **kw: cloud)
        snap = SimpleNamespace(color=np.full((40,40,3), [255,0,0], np.uint8),
                               depth=np.ones((40,40)), timestamp=10.,
                               state={'trans': [0,0,0], 'orient':[0,0,0,1]})
        batch = SimpleNamespace(snapshot=snap, detections=[Detection((0,0,40,40), 'block', .9, 'blue', 1.)])
        diagnostics = {}
        result = measured_blocks(batch,camera,np.eye(4),DEFAULT_DETECTION_CONFIG,{**load_settings(), "block_size": [.04]*3},
                                 diagnostics=diagnostics)
        self.assertEqual(result['blue']['quality'], 'estimated')
        np.testing.assert_allclose(result['blue']['position'], [.3,-.1,.02], atol=.001)
        self.assertIn('半透明', diagnostics['blue'])
        snap.depth[:] = 0
        self.assertEqual(measured_blocks(batch,camera,np.eye(4),DEFAULT_DETECTION_CONFIG,{**load_settings(), "block_size": [.04]*3},
                                        diagnostics=diagnostics), {})
        self.assertIn('深度点不足', diagnostics['blue'])

    def test_display_explains_stale_and_missing_objects(self):
        from live_mirror_window import object_status
        packet = {'objects': {'blue': {'position':[.3,-.1,.02], 'timestamp':10., 'quality':'estimated'}},
                  'diagnostics': {'green':'有效彩色深度点不足'}}
        text = object_status(packet, load_settings(), now=10.5)
        self.assertIn('半透明', text)
        self.assertIn('深度点不足', text)
        self.assertIn('已过期', object_status(packet, load_settings(), now=14.))
