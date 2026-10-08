import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from block_geometry import fit_block_top, fit_known_cube
from airbot_grasp_simple import SimpleGrasp


class BlockGeometryTest(unittest.TestCase):
    def top(self, degrees, size=(.04, .04)):
        x, y = np.meshgrid(np.linspace(-size[0]/2, size[0]/2, 45),
                           np.linspace(-size[1]/2, size[1]/2, 45))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        return points @ Rotation.from_euler('z', degrees, degrees=True).as_matrix().T + [.3, -.1, .04]

    def test_square_edges_not_diagonals_with_noise_and_side(self):
        rng = np.random.default_rng(3)
        for degrees in [0, 17, 44, 71, 89]:
            points = self.top(degrees)
            side = points[:45].repeat(12, axis=0)
            side[:, 2] = np.tile(np.linspace(.005, .035, 12), 45)
            cloud = np.vstack((points, side)) + rng.normal(0, .0002, (len(points)+len(side), 3))
            center, height, rotation, width = fit_block_top(cloud, [0, 1, 0])
            np.testing.assert_allclose(center, [.3, -.1], atol=.001)
            self.assertAlmostEqual(height, .04, delta=.001)
            self.assertAlmostEqual(width, .04, delta=.002)
            edge = Rotation.from_euler('z', degrees, degrees=True).as_matrix()
            self.assertGreater(np.max(np.abs(edge[:, :2].T @ rotation[:, 1])), .999)
            np.testing.assert_allclose(rotation[:, 0], [0, 0, -1])
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_oblique_camera_chain_uses_base_edges(self):
        grasp = SimpleGrasp()
        grasp.dump = False
        grasp.grasp_offset_base_xy_m = np.zeros(2)
        grasp.grasp_offset_camera_xy_m = np.zeros(2)
        grasp.position_offset = {'enabled': False}
        end_rotation = Rotation.from_euler('xyz', [23, 38, -27], degrees=True)
        pose = [[.23, .02, .3], end_rotation.as_quat()]
        end = np.eye(4)
        end[:3, :3] = end_rotation.as_matrix()
        end[:3, 3] = pose[0]
        # A downward camera with an oblique roll/yaw relative to the tool.
        cam = np.eye(4)
        cam[:3, :3] = Rotation.from_euler('xyz', [160, 10, 31], degrees=True).as_matrix()
        cam[:3, 3] = [.3, -.1, .35]
        grasp.cam2end = np.linalg.inv(end) @ cam
        base = self.top(27, (.025, .025))
        camera = (base - cam[:3, 3]) @ cam[:3, :3]
        xyz = camera.reshape(45, 45, 3)
        trans, quat, _ = grasp.inference(np.zeros((45,45,3), np.uint8), xyz[...,2], xyz,
                                         pose, mask=np.ones((45,45), bool), save_cloud=False)
        np.testing.assert_allclose(trans[:2], [.3,-.1], atol=.001)
        rotation = Rotation.from_quat(quat).as_matrix()
        edge = Rotation.from_euler('z', 27, degrees=True).as_matrix()
        self.assertGreater(np.max(np.abs(edge[:, :2].T @ rotation[:, 1])), .999)
        self.assertEqual(grasp.geometry_diagnostics['method'], 'known_cube_visible_faces')
        grasp.grasp_offset_base_xy_m = np.array([-.010, 0.])
        for degrees in (27, 63):
            base = self.top(degrees, (.025, .025))
            xyz = ((base - cam[:3, 3]) @ cam[:3, :3]).reshape(45, 45, 3)
            shifted, _, _ = grasp.inference(
                np.zeros((45,45,3), np.uint8), xyz[...,2], xyz, pose,
                mask=np.ones((45,45), bool), save_cloud=False)
            np.testing.assert_allclose(shifted[:2], [.29, -.1], atol=.001)
            self.assertAlmostEqual(shifted[2], trans[2])

        # Position profile replaces a fixed camera correction, and is mapped
        # through the capture camera while preserving the commanded base Z.
        grasp.grasp_offset_base_xy_m = np.zeros(2)
        grasp.grasp_offset_camera_xy_m = np.array([.1, .1])
        grasp.position_offset = {'enabled': True, 'image_x_fractions': [.4,.6],
                                 'offsets_camera_xy_m': [[-.005,.006],[-.006,.019]]}
        corrected, _, _ = grasp.inference(
            np.zeros((45,45,3), np.uint8), xyz[...,2], xyz, pose,
            mask=np.ones((45,45), bool), save_cloud=False)
        weight = ((22/45) - .4) / .2
        expected_camera = (1-weight)*np.array([-.005,.006]) + weight*np.array([-.006,.019])
        raw = np.array([*grasp.preview_points_base[0][:2], corrected[2]])
        camera_delta = cam[:3,:3].T @ (corrected - raw)
        np.testing.assert_allclose(camera_delta[:2], expected_camera, atol=1e-9)
        self.assertAlmostEqual(corrected[2], trans[2])

    def test_rectangle_uses_narrow_width_and_bad_cloud_rejected(self):
        _, _, _, width = fit_block_top(self.top(32, (.025,.06)), [0,1,0])
        self.assertAlmostEqual(width, .025, delta=.001)
        for points in [np.zeros((10,3)), np.tile([.3,.1,.04], (100,1))]:
            with self.assertRaises(ValueError):
                fit_block_top(points, [0,1,0])

    def test_high_slice_of_sloped_top_is_rejected(self):
        points = self.top(15)
        points[:, 2] += .4 * (points[:, 0] - .3)
        with self.assertRaisesRegex(ValueError, '未覆盖目标'):
            fit_block_top(points, [0, 1, 0])

    def test_cube_partial_top_and_visible_side(self):
        rng = np.random.default_rng(42)
        for degrees in (0, 27, 63):
            for partial in (False, True):
                x, y = np.meshgrid(np.linspace(-.0125 if not partial else -.004, .0125, 35),
                                   np.linspace(-.0125, .0125, 35))
                top = np.c_[x.ravel(), y.ravel(), np.full(x.size, .05)]
                y, z = np.meshgrid(np.linspace(-.0125, .0125, 35), np.linspace(.027, .048, 35))
                side = np.c_[np.full(y.size, .0125), y.ravel(), z.ravel()]
                rot = Rotation.from_euler('z', degrees, degrees=True).as_matrix()
                cloud = np.vstack((top, side)) @ rot.T + [.3, -.1, 0]
                cloud += rng.normal(0, .00025, cloud.shape)
                camera = np.array([.25, 0, .3]) @ rot.T + [.3, -.1, 0]
                center, height, rotation, width = fit_known_cube(cloud, [0,1,0], camera)
                np.testing.assert_allclose(center, [.3,-.1], atol=.001)
                self.assertAlmostEqual(height, .05, delta=.001)
                self.assertEqual(width, .025)
                self.assertGreater(np.max(np.abs(rot[:, :2].T @ rotation[:, 1])), .998)

    def test_cube_missing_constraints_and_wrong_size_rejected(self):
        for cloud in (self.top(0, (.012,.025)), self.top(0, (.04,.04))):
            with self.assertRaises(ValueError):
                fit_known_cube(cloud, [0,1,0], [.5,-.1,.3])

    def test_cube_side_only_and_sloped_cloud_rejected(self):
        side = self.top(0, (.025, .025))
        side[:, 2] = side[:, 0] - .3 + .04
        side[:, 0] = .3125
        sloped = self.top(15, (.025, .025))
        sloped[:, 2] += .4 * (sloped[:, 0] - .3)
        for cloud in (side, sloped):
            with self.assertRaises(ValueError):
                fit_known_cube(cloud, [0,1,0], [.5,-.1,.3])

    def test_depth_outliers_do_not_bias_supported_top(self):
        rng = np.random.default_rng(73)
        top = self.top(27, (.025, .025))
        top += rng.normal(0, .0006, top.shape)
        outliers = rng.uniform([.289, -.111, .022], [.311, -.089, .032], (130, 3))
        diagnostics = {}
        center, height, rotation, _ = fit_known_cube(
            np.vstack((top, outliers)), [0,1,0], [.3,-.1,.3], diagnostics=diagnostics)
        np.testing.assert_allclose(center, [.3,-.1], atol=.0015)
        self.assertAlmostEqual(height, .04, delta=.001)
        edge = Rotation.from_euler('z', 27, degrees=True).as_matrix()
        self.assertGreater(np.max(np.abs(edge[:, :2].T @ rotation[:, 1])), .995)
        self.assertLess(diagnostics['inlier_fraction'], .95)
        with self.assertRaises(ValueError):
            fit_known_cube(np.vstack((top, rng.uniform(
                [.288,-.112,.020], [.312,-.088,.036], (1500,3)))), [0,1,0], [.3,-.1,.3])
