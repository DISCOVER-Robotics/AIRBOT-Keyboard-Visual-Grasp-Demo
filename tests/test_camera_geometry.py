import unittest

import cv2
import numpy as np

from airbot_camera import RealsenseCamera
from camera_geometry import calibrated_point_cloud, camera_xy_offset_in_base, position_camera_offset
from scipy.spatial.transform import Rotation


class CameraGeometryTest(unittest.TestCase):
    def test_position_offsets_interpolate_totals_and_do_not_extrapolate(self):
        profile = {'image_x_fractions': [.4, .6],
                   'offsets_camera_xy_m': [[-.005,.006],[-.006,.019]]}
        for fraction, expected in [(0,[-.005,.006]), (.4,[-.005,.006]),
                                   (.5,[-.0055,.0125]), (.6,[-.006,.019]),
                                   (1,[-.006,.019])]:
            np.testing.assert_allclose(position_camera_offset(profile, fraction), expected)
        for anchors in ([.7,.3], [.5,.5], [-1,.5], [.3,float('nan')]):
            with self.assertRaises(ValueError):
                position_camera_offset(dict(profile, image_x_fractions=anchors), .5)

    def test_camera_offset_preserves_requested_components_and_base_height(self):
        for yaw in (0, 60, -40):
            end = Rotation.from_euler('z', yaw, degrees=True)
            extrinsic = np.eye(4)
            extrinsic[:3,:3] = Rotation.from_euler('x', 150, degrees=True).as_matrix()
            offset = camera_xy_offset_in_base([-.003,.006], [[0,0,0],end.as_quat()], extrinsic)
            camera_delta = (end.as_matrix() @ extrinsic[:3,:3]).T @ np.r_[offset,0.]
            np.testing.assert_allclose(camera_delta[:2],[-.003,.006],atol=1e-12)

    def test_horizontal_view_rejects_unreliable_offset(self):
        extrinsic=np.eye(4)
        extrinsic[:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError,'倾斜'):
            camera_xy_offset_in_base([-.003,.006],[[0,0,0],[0,0,0,1]],extrinsic)

    def setUp(self):
        self.camera = RealsenseCamera.__new__(RealsenseCamera)
        self.camera.profile = [80, 60, 30]
        self.camera.intrinsic = np.array([[60., 0., 39.], [0., 62., 29.], [0., 0., 1.]])
        self.camera.distortion = [.14, -.12, .001, -.001, .04]
        self.camera.depth_factor = 1000.

    def test_distorted_rays_project_back_to_same_pixels(self):
        # Forward project independently: corrected 3D must map to original
        # RGB/SAM indices on both sides of the image, with unchanged axial Z.
        depth = np.full((60, 80), 400., dtype=np.float32)
        cloud = self.camera.create_point_cloud(depth)
        pixels, _ = cv2.projectPoints(cloud.reshape(-1, 3), np.zeros(3), np.zeros(3),
                                      self.camera.intrinsic, np.array(self.camera.distortion))
        y, x = np.indices(depth.shape)
        np.testing.assert_allclose(pixels.reshape(60, 80, 2), np.stack((x, y), -1), atol=.001)
        np.testing.assert_allclose(cloud[..., 2], .4)
        np.testing.assert_allclose(self.camera.create_point_cloud(depth, organized=False),
                                   cloud.reshape(-1, 3))

    def test_zero_distortion_preserves_pinhole_and_invalid_depth(self):
        self.camera.distortion = [[0., 0., 0., 0., 0.]]
        depth = np.full((60, 80), 500.)
        depth[0, 0] = 0
        depth[1, 1] = np.nan
        cloud = self.camera.create_point_cloud(depth)
        y, x = np.indices(depth.shape)
        expected = np.stack(((x-39)/60, (y-29)/62, np.ones(depth.shape)), -1) * (depth/1000)[..., None]
        np.testing.assert_allclose(cloud, expected, equal_nan=True)
        np.testing.assert_array_equal(cloud[0, 0], [0, 0, 0])

    def test_wrong_resolution_and_invalid_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError, '分辨率'):
            self.camera.create_point_cloud(np.ones((30, 40)))
        for distortion, factor in [([np.nan]*5, 1000), ([0]*3, 1000), ([0]*5, 0)]:
            with self.assertRaises(ValueError):
                calibrated_point_cloud(np.ones((60, 80)), self.camera.intrinsic,
                                       distortion, factor, self.camera.profile)

    def test_calibration_change_does_not_reuse_old_rays(self):
        depth = np.full((60, 80), 400.)
        first = self.camera.create_point_cloud(depth)
        self.camera.distortion = [0.]*5
        second = self.camera.create_point_cloud(depth)
        self.assertGreater(np.max(np.abs(first-second)), .001)
        self.camera.intrinsic[0, 2] += 2
        third = self.camera.create_point_cloud(depth)
        self.assertGreater(np.max(np.abs(second-third)), .001)
