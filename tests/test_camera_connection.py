import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch
from airbot_camera import check_realsense_connection, CameraConnectionError, RealsenseCamera


class ConnectionTest(unittest.TestCase):
    def test_start_uses_sensor_depth_units_and_records_active_geometry(self):
        sdk = Mock()
        sdk.context.return_value.query_devices.return_value = [object()]
        cfg = sdk.pipeline.return_value.start.return_value
        cfg.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = .0001
        native = SimpleNamespace(width=80, height=60, fx=60., fy=62., ppx=39.,
                                 ppy=29., coeffs=[0.]*5, model='none')
        cfg.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value = native
        camera = RealsenseCamera.__new__(RealsenseCamera)
        camera.profile = [80, 60, 30]
        camera.intrinsic = [[60.,0,39.],[0,62.,29.],[0,0,1.]]
        camera.distortion = [0.]*5
        with patch('airbot_camera.rs', sdk):
            camera.init()
        self.assertEqual(camera.depth_factor, 10000.)
        self.assertFalse(camera.geometry_metadata['depth_hole_filling'])
        cloud = camera.create_point_cloud(np.full((60,80), 4000.))
        np.testing.assert_allclose(cloud[...,2], .4)

    def test_capture_preserves_missing_depth_without_hole_filling(self):
        camera = RealsenseCamera.__new__(RealsenseCamera)
        camera.pipeline = Mock()
        camera.aligner = Mock()
        frames = camera.aligner.process.return_value
        raw_depth = np.array([[400, 0], [0, 500]], dtype=np.uint16)
        frames.get_depth_frame.return_value.get_data.return_value = raw_depth
        frames.get_color_frame.return_value.get_data.return_value = np.zeros((2,2,3),np.uint8)
        camera.colorizer = Mock()
        camera.colorizer.colorize.return_value.get_data.return_value = np.zeros((2,2,3),np.uint8)
        np.testing.assert_array_equal(camera.get_frame('depth'), raw_depth)

    def test_missing_device_reports_actionable_error_without_starting_stream(self):
        sdk = Mock()
        sdk.context.return_value.query_devices.return_value = []
        with patch('airbot_camera.rs', sdk), self.assertRaisesRegex(CameraConnectionError, '设备数为 0'):
            check_realsense_connection()
        sdk.pipeline.assert_not_called()

    def test_present_device_only_enumerates(self):
        sdk = Mock()
        sdk.context.return_value.query_devices.return_value = [object()]
        with patch('airbot_camera.rs', sdk):
            check_realsense_connection()
        sdk.pipeline.assert_not_called()

    def test_disconnect_during_stream_start_is_explained(self):
        sdk = Mock()
        sdk.context.return_value.query_devices.return_value = [object()]
        sdk.pipeline.return_value.start.side_effect = RuntimeError('No device connected')
        camera = RealsenseCamera.__new__(RealsenseCamera)
        camera.profile = [640, 480, 30]
        with patch('airbot_camera.rs', sdk), self.assertRaisesRegex(CameraConnectionError, '视频流启动失败'):
            camera.init()
