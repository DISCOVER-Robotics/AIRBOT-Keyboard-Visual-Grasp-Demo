import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from grasp_preview import project_base_points, draw_grasp_preview, target_mask_clearance_mm


class PreviewTest(unittest.TestCase):
    def test_projects_known_camera_point_through_rotated_translated_frames(self):
        intrinsic=np.array([[200.,0,100],[0,200,80],[0,0,1]])
        end=np.eye(4)
        end[:3,:3]=Rotation.from_euler('xyz',[20,30,40],degrees=True).as_matrix()
        end[:3,3]=[.2,.1,.4]
        extrinsic=np.eye(4)
        extrinsic[:3,:3]=Rotation.from_euler('y',35,degrees=True).as_matrix()
        extrinsic[:3,3]=[.04,-.02,.03]
        transform=end@extrinsic
        points=np.array([[0,0,.5],[.025,-.05,.5]])@transform[:3,:3].T+transform[:3,3]
        pose=[end[:3,3],Rotation.from_matrix(end[:3,:3]).as_quat()]
        pixels=project_base_points(points,pose,extrinsic,intrinsic,np.zeros(5))
        np.testing.assert_allclose(pixels,[[100,80],[110,60]],atol=1e-8)

    def test_draw_preserves_input_and_does_not_fill_mask(self):
        image=np.full((100,100,3),80,np.uint8); mask=np.zeros((100,100),bool)
        mask[20:80,20:80]=True
        original=image.copy()
        preview,pixels=draw_grasp_preview(image,mask,[[0,0,.5],[.04,0,.5]],
            [[0,0,0],[0,0,0,1]],np.eye(4),[[100,0,50],[0,100,50],[0,0,1]],np.zeros(5))
        np.testing.assert_array_equal(image,original)
        # Text/markers may cross the mask interior, but the object itself
        # must remain visible instead of being painted with an opaque fill.
        unchanged = np.all(preview == original, axis=-1)
        self.assertGreater(np.mean(unchanged[mask]), .70)
        np.testing.assert_allclose(pixels,[[50,50],[58,50]])
        self.assertTrue(np.any(preview!=image))

    def test_behind_camera_is_rejected(self):
        with self.assertRaises(ValueError):
            project_base_points([[0,0,-1]],[[0,0,0],[0,0,0,1]],np.eye(4),np.eye(3),np.zeros(5))

    def test_mask_clearance_uses_projected_metric_scale(self):
        mask=np.zeros((100,100),bool); mask[25:75,25:75]=True
        args=([[0,0,0],[0,0,0,1]],np.eye(4),
              [[1000,0,50],[0,1000,50],[0,0,1]],np.zeros(5))
        center=target_mask_clearance_mm(mask,[0,0,1],*args)
        edge=target_mask_clearance_mm(mask,[.024,0,1],*args)
        self.assertGreater(center,20.)
        self.assertLess(edge,2.)
        self.assertEqual(target_mask_clearance_mm(mask,[.03,0,1],*args),0.)
