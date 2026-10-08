import numpy as np
import torch
import yaml
import os
import time
import json
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from block_geometry import fit_block_top, fit_known_cube
from camera_geometry import camera_xy_offset_in_base, position_camera_offset
from utils import plot_gripper, masks_to_boxes


def clamp_grasp_z(predicted_z, min_grasp_z):
    """Keep the commanded tool-center height above the configured table limit."""
    predicted_z = float(predicted_z)
    min_grasp_z = float(min_grasp_z)
    if not np.isfinite(predicted_z) or not np.isfinite(min_grasp_z):
        raise ValueError("抓取高度和最低抓取高度必须是有限数值")
    if min_grasp_z < 0:
        raise ValueError("最低抓取高度不能小于 0")
    return max(predicted_z, min_grasp_z)


class SimpleGrasp:
    def __init__(self, config=None):
        self.safe_height_bias = None
        self.base_height = None
        self.gripper_width = None
        self.gripper_length = None
        self.cam2end = None
        self.grasp_depth = None
        self.min_grasp_z = None
        self.angle = None
        self.width_m = None

        if config is None:
            with open("configs/config_file.yaml", "r") as file:
                config_path = yaml.safe_load(file)["Path"]
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

        self.robot_port = config["ArmParams"]["port"]
        self.base_height = config["ArmParams"]["base_height"]
        self.gripper_width = config["ArmParams"]["gripper_width"]
        self.gripper_length = config["ArmParams"]["gripper_length"]

        camera_type = config.get("Camera", {}).get(
            "calibration_type", config["AirbotGrasp"]["camera_type"])
        resolution = config[camera_type]["resolution"]
        self.cam2end = config[camera_type][resolution]["extrinsic"]
        self.grasp_depth = config["AirbotGrasp"]["grasp_depth"]
        self.safe_height_bias = config["AirbotGrasp"]["safe_height_bias"]
        self.min_grasp_z = config["AirbotGrasp"].get(
            "min_grasp_z", self.safe_height_bias
        )
        self.block_size_m = config["AirbotGrasp"].get("block_size_m")
        self.position_offset = config['AirbotGrasp'].get('position_offset', {})
        if self.position_offset.get('enabled', False):
            position_camera_offset(self.position_offset, .5)
        self.grasp_offset_camera_xy_m = np.asarray(
            config['AirbotGrasp'].get('grasp_offset_camera_xy_m', [0., 0.]), dtype=float)
        if (self.grasp_offset_camera_xy_m.shape != (2,)
                or not np.isfinite(self.grasp_offset_camera_xy_m).all()):
            raise ValueError('grasp_offset_camera_xy_m 必须是相机右/下方向的两个有限数值（米）')
        self.grasp_offset_base_xy_m = np.asarray(
            config["AirbotGrasp"].get("grasp_offset_base_xy_m", [0., 0.]), dtype=float)
        if (self.grasp_offset_base_xy_m.shape != (2,)
                or not np.isfinite(self.grasp_offset_base_xy_m).all()):
            raise ValueError('grasp_offset_base_xy_m 必须是基座 XY 的两个有限数值（米）')
        self.observe_pose = config["AirbotGrasp"]["observe_pose"]
        self.place_pose = config["AirbotGrasp"]["place_pose"]
        self.pre_place_pose = config["AirbotGrasp"]["pre_place_pose"]

        self.dump = config["RunTime"]["dump"]
        self.dump_path = config["RunTime"]["dump_path"]
        os.makedirs(self.dump_path, exist_ok=True)

    def cam_cloud_to_base(self, cloud_cam, end_pose):
        cloud_cam_homogeneous = np.hstack((cloud_cam, np.ones((cloud_cam.shape[0], 1))))
        gripper2base = np.eye(4)
        gripper2base[:3, :3] = Rotation.from_quat(end_pose[1]).as_matrix()
        gripper2base[:3, 3] = end_pose[0]
        cloud_base_homogeneous = (
            gripper2base @ self.cam2end @ cloud_cam_homogeneous.T
        ).T
        cloud_base = cloud_base_homogeneous[:, :3]
        return cloud_base

    def inference(self, color_image, depth_image, cloud_cam_raw, end_pose, mask=None, bbox=None, preview_cloud = False, save_cloud = True, observation=None):
        color = color_image.astype(np.float32) / 255.0
        depth = depth_image.astype(np.float32)
        # filter valid cloud
        time1 = time.time()
        valid = np.asarray(mask, dtype=bool) & np.isfinite(cloud_cam_raw).all(axis=2) & (cloud_cam_raw[..., 2] > 0)
        cloud_cam = cloud_cam_raw[valid]
        # print("cloud_cam.shape: ", cloud_cam.shape)
        cloud_color = color[valid]
        print("time mask: ", time.time() - time1)
        time1 = time.time()
        # transform to base
        cloud_base = self.cam_cloud_to_base(cloud_cam, end_pose)
        # print(f"cloud_base.shape: {cloud_base.shape}, len: {len(cloud_base)}")
        above_table = cloud_base[:, 2] > 0
        cloud_base = cloud_base[above_table]
        cloud_color = cloud_color[above_table]
        print("time cloud_base: ", time.time() - time1)
        if len(cloud_base) == 0:
            return None, None, None
        # build trans
        time1 = time.time()
        reference = Rotation.from_quat(self.observe_pose[1]).as_matrix()[:, 1]
        self.geometry_diagnostics = {}
        if self.block_size_m is None:
            center, z_top, rotation, self.width_m = fit_block_top(cloud_base, reference)
        else:
            camera_origin = self.cam_cloud_to_base(np.zeros((1, 3)), end_pose)[0]
            try:
                center, z_top, rotation, self.width_m = fit_known_cube(
                    cloud_base, reference, camera_origin, self.block_size_m,
                    diagnostics=self.geometry_diagnostics)
            except ValueError as exc:
                if self.dump:
                    prefix = os.path.join(self.dump_path, time.strftime('%Y%m%d%H%M%S') + '_rejected')
                    try:
                        np.savez_compressed(prefix + '.npz', points=cloud_base,
                                            camera_origin=camera_origin, reference_closing=reference,
                                            size_m=self.block_size_m)
                        with open(prefix + '.json', 'w', encoding='utf-8') as file:
                            json.dump(dict(error=str(exc), **self.geometry_diagnostics),
                                      file, ensure_ascii=False, indent=2)
                    except OSError as save_error:
                        print(f'保存拒绝点云失败：{save_error}')
                raise
        x, y = center
        self.angle = float(np.degrees(np.arctan2(rotation[1, 1], rotation[0, 1])))

        gripper_bottom = z_top - self.gripper_length
        predicted_z = max(gripper_bottom, self.safe_height_bias)
        z = clamp_grasp_z(predicted_z, self.min_grasp_z)
        if z > predicted_z:
            print(
                "grasp z safety clamp: "
                f"predicted={predicted_z:.6f}, commanded={z:.6f}"
            )
        trans = np.array([x, y, z])
        raw_trans = trans.copy()
        camera_offset = self.grasp_offset_camera_xy_m.copy()
        offset_x_fraction = None
        if self.position_offset.get('enabled', False):
            # Use the actual segmentation extent for all entry points; labels
            # and smoothed detector boxes must not affect the correction.
            _, mask_x = np.nonzero(mask)
            offset_x_fraction = float((mask_x.min() + mask_x.max()) / 2 / mask.shape[1])
            camera_offset = position_camera_offset(self.position_offset, offset_x_fraction)
        camera_offset_base = camera_xy_offset_in_base(
            camera_offset, end_pose, self.cam2end)
        applied_offset = self.grasp_offset_base_xy_m + camera_offset_base
        trans[:2] += applied_offset
        # Both diagnostic points lie on the fitted top plane. Neither is an
        # observed fingertip or the commanded tool-origin Z.
        self.preview_points_base = [[float(x), float(y), float(z_top)],
                                    [float(trans[0]), float(trans[1]), float(z_top)]]
        print(f'目标观测：{observation}；画面横坐标比例：{offset_x_fraction}；相机右/下补偿（mm）：{camera_offset * 1000}；'
              f'合计抓取位置补偿（基座 XY，mm）：{applied_offset * 1000}；'
              f'原始位置 {raw_trans.tolist()}；目标位置 {trans.tolist()}')
        # Tool X points down; tool Y is the jaw closing direction in base XY.
        orient = Rotation.from_matrix(rotation).as_quat()
        if preview_cloud or save_cloud:
            # visualize points cloud
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_base.astype(np.float32))
            cloud.colors = o3d.utility.Vector3dVector(cloud_color.astype(np.float32))
            # visualize gripper
            g = plot_gripper(
                trans,
                Rotation.from_quat(orient).as_matrix(),
                self.gripper_width,
                self.gripper_length,
            )
            if preview_cloud:
                o3d.visualization.draw_geometries([cloud, g])
            if self.dump:
                combined_cloud = o3d.geometry.PointCloud()
                combined_cloud.points = cloud.points
                combined_cloud.colors = cloud.colors

                # 转换gripper mesh为点云
                gripper_points = np.asarray(g.sample_points_uniformly(number_of_points=10000).points)
                gripper_colors = np.tile(np.array([0.5, 0.5, 0.5]), (gripper_points.shape[0], 1)) # 设置夹爪颜色为灰色

                combined_cloud.points.extend(o3d.utility.Vector3dVector(gripper_points))
                combined_cloud.colors.extend(o3d.utility.Vector3dVector(gripper_colors))

                file_name_prefix = os.path.join(self.dump_path, time.strftime("%Y%m%d%H%M%S", time.localtime()))

                # Keep the capture pose and transform with the raw depth so
                # camera-fixed and tool-fixed errors can be separated offline.
                with open(f"{file_name_prefix}_geometry.json", "w", encoding="utf-8") as record:
                    json.dump({
                        'observation': observation,
                        'preview_top_points_base': self.preview_points_base,
                        "capture_end_pose": [np.asarray(v).tolist() for v in end_pose],
                        "camera_to_end": np.asarray(self.cam2end).tolist(),
                        "target_position_base": trans.tolist(),
                        "raw_target_position_base": raw_trans.tolist(),
                        "grasp_offset_base_xy_m": self.grasp_offset_base_xy_m.tolist(),
                        "grasp_offset_camera_xy_m": camera_offset.tolist(),
                        "position_offset_image_x_fraction": offset_x_fraction,
                        "position_offset_profile": self.position_offset,
                        "applied_offset_base_xy_m": applied_offset.tolist(),
                        "target_orientation_xyzw": orient.tolist(),
                        "closing_direction_base": rotation[:, 1].tolist(),
                        "width_m": self.width_m,
                        "geometry_fit": self.geometry_diagnostics,
                        "top_z_base_m": z_top,
                        "gripper_length_m": self.gripper_length,
                        "note": "Commanded pose, not measured physical fingertip center",
                    }, record, ensure_ascii=False, indent=2)
                o3d.io.write_point_cloud(f"{file_name_prefix}_cloud.ply", combined_cloud)
                cv2.imwrite(f"{file_name_prefix}_mask.png", mask.astype(np.uint8) * 255)
                cv2.imwrite(f"{file_name_prefix}_color.png", color_image)
                cv2.imwrite(f"{file_name_prefix}_depth.png", depth_image.astype(np.uint16))
                masked_color = color_image.copy()
                masked_color[mask] = [191, 214, 238]
                cv2.imwrite(f"{file_name_prefix}_masked_color.png", masked_color)

        return trans, orient, cloud_base


if __name__ == "__main__":
    import cv2
    import time
    import yaml
    import os
    from airbot_arm import AirbotArm
    from airbot_camera import RealsenseCamera
    from airbot_segment import AirbotSegment

    realsense = RealsenseCamera()
    airbot_segment = AirbotSegment()
    airbot_grasp = SimpleGrasp()
    color = None
    depth = None
    mask = None
    bbox = None
    masked_color = None



    # observe pose
    robot = AirbotArm(port=airbot_grasp.robot_port)
    try:
        robot.move_end_pose(airbot_grasp.observe_pose)
    finally:
        robot.close()
    while True:
        # get image
        while True:
            color_frame, depth_frame = realsense.get_frame(align=True)
            depth_map = realsense.get_frame(frame_type="depth_map", align=True)
            if color_frame is not None and depth_frame is not None:
                cv2.imshow("color", color_frame)
            key = cv2.waitKey(1)
            if key == 27:
                color = color_frame
                depth = depth_frame
                cv2.destroyAllWindows()
                airbot_segment.set_image(color)
                airbot_segment.input_labels.clear()
                airbot_segment.input_points.clear()
                break
        # get bbox
        masked_color = color.copy()

        cv2.namedWindow("Segment")
        def mouse_callback(event, x, y, flags, param):
            global color, mask, bbox
            masked_color = color.copy()
            if event == cv2.EVENT_LBUTTONDOWN:
                airbot_segment.add_point(x, y, True)
                print("({}, {}) : Left clicked, add as positive".format(x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                airbot_segment.add_point(x, y, False)
                print("({}, {}) : Right clicked, add as negative".format(x, y))
            elif event == cv2.EVENT_MBUTTONDOWN:
                airbot_segment.clear_prompt()
                print("Middle mouse button clicked, clearing points")
                cv2.imshow("Segment", masked_color)
                return
            else:
                return
            masks, _, _ = airbot_segment.inference()
            mask = masks[0]
            masked_color[mask] = [191, 214, 238]  # preview mask
            cv2.imshow("Segment", masked_color)
            bboxes = masks_to_boxes(torch.from_numpy(masks))
            bbox = bboxes[0]

        cv2.setMouseCallback("Segment", mouse_callback)

        while True:
            cv2.imshow("Segment", masked_color)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break
