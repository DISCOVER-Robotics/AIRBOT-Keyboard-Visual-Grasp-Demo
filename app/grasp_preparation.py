"""Shared color-target preparation used by the hardware and simulation demos."""
import numpy as np
from airbot_yolo import draw_detections
from grasp_preview import draw_grasp_preview, target_mask_clearance_mm


def prepare_color_grasp(snapshot, detection, detections, segment, camera, grasp,
                        settings, *, save_cloud=True, execute_motion=False):
    mask = segment.inference_bbox(snapshot.color, list(detection.bbox))
    if mask is None or not np.any(mask):
        raise ValueError("MobileSAM 未生成有效目标掩码")
    mask_area = int(np.count_nonzero(mask))
    if mask_area < 100:
        raise ValueError("目标掩码面积过小")
    x1, y1, x2, y2 = detection.bbox
    in_box = np.zeros_like(mask, dtype=bool)
    in_box[y1:y2, x1:x2] = True
    mask_in_box_ratio = np.count_nonzero(mask & in_box) / mask_area
    required_ratio = float(settings.get(
        "min_mask_in_box_ratio", 0.50))
    if mask_in_box_ratio < required_ratio:
        raise ValueError("分割掩码与选中检测框不一致")
    if camera.has_hardware_depth:
        valid_depth = snapshot.depth[mask] > 0
        if np.count_nonzero(valid_depth) < max(20, int(mask_area * 0.25)):
            raise ValueError("目标区域没有足够的有效深度")
    cloud = camera.create_point_cloud(
        snapshot.depth, end_pose=[snapshot.state["trans"], snapshot.state["orient"]])
    pose = [snapshot.state["trans"], snapshot.state["orient"]]
    geometry_metadata = getattr(camera, 'geometry_metadata', None)
    trans, orient, cloud_base = grasp.inference(
        color_image=snapshot.color, depth_image=snapshot.depth,
        end_pose=pose, cloud_cam_raw=cloud, mask=mask, save_cloud=save_cloud,
        observation={'color': detection.color, 'bbox': list(detection.bbox),
                     'camera_geometry': geometry_metadata if isinstance(geometry_metadata, dict) else None,
                     'confidence': float(detection.confidence),
                     'snapshot_timestamp': float(snapshot.timestamp),
                     'image_size_wh': [snapshot.color.shape[1], snapshot.color.shape[0]]})
    if trans is None or orient is None or cloud_base is None or len(cloud_base) == 0:
        raise ValueError("抓取位姿计算失败")
    width = float(grasp.width_m)
    info = {
        "trans": np.asarray(trans).tolist(),
        "orient": np.asarray(orient).tolist(),
        "o_height": float(np.max(cloud_base[:, 2])),
        "o_width": float(width),
        "o_angle": float(grasp.angle),
        "o_label": detection.display_label,
        "o_bbox": list(detection.bbox),
        "snapshot_timestamp": snapshot.timestamp,
    }
    preview = draw_detections(snapshot.color, detections, detection.bbox)
    preview, pixels = draw_grasp_preview(
        preview, mask, grasp.preview_points_base, pose, grasp.cam2end,
        camera.intrinsic, camera.distortion)
    info['preview_pixels_raw_target'] = pixels
    clearance = target_mask_clearance_mm(
        mask, grasp.preview_points_base[1], pose, grasp.cam2end,
        camera.intrinsic, camera.distortion)
    info['target_mask_clearance_mm'] = clearance
    required = float(settings.get('min_target_mask_clearance_mm', 5.0))
    if not np.isfinite(required) or required < 0:
        raise ValueError('抓取目标边缘余量配置无效')
    if execute_motion and clearance < required:
        raise ValueError(
            f'目标抓取点离积木轮廓仅 {clearance:.1f} mm，要求至少 {required:.1f} mm；'
            '已取消自动下降。请查看预测图中的 RAW/TARGET，核对标定和抓取补偿')
    return info, preview
