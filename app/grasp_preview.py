"""Diagnostic projection only; never changes a grasp or commands motion."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def project_base_points(points, end_pose, camera_to_end, intrinsic, distortion):
    end = np.eye(4)
    end[:3, :3] = Rotation.from_quat(end_pose[1]).as_matrix()
    end[:3, 3] = end_pose[0]
    camera_to_base = end @ np.asarray(camera_to_end, dtype=float)
    camera = (np.asarray(points, dtype=float) - camera_to_base[:3, 3]) @ camera_to_base[:3, :3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError('诊断投影点不在相机前方')
    pixels, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3),
                                  np.asarray(intrinsic, dtype=float),
                                  np.asarray(distortion, dtype=float).reshape(-1))
    return pixels.reshape(-1, 2)


def draw_grasp_preview(image, mask, points_base, end_pose, camera_to_end, intrinsic, distortion):
    preview = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 1)
    pixels = project_base_points(points_base, end_pose, camera_to_end, intrinsic, distortion)
    height, width = image.shape[:2]
    for index, (pixel, label, color) in enumerate(zip(pixels, ('RAW', 'TARGET'), ((255,255,0), (255,0,255)))):
        if not np.isfinite(pixel).all() or not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            continue
        xy = tuple(np.rint(pixel).astype(int))
        cv2.drawMarker(preview, xy, color, cv2.MARKER_CROSS, 15, 2)
        text_xy = (min(max(0, xy[0] + 8), max(0, width - 80)),
                   min(height - 5, max(15, xy[1] + (-12 if index == 0 else 22))))
        cv2.putText(preview, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return preview, pixels.tolist()


def target_mask_clearance_mm(mask, target_base, end_pose, camera_to_end, intrinsic,
                             distortion):
    """Conservative image-mask clearance at the fitted top-plane height."""
    target = np.asarray(target_base, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('目标抓取点无效')
    offsets = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0]])
    pixels = project_base_points(target + offsets, end_pose, camera_to_end,
                                 intrinsic, distortion)
    x, y = np.rint(pixels[0]).astype(int)
    mask = np.asarray(mask, dtype=bool)
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or not mask[y, x]:
        return 0.
    jacobian = np.column_stack((pixels[1] - pixels[0], pixels[2] - pixels[0]))
    pixels_per_mm = np.linalg.svd(jacobian, compute_uv=False).max()
    if pixels_per_mm <= 0 or not np.isfinite(pixels_per_mm):
        raise ValueError('目标投影尺度无效')
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)[y, x]
    # Use the largest local scale and reserve a pixel for rounding/boundaries.
    return float(max(0., distance - 1.) / pixels_per_mm)
