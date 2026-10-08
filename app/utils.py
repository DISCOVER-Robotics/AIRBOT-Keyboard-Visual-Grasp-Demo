import open3d as o3d
import numpy as np
import cv2

def plot_gripper(center, R, width, length, score=1.0):
    '''
    可视化一个完整的夹爪模型（含尾部），抓取点为 center（前端中点）

    参数:
    - center: np.ndarray(3,), 抓取点位置
    - R: np.ndarray(3,3), 姿态矩阵
    - width: float, 两个手指间的间距
    - length: float, 手指长度
    - tail_length: float, 尾部长度
    - finger_thickness: float, 手指厚度
    - finger_height: float, 手指高度
    - score: float, 抓取得分 (影响颜色)
    '''
    finger_thickness=0.004
    finger_height=0.01

    # 颜色（红蓝渐变）
    color_r = score
    color_g = 0
    color_b = 1 - score
    gripper_color = [color_r, color_g, color_b]

    def create_box(w, h, d):
        return o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)

    # 创建左/右手指
    def create_finger(offset_y):
        finger = create_box(length, finger_thickness, finger_height)
        finger.translate([-length,
                          offset_y,
                          -finger_height / 2])
        return finger

    finger_left = create_finger(-(width / 2 + finger_thickness / 2))
    finger_right = create_finger((width / 2 + finger_thickness / 2))

    # 创建底部连接的尾部
    tail = create_box(finger_thickness, width, finger_height)
    tail.translate([
        -finger_thickness / 2 - length,
        - width /  2,
        -finger_height / 2
    ])

    # 合并 mesh
    for part in [finger_left, finger_right, tail]:
        part.paint_uniform_color(gripper_color)
    gripper = finger_left + finger_right + tail

    # 添加抓取点球体
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.paint_uniform_color([1, 1, 0])  # 黄色
    sphere.translate(center)

    # 应用旋转和平移（夹爪原点在 tip）
    gripper.rotate(R, center=[0, 0, 0])
    gripper.translate(center)

    # 合并成整体
    full_gripper = gripper + sphere

    return full_gripper


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    参数:
        masks: (N, H, W) 的二值掩码数组，N 是 mask 数量
    返回:
        boxes: (N, 4) 的边界框数组，格式为 [x_min, y_min, x_max, y_max]
    """
    boxes = []
    for mask in masks:
        points = cv2.findNonZero(mask.astype(np.uint8))
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            boxes.append([x, y, x + w, y + h])
        else:
            boxes.append([0, 0, 0, 0])
    return np.array(boxes, dtype=np.float32)