import argparse
import sys
import os
import datetime
import numpy as np
import cv2

import threading
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

try:
    from airbot_camera import RealsenseCamera, USBCamera
    from airbot_arm import AirbotArm
except ImportError as e:
    print(f"ImportError: Failed to import arm_sdk 5.2.2 integration: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Airbot Calibration Tool")
parser.add_argument(
    "-t", "--type", type=str, default="hand_eye", choices=["hand_eye", "intrinsic"], help="Calibartion type, available calibration type: [hand_eye, intrinsic], default is hand_eye"
)
parser.add_argument(
    "-c", "--camera-type", type=str, default="realsense", choices=["usbcam", "realsense", "ros"], help="Camera type, available camera type: [usbcam, realsense], default is realsense"    
)
parser.add_argument(
    "-o", "--output-path", type=str, default="calib/", help="Directory to save the generated calibration data.",
)
parser.add_argument(
    "-p", "--port", type=int, default=50051, help="Robot port number.",
)
parser.add_argument(
    "--ros-topic", type=str, default="/camera/image_raw", help="ROS image topic (valid when using ros cam)"
)
parser.add_argument(
    "--no-display-mode", action="store_true", help="Calibrate in no display mode."
)

args = parser.parse_args()
space_len = 6

if args.camera_type == "ros":
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from rclpy.executors import MultiThreadedExecutor
    class RosCamera:
        
        def __init__(self, topic: str, node_name: str = "airbot_calib_node"):
            # 如果外部还没初始化 rclpy，就先初始化
            if not rclpy.ok():
                rclpy.init()

            # 创建 ROS 2 节点
            self.node = Node(node_name)
            self.bridge = CvBridge()
            self.topic = topic

            # 缓存帧 & 锁
            self.latest_frame = None
            self.lock = threading.Lock()
            
            self.WIDTH = 640
            self.HEIGHT = 480

            # 订阅 Image 话题
            # 10 深度的 QoS 对实时性有帮助
            qos = rclpy.qos.QoSProfile(depth=10)
            self.subscription = self.node.create_subscription(
                Image,
                self.topic,
                self._callback,
                qos
            )

            # MultiThreadedExecutor + 背景线程，用于持续 spin
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

        def _spin(self):
            """在后台线程里不断 spin"""
            try:
                self.executor.spin()
            except Exception as e:
                self.node.get_logger().error(f"Executor spin error: {e}")

        def _callback(self, msg: Image):
            """收到 ROS 图像消息后，用 CvBridge 转成 OpenCV，并缓存"""
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                self.node.get_logger().error(f"CvBridgeError: {e}")
                return

            with self.lock:
                self.latest_frame = cv_img
                self.WIDTH, self.HEIGHT = cv_img.shape[1], cv_img.shape[0]

        def get_frame(self, frame_type="bgr", align=False, timeout: float = 5.0):
            """
            拉取最新一帧（最多等待 timeout 秒）
            返回：OpenCV BGR 或 RGB 图像
            """
            start = time.time()
            while rclpy.ok() and (time.time() - start) < timeout:
                with self.lock:
                    if self.latest_frame is not None:
                        img = self.latest_frame.copy()
                        break
                time.sleep(0.01)
            else:
                raise RuntimeError(f"No image received on '{self.topic}' within {timeout}s")

            if frame_type.lower() == "rgb":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def destroy(self):
            """退出时清理资源"""
            # 停掉 executor
            self.executor.shutdown()
            # 销毁节点
            self.node.destroy_node()
            # 如果这就是唯一的节点，可以 shutdown rclpy
            # rclpy.shutdown()

class ChessBoard:
    def __init__(self):
        self.rows = 10
        self.cols = 8
        self.square_size = 0.02 # m
        self.number_of_image_needed = 15


class AirbotCalibration:
    def __init__(self):
        self.type = args.type
        self.camera = None
        self.cam_intrinsic = None
        self.cam_distortion = None
        self.cam2end = None
        self.project_error = None
        
        self.images = []
        self.end_pose_matrixes = []
        
        if args.camera_type == "realsense":
            try:
                self.camera = RealsenseCamera()
            except ImportError as e:
                print(f"Error importing RealsenseCamera: {e}")
                print("Please make sure airbot_realsense module is installed.")
                sys.exit(1)
        elif args.camera_type == "usbcam":
            self.camera = USBCamera()
        elif args.camera_type == "ros":
            self.camera = RosCamera(args.ros_topic)
        else:
            raise ValueError(f"Unsupported camera type: {args.camera_type}")
        
        self.chessboard = ChessBoard()
        self.time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = os.path.join(args.output_path, f"{self.type}_{self.camera.WIDTH}x{self.camera.HEIGHT}", self.time_str)
        os.makedirs(self.save_path, exist_ok=True)
        
    def choose_image(self, name="Image"):
        if args.no_display_mode:
            input("\nNo-display mode, Press Enter to capture image.")
            return self.camera.get_frame(frame_type="bgr", align=True)
        else:
            # In display mode, show the image and wait for ESC key
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            while True:
                image = self.camera.get_frame(frame_type="bgr", align=True)
                cv2.imshow(name, image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    return image
        
    def data_collect(self):
        robot = AirbotArm(port=args.port)
        try:
            robot.enter_gravity_compensation()
            print("Robot switched to GRAVITY_COMP mode.")
            if args.no_display_mode:
                print("Running in no-display mode.")
            else:
                print("Move the robot to capture positions. Press ESC to capture each position.")

            for i in range(self.chessboard.number_of_image_needed):
                image = self.choose_image(f"Collect data {i+1}/{self.chessboard.number_of_image_needed}")
                self.images.append(image)
                image_name = os.path.join(self.save_path, f"image{i}.png")
                cv2.imwrite(image_name, image)
                if self.type == "hand_eye":
                    pose = robot.get_end_pose()
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = R.from_quat(pose[1]).as_matrix()
                    pose_matrix[:3, 3] = pose[0]
                    self.end_pose_matrixes.append(pose_matrix)
                    print(f"--Data{i} Saved--\n  Image: {image_name}\n  Pose: {pose_matrix.flatten()}")
                elif self.type == "intrinsic":
                    print(f"--Data{i} Saved--\n  Image: {image_name}")
                else:
                    raise ValueError("Unsupported calibration type")

                cv2.destroyAllWindows()
        finally:
            robot.close()
    
    def plot_calibration_result(self, project_errors, image_points, object_points, rvecs, tvecs, mtx, dist):
        plt.figure(figsize=(15,5))
        # Per-image error curve
        plt.subplot(131)
        plt.plot(project_errors, 'b-')
        plt.xlabel('Image Index'), plt.ylabel('Error (pixels)')
        plt.title('Per-image Reprojection Error')
        plt.grid(True)
        # Error histogram
        plt.subplot(132)
        all_errors = np.sqrt(np.sum((np.array(image_points)-np.array([cv2.projectPoints(o, r, t, mtx, dist)[0] 
                                for o,r,t in zip(object_points, rvecs, tvecs)]))**2, axis=2))
        plt.hist(all_errors.ravel(), bins=50, color='g')
        plt.xlabel('Error (pixels)'), plt.ylabel('Count')
        plt.title('Error Histogram')
        plt.grid(True)
        # Error distribution box plot
        plt.subplot(133)
        plt.boxplot(all_errors.ravel(), showfliers=False)
        plt.ylabel('Error (pixels)')
        plt.title('Error Distribution')

        plt.tight_layout()
        # Save the analysis plot
        save_dir = os.path.join(self.save_path, "error_analysis.jpg")
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to: {save_dir}\n\n")
        
    def calibrate_camera(self):
        print("\nStarting camera calibration...")
        # 3D object points of the chessboard
        object_point = np.zeros((self.chessboard.rows * self.chessboard.cols, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.chessboard.cols, 0:self.chessboard.rows].T.reshape(-1, 2)
        object_point *= self.chessboard.square_size
        
        object_points = []
        image_points = []
        for image in self.images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard.cols, self.chessboard.rows), None)
            if ret:
                # optimize the corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                object_points.append(object_point)
                image_points.append(corners2)
                
                img_show = image.copy()
                cv2.drawChessboardCorners(img_show, (self.chessboard.cols, self.chessboard.rows), corners2, ret)
                cv2.imshow('Corners', img_show)
                cv2.waitKey(50)
            else:
                print("Chessboard pattern not found in image")
        
        cv2.destroyAllWindows()
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (self.camera.WIDTH, self.camera.HEIGHT), None, None)
        
        self.cam_intrinsic = mtx
        self.cam_distortion = dist
        
        project_errors = []
        for i in range(len(object_points)):
            imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            project_errors.append(error)
        self.project_error = np.mean(np.array(project_errors))
        
        self.plot_calibration_result(project_errors, image_points, object_points, rvecs, tvecs, mtx, dist)
        
        
    def calibrate_hand_eye(self, intrinsic, distortion):
        # 3D object points of the chessboard
        object_point = np.zeros((self.chessboard.rows * self.chessboard.cols, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.chessboard.cols, 0:self.chessboard.rows].T.reshape(-1, 2)
        object_point *= self.chessboard.square_size
        
        R_checkerboard_to_camera_poses = []
        T_checkerboard_to_camera_poses = []
        R_end_to_base_poses = []
        T_end_to_base_poses = []
        
        for i in range(len(self.images)):
            gray = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard.cols, self.chessboard.rows), None)
            if not ret:
                print("Chessboard pattern not found in image")
                continue
            else:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                ret, rvec, tvec = cv2.solvePnP(object_point, corners2, intrinsic, distortion)
                R_cam_pose, _ = cv2.Rodrigues(rvec)
                R_checkerboard_to_camera_poses.append(R_cam_pose)
                T_checkerboard_to_camera_poses.append(tvec.flatten())
                
                R_end_to_base_poses.append(self.end_pose_matrixes[i][:3, :3])
                T_end_to_base_poses.append(self.end_pose_matrixes[i][:3, 3])
            
        R_cam2end, T_cam2end = cv2.calibrateHandEye(
            R_end_to_base_poses, T_end_to_base_poses, 
            R_checkerboard_to_camera_poses, T_checkerboard_to_camera_poses,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        self.cam2end = np.eye(4)
        self.cam2end[:3, :3] = R_cam2end
        self.cam2end[:3, 3] = T_cam2end.flatten()
        
        return self.cam2end
    
    # def verify_hand_eye(self):
    #     image = self.choose_image("Verify Hand Eye Calibration")
    #     # 3D object points of the chessboard
    #     object_point = np.zeros((self.chessboard.rows * self.chessboard.cols, 3), np.float32)
    #     object_point[:, :2] = np.mgrid[0:self.chessboard.cols, 0:self.chessboard.rows].T.reshape(-1, 2)
    #     object_point *= self.chessboard.square_size
        calibrate_camera
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     ret, corners = cv2.findChessboardCorners(gray, (self.chessboard.cols, self.chessboard.rows), None)
    #     if ret:
    #         # optimize the corner positions
    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
    def report_calibration(self):
        reporter_head = f"""---Calibration Report---
Camera Type: {args.camera_type}
Resolution: {self.camera.WIDTH}x{self.camera.HEIGHT}
Chessboard: {self.chessboard.rows}x{self.chessboard.cols}-{self.chessboard.square_size}m
"""
        if self.project_error is not None:
            reporter_head += f"Project Error: {self.project_error}\n"
            
        print(reporter_head)
        
        if self.cam_intrinsic is not None:
            print("--Intrinsic--")
            MatrixPrinter.print_matrix(self.cam_intrinsic)
        
        if self.cam_distortion is not None:
            print("\n--Distortion--")
            MatrixPrinter.print_matrix(self.cam_distortion)
        
        if self.cam2end is not None:
            print("\n--Extrinsic--")
            MatrixPrinter.print_matrix(self.cam2end)
        
        file_name = os.path.join(self.save_path, "Calibration_Report.txt")
        with open(file_name, "w") as f:
            f.write(reporter_head)
            
        if self.cam_intrinsic is not None:
            MatrixPrinter.save_matrix(self.cam_intrinsic, "Intrinsic", file_name)
            
        if self.cam_distortion is not None:
            MatrixPrinter.save_matrix(self.cam_distortion, "Distortion", file_name)
            
        if self.cam2end is not None:
            MatrixPrinter.save_matrix(self.cam2end, "Extrinsic", file_name)
            
        print(f"\nCalibration report saved to: {file_name}")
        
        
class MatrixPrinter:
    """Utilities for formatted printing of matrices"""
    
    @staticmethod
    def format_number(v: float) -> str:
        """Format a number for display with consistent spacing"""
        if abs(v - 0) < 1e-12:
            return "0.        "
        elif abs(v - 1) < 1e-12:
            return "1.        "
        else:
            return f"{v:.8f}"
    
    @staticmethod
    def print_matrix(matrix: np.ndarray) -> None:
        """Print a matrix in two different formats for readability"""
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")

        # Format all values
        str_matrix = [[MatrixPrinter.format_number(val) for val in row] for row in matrix]
        
        # Calculate max width per column for alignment
        col_widths = [max(len(row[i]) for row in str_matrix) for i in range(matrix.shape[1])]

        # Create formatted row strings
        def format_row(row):
            return ", ".join(f"{val:>{col_widths[i]}}" for i, val in enumerate(row))

        print("\nlist format:")
        if matrix.shape[0] == 1:
            print(f"[{format_row(str_matrix[0])}]")
        else:
            print("[")
            for i, row in enumerate(str_matrix):
                comma = "," if i < len(str_matrix) - 1 else ""
                print(f" [{format_row(row)}]{comma}")
            print("]")

        print("\nyaml format:")
        for row in str_matrix:
            print(" " * space_len + f"- [{format_row(row)}]")
    
    @staticmethod
    def save_matrix(matrix: np.ndarray, segment_name, file_path: str) -> None:
        """Save matrix to file in both list and yaml formats"""
        with open(file_path, "a") as f:
            if not isinstance(matrix, np.ndarray):
                raise TypeError("Input must be a numpy ndarray.")

            # Format all values
            str_matrix = [[MatrixPrinter.format_number(val) for val in row] for row in matrix]
            
            # Calculate max width per column for alignment
            col_widths = [max(len(row[i]) for row in str_matrix) for i in range(matrix.shape[1])]

            # Create formatted row strings
            def format_row(row):
                return ", ".join(f"{val:>{col_widths[i]}}" for i, val in enumerate(row))

            f.write(f"\n--{segment_name}--\n")
            f.write("list format:\n")
            if matrix.shape[0] == 1:
                f.write(f"[{format_row(str_matrix[0])}]\n")
            else:
                f.write("[\n")
                for i, row in enumerate(str_matrix):
                    comma = "," if i < len(str_matrix) - 1 else ""
                    f.write(f" [{format_row(row)}]{comma}\n")
                f.write("]\n")

            f.write("yaml format:\n")
            for row in str_matrix:
                f.write(f"    - [{format_row(row)}]\n")

def draw_frame(T, ax=None, name='frame'):
    length=0.1
    linewidth=1.0
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    origin = T[:3, 3]
    x_axis = T[:3, 0] * length
    y_axis = T[:3, 1] * length
    z_axis = T[:3, 2] * length

    ax.quiver(*origin, *x_axis, color='r', linewidth=linewidth)
    ax.quiver(*origin, *y_axis, color='g', linewidth=linewidth)
    ax.quiver(*origin, *z_axis, color='b', linewidth=linewidth)

    if origin[0] == 0 and origin[1] == 0 and origin[2] == 0:
        coord_str = f'{name} (0,0,0)'
    else:
        coord_str = f'{name} ({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f})'
    ax.text(*origin, coord_str, fontsize=10)


    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.25, 0.25])
    ax.set_zlim([-0.25, 0.25])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([0.5,0.5,0.5])

    return ax

if __name__ == "__main__":
    try:
        calibrator = AirbotCalibration()
        print(f"\n{'='*60}")
        print(f"Airbot Calibration Tool")
        print(f"{'='*60}")
        print(f"Calibration type: {args.type}")
        print(f"Camera type: {args.camera_type}")
        print(f"Resolution: {calibrator.camera.resolution}")
        print(f"Output path: {args.output_path}")
        print(f"Robot port: {args.port}")
        if args.camera_type == "usbcam":
            print(f"USB camera device ID: {calibrator.camera.device_id}")
        print(f"{'='*60}\n")
    
        print("Initialized calibrator successfully.")
        
        calibrator.data_collect()
        print("Data collection complete.")
        
        calibrator.calibrate_camera()
            
        if args.type == "hand_eye":
            calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)
            
        calibrator.report_calibration()
        print("\nCalibration process completed successfully.")
        
        ax = draw_frame(np.eye(4), name='eef')
        draw_frame(calibrator.cam2end, ax=ax, name='cam')
        ax.view_init(elev=20, azim=70)

        plt.show()
                
    except Exception as e:
        print(f"\nError during calibration: {e}")
