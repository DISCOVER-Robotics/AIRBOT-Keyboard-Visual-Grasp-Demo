import numpy as np
import yaml
import cv2
from camera_geometry import calibrated_point_cloud
from scipy.spatial.transform import Rotation

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class CameraConnectionError(RuntimeError):
    """Actionable camera startup error, before robot initialization."""


def check_realsense_connection():
    """Enumerate only: do not start streams or connect to a robot."""
    if rs is None:
        raise CameraConnectionError('未安装 pyrealsense2，无法使用 RealSense 相机。')
    if len(rs.context().query_devices()) == 0:
        raise CameraConnectionError(
            '未检测到 RealSense 深度相机（设备数为 0）。\n'
            '请检查相机供电和 USB 数据线，优先直连 USB 3 接口；用 lsusb 或 realsense-viewer 确认设备。\n'
            '--with-sim 需要真实相机和机械臂；笔记本内置 RGB 摄像头不能替代深度相机。\n'
            '仅体验仿真请运行 ./run_sim.sh；仅文字控制可加 --no-voice。')


def check_camera_connection():
    with open('configs/config_file.yaml', encoding='utf-8') as file:
        config_path = yaml.safe_load(file)['Path']
    with open(config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if str(config.get('Camera', {}).get('type', 'realsense')).lower() == 'realsense':
        check_realsense_connection()


class RealsenseCamera:
    has_hardware_depth = True
    __unaligned_warning_printed = False

    def __init__(self) -> None:
        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        config = yaml.safe_load(open(config_path, "r"))

        self.inited: bool = False
        self.resolution = config["Realsense"]["resolution"]
        self.depth_factor = config["Realsense"]["depth_factor"]
        self.profile = config["Realsense"][self.resolution]["profile"]
        self.intrinsic = config["Realsense"][self.resolution]["intrinsic"]
        self.distortion = config["Realsense"][self.resolution]["distortion"]

        self.init()

    def init(self):
        check_realsense_connection()
        # Configure streams
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth,
            self.profile[0],
            self.profile[1],
            rs.format.z16,
            self.profile[2],
        )
        rs_config.enable_stream(
            rs.stream.color,
            self.profile[0],
            self.profile[1],
            rs.format.bgr8,
            self.profile[2],
        )
        # Start streaming
        try:
            cfg = self.pipeline.start(rs_config)
        except RuntimeError as exc:
            raise CameraConnectionError(
                f'RealSense 视频流启动失败：{exc}。请检查 USB 连接、权限、分辨率及是否被其他程序占用。'
            ) from exc
        color_profile = cfg.get_stream(rs.stream.color)
        depth_profile = cfg.get_stream(rs.stream.depth)
        # z16 units belong to the connected depth sensor, not the YAML file.
        try:
            depth_scale = float(cfg.get_device().first_depth_sensor().get_depth_scale())
            if not np.isfinite(depth_scale) or depth_scale <= 0:
                raise ValueError('设备返回了无效的深度单位')
            self.depth_factor = 1.0 / depth_scale
            native = color_profile.as_video_stream_profile().get_intrinsics()
            if (native.width, native.height) != tuple(self.profile[:2]):
                raise ValueError('实际彩色流尺寸与标定分辨率不一致')
            self.geometry_metadata = {
                'depth_m_per_unit': depth_scale,
                'depth_hole_filling': False,
                'depth_alignment': 'color',
                'intrinsic': np.asarray(self.intrinsic).tolist(),
                'distortion': np.asarray(self.distortion).reshape(-1).tolist(),
                'device_color_intrinsic': [[native.fx, 0., native.ppx],
                                           [0., native.fy, native.ppy], [0., 0., 1.]],
                'device_color_distortion': list(native.coeffs),
                'device_color_distortion_model': str(native.model),
                'profile': list(self.profile),
            }
            print(f'RealSense 几何参数：{self.geometry_metadata}')
        except Exception:
            self.pipeline.stop()
            raise
        print(
            f"color profile:{color_profile.as_video_stream_profile()}\ndepth profile:{depth_profile.as_video_stream_profile()}"
        )

        # Set processers
        self.aligner = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

        self.inited = True

    @property
    def WIDTH(self) -> int:
        return self.profile[0]

    @property
    def HEIGHT(self) -> int:
        return self.profile[1]

    @property
    def FPS(self) -> int:
        return self.profile[2]

    @property
    def INTRINSIC(self) -> np.ndarray:
        return self.intrinsic

    @property
    def DISTORTION(self) -> np.ndarray:
        return self.distortion

    def deinit(self) -> bool:
        self.inited = False
        self.pipeline.stop()
        return True

    def get_rgb(self) -> np.ndarray:
        return self.get_frame("rgb")

    def get_bgr(self) -> np.ndarray:
        return self.get_frame("bgr")

    def get_depth(self) -> np.ndarray:
        return self.get_frame("depth")

    def get_depth_map(self) -> np.ndarray:
        return self.get_frame("depth_map")

    def get_frame(self, frame_type: str | list[str] = ["bgr","depth"], align: bool = True):
        frames = self.pipeline.wait_for_frames()
        if align:
            frames = self.aligner.process(frames)
        elif not self.__unaligned_warning_printed:
            print("\033[93mWarning: get unaligned frame\033[0m")  # 黄色警告
            self.__unaligned_warning_printed = True

        depth_frame = frames.get_depth_frame()
        # Preserve missing measurements as zero. Filling RGB-D edges with a
        # neighbour's depth can turn table pixels into false object surfaces.
        color_frame = frames.get_color_frame()

        depth_image = np.array(depth_frame.get_data()).astype(np.float32)
        bgr_image = np.array(color_frame.get_data())
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        depth_map = np.array(self.colorizer.colorize(depth_frame).get_data())

        frames = {
            "rgb": rgb_image,
            "bgr": bgr_image,
            "depth": depth_image,
            "depth_map": depth_map
        }

        if isinstance(frame_type, list):
            result = []
            for i in frame_type:
                if i not in frames.keys():
                    raise TypeError(
                        "Invalid frame_type, candidate type: ['rgb', 'bgr', 'depth', 'depth_map']"
                    )
                else:
                    result.append(frames[i])
            return result
        elif isinstance(frame_type, str):
            if frame_type not in frames.keys():
                raise TypeError(
                    "Invalid frame_type, candidate type: ['rgb', 'bgr', 'depth', 'depth_map']"
                )
            return frames[frame_type]
        else:
            raise TypeError(
                "Param frame_type should be 'str | list[str]'"
            )

    def create_point_cloud(self, depth: np.ndarray, organized: bool = True,
                           end_pose=None):
        """Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        # Depth is aligned to raw color pixels. Correct their rays, not just
        # the displayed image, so YOLO/SAM and depth retain identical indexing.
        cloud = calibrated_point_cloud(
            depth, self.intrinsic, self.distortion, self.depth_factor, self.profile)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        return cloud


class USBCamera:
    """普通 USB RGB 相机；抓取时使用已标定的固定桌面平面。"""
    has_hardware_depth = False

    def __init__(self):
        """Initialize USB camera with resolution settings"""

        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        config = yaml.safe_load(open(config_path, "r"))

        self.device_id = config["UsbCam"]["device_id"]
        self.resolution = config["UsbCam"]["resolution"]
        self.profile = config["UsbCam"][self.resolution]["profile"]
        self.intrinsic = np.asarray(config["UsbCam"][self.resolution]["intrinsic"], dtype=float)
        self.distortion = np.asarray(config["UsbCam"][self.resolution]["distortion"], dtype=float)
        profile_config = config["UsbCam"][self.resolution]
        # Existing calibration files used both spellings; accept either so
        # USB-camera setup does not fail before the first frame is read.
        extrinsic = profile_config.get("extrinsic", profile_config.get("Extrinsic"))
        if extrinsic is None:
            raise ValueError("UsbCam calibration requires an extrinsic matrix")
        self.cam2end = np.asarray(extrinsic, dtype=float)
        self.table_z_base = float(config["UsbCam"].get("table_z_base", 0.01338870424854599))
        self.depth_factor = 1.0
        self.WIDTH, self.HEIGHT, self.FPS = self.profile

        self.cap = None
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the camera capture"""
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera with device ID {self.device_id}")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        # Check if resolution was set correctly
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if abs(actual_width - self.WIDTH) > 10 or abs(actual_height - self.HEIGHT) > 10:
            print(f"Warning: Requested resolution {self.WIDTH}x{self.HEIGHT} not available.")
            print(f"Using {actual_width}x{actual_height} instead.")
            self.WIDTH, self.HEIGHT = int(actual_width), int(actual_height)

    def get_frame(self, frame_type="bgr", align=False):
        """
        Capture a frame from the USB camera

        Args:
            frame_type: The color format of the returned frame, 'bgr' by default
            align: Parameter kept for compatibility with RealSense interface, not used for USB camera

        Returns:
            The captured frame image
        """
        if not self.cap or not self.cap.isOpened():
            self._initialize_camera()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from USB camera")

        frames = {
            "bgr": frame,
            "rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            "depth": np.zeros(frame.shape[:2], dtype=np.float32),
            "depth_map": np.zeros_like(frame),
        }
        if isinstance(frame_type, list):
            return [frames[item] for item in frame_type]
        if frame_type not in frames:
            raise TypeError("Invalid frame_type, candidate type: ['rgb', 'bgr', 'depth', 'depth_map']")
        return frames[frame_type]

    def create_point_cloud(self, depth, organized=True, end_pose=None):
        """Create points by intersecting camera rays with the calibrated table plane."""
        if end_pose is None:
            raise ValueError("普通 RGB 相机需要机械臂末端位姿才能计算桌面坐标")
        end2base = np.eye(4)
        end2base[:3, :3] = Rotation.from_quat(end_pose[1]).as_matrix()
        end2base[:3, 3] = end_pose[0]
        cam2base = end2base @ self.cam2end
        xmap, ymap = np.meshgrid(np.arange(self.WIDTH), np.arange(self.HEIGHT))
        rays = np.stack(((xmap - self.intrinsic[0, 2]) / self.intrinsic[0, 0],
                         (ymap - self.intrinsic[1, 2]) / self.intrinsic[1, 1],
                         np.ones_like(xmap, dtype=float)), axis=-1)
        rays_base = rays @ cam2base[:3, :3].T
        scale = np.divide(self.table_z_base - cam2base[2, 3], rays_base[..., 2],
                          out=np.full(rays_base.shape[:2], np.nan),
                          where=np.abs(rays_base[..., 2]) > 1e-8)
        scale[scale <= 0] = np.nan
        points = rays * scale[..., None]
        points[~np.isfinite(points)] = 0
        return points.astype(np.float32) if organized else points.reshape(-1, 3).astype(np.float32)

    def deinit(self) -> bool:
        """Release the USB camera using the same lifecycle as RealSense."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return True

    def __del__(self):
        """Clean up camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()


def create_camera():
    with open("configs/config_file.yaml", "r") as file:
        config_path = yaml.safe_load(file)["Path"]
    config = yaml.safe_load(open(config_path, "r"))
    camera_type = str(config.get("Camera", {}).get("type", "realsense")).lower()
    if camera_type == "realsense":
        return RealsenseCamera()
    if camera_type in {"usb_rgb", "rgb", "usbcam"}:
        return USBCamera()
    raise ValueError("Camera.type 仅支持 realsense 或 usb_rgb")


if __name__ == "__main__":
    import datetime
    import os

    camera = RealsenseCamera()
    frame_cnt = 0
    dir_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join("yolo_dataset", dir_name)
    os.makedirs(save_path, exist_ok=True)
    while True:
        color_image = camera.get_frame(frame_type="bgr")
        cv2.imshow(f"Color Image {frame_cnt}", color_image)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord(" "):
            name = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
            img_path = os.path.join(save_path, name)
            cv2.imwrite(img_path, color_image)
            frame_cnt += 1
            frame_cnt = frame_cnt % 20
            cv2.destroyAllWindows()
    camera.deinit()
