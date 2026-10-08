"""Read-only board capture. Never acquires robot control or writes calibration."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description='标定板验证采集：不移动机械臂、不修改标定')
    parser.add_argument('--square-mm', type=float, required=True, help='实测单格边长，毫米')
    parser.add_argument('--cols', type=int, default=8, help='内角点列数')
    parser.add_argument('--rows', type=int, default=10, help='内角点行数')
    args = parser.parse_args()
    if not np.isfinite(args.square_mm) or args.square_mm <= 0 or min(args.cols, args.rows) < 3:
        parser.error('格长须为正数，内角点行列数须至少为 3')
    # Delay all hardware imports/initialization until arguments are validated.
    from airbot_camera import RealsenseCamera
    from airbot_arm import AirbotMonitor

    config = yaml.safe_load(Path('configs/config_file.yaml').read_text())
    config = yaml.safe_load(Path(config['Path']).read_text())
    if config.get('Camera', {}).get('type', 'realsense') != 'realsense':
        parser.error('此验证工具需要 RealSense 原始深度')
    directory = Path('runtime/board_check') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    (directory / 'config_snapshot.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    camera = None
    monitor = None
    try:
        camera = RealsenseCamera()
        monitor = AirbotMonitor(host=config['ArmParams'].get('host', 'localhost'),
                               port=config['ArmParams']['port'])
        print(f'只读采集已启动，保存目录：{directory}\n'
              '机械臂保持静止；移动标定板后等待稳定。按 1/2/3 分别保存左/中/右；Q 退出。')
        sequence = 0
        while True:
            before = monitor.read_state()
            color, depth = camera.get_frame(['bgr', 'depth'])
            after = monitor.read_state()
            cv2.imshow('Board check - 1:left 2:center 3:right Q:quit', color)
            key = cv2.waitKey(20) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                break
            if key not in (ord('1'), ord('2'), ord('3')):
                continue
            delta = np.linalg.norm(np.array(after['trans']) - before['trans'])
            angle = (Rotation.from_quat(after['orient']).inv()
                     * Rotation.from_quat(before['orient'])).magnitude()
            if not np.isfinite(delta + angle) or delta > .0005 or angle > np.deg2rad(.2):
                print('机械臂反馈仍在变化，本帧不保存；请静止后重试。')
                continue
            found, corners = cv2.findChessboardCornersSB(
                cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (args.cols, args.rows))
            if not found:
                print('未找到完整棋盘内角点；检查板规格、反光和遮挡后重试。')
                continue
            sequence += 1
            label = {ord('1'): 'left', ord('2'): 'center', ord('3'): 'right'}[key]
            prefix = directory / f'{sequence:03d}_{label}'
            np.savez_compressed(str(prefix) + '.npz', color=color, depth=depth,
                                corners=corners)
            profile = config['Realsense'][config['Realsense']['resolution']]
            record = dict(label=label, captured_at=time.time(), square_mm=args.square_mm,
                          pattern_size=[args.cols, args.rows], state_before=before,
                          state_after=after, camera_geometry=camera.geometry_metadata,
                          camera_to_end=profile['extrinsic'],
                          note='Board moved between regions; not known base coordinates. No grasp offsets applied.')
            Path(str(prefix) + '.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
            print(f'已保存 {label}：{prefix}（图像、未补洞深度、角点、前后位姿）')
    finally:
        if monitor is not None:
            monitor.close()
        if camera is not None:
            camera.deinit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
