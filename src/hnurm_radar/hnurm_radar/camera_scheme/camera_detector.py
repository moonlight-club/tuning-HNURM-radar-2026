"""
camera_detector.py — 纯相机透视变换雷达站检测节点
功能：调用海康工业相机，使用 YOLO 识别赛场地面机器人，
      通过透视变换（Homography）将图像像素坐标映射到赛场坐标，
      并将坐标在小地图中实时显示。

与已有代码（相机+激光雷达方案）的区别：
  - 不依赖激光雷达点云和外参矩阵
  - 不依赖 ICP / TF 重定位
  - 使用 4 组以上「图像像素 ↔ 赛场坐标」对应点计算单应性矩阵（Homography）
  - 检测到机器人后取包围框底部中心点，通过透视变换直接得到赛场 (x, y)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from detect_result.msg import Location, Locations
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ruamel.yaml import YAML

# from ..Camera.HKCam import *  # 使用海康工业相机时取消注释
from ..Car.Car import CarList
from ..camera_locator.anchor import Anchor
from ..camera_locator.point_picker import PointsPicker
from ..filters.kalman_filter import KalmanFilterWrapper
from .hungarian_tracker import HungarianTracker  # tunning: 新增匈牙利关联模块导入
from ..shared.type import SingleDetectionResult, TrackingState # tunning: 引入底层数据协议与状态机
from .guess_pts import PointGuesser  # tunning: 引入长时间丢失惯性推演

import cv2
import numpy as np
import threading
import time
import math
import sys
import os
import json
from collections import deque
from ultralytics import YOLO
from ..shared.paths import (
    MAIN_CONFIG_PATH, DETECTOR_CONFIG_PATH, PERSPECTIVE_CALIB_PATH,
    PFA_MAP_2025_PATH, PFA_MAP_MASK_2025_PATH, TEST_RESOURCES_DIR,
    resolve_path,
)

# ======================== 配置路径 ========================
# 地图图片（当前使用 PFA 2025 赛季地图，后续替换为 2026 赛季地图）
MAP_IMAGE_PATH = PFA_MAP_2025_PATH
# 分区掩码图（PFA 2025 赛季掩码，后续用 make_mask 重新绘制 2026 版本）
MASK_IMAGE_PATH = PFA_MAP_MASK_2025_PATH

# ======================== 视频输入源 ========================
# 现在通过 main_config.yaml 中的 camera_mode 配置项控制：
#   'test'  — 测试模式，使用静态图片反复推理
#   'video' — USB 摄像头 / 视频文件
#   'hik'   — 海康工业相机


class CameraDetector(Node):
    """
    纯相机透视变换雷达站节点
    ──────────────────────
    启动后流程：
      1. 打开海康相机并持续读取图像
      2. （首次/需要时）标定：用户在图像上选 4 个点，输入对应赛场坐标，
         计算 Homography 矩阵并保存
      3. YOLO 三阶段推理（复用已有 detector_node 的逻辑）识别机器人
      4. 取每个机器人检测框底部中心的像素坐标，通过 H 矩阵变换到赛场坐标
      5. 将结果发布到 ROS2 话题 & 在小地图窗口中实时绘制
    """

    def __init__(self):
        super().__init__('camera_detector')
        self.get_logger().info("CameraDetector 节点启动 ...")

        # ---------- 加载配置 ----------
        self.main_cfg = YAML().load(
            open(MAIN_CONFIG_PATH, encoding='Utf-8', mode='r'))
        self.det_cfg = YAML().load(
            open(DETECTOR_CONFIG_PATH, encoding='Utf-8', mode='r'))

        self.my_color = self.main_cfg['global']['my_color']
        self.is_debug = self.main_cfg['global']['is_debug']
        self.bridge = CvBridge()

        # ---------- 初始化 CarList ----------
        self.carList = CarList(self.main_cfg)

        # ---------- 加载 YOLO 模型 ----------
        self.get_logger().info('正在加载 YOLO 模型 ...')
        self.model_car = YOLO(resolve_path(self.det_cfg['path']['stage_one_path']), task="detect")
        self.model_car.overrides['imgsz'] = 1280
        self.model_car2 = YOLO(resolve_path(self.det_cfg['path']['stage_two_path']))
        self.model_car2.overrides['imgsz'] = 256
        self.model_car3 = YOLO(resolve_path(self.det_cfg['path']['stage_three_path']))
        self.model_car3.overrides['imgsz'] = 256
        self.get_logger().info('YOLO 模型加载完毕。')

        self.tracker_path = resolve_path(self.det_cfg['path']['tracker_path'])
        self.stage_one_conf = self.det_cfg['params']['stage_one_conf']
        self.stage_two_conf = self.det_cfg['params']['stage_two_conf']
        self.stage_three_conf = self.det_cfg['params']['stage_three_conf']
        self.life_time = self.det_cfg['params']['life_time']
        self.labels = self.det_cfg['params']['labels']
        self.class_num = len(self.labels)

        # 追踪器投票表
        self.Track_value = {}
        self.Status = [0] * 10000
        for i in range(10000):
            self.Track_value[i] = [0] * self.class_num
        self.id_candidate = [0] * 10000
        self.loop_times = 0

        # 灰色装甲板映射（与 detector_node 一致）
        self.Gray2Blue = {12: 5, 13: 1, 14: 0, 15: 3, 16: 2, 17: 4}
        self.Gray2Red = {12: 11, 13: 7, 14: 6, 15: 9, 16: 8, 17: 10}
        self.gray2gray = {0: 14, 1: 13, 2: 16, 3: 15, 4: 17, 5: 12}
        self.Blue2Gray = {v: k for k, v in self.Gray2Blue.items()}
        self.Red2Gray = {v: k for k, v in self.Gray2Red.items()}

        # ---------- QoS ----------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=3
        )

        # ---------- ROS2 发布 ----------
        self.pub_location = self.create_publisher(Locations, "location", qos_profile)
        self.pub_detect_view = self.create_publisher(Image, 'detect_view', qos_profile)

        # ---------- 视频输入源（根据配置选择模式） ----------
        self.frame = None
        self._frame_lock = threading.Lock()
        self.use_hkcam = False
        self.video_cap = None
        self.cam = None
        self.test_image = None  # test 模式下的静态图片

        camera_cfg = self.main_cfg.get('camera', {})
        self.camera_mode = camera_cfg.get('mode', 'video')
        video_source = camera_cfg.get('video_source', 0)
        test_img_path = resolve_path(camera_cfg.get('test_image',
            os.path.join(TEST_RESOURCES_DIR, 'pfa_test_image.jpg')))

        if self.camera_mode == 'test':
            # ====== 测试模式：使用静态图片反复推理 ======
            self.test_image = cv2.imread(test_img_path)
            if self.test_image is None:
                self.get_logger().error(f"测试图片无法加载: {test_img_path}")
            else:
                self.get_logger().info(f"测试模式：使用静态图片 {test_img_path} "
                                       f"({self.test_image.shape[1]}×{self.test_image.shape[0]})")
        elif self.camera_mode == 'hik':
            # ====== 海康工业相机（比赛时使用） ======
            try:
                from ..Camera.HKCam import HKCam
                self.cam = HKCam(0)
                self.use_hkcam = True
                self.get_logger().info("使用海康工业相机 (HKCam)")
            except Exception as e:
                self.get_logger().error(f"海康相机初始化失败: {e}，回退到测试模式")
                self.camera_mode = 'test'
                self.test_image = cv2.imread(test_img_path)
        else:
            # ====== video 模式：USB 摄像头 / 视频文件 ======
            self.video_cap = cv2.VideoCapture(video_source)
            if not self.video_cap.isOpened():
                self.get_logger().error(
                    f"无法打开视频源: {video_source}，回退到测试模式")
                self.camera_mode = 'test'
                self.test_image = cv2.imread(test_img_path)
            else:
                src_name = (f"USB 摄像头 {video_source}"
                            if isinstance(video_source, int) else video_source)
                self.get_logger().info(f"使用视频源: {src_name}")

        # ---------- 透视变换矩阵（多层） ----------
        self.H_ground = None    # 地面层 Homography (相机像素→地图像素)
        self.H_highland = None  # 高地层 Homography
        self.H = None           # 兼容：默认指向地面层
        self.calib_map_w = None # 标定时使用的地图宽度(px)
        self.calib_map_h = None # 标定时使用的地图高度(px)
        self.calib_map_portrait = False  # 标定地图是否竖版
        self.mask_img = None    # 分区掩码图
        self._load_or_calibrate_homography()
        self._load_mask()

       # ---------- 坐标卡尔曼滤波器 ----------
        # 修改：注释kalman_filter 开始
        # filter_cfg = self.det_cfg.get('filter', {})
        # self.kf_wrapper = KalmanFilterWrapper(
        #     process_noise=float(filter_cfg.get('process_noise', 1e-2)),
        #     measurement_noise=float(filter_cfg.get('measurement_noise', 1e-1)),
        #     jump_threshold=float(filter_cfg.get('jump_threshold', 1.0)),
        #     max_velocity=float(filter_cfg.get('max_velocity', 5.0)),
        #     max_inactive_time=float(filter_cfg.get('max_inactive_time', 3.0)),
        # )
        # self._cleanup_counter = 0
        # 修改：注释kalman_filter 结束

        

        # ---------- 小地图 ----------
        self.map_img = cv2.imread(MAP_IMAGE_PATH)
        if self.map_img is None:
            self.get_logger().warn(f"无法读取地图图片: {MAP_IMAGE_PATH}，将使用纯色底图")
            self.map_img = np.zeros((1500, 2800, 3), dtype=np.uint8)

        # ---------- 启动线程 ----------
        self.frame_thread = threading.Thread(target=self._sync_frame, daemon=True)
        self.frame_thread.start()

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        # tunning: 新增匈牙利 tracker，用于检测-轨迹关联，max_miss 配置可通过 detector config 调整
        track_params = self.det_cfg.get('track', {})
        # // tunning: 从 config 读取完整的三段式生命周期参数
        self.hungarian = HungarianTracker(
            iou_thr=float(track_params.get('iou_thr', 0.05)),
            dist_thr=float(track_params.get('dist_thr', 200)),
            max_miss=int(track_params.get('max_miss', 84)),    # // tunning: 最大容忍 84 帧 (2.8s)
            lost_thr=int(track_params.get('lost_thr', 0)),     # // tunning: 浅层遮挡阈值
            guess_thr=int(track_params.get('guess_thr', 3))    # // tunning: 深度遮挡物理推演阈值
        )
        # tunning: 当无检测时是否发布短期预测/历史位置信息（YAML 可配置）
        self.publish_predict_when_no_det = bool(self.det_cfg.get('publish_predict_when_no_det', True))
        
       # // tunning: 从参数文件读取盲猜配置，保持逻辑与参数绝对统一
        self.guesser = PointGuesser(
            decay_factor=float(track_params.get('decay_factor', 0.94)),
            max_guess_sec=float(track_params.get('max_guess_sec', 2.8))
        )
        self.get_logger().info("CameraDetector 初始化完成。")

    # ================================================================
    #  透视变换标定
    # ================================================================
    def _load_or_calibrate_homography(self):
        """尝试从文件加载 Homography（支持多层），若不存在则启动交互式标定"""
        if os.path.exists(PERSPECTIVE_CALIB_PATH):
            self.get_logger().info(f"从 {PERSPECTIVE_CALIB_PATH} 加载透视变换标定 ...")
            with open(PERSPECTIVE_CALIB_PATH, 'r') as f:
                data = json.load(f)
            # 兼容旧格式（单层 H）和新格式（H_ground + H_highland）
            if 'H_ground' in data:
                self.H_ground = np.array(data['H_ground'], dtype=np.float64)
                self.get_logger().info("地面层透视变换矩阵加载成功。")
            if 'H_highland' in data:
                self.H_highland = np.array(data['H_highland'], dtype=np.float64)
                self.get_logger().info("高地层透视变换矩阵加载成功。")
            if 'H' in data and self.H_ground is None:
                # 旧格式兼容：只有单个 H，作为地面层使用
                self.H_ground = np.array(data['H'], dtype=np.float64)
                self.get_logger().info("兼容旧格式：单层 H 作为地面层矩阵。")
            self.H = self.H_ground  # 默认指向地面层
            # 读取标定时的地图尺寸信息（新格式）
            if 'map_w' in data and 'map_h' in data:
                self.calib_map_w = int(data['map_w'])
                self.calib_map_h = int(data['map_h'])
                self.calib_map_portrait = data.get('map_is_portrait', False)
                self.get_logger().info(
                    f"标定地图尺寸: {self.calib_map_w}×{self.calib_map_h}, "
                    f"竖版={self.calib_map_portrait}")
            else:
                self.get_logger().info("旧格式标定文件，无地图尺寸信息（假设H直接输出赛场米坐标）")
        else:
            self.get_logger().info("未找到透视变换标定文件，进入交互式标定 ...")
            self._interactive_calibrate()

    def _load_mask(self):
        """加载分区掩码图（黑色=地面，非黑色=高地）"""
        if os.path.exists(MASK_IMAGE_PATH):
            self.mask_img = cv2.imread(MASK_IMAGE_PATH)
            if self.mask_img is not None:
                h, w = self.mask_img.shape[:2]
                # 判断掩码方向：竖版(H>W) 或 横版(W>=H)
                self.mask_is_portrait = (h > w)
                if self.mask_is_portrait:
                    self.get_logger().info(
                        f"掩码图为竖版({w}×{h})，H→28m, W→15m")
                else:
                    self.get_logger().info(
                        f"掩码图为横版({w}×{h})，W→28m, H→15m")
                self.get_logger().info(f"分区掩码已加载: {MASK_IMAGE_PATH}")
            else:
                self.get_logger().warn(f"无法读取掩码图: {MASK_IMAGE_PATH}，将仅使用地面层")
        else:
            self.mask_is_portrait = False
            self.get_logger().info("未找到分区掩码图，将仅使用地面层透视变换。"
                                   f"（可使用 make_mask 工具生成: {MASK_IMAGE_PATH}）")

    def _interactive_calibrate(self):
        """
        交互式标定：提示用户运行独立的 GUI 标定工具。
        不再在 camera_detector 节点内做标定，避免阻塞 ROS2 节点。
        """
        self.get_logger().error(
            "═" * 60 + "\n"
            "  未找到透视变换标定文件！\n"
            f"  需要的文件: {PERSPECTIVE_CALIB_PATH}\n"
            "\n"
            "  请先运行 GUI 标定工具进行标定：\n"
            "    ros2 run hnurm_radar perspective_calibrator\n"
            "\n"
            "  标定完成后重新启动 camera_detector。\n"
            + "═" * 60
        )
        self.get_logger().info("节点将以无标定模式继续运行（坐标映射不可用）。")

    def recalibrate(self):
        """重新标定（可在运行时调用）"""
        if os.path.exists(PERSPECTIVE_CALIB_PATH):
            os.remove(PERSPECTIVE_CALIB_PATH)
        self.H_ground = None
        self.H_highland = None
        self.H = None
        self._interactive_calibrate()

    # ================================================================
    #  透视变换：像素 → 赛场坐标
    # ================================================================
    def _map_pixel_to_field(self, map_px, map_py):
        """
        将地图像素坐标转换为赛场米坐标。
        与原始 PFA 标定方案一致：H 矩阵输出地图像素，运行时再转赛场米。

        关键：红方和蓝方的地图是 180° 旋转关系，坐标转换公式不同！
        （参考 pfa_vision_radar/main.py 中 send_point_B / send_point_R）

        蓝方竖版地图:
          field_x = map_py / map_h * 28    (地图纵轴→赛场x)
          field_y = map_px / map_w * 15    (地图横轴→赛场y)
        红方竖版地图（相对蓝方旋转180°）:
          field_x = (map_h - map_py) / map_h * 28
          field_y = (map_w - map_px) / map_w * 15
        """
        if self.calib_map_w is None:
            # 旧格式：H 直接输出赛场米坐标，无需转换
            return map_px, map_py

        if self.calib_map_portrait:
            if self.my_color == 'Red':
                # 红方地图：坐标需要翻转（180°旋转关系）
                field_x = (self.calib_map_h - map_py) / self.calib_map_h * 28.0
                field_y = (self.calib_map_w - map_px) / self.calib_map_w * 15.0
            else:
                # 蓝方地图
                field_x = map_py / self.calib_map_h * 28.0
                field_y = map_px / self.calib_map_w * 15.0
        else:
            field_x = map_px / self.calib_map_w * 28.0
            field_y = (self.calib_map_h - map_py) / self.calib_map_h * 15.0
        return field_x, field_y

    def pixel_to_field(self, px, py):
        """
        将图像像素坐标 (px, py) 通过多层 Homography 变换到赛场坐标 (fx, fy)。

        流程:
          1. H 矩阵将相机像素变换到地图像素坐标
          2. 用地图像素坐标查掩码判定高度层
          3. 地图像素坐标转换为赛场米坐标

        参数:
            px, py: 原始分辨率下的像素坐标
        返回:
            (field_x, field_y): 赛场坐标（单位 m），若变换失败返回 None
        """
        if self.H_ground is None:
            return None

        pt = np.array([[[px, py]]], dtype=np.float64)

        # Step 1: 地面层变换 → 地图像素坐标
        transformed = cv2.perspectiveTransform(pt, self.H_ground)
        map_x = transformed[0][0][0]
        map_y = transformed[0][0][1]

        # ★ 异常值检测：地图像素坐标超出合理范围则返回 None
        # 允许一定的边界外扩，但不能太离谱
        margin = 500  # 像素边界外扩
        if self.calib_map_w is not None:
            if (map_x < -margin or map_x > self.calib_map_w + margin or
                map_y < -margin or map_y > self.calib_map_h + margin):
                if self.is_debug:
                    self.get_logger().warn(
                        f"透视变换异常: px=({px:.0f},{py:.0f}) → map=({map_x:.1f},{map_y:.1f}) 超出范围")
                return None

        # Step 2: 查掩码判定高度层（直接用地图像素坐标查掩码）
        if self.mask_img is not None and self.H_highland is not None:
            mask_h, mask_w = self.mask_img.shape[:2]
            # 掩码与标定地图同方向同尺寸，直接按比例映射
            if self.calib_map_w is not None:
                mx = int(map_x * mask_w / self.calib_map_w)
                my = int(map_y * mask_h / self.calib_map_h)
            else:
                # 旧格式：map_x/map_y 是赛场米坐标
                if self.mask_is_portrait:
                    mx = int(map_y * mask_w / 15.0)
                    my = int(map_x * mask_h / 28.0)
                else:
                    mx = int(map_x * mask_w / 28.0)
                    my = int(mask_h - map_y * mask_h / 15.0)
            mx = max(0, min(mx, mask_w - 1))
            my = max(0, min(my, mask_h - 1))

            pixel_color = self.mask_img[my, mx]
            is_highland = not (pixel_color[0] == 0 and
                               pixel_color[1] == 0 and
                               pixel_color[2] == 0)

            if is_highland:
                # 用高地层矩阵重新变换
                transformed_h = cv2.perspectiveTransform(pt, self.H_highland)
                map_x = transformed_h[0][0][0]
                map_y = transformed_h[0][0][1]
                # 高地层变换后也要检查范围
                if (map_x < -margin or map_x > self.calib_map_w + margin or
                    map_y < -margin or map_y > self.calib_map_h + margin):
                    if self.is_debug:
                        self.get_logger().warn(
                            f"高地层透视变换异常: px=({px:.0f},{py:.0f}) → map=({map_x:.1f},{map_y:.1f})")
                    return None

        # Step 3: 地图像素 → 赛场米坐标
        fx, fy = self._map_pixel_to_field(map_x, map_y)
        return (fx, fy)

    def get_box_bottom_center(self, xyxy_box):
        """
        从检测框 [x1, y1, x2, y2] 中取底部中心点的像素坐标。
        底部中心更接近地面，透视变换精度更高。
        """
        x1, y1, x2, y2 = xyxy_box
        cx = (x1 + x2) / 2.0
        cy = y2  # 底部
        return cx, cy

    # ================================================================
    #  相机图像同步
    # ================================================================
    def _sync_frame(self):
        """持续从视频源获取图像（支持 test/video/hik 三种模式）"""
        while rclpy.ok():
            cam_frame = None

            if self.camera_mode == 'test':
                # 测试模式：反复使用同一张图片
                if self.test_image is not None:
                    cam_frame = self.test_image.copy()
                time.sleep(0.05)  # 模拟 ~20fps

            elif self.camera_mode == 'hik' and self.use_hkcam and self.cam is not None:
                # 海康工业相机
                cam_frame = self.cam.getFrame()

            elif self.video_cap is not None and self.video_cap.isOpened():
                # USB 摄像头 / 视频文件
                ret, cam_frame = self.video_cap.read()
                if not ret:
                    # 视频播放完毕，循环播放
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, cam_frame = self.video_cap.read()
                    if not ret:
                        cam_frame = None
                    # ★ 重置跟踪器和滤波器状态，避免状态残留
                    self._reset_tracking_state()
                time.sleep(0.033)  # 30fps 视频帧率控制

            if cam_frame is not None:
                with self._frame_lock:
                    self.frame = cam_frame.copy()
            else:
                time.sleep(0.03)  # 没有画面时短暂等待

    def _get_frame(self):
        with self._frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def _reset_tracking_state(self):
        """
        重置跟踪器和滤波器状态。
        在视频循环播放时调用，避免状态残留导致坐标错误。
        """
        # 修改：注释kalman_filter 开始
        # # 重置卡尔曼滤波器
        # self.kf_wrapper.reset()

        # 重置跟踪投票表和状态
        for i in range(10000):
            self.Track_value[i] = [0] * self.class_num
            self.Status[i] = 0
        self.id_candidate = [0] * 10000

        # 重置 ByteTrack 跟踪器（通过清除 predictor 中的 trackers）
        if hasattr(self.model_car, 'predictor') and self.model_car.predictor is not None:
            if hasattr(self.model_car.predictor, 'trackers'):
                self.model_car.predictor.trackers = None

        if self.is_debug:
            self.get_logger().info("跟踪器和滤波器状态已重置")

    # ================================================================
    #  YOLO 推理（复用已有 detector_node 的三阶段逻辑）
    # ================================================================
    def _is_results_empty(self, results):
        if results is None:
            return True
        if results[0].boxes.id is None:
            return True
        return False

    def _parse_results(self, results):
        confidences = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        return confidences, boxes, track_ids

    def _track_infer(self, frame):
        results = self.model_car.track(
            frame, persist=True, tracker=self.tracker_path, verbose=False)
        return results

    def _classify_infer(self, roi_list):
        """二/三阶段分类推理"""
        results = []
        gray_results = []
        for roi in roi_list:
            results.extend(self.model_car2.predict(
                roi, conf=self.stage_two_conf, device=0, verbose=False))
            gray_results.extend(self.model_car3.predict(
                roi, device=0, verbose=False))

        if len(results) == 0:
            return -1

        label_list = []
        conf_list = []
        for result, gray_result, roi in zip(results, gray_results, roi_list):
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            data = result.boxes.data
            gray_data = gray_result.boxes.data
            maxConf = 0
            maxGrayConf = 0
            label = -1
            gray_label = -1
            gray_tmp = []
            tmp = []

            for i in range(len(gray_data)):
                if gray_data[i][4] > maxGrayConf:
                    maxGrayConf = gray_data[i][4]
                    gray_label = gray_data[i][5]
                    gray_tmp = gray_data[i]

            for i in range(len(data)):
                if data[i][4] > maxConf:
                    maxConf = data[i][4]
                    label = data[i][5]
                    tmp = data[i]

            if len(tmp) != 0:
                x1, y1, x2, y2 = tmp[:4]
                gray_region = roi_gray[int(y1):int(y2), int(x1):int(x2)]
                if label < 6 and gray_region.size > 0 and cv2.mean(gray_region)[0] < 35:
                    label = self.Blue2Gray[int(label)]
                elif label > 5 and gray_region.size > 0 and cv2.mean(gray_region)[0] < 15:
                    label = self.Red2Gray[int(label)]
            else:
                if len(gray_tmp) != 0:
                    x1, y1, x2, y2 = gray_tmp[:4]
                    gray_region = roi_gray[int(y1):int(y2), int(x1):int(x2)]
                    if gray_region.size > 0 and cv2.mean(gray_region)[0] < 30:
                        label = self.gray2gray[int(gray_label)]

            label_list.append(int(label))
            conf_list.append(float(maxConf))
        return label_list, conf_list

    def _infer(self, frame):
        """
        三阶段推理（与 detector_node.infer 逻辑一致）
        返回: (绘制后的图像, 结果列表)
        结果列表中每个元素: [xyxy_box, xywh_box, track_id, label_str]
        """
        if frame is None:
            return None, None
        results = self._track_infer(frame)
        if self._is_results_empty(results):
            return frame, None

        exist_armor = [-1] * (self.class_num + 6)
        draw_candidate = []
        confidences, boxes, track_ids = self._parse_results(results)
        zip_results = []
        roi_list = []
        id_list = []
        box_list = []

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if self.loop_times % self.life_time == 1:
                for i in range(self.class_num):
                    self.Track_value[int(track_id)][i] = math.floor(
                        self.Track_value[int(track_id)][i] / 10)
            x, y, w, h = box
            x_left = x - w / 2
            y_left = y - h / 2
            roi = frame[int(y_left):int(y_left + h), int(x_left):int(x_left + w)]
            if roi.size == 0:
                continue
            roi_list.append(roi)
            id_list.append(track_id)
            box_list.append(box)

        if len(roi_list) == 0:
            self.loop_times += 1
            return frame, None

        classify_result = self._classify_infer(roi_list)
        if classify_result == -1:
            self.loop_times += 1
            return frame, None

        label_list, conf_list = classify_result
        index = 0
        for i in range(len(roi_list)):
            classify_label = label_list[i]
            conf = conf_list[i]
            track_id = id_list[i]
            box = box_list[i]
            x, y, w, h = box
            status = 0

            if classify_label != -1:
                label = self.Track_value[int(track_id)].index(
                    max(self.Track_value[int(track_id)]))
                if classify_label > 11:  # Gray
                    status = 1
                    if self.Status[track_id] < 6:
                        self.Status[track_id] += status
                    if label < 6 and self.Gray2Blue.get(classify_label) == label:
                        self.Track_value[int(track_id)][int(float(
                            self.Gray2Blue[classify_label]))] += 0.5 + conf * 0.5
                    elif label > 5 and self.Gray2Red.get(classify_label) == label:
                        self.Track_value[int(track_id)][int(float(
                            self.Gray2Red[classify_label]))] += 0.5 + conf * 0.5
                    else:
                        classify_label = -1
                else:
                    if self.Status[int(track_id)] > 0:
                        self.Status[int(track_id)] -= 1
                    if self.Status[int(track_id)] < 4:
                        self.Status[int(track_id)] = 0
                    # 高置信度纠正：当前帧分类与投票结果不一致时，加大权重
                    vote_weight = 0.5 + conf * 0.5
                    if conf > 0.85 and label != int(float(classify_label)) and max(self.Track_value[int(track_id)]) > 0:
                        vote_weight = 2.0 + conf * 2.0  # 高置信度不一致时给 4 倍权重纠正
                    self.Track_value[int(track_id)][int(float(
                        classify_label))] += vote_weight

            label = self.Track_value[int(track_id)].index(
                max(self.Track_value[int(track_id)]))

            # 判重
            if label < len(exist_armor) and exist_armor[label] != -1:
                old_id = exist_armor[label]
                if self.Track_value[int(track_id)][label] < self.Track_value[int(old_id)][label]:
                    self.Track_value[int(track_id)][label] = 0
                    label = "NULL"
                else:
                    self.Track_value[int(old_id)][label] = 0
                    old_id_index = self.id_candidate[old_id]
                    if old_id_index < len(draw_candidate):
                        draw_candidate[old_id_index][5] = "NULL"
                    exist_armor[label] = track_id
            else:
                if label < len(exist_armor):
                    exist_armor[label] = track_id

            pd = self.Track_value[int(track_id)][0]
            same = True
            for j in range(self.class_num - 1):
                if pd != self.Track_value[int(track_id)][j + 1]:
                    same = False
                    break
            if not same and label != "NULL":
                label = str(self.labels[label])
            else:
                label = "NULL"

            x_left = int(x - w / 2)
            y_left = int(y - h / 2)
            x_right = int(x + w / 2)
            y_right = int(y + h / 2)
            xywh_box = [x, y, w, h]
            xyxy_box = [x_left, y_left, x_right, y_right]
            draw_candidate.append([track_id, x_left, y_left, x_right, y_right, label])
            zip_results.append([xyxy_box, xywh_box, track_id, label])
            self.id_candidate[track_id] = index
            index += 1

        # 在图像上画出检测结果
        for box in draw_candidate:
            tid, x1, y1, x2, y2, lbl = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 3)
            cv2.putText(frame, lbl, (x1 - 10, y2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
            cv2.putText(frame, str(tid), (x2 + 5, y2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)

        self.loop_times += 1
        return frame, zip_results

    # ================================================================
    #  主推理循环
    # ================================================================
    def _infer_loop(self):
        """主循环：推理 → 透视变换 → 发布位置 → 显示小地图"""
        # 等待标定完成
        while self.H_ground is None and rclpy.ok():
            self.get_logger().warn("等待透视变换标定 ...")
            time.sleep(1.0)

        cv2.namedWindow("CameraDetector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CameraDetector", 1920, 1080)
        cv2.namedWindow("MiniMap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MiniMap", 800, 471)

        start_time = time.time()
        # 推理用分辨率
        INFER_W, INFER_H = 1920, 1080

        while rclpy.ok():
            now = time.time()
            fps = 1.0 / max(now - start_time, 1e-6)
            start_time = now

            try:
                cv_image = self._get_frame()
                if cv_image is None:
                    time.sleep(0.01)
                    continue

                # 动态获取原始图像分辨率
                ORIG_H, ORIG_W = cv_image.shape[:2]

                # 缩放到推理分辨率
                infer_image = cv2.resize(cv_image, (INFER_W, INFER_H))

                # 推理
                result_img, results = self._infer(infer_image)

                # ---------- 透视变换 + 发布 ----------
                allLocation = Locations()
                carList_results = []

                if results is not None:
                    # ==========================================
                    # tunning: 1. 数据标准化：将 YOLO 结果打包为基建协议格式
                    # ==========================================
                    detections = []
                    if results is not None:
                        for res in results:
                            xyxy_box, xywh_box, track_id, label = res
                            detections.append(SingleDetectionResult(
                                xyxy=xyxy_box,
                                xywh=xywh_box,
                                label=label,
                                conf=1.0,  # tunning: 此处复用已过滤好的框，默认给1.0
                                track_id=track_id
                            ))

                    # ==========================================
                    # tunning: 2. 匈牙利核心推演 (卡尔曼平滑 + ID关联)
                    # ==========================================
                    # tunning: 无论当前帧有无 YOLO 检测，都必须调用 update 推动卡尔曼滤波器向前推演
                    active_robots = self.hungarian.update(detections)

                    # ==========================================
                    # tunning: 3. 坐标投影与下发：仅使用平滑后的卡尔曼预测框
                    # ==========================================
                    # // tunning: 新增防分身集合，记录本帧已经露头的真实车辆
                    published_car_ids = set()
                    for robot in active_robots:
                        # tunning: 如果目标已经深度丢失（丢失帧数超过阈值） (进入掩体吸附状态)，由下游的 point_guesser 节点接管，相机节点停止发布
                        if robot.state == TrackingState.GUESSING:
                            continue
                            
                        # tunning: 身份投票决议：取出历史投票中最多的 label 作为最终身份
                        best_label = max(robot.vote_pool, key=robot.vote_pool.get) if robot.vote_pool else "NULL"
                        
                        # tunning: 过滤己方车辆 (复用原有逻辑)
                        car_id = self.carList.get_car_id(best_label) if best_label != "NULL" else -1
                        if car_id == -1 and best_label == "NULL":
                            car_id = 9000 + robot.id  # tunning: 测试模式临时ID
                        elif car_id == -1:
                            continue
                        # // tunning: ★ 判定敌我身份，但不再 continue
                        # 敌人的定义：颜色不同 且 不是 -1 (1-7 是红，101-107 是蓝)
                        is_enemy = (self.my_color == "Red" and car_id >= 100) or \
                                   (self.my_color == "Blue" and car_id < 100 and car_id != -1)
                        robot.is_enemy = is_enemy # 标记身份，给后面的 guesser 用

                        # ==========================================
                        # 你原有的核心坐标计算逻辑 (保持原封不动，确保 field_x 不为 None)
                        # ==========================================
                        kf_cx, kf_cy, kf_w, kf_h = robot.bbox_kf_state[:4]
                        orig_cx = kf_cx * ORIG_W / INFER_W
                        orig_cy = kf_cy * ORIG_H / INFER_H
                        orig_h = kf_h * ORIG_H / INFER_H
                        orig_w = kf_w * ORIG_W / INFER_W
                        
                        px = orig_cx
                        py = orig_cy + (orig_h / 2.0)

                        field_coord = self.pixel_to_field(px, py)
                        if field_coord is None:
                            continue
                        field_x, field_y = field_coord

                        field_x = max(0.0, min(28.0, field_x))
                        field_y = max(0.0, min(15.0, field_y))

                        # 修改：注释kalman_filter 开始
                        # field_x, field_y = self.kf_wrapper.update(car_id, field_x, field_y)
                        # field_x = max(0.0, min(28.0, field_x))
                        # field_y = max(0.0, min(15.0, field_y))

                        field_xyz = np.array([field_x, field_y, 0.0])

                        if robot.state == TrackingState.TRACKING and robot.field_x is not None:
                            dx, dy = field_x - robot.field_x, field_y - robot.field_y
                            jump_dist = (dx**2 + dy**2)**0.5
                            if jump_dist < 0.3:
                                raw_vx, raw_vy = dx / 0.033, dy / 0.033
                                robot.field_vx = 0.5 * getattr(robot, 'field_vx', 0.0) + 0.5 * raw_vx
                                robot.field_vy = 0.5 * getattr(robot, 'field_vy', 0.0) + 0.5 * raw_vy
                        
                        robot.field_x, robot.field_y = field_x, field_y

                        # // tunning: ★ 发布门控 - 只有敌人或己方7号/107号才发布坐标
                        is_my_7 = (self.my_color == "Red" and car_id == 7) or \
                                  (self.my_color == "Blue" and car_id == 107)
                        
                        if is_enemy or is_my_7:
                            # 标注、组装 CarList、记录 ID
                            cv2.putText(result_img, f"({field_x:.1f},{field_y:.1f}) {robot.state.name}",
                                        (int(kf_cx - kf_w / 2.0), int(kf_cy - kf_h / 2.0) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            
                            if best_label != "NULL":
                                carList_results.append([robot.id, car_id, [float(orig_cx), float(orig_cy), float(orig_w), float(orig_h)], 1, [0.0,0.0,0.0], field_xyz])

                            published_car_ids.add(car_id)
                            loc = Location()

                            loc.x, loc.y, loc.z = float(field_x), float(field_y), 0.0
                            loc.id, loc.label = int(car_id), ("Red" if car_id < 100 else "Blue")
                            allLocation.locs.append(loc)


                    # ==========================================
                    # tunning: 4. guess_pts.py 接管深度丢失目标，执行带衰减的惯性外推
                    # ==========================================
                    # 此处放在循环外，确保所有可见机器人的 field_vx/vy 已更新完毕
                    # // tunning: ★ 只有敌方机器人才传给 guesser，己方不参与预测
                    enemies = [r for r in active_robots if getattr(r, 'is_enemy', False)]
                    self.guesser.update(enemies)
 

                    # tunning: 5. 遍历列表，将处于 GUESSING 状态的推演坐标也压入发布队列
                    for robot in active_robots:
                        if robot.state == TrackingState.GUESSING:
                            # // tunning: ★ 核心过滤 - 如果是己方机器人（包括 7 号），钻进掩体就消失，不许发布紫色盲猜坐标
                            if not getattr(robot, 'is_enemy', False):
                                continue
                            # 身份逻辑：盲猜期间沿用最后一次确定的身份
                            best_label = max(robot.vote_pool, key=robot.vote_pool.get) if robot.vote_pool else "NULL"
                            car_id = self.carList.get_car_id(best_label) if best_label != "NULL" else (9000 + robot.id)
                            
                            # // tunning: ★防分身拦截核心！如果该 ID 的真身已被观测到，或身份无效，立刻停止发布它的盲猜幽灵
                            if car_id in published_car_ids or car_id == -1:
                                continue
                                
                            # 3秒门控由 guess_pts.py 内部控制，此处直接打包
                            loc = Location()
                            loc.x = float(robot.field_x)
                            loc.y = float(robot.field_y)
                            loc.z = -1.0 # // tunning: 悄悄把 z 设为 -1.0，作为给 display_panel 的“盲猜暗号”
                            
                            loc.id = int(car_id)
                            loc.label = "Red" if loc.id < 100 else "Blue"
                            allLocation.locs.append(loc)

                    # ==========================================
                    # tunning : 更新 CarList 与 发布 (原有逻辑)
                    # ==========================================
                    if carList_results:
                        self.carList.update_car_info(carList_results)
                    
                    # 此时 allLocation 已经包含了“观测点”和“盲猜点”
                    self.pub_location.publish(allLocation)

                # 修改：注释kalman_filter 开始
                # # ★ 定期清理超时的卡尔曼滤波器
                # self._cleanup_counter += 1
                # if self._cleanup_counter % 100 == 0:
                #     self.kf_wrapper.cleanup()

                # ---------- 在推理图像上显示 FPS ----------
                cv2.putText(result_img, f"FPS: {fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.imshow("CameraDetector", result_img)

                # ---------- 小地图可视化 ----------
                self._draw_minimap(allLocation)

                cv2.waitKey(1)

            except Exception as e:
                self.get_logger().error(f"推理循环异常: {e}")
                import traceback
                traceback.print_exc()

    # ================================================================
    #  小地图可视化
    # ================================================================
    def _draw_minimap(self, locations):
        """
        在小地图上绘制机器人位置。
        地图图片尺寸映射: 赛场 28m×15m → 图片 2800×1500 像素（100 px/m）

        地图方向：红方在左(x=0)，蓝方在右(x=28)，与世界坐标系一致。赛场坐标直接映射到地图像素，无需镜像。
        """
        show_map = self.map_img.copy()

        for loc in locations.locs:
            x = round(loc.x, 2)
            y = round(loc.y, 2)

            # // tunning: 提取 z 坐标，接收盲猜暗号
            z = round(loc.z, 2)

            # 赛场坐标 → 地图像素（地图与世界坐标系方向一致，无需镜像）
            map_xx = max(0, min(int(x * 100), 2800))
            map_yy = max(0, min(int(1500 - y * 100), 1500))
            disp_x = x
            disp_y = y

            # // tunning: 判断是否为视觉推演的“幽灵点”
            is_guessing = (z < 0.0)

            # // tunning: 机器人颜色不仅决定阵营，还决定是否为盲猜状态
            if loc.label == 'Red':
                # 红方：盲猜点画紫色 (BGR: 255, 0, 255)，正常画红色
                color = (255, 0, 255) if is_guessing else (0, 0, 255)
            else:
                # 蓝方：盲猜点画青蓝色 (BGR: 255, 255, 0)，正常画蓝色
                color = (255, 255, 0) if is_guessing else (255, 100, 0) # tunning: 蓝色改为更亮的橙蓝色，增强视觉区分度

            # 绘制圆圈和编号
            cv2.circle(show_map, (map_xx, map_yy), 60, color, 4)
            cv2.putText(show_map, str(loc.id), (map_xx - 15, map_yy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            cv2.putText(show_map,
                        f"{disp_x},{disp_y}",
                        (map_xx, map_yy - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        cv2.imshow("MiniMap", show_map)

    # ================================================================
    #  析构
    # ================================================================
    def __del__(self):
        self.get_logger().info('CameraDetector 节点正在销毁。')


# ================================================================
#  入口
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = CameraDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
