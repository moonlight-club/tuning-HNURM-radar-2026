"""
type.py — 雷达站视觉感知与滤波基建数据类型模块

所有涉及纯相机透视方案的视觉检测、目标追踪与坐标滤波的节点，统一从此模块导入基础数据结构与状态机枚举，
避免在各层之间传递松散的列表（如 [x, y, w, h, id]），确保数据类型安全与跨层接口的一致性。

模块交互与核心逻辑映射：

    1. camera_scheme/camera_detector.py（纯相机透视主节点）：
       - 将 YOLO 检测器的原始输出封装为 SingleDetectionResult。
       - 实例化并维护所有赛场实体的 RobotState。
       - 提取 RobotState 中平滑后的底边中心点执行透视变换。

    2. camera_scheme/hungarian_tracker.py（匈牙利匹配tracker维护节点）：
       - 读取 RobotState 中的 bbox_kf_state（卡尔曼预测框）。
       - 与当前帧的 SingleDetectionResult 计算代价矩阵，完成数据关联。

    3. filters/bbox_kalman.py（对YOLO检测框进行滤波并进行短时间遮挡的预测）：
       - 专职读取和更新 RobotState 内部的 8 维 bbox_kf_state 数组。
       - 不涉及任何物理坐标计算。

    4. ekf/ekf_node.py & ekf/guess_pts.py（坐标与目标丢失长时间预测）：
       - 核心职责：作为系统的“战术大脑”，严格依据 RobotState.state 执行差异化的物理学解算。
       - 动作映射：
         * 状态 == TRACKING    -> 执行标准 EKF 观测更新，输出高精度平滑赛场坐标。
         * 状态 == LOST        -> 视觉短暂丢失，EKF 无观测输入，仅执行运动学惯性推演 (Predict)。
         * 状态 == GUESSING    -> 视觉长时丢失，触发 guess_pts 盲猜，向掩体中心输出伪观测值，并持续膨胀误差协方差矩阵 P。
         * 状态 == RE_ACQUIRED -> 目标重新露头，利用极度膨胀的 P 矩阵和卡尔曼增益特性，瞬间将预测坐标拉回至真实观测位置（跨度过大时触发软重置）。

维护须知：
    1. 状态机扩展：若需新增目标状态（如反陀螺状态），必须在 TrackingState 中添加枚举，并同步修改 ekf_node.py 的处理分支。
    2. 维度升级：当前 bbox_kf_state 默认为 8 维 [cx, cy, w, h, vx, vy, vw, vh]。若引入加速度等更高阶状态，需同步修改此处 default_factory 初始化长度以及 bbox_kalman.py 的矩阵维度。
    3. 职责隔离：在 RobotState 中新增数据字段时，务必在注释中标明该字段由哪个节点负责更新，严禁不同节点跨层覆写同一字段。
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

class TrackingState(Enum):
    """
    目标追踪状态机枚举
    用于下游节点判定是否需要触发特殊补偿逻辑
    """
    TRACKING = auto()    # 视觉正常可见并追踪
    LOST = auto()        # 视觉短时丢失（由 bbox_kalman 预测续命）
    GUESSING = auto()    # 视觉长时丢失（由 guess_pts 执行掩体中心吸附）
    RE_ACQUIRED = auto() # 重捕获瞬间（用于触发协方差膨胀与软重置机制）

@dataclass
class SingleDetectionResult:
    """YOLO 单帧检测结果的标准化封装"""
    xyxy: List[float]    # 像素坐标 [x1, y1, x2, y2]
    xywh: List[float]    # 像素中心坐标 [cx, cy, w, h]
    label: str           # 原始兵种标签 (如 "Red 3")
    conf: float          # 检测置信度
    track_id: int        # 原始分配 ID

@dataclass
class RobotState:
    """机器人全局追踪状态容器"""
    id: int              # 最终决议的兵种 ID
    state: TrackingState = TrackingState.TRACKING
    
    # 图像像素空间状态 [cx, cy, w, h, vx, vy, vw, vh]
    # 【写权限】：仅限 filters/bbox_kalman.py
    bbox_kf_state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    # 图像像素空间状态误差协方差矩阵 (8x8)
    # 【写权限】：仅限 filters/bbox_kalman.py
    bbox_kf_cov: np.ndarray = field(default_factory=lambda: np.eye(8))
    
    # 赛场物理空间绝对坐标
    # 【写权限】：仅限 ekf/ekf_node.py (透视变换后)
    field_x: float = None
    field_y: float = None

    # 赛场物理空间瞬时速度 (m/s)
    # 【写权限】：仅限 camera_scheme/camera_detector.py (基于位移差计算)
    # 【读权限】：仅限 camera_scheme/guess_pts.py (用于惯性外推)
    field_vx: float = 0.0
    field_vy: float = 0.0
    
    # 连续漏检帧数计数值
    # 【写权限】：仅限 camera_scheme/camera_detector.py
    miss_cnt: int = 0

    # BoT-SORT 底层帧间关联 ID (物理连续性标识, 非最终兵种身份)
    # 【写权限】：仅限 camera_scheme/hungarian_tracker.py
    # 【用途】：作为匈牙利匹配代价矩阵的辅助维度, 提升遮挡后重关联准确率
    bot_id: int = -1

    # 累计物理命中计数 (被 YOLO 检测到的总帧数)
    # 【写权限】：仅限 camera_scheme/hungarian_tracker.py
    # 【用途】：区分"真车"与"噪音", 决定垃圾回收时的容忍帧数
    hit_cnt: int = 0

    last_seen_time: float = 0.0 # 上次真实观测的绝对时间戳
    
    # 兵种身份投票池: {label_name: count}
    # // tunning: 将 count (int) 升级为 score (float)，以支持基于检测置信度的加权投票。
    vote_pool: Dict[str, float] = field(default_factory=dict)