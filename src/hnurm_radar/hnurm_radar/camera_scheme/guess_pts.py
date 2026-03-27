"""
guess_pts.py — 视觉惯性推演器，用于目标深度丢失时的位置预测
==========================================================
当被跟踪的目标进入 GUESSING 状态（图像连续漏检）时，接管其赛场物理坐标。
基于机器人消失前的瞬时物理速度 (vx, vy)，在绝对坐标系下进行惯性外推。

状态接管: TrackingState.GUESSING
推演模型: 匀速运动学模型 + 速度指数衰减
坐标系:   赛场绝对物理坐标系 (原点为红方补给站交点)

特性:
  - 速度衰减机制: 随漏检帧数增加执行速度衰减，防止预测轨迹无限发散
  - 物理边界约束: 强制将预测坐标限制在 28m x 15m 赛场围栏内
  - 最大时长门控: 连续丢失超过预设时间（如 3s）后停止推演，防止产生过期幻影
  - 阵营无关性: 内部计算不涉及镜像，镜像逻辑由下游绘图根据 label 判定
  - 伪观测输出: 为下游 EKF 提供持续的推演输入，维持全局轨迹平滑
"""

import numpy as np
from hnurm_radar.shared.type import RobotState, TrackingState

class PointGuesser:
    def __init__(self, decay_factor: float = 0.96, max_guess_sec: float = 3.0):
        """
        初始化推演器。
        参数:
            decay_factor: 每帧速度衰减系数，取值范围 [0, 1]
            max_guess_sec: 最大允许盲猜的持续时间（秒）
        """
        self.decay_factor = decay_factor
        self.max_guess_sec = max_guess_sec
        self.fps = 30.0
        self.dt = 1.0 / self.fps
        
        # 赛场绝对物理边界约束 (单位: 米)
        self.x_limit = (0.0, 28.0)
        self.y_limit = (0.0, 15.0)

     # --- [修改后] ---
    # // tunning: 接收主循环传来的动态真实时间差 dt
    def update(self, active_robots: list, dt: float = 0.033):
        for robot in active_robots:
            if robot.state == TrackingState.GUESSING:
                if robot.field_x is None or robot.field_y is None:
                    continue

                # // tunning: 使用实际动态时间步长计算丢失时间，更加严谨
                lost_duration = robot.miss_cnt * dt
                if lost_duration > self.max_guess_sec:
                    continue

                if hasattr(robot, 'field_vx') and hasattr(robot, 'field_vy'):
                    # 速度衰减机制（可依据真实 dt 做进一步非线性优化，此处保持原有逻辑）
                    robot.field_vx *= self.decay_factor
                    robot.field_vy *= self.decay_factor

                    # // tunning: 严格依据真实物理时间步 dt 执行积分位移
                    robot.field_x += robot.field_vx * dt
                    robot.field_y += robot.field_vy * dt

                robot.field_x = max(0.1, min(self.x_limit[1] - 0.1, robot.field_x))
                robot.field_y = max(0.1, min(self.y_limit[1] - 0.1, robot.field_y))