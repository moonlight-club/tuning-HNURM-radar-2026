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
    def __init__(self, decay_factor: float = 0.94, max_guess_sec: float = 1.3, max_speed: float = 6.0):
        """
        初始化推演器。
        参数:
            decay_factor: 每帧速度衰减系数，取值范围 [0, 1]
            max_guess_sec: 最大允许盲猜的持续时间（秒）
        """
        self.decay_factor = decay_factor
        self.max_guess_sec = max_guess_sec
        self.max_speed = max_speed  # // tunning: 盲猜速度上限(m/s)，防止飞点
        self.fps = 30.0
        self.dt = 1.0 / self.fps
        
        # 赛场绝对物理边界约束 (单位: 米)
        self.x_limit = (0.0, 28.0)
        self.y_limit = (0.0, 15.0)

    # // tunning: 接收主循环传来的动态真实时间差 dt
    def update(self, active_robots: list, dt: float = 0.033):
        """
        执行一帧物理推演，并引入空间竞争排他机制。
        """
        # // tunning: 限幅 dt，避免卡顿帧导致一次积分位移过大
        dt = max(0.01, min(0.1, float(dt)))

        # 提取当前所有视觉锁定的真实机器人坐标
        tracking_positions = [
            (r.field_x, r.field_y) for r in active_robots 
            if r.state == TrackingState.TRACKING and r.field_x is not None
        ]

        for robot in active_robots:
            if robot.state == TrackingState.GUESSING:
                if robot.field_x is None or robot.field_y is None:
                    # // tunning: 超过最大盲猜时长，强行抹除坐标信息
                    robot.field_x, robot.field_y = None, None
                    continue

                # // tunning: 使用实际动态时间步长计算丢失时间，更加严谨
                lost_duration = robot.miss_cnt * dt
                if lost_duration > self.max_guess_sec:
                    # // tunning: 超时后停止发布并清空速度，防止“飞天残留”
                    robot.field_x, robot.field_y = None, None
                    if hasattr(robot, 'field_vx'):
                        robot.field_vx = 0.0
                    if hasattr(robot, 'field_vy'):
                        robot.field_vy = 0.0
                    continue

                # 2. // tunning: 幽灵劫持抑制逻辑 (Ghost Suppression)
              # 若推演坐标与任一视觉追踪目标的距离小于 0.3m，则判定该推演轨迹冗余，停止输出。
                is_conflicted = False
                for tx, ty in tracking_positions:
                    dist = np.hypot(robot.field_x - tx, robot.field_y - ty)
                    if dist < 0.3: 
                        is_conflicted = True
                        break
                
                if is_conflicted:
                    # 判定为身份劫持，立即终止该预测轨迹
                    robot.field_x, robot.field_y = None, None
                    if hasattr(robot, 'field_vx'):
                        robot.field_vx = 0.0
                    if hasattr(robot, 'field_vy'):
                        robot.field_vy = 0.0
                    continue

                if hasattr(robot, 'field_vx') and hasattr(robot, 'field_vy'):
                    # 速度衰减机制（可依据真实 dt 做进一步非线性优化，此处保持原有逻辑）
                    robot.field_vx *= self.decay_factor
                    robot.field_vy *= self.decay_factor

                    # // tunning: 速度向量限幅，防止异常速度导致预测爆炸
                    v_norm = np.hypot(robot.field_vx, robot.field_vy)
                    if v_norm > self.max_speed and v_norm > 1e-6:
                        scale = self.max_speed / v_norm
                        robot.field_vx *= scale
                        robot.field_vy *= scale

                    # // tunning: 严格依据真实物理时间步 dt 执行积分位移
                    robot.field_x += robot.field_vx * dt
                    robot.field_y += robot.field_vy * dt
                else:
                    # // tunning: 无有效速度时不推演，避免随机漂移
                    robot.field_x, robot.field_y = None, None
                    continue

                robot.field_x = max(0.1, min(self.x_limit[1] - 0.1, robot.field_x))
                robot.field_y = max(0.1, min(self.y_limit[1] - 0.1, robot.field_y))