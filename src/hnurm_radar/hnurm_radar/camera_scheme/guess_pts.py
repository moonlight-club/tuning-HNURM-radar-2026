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
说明：
  -目前guessing_pts.py的推演逻辑仍然保持相对简单，主要依赖于速度衰减和时间门控来抑制预测发散,实际预测效果不理想。
  - 未来可考虑引入更复杂的运动模型（如加速度项、转向模型）或基于历史轨迹的学习式预测，以提升盲猜阶段的准确性和鲁棒性。
"""

import numpy as np
from hnurm_radar.shared.type import RobotState, TrackingState

class PointGuesser:
    """GUESSING 阶段的赛场坐标推演器。"""

    def __init__(self, decay_factor: float = 0.94, max_guess_sec: float = 1.3, max_speed: float = 6.0):
        """
        初始化推演器的动力学约束与安全门控参数。

        机制说明：
            - 仅在目标处于 `TrackingState.GUESSING` 时生效。
            - 通过指数衰减抑制速度外推的长期发散。
            - 通过最大盲猜时长与速度上限，限制“过期轨迹”和“飞点”风险。

        参数:
            decay_factor (float): 每帧速度衰减系数，范围建议为 [0, 1]。
                数值越小，速度衰减越快，轨迹越保守。
            max_guess_sec (float): 单条轨迹连续盲猜的最大持续时间（秒）。
                超时后清空位置与速度，停止该轨迹的推演输出。
            max_speed (float): 盲猜阶段允许的速度模长上限（m/s）。
                当速度超过该阈值时会按比例缩放，保持方向不变。

        成员约束:
            - 坐标边界固定为赛场范围 x∈[0, 28], y∈[0, 15]。
            - 内部保留 `fps/dt` 仅用于兼容历史接口；当前推演以 `update()` 传入的实时 `dt` 为准。
        """
        self.decay_factor = decay_factor
        self.max_guess_sec = max_guess_sec
        self.max_speed = max_speed  # 盲猜阶段速度上限 (m/s)，用于抑制异常外推位移
        self.fps = 30.0  # 历史兼容字段：固定帧率参数，当前推演流程不直接使用
        self.dt = 1.0 / self.fps  # 历史兼容默认步长：当前以 update(dt) 实时入参为准
        
        # 赛场绝对物理边界约束 (单位: 米)
        self.x_limit = (0.0, 28.0)
        self.y_limit = (0.0, 15.0)

    def update(self, active_robots: list, dt: float = 0.033):
        """
        对活跃轨迹执行一帧 GUESSING 推演。

        逻辑概要：
            - 限幅 `dt`，避免异常帧引起位移突增；
            - 仅更新 `GUESSING` 目标，超时或冲突时直接失效；
            - 对有效速度做衰减与限幅，再按 `dt` 积分；
            - 结果坐标最终钳制在赛场边界内。

        参数:
            active_robots (list): 当前帧机器人状态列表（原地更新）。
            dt (float): 当前帧实时时间步长（秒），内部限幅到 [0.01, 0.1]。
        """
        # 对 dt 做数值限幅，避免异常帧导致单步积分位移过大
        dt = max(0.01, min(0.1, float(dt)))

        # 提取 TRACKING 目标坐标，作为 GUESSING 轨迹的空间冲突基准
        tracking_positions = [
            (r.field_x, r.field_y) for r in active_robots 
            if r.state == TrackingState.TRACKING and r.field_x is not None
        ]

        for robot in active_robots:
            if robot.state == TrackingState.GUESSING:
                if robot.field_x is None or robot.field_y is None:
                    # 坐标缺失时保持失效状态并跳过推演
                    robot.field_x, robot.field_y = None, None
                    continue

                # 使用实际动态时间步长估算丢失时长
                lost_duration = robot.miss_cnt * dt
                if lost_duration > self.max_guess_sec:
                    # 超过最大盲猜时长后终止该轨迹，并清零速度状态
                    robot.field_x, robot.field_y = None, None
                    if hasattr(robot, 'field_vx'):
                        robot.field_vx = 0.0
                    if hasattr(robot, 'field_vy'):
                        robot.field_vy = 0.0
                    continue

                # 空间冲突抑制：若与任一 TRACKING 目标距离小于阈值，则终止该推演轨迹
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
                    # 对速度执行指数衰减，降低长时外推误差
                    robot.field_vx *= self.decay_factor
                    robot.field_vy *= self.decay_factor

                    # 对速度模长限幅，抑制异常速度引起的外推发散
                    v_norm = np.hypot(robot.field_vx, robot.field_vy)
                    if v_norm > self.max_speed and v_norm > 1e-6:
                        scale = self.max_speed / v_norm
                        robot.field_vx *= scale
                        robot.field_vy *= scale

                    # 使用实时 dt 执行位置积分
                    robot.field_x += robot.field_vx * dt
                    robot.field_y += robot.field_vy * dt
                else:
                    # 无有效速度输入时终止推演，避免生成不可信位移
                    robot.field_x, robot.field_y = None, None
                    continue

                # 边界钳制：预留 0.1m 安全边距，避免坐标落在场地边界外
                robot.field_x = max(0.1, min(self.x_limit[1] - 0.1, robot.field_x))
                robot.field_y = max(0.1, min(self.y_limit[1] - 0.1, robot.field_y))