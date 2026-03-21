#!/usr/bin/env python3
"""
ekf_node.py — 扩展卡尔曼滤波（EKF）节点

功能：
  订阅检测节点发布的敌方机器人赛场坐标（location 话题），
  对每辆敌方机器人独立运行 EKF 滤波，平滑位置并估计速度，
  将滤波后的坐标通过 ekf_location_filtered 话题发布。

数据流：
  detector_node → [location] → ekf_node → [ekf_location_filtered] → judge_messager

订阅话题：
  - location (Locations) — 原始检测坐标

发布话题：
  - ekf_location_filtered (Locations) — 滤波后坐标

核心逻辑：
  - 维护 6 个独立的 RobotEKF 实例（对应敌方 6 辆机器人）
  - 每 50ms 定时触发一次滤波步进
  - 状态向量 [x, vx, y, vy]，观测向量同维
  - 通过 RobotInfo 计算帧间速度和加速度，作为过程模型输入

依赖：
  - ekf.RobotEKF — 基于 tinyekf 的 EKF 实现
  - detect_result.msg — Location / Locations 自定义消息

tunning 说明：
    - 2025-03-17: 初始修改版本，在time!=0时仍执行predict-only，修正历史队列更新逻辑
"""

import os
import time  # tunning: 新增 time 模块，用于遮挡预测的超时控制

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from detect_result.msg import Location, Locations
from detect_result.msg import DetectResult
from detect_result.msg import Robots
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from ekf.RobotEKF import RobotEKF

# ---------- 读取 main_config.yaml ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', '..'))
_MAIN_CONFIG = os.path.join(_PROJECT_ROOT, 'configs', 'main_config.yaml')

def _load_debug_flag() -> bool:
    """从 main_config.yaml 读取 global.debug_coordinate_publish 配置。
    缺省值 False（赛场安全）。"""
    try:
        import yaml
        with open(_MAIN_CONFIG, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return bool(cfg.get('global', {}).get('debug_coordinate_publish', False))
    except Exception:
        return False

# 机器人信息类
class RobotInfo(object):

    def __init__(self, time, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.time = time
        self.v_x = 0
        self.v_y = 0
        self.last_v_x = 0 # 上一次的速度
        self.last_v_y = 0 # 上一次的速度
        self.a_x = 0   # V1 = V0 + at   -> a = (v1-v0)/t
        self.a_y = 0

    def __str__(self):
        return '%4d %4d %4d %4d' % (self.time, self.x, self.y, self.z)
    
    # 计算速度和加速度
    def calculateInfo(self, last_info):

        if last_info.x > 0 and last_info.y > 0:
            t = (self.time - last_info.time) / 1000.0  # tunning: ms转s
            if abs(t) < 1e-6:  # 避免除零
                return

            self.v_x = (self.x - last_info.x) / t
            self.v_y = (self.y - last_info.y) / t

            self.a_x = (self.v_x - last_info.v_x) / t
            self.a_y = (self.v_y - last_info.v_y) / t

            # tunning: 保存当前速度供下一帧计算加速度
            self.last_v_x = self.v_x
            self.last_v_y = self.v_y





# EKF节点类
class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        
        # QoS 设置
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # 记录敌方颜色
        self.eneny_color = 'NULL'
        # ── 调试坐标发布开关（从 main_config.yaml 读取）──
        # True  = 调试: 所有检测坐标都发布（空中ID映射到6/106, 未知ID透传）
        # False = 赛场: 仅标准ID (1-7/101-107) 经EKF发布, 其余丢弃
        self.debug_coordinate_publish = _load_debug_flag()
        self.get_logger().info(
            f'📋 debug_coordinate_publish = {self.debug_coordinate_publish} '
            f'{"(调试模式: 全部坐标发布)" if self.debug_coordinate_publish else "(赛场模式: 仅标准ID)"}')
        # 机器人ID映射到数组下标
        # 地面机器人: 1-5/101-105 → slot 1-5
        # 哨兵:     7/107       → slot 6
        # 空中机器人: 6/106       → slot 7
        self.transform_to_th = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 7,       # 红方空中机器人 → 第7个滤波器
            7: 6,
            101: 1,
            102: 2,
            103: 3,
            104: 4,
            105: 5,
            106: 7,     # 蓝方空中机器人 → 第7个滤波器
            107: 6
        }
        # 初始化7个机器人的EKF滤波器（6个地面 + 1个空中），修改噪声参数q，r减小突变对数据的影响
        self.kalfilt = [
            RobotEKF(4, 4, pval=0.001, qval=1e-4, rval=0.0005, interval=0.05) for _ in range(7)
        ]

        # 创建发布者和订阅者
        self.pub_location_filtered = self.create_publisher(Locations, 'ekf_location_filtered', qos_profile)
        self.sub_location = self.create_subscription(Locations, "location", self.location_callback, qos_profile)
        # 初始化变量
        self.recv_location = Locations()
        self.last_location = Locations()
        # 创建定时器
        self.timer = self.create_timer(0.05, self.timer_callback)  # 50 ms
        
        # 初始化位置队列，保存上次和当前的机器人位置信息 [0]是上次的， [1]是当前的
        self.locations_queue = [[RobotInfo(0,0,0, 0), RobotInfo(0,0,0,0), RobotInfo(0,0,0,0), RobotInfo(0,0,0,0),RobotInfo(0,0,0,0),RobotInfo(0,0,0,0),RobotInfo(0,0,0,0)] for i in range(2)]

    # 获取当前时间的毫秒数    
    def get_current_time_ms(self):
        current_time = self.get_clock().now()
        time_msg = current_time.to_msg()
        milliseconds = time_msg.sec * 1000 + time_msg.nanosec / 1e6
        return milliseconds

    # 定时器回调函数    
    def timer_callback(self):
        # tuning： timer_callback 改为 DEBUG，不阻塞定时器的正常运行，保留时间戳输出以便调试
        # 运行时控制日志级别： ros2 run ekf_node --ros-args --log-level ekf_node:=INFO  # 隐藏 DEBUG
        self.get_logger().debug(f"Timestamp: {self.get_clock().now()}")  # 只在调试时打印


        # self.get_logger().info(f"Timestamp: {self.get_current_time_ms()}, Location: {self.recv_location.__str__()}")
        locations = self.recv_location

            # tunning: 清空本帧的观测时间戳，防止上一帧数据被复用
        for i in range(7):
            self.locations_queue[1][i].time = 0
        # 更新测量值
        for i in range(len(locations.locs)):
            location = locations.locs[i]
            robot_id = location.id
            # air_scheme 非严格模式产生的 ID 范围映射
            if 600 <= robot_id < 1000:
                if not self.debug_coordinate_publish:
                    continue            # 赛场模式: 丢弃, 不发裁判系统
                robot_id = 6            # 调试模式: 映射为红方空中标准 ID
            elif 1600 <= robot_id < 2000:
                if not self.debug_coordinate_publish:
                    continue            # 赛场模式: 丢弃
                robot_id = 106          # 调试模式: 映射为蓝方空中标准 ID
            # 跳过 ID 不在映射表中的机器人
            if robot_id not in self.transform_to_th:
                continue
            x = location.x
            y = location.y
            time = self.get_current_time_ms()
            if self.eneny_color == "NULL":
                self.eneny_color = location.label
            self.locations_queue[1][self.transform_to_th[robot_id] - 1].x = x
            self.locations_queue[1][self.transform_to_th[robot_id] - 1].y = y
            self.locations_queue[1][self.transform_to_th[robot_id] - 1].time = time
            self.locations_queue[1][self.transform_to_th[robot_id] - 1].z = location.z
        estimated_locations = [[0,0,0,0], [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        # 对每个机器人进行卡尔曼滤波
        for i in range(len(self.locations_queue[0])):
         # tunning: BUG FIX 核心改动——修正条件判断逻辑，启用predict-only模式
            # 原代码: if self.locations_queue[1][i].time == 0: continue
            # 问题: time==0表示无新观测，却跳过整个滤波，导致无观测时不做预测
            # 改正: 改为 if self.locations_queue[1][i].time != 0，有新观测才进行 update
            if self.locations_queue[1][i].time != 0:
                # 有新观测：执行完整的 update 步骤
                # 计算加速度
                self.locations_queue[1][i].calculateInfo(self.locations_queue[0][i])
                self.kalfilt[i].update_acceleration(self.locations_queue[1][i].a_x, self.locations_queue[1][i].a_y)
                estimated_locations[i] = (self.kalfilt[i].step((self.locations_queue[1][i].x, self.locations_queue[1][i].v_x, self.locations_queue[1][i].y, self.locations_queue[1][i].v_y)))
            else:
                # tunning: 无新观测时，EKF 自动进行 predict-only
                estimated_locations[i] = self.kalfilt[i].predict_only()
            
        ## tunning: BUG FIX 历史队列更新逻辑反转
        # 原代码: if self.locations_queue[1][i].time == 0: 保存历史
        # 问题: 逻辑反了！time==0表示无观测，不应该保存
        #       这导致历史队列被无观测数据污染，下一帧计算速度/加速度就错了
        # 改正: 改为 if self.locations_queue[1][i].time != 0，只在有新观测时保存历史
        for i in range(len(self.locations_queue[0])):
            if self.locations_queue[1][i].time != 0:
                self.locations_queue[0][i].time = self.locations_queue[1][i].time
                self.locations_queue[0][i].x = self.locations_queue[1][i].x
                self.locations_queue[0][i].y = self.locations_queue[1][i].y
                self.locations_queue[0][i].z = self.locations_queue[1][i].z
                self.locations_queue[0][i].v_x = self.locations_queue[1][i].v_x
                self.locations_queue[0][i].v_y = self.locations_queue[1][i].v_y
            self.get_logger().debug(f"Estimated locations: {estimated_locations}") # tunning: 等级降级为DEBUG,防止阻塞定时器
        #     # 如果没有新的数据,则跳过
        #     if self.locations_queue[1][i].time == 0:
        #         continue
        #     # 计算加速度
        #     self.locations_queue[1][i].calculateInfo(self.locations_queue[0][i])
        #     self.kalfilt[i].update_acceleration(self.locations_queue[1][i].a_x, self.locations_queue[1][i].a_y)
        #     estimated_locations[i] = (self.kalfilt[i].step((self.locations_queue[1][i].x, self.locations_queue[1][i].v_x, self.locations_queue[1][i].y, self.locations_queue[1][i].v_y)))
            
        # ## 保存历史位置（每帧更新，用于下一帧速度/加速度计算） 等测试完成之后再考虑优化
        # for i in range(len(self.locations_queue[0])):
        #     # if self.locations_queue[1][i].time > 0:
        #     if self.locations_queue[1][i].time == 0:
        #         self.locations_queue[0][i].time = self.locations_queue[1][i].time
        #         self.locations_queue[0][i].x = self.locations_queue[1][i].x
        #         self.locations_queue[0][i].y = self.locations_queue[1][i].y
        #         self.locations_queue[0][i].z = self.locations_queue[1][i].z
        #         # self.locations_queue[0][i].v_x = self.locations_queue[1][i].v_x # 等测试完成之后再考虑优化
        #         # self.locations_queue[0][i].v_y = self.locations_queue[1][i].v_y # 等测试完成之后再考虑优化
        
        
        
        
        # 发布滤波后的位置信息
        # 数组下标到 ID 的映射:
        #   index 0-4 → 地面机器人 1-5 / 101-105
        #   index 5   → 哨兵 7 / 107
        #   index 6   → 空中机器人 6 / 106
        th_to_id_red = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 6}
        th_to_id_blue = {0: 101, 1: 102, 2: 103, 3: 104, 4: 105, 5: 107, 6: 106}
        locations = Locations()
        for i in range(len(estimated_locations)):
            location = Location()
            if estimated_locations[i][0] == 0 or estimated_locations[i][2] == 0:
                continue
            if self.eneny_color == 'Blue':
                location.id = th_to_id_blue.get(i, i + 101)
            elif self.eneny_color == 'Red':
                location.id = th_to_id_red.get(i, i + 1)
            else:
                continue
            # 空中机器人 (id=6 或 106) 用 "Air" 标签区分
            if location.id in (6, 106):
                location.label = self.eneny_color
            else:
                location.label = self.eneny_color
            location.x = float(estimated_locations[i][0])
            location.y = float(estimated_locations[i][2])
            location.z = float(self.locations_queue[1][i].z)
            locations.locs.append(location)
        # 非标准 ID 处理：仅调试模式下透传, 赛场模式全部丢弃
        if self.debug_coordinate_publish:
            for loc in self.recv_location.locs:
                if loc.id not in self.transform_to_th:
                    # 空中 ID 已在上方映射处理, 此处跳过避免重复
                    if 600 <= loc.id < 1000 or 1600 <= loc.id < 2000:
                        continue
                    locations.locs.append(loc)
        # 通过话题发布
        self.pub_location_filtered.publish(locations)
            
                
    # 订阅位置信息的回调函数
    def location_callback(self, msg):
        # self.get_logger().info(f"Received location: {msg}")
        self.recv_location = msg
        
    

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
