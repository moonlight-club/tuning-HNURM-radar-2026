"""
display_panel.py — 小地图可视化面板节点

功能：
  订阅 EKF 滤波后的机器人位置信息（/ekf_location_filtered），
  在标准赛场地图（std_map.png）上实时绘制敌我双方机器人位置，
  以 ~15fps 刷新 OpenCV 窗口显示。

坐标映射：
  赛场坐标 (m) → 像素坐标：x_px = x * 100, y_px = 1500 - y * 100
  红方坐标需做对称翻转（28-x, 15-y）以统一到蓝方视角

订阅话题：
  - /ekf_location_filtered (Locations) — EKF 滤波后的机器人位置

依赖：
  - map/std_map.png — 标准赛场底图（2800×1500 像素）
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from detect_result.msg import Location, Locations
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
import threading
import time
import cv2
import numpy as np


class DisplayPanel(Node):

    def __init__(self):
        super().__init__('display_panel')
        self.get_logger().info('Display Panel Node is running')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=3
        )
        self.locations = Locations()
        from .paths import STD_MAP_PATH
        self.map = cv2.imread(STD_MAP_PATH)
        
        self.sub_location = self.create_subscription(Locations, "/ekf_location_filtered", self.location_callback, qos_profile)
        # 创建一个锁
        self.lock = threading.Lock()
        
        cv2.namedWindow('map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('map', 800, 471)
        
    def display_locations(self):
        show_map = self.map.copy()

        cur_locations = []
        with self.lock:
            cur_locations = self.locations

        for i in range(len(cur_locations.locs)):
            location = cur_locations.locs[i]
            x = round(location.x, 2)
            y = round(location.y, 2)
            z = round(location.z, 2)
            
            xx = int( x * 100)
            yy = int( 1500 - y * 100)
            # # 限定xx,yy范围
            # if xx < 0 or xx > 2800 or yy < 0 or yy > 1500:
            #     xx = max(0, min(xx, 2800))
            #     yy = max(0, min(yy, 1500))
                
            # 判断是否为空中机器人：正式ID(6/106) + 调试多目标ID段(600+/1600+)
            is_air_robot = location.id in (6, 106) or location.id >= 600
            air_suffix = " UAV" if is_air_robot else ""
            
            # // tunning: 接收暗号，判断是否为视觉推演的“幽灵点”
            is_guessing = (z < 0.0)

            if location.label == 'Red':
                x = 28 - x
                y = 15 - y
                xx = 2800 - xx
                yy = 1500 - yy
                # // tunning: 如果是盲猜点，画紫色 (BGR格式: 255, 0, 255)；否则画红色
                color = (255, 0, 255) if is_guessing else (0, 0, 255)  
                
                cv2.putText(show_map, str(location.id) + air_suffix, (xx - 15, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                cv2.putText(show_map, str((x)) + ',' + str((y)) + ',' + str(z), (xx, yy - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                if is_air_robot:
                    pts = np.array([[xx, yy-50], [xx+50, yy], [xx, yy+50], [xx-50, yy]], np.int32)
                    cv2.polylines(show_map, [pts], True, color, 3)
                    cv2.putText(show_map, f"h={z:.1f}m", (xx + 55, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.circle(show_map, (xx, yy), 60, color, 4)
                    
            elif location.label == 'Blue':
                # // tunning: 蓝方盲猜点画青蓝色 (BGR: 255, 255, 0) 以示区分，正常画蓝色
                color = (255, 255, 0) if is_guessing else (255, 0, 0)  
                
                cv2.putText(show_map, str(location.id) + air_suffix, (xx - 15, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                cv2.putText(show_map, str((x)) + ',' + str((y)) + ',' + str(z), (xx, yy - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                if is_air_robot:
                    pts = np.array([[xx, yy-50], [xx+50, yy], [xx, yy+50], [xx-50, yy]], np.int32)
                    cv2.polylines(show_map, [pts], True, color, 3)
                    cv2.putText(show_map, f"h={z:.1f}m", (xx + 55, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.circle(show_map, (xx, yy), 60, color, 4)
            else:
                # 颜色标签为 null/未知的机器人，灰色显示，不做坐标对称变换
                cv2.putText(show_map, 'null', (xx - 15, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 4)
                cv2.putText(show_map, str((x)) + ',' + str((y)) + ',' + str(z), (xx, yy - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                cv2.circle(show_map, (xx, yy), 60, (128, 128, 128), 4)
        cv2.imshow('map', show_map)
        cv2.waitKey(16)
        
    def location_callback(self, msg):
        # 更新locations
        with self.lock:
            self.locations = msg

def spin_thread(node):
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)
    disPanel = DisplayPanel()
    
    # 创建一个线程来读取节点内容
    thread = threading.Thread(target=spin_thread, args=(disPanel,))
    thread.start()
    
    try:
        while rclpy.ok():
            disPanel.display_locations()
            time.sleep(1/15)  # 控制显示频率
    except KeyboardInterrupt:
        pass
    
    disPanel.destroy_node()
    rclpy.shutdown()
    thread.join()

if __name__ == '__main__':
    main()
