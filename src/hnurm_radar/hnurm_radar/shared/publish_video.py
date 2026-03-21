"""
publish_video.py — 视频文件发布节点

功能：
  读取本地视频文件，以 ~60fps 的频率将每帧图像通过 ROS2 话题发布，
  用于离线调试和回放测试（替代实际相机输入）。
  视频播放完毕后自动循环。

发布话题：
  - image (Image) — BGR8 编码的视频帧

参数：
  - video_file (string) — 视频文件路径，通过 ROS2 参数传入

使用场景：
  配合 hnurm_radar_video_launch.py 启动，用于无硬件环境下的
  camera_scheme 方案调试。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy ,qos_profile_sensor_data
import time
class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.declare_parameter('video_file', 'path/to/video.mp4')
        video_file = self.get_parameter('video_file').get_parameter_value().string_value
        self.get_logger().info('video_file: %s' % video_file)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=3
        )
        self.last = time.time()
        self.publisher_ = self.create_publisher(Image, 'image', qos_profile)
        self.timer = self.create_timer(0.033, self.timer_callback) # 这里可以更改视频发布的频率，即视频播放速度，越小越快，单位为秒
        self.cap = cv2.VideoCapture(video_file)
        self.bridge = CvBridge()
        self.get_logger().info('Publishing video from: %s' % video_file)
    def timer_callback(self):
        self.cur = time.time()
        delta = self.cur - self.last
        self.last = self.cur
        # self.get_logger().info('Publishing video, frequency: %.2f Hz' % (1 / delta))
        ret, frame = self.cap.read()
        # 将frame resize 为1920*1080
        # cv2.resize(frame, (1920, 1080))
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
