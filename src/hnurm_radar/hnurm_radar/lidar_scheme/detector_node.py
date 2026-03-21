"""
detector_node.py — 激光雷达方案的视觉检测节点（方案二）

功能：
  调用海康工业相机获取图像，使用 YOLO 三阶段推理识别赛场机器人，
  将检测结果（bbox + track_id + 分类标签）通过 ROS2 话题发布给
  下游的 radar_node 进行点云投影定位。

三阶段推理流程：
  Stage 1 — _track_infer():  全图目标检测 + ByteTrack 多目标追踪（stage_one.pt, imgsz=1280）
  Stage 2 — classify_infer(): ROI 装甲板颜色+编号分类（stage_two.pt, imgsz=256）
  Stage 3 — classify_infer(): ROI 灰色装甲板专用分类（stage_three.pt, imgsz=256）

投票机制：
  每个 track_id 维护一个长度为 class_num 的投票表 Track_value，
  每帧分类结果按 (0.5 + conf * 0.5) 累加投票，每 life_time 帧衰减为 1/10，
  最终取投票最高的类别作为该目标的标签。

与 camera_detector（方案一）的区别：
  - 本节点不做透视变换定位，仅输出 2D 检测框
  - 定位由下游 radar_node 结合点云完成（3D 投影 + DBSCAN 聚类）
  - 直接依赖海康工业相机硬件（HKCam），无 test/video 模式

发布话题：
  - detect_result (Robots)   — 所有检测到的机器人 bbox + 标签
  - detect_view   (Image)    — 带标注的检测结果图像

配置文件：
  - configs/detector_config.yaml — 模型路径、置信度阈值、生命周期等
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
from ruamel.yaml import YAML

from detect_result.msg import DetectResult
from detect_result.msg import Robots
from sensor_msgs.msg import Image
from ..Camera.HKCam import *
from cv_bridge import CvBridge
import cv2
import time
import math
import threading
import sys
import os
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from ..shared.paths import PROJECT_ROOT, DETECTOR_CONFIG_PATH, RECORD_DIR, ULTRALYTICS_DIR, resolve_path
sys.path.append(ULTRALYTICS_DIR)
from ultralytics import YOLO

# 检测节点类
class Detector(Node):
    def __init__(self):
        super().__init__('detector')
        
        self.start_time = time.time()
        self.bridge = CvBridge()
        # 加载配置文件
        self.cfg = YAML().load(open(DETECTOR_CONFIG_PATH, encoding='Utf-8', mode='r'))
        # 记录相关参数 flag and fps
        self.is_record = self.cfg['is_record']
        self.record_fps = self.cfg['record_fps']
        
        # 初始化并加载 3 个 YOLO 模型，分别用于：目标检测 → 目标分类 → 灰度/细分类
        self.get_logger().info('Loading Yolo models...')
        self.model_car = YOLO(resolve_path(self.cfg['path']['stage_one_path']), task="detect")
        self.model_car.overrides['imgsz'] = 1280
        self.model_car2 = YOLO(resolve_path(self.cfg['path']['stage_two_path']))
        self.model_car2.overrides['imgsz'] = 256
        self.model_car3 = YOLO(resolve_path(self.cfg['path']['stage_three_path']))
        self.model_car3.overrides['imgsz'] = 256
        self.get_logger().info('Yolo models loaded.')

        # 读取追踪器路径和各阶段 YOLO 的置信度阈值，设置目标生命周期，并初始化一个用于分配目标 ID 的缓存表
        self.tracker_path = resolve_path(self.cfg['path']['tracker_path'])
        self.stage_one_conf = self.cfg['params']['stage_one_conf']
        self.stage_two_conf = self.cfg['params']['stage_two_conf']
        self.stage_three_conf = self.cfg['params']['stage_three_conf']
        self.life_time = self.cfg['params']['life_time'] # 生命周期
        self.id_candidate = [0] * 1000
        
        
        self.labels = self.cfg['params']['labels'] # 类别标签列表
        # 由labels长度初始化class_num
        self.class_num = len(self.labels) # 类别数量
        self.Status = [0] * 10000
        self.Track_value = {} # 这是tracker的track_id和类别的字典，主键是追踪器编号，值是一个列表，记录每个类别的识别次数。每个追踪器有所有类别的识别次数
        for i in range(10000):
            self.Track_value[i] = [0] * self.class_num
        self.id_candidate = [0] * 10000
        
        # 灰色装甲板的类别映射
        self.Gray2Blue = {12:5,13:1,14:0,15:3,16:2,17:4}
        self.Gray2Red = {12:11,13:7,14:6,15:9,16:8,17:10}
        self.gray2gray = {0:14,1:13,2:16,3:15,4:17,5:12}
        self.Blue2Gray ={v: k for k, v in self.Gray2Blue.items()}
        self.Red2Gray ={v: k for k, v in self.Gray2Red.items()}
        self.gray_thresh = 15
        
        # 设置计数器
        self.loop_times = 0
        # QoS 设置
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=3
        )
        # 打开相机
        self.cam = HKCam(0)
        self.frame = None
        # 启动图像同步线程
        self.frame_threading = threading.Thread(target=self.sync_frame)
        self.frame_threading.start()
        # 定期发送检测结果
        self.publisher_ = self.create_publisher(Robots, 'detect_result', qos_profile)
        # 检测标定是否完成
        # self.sub_init = self.create_subscription(Bool, "inited", self.before_initialization, qos_profile)
        # self.pub_image = self.create_publisher(Image, 'image', qos_profile)
        # 用于发布检测图像结果节点
        self.pub_res = self.create_publisher(Image, 'detect_view', qos_profile)
        
        # 创建图像资源锁
        self._frame_lock = threading.Lock()
        # 创建线程用于推理
        self.threading = threading.Thread(target=self.infer_loop)
        self.threading.start()
        # cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Window", 1920, 1080)
        self.init_flag = False
        
        # 创建一个图像队列
        self.image_queue = deque(maxlen=10)
        self.timestamp_queue = deque(maxlen=10)
        # 创建一个线程用于保存图像，录制文件存放在 record/ 文件夹下
        os.makedirs(RECORD_DIR, exist_ok=True)
        cur_date = time.strftime("%Y-%m-%d-%h-%s", time.localtime())
        self.video_writer = cv2.VideoWriter(os.path.join(RECORD_DIR, cur_date + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), 60, (3072, 2048))
        # 打开文件用于保存时间戳
        self.timestamp_file = open(os.path.join(RECORD_DIR, cur_date + ".txt"), "w")
        self.save_thread = threading.Thread(target=self.save_image)
        self.save_thread.start()

    # 保存图像函数    
    def save_image(self):
        while rclpy.ok():
            if self.is_record:
                if len(self.image_queue) > 0 and len(self.timestamp_queue) > 0:
                    # self.get_logger().info("Saving image...")
                    img = self.image_queue.popleft()
                    timestp = self.timestamp_queue.popleft()
                    
                    self.video_writer.write(img)
                    self.timestamp_file.write(str(timestp) + "\n")
                    
                    # 刷新
                    self.timestamp_file.flush()
                    time.sleep(1/self.record_fps)
            else:
                time.sleep(0.1)

    # 同步图像函数    
    def sync_frame(self):
        while rclpy.ok():
            cam_frame = self.cam.getFrame()
            self.image_queue.append(cam_frame)
            self.timestamp_queue.append(time.time())
            # 加锁防止多线程同时访问
            with self._frame_lock:
                self.frame = cam_frame.copy()

    # 获取图像函数    
    def getFrame(self):
        with self._frame_lock:
            if self.frame is None:
                return None
            else:
                return self.frame
    # 对results的结果进行判空
    def is_results_empty(self, results):
        if results is None:
            # print("No results!")
            return True
        if results[0].boxes.id is None:
            # print("No detect!")
            return True
        return False

    # 解析results
    def parse_results(self, results):
        confidences = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        return confidences, boxes, track_ids
    
    # 一阶段追踪推理
    def track_infer(self, frame):
        results = self.model_car.track(frame, persist=True,tracker=self.tracker_path,verbose = False)
        return results
    
    # 二阶段分类推理Classify
    def classify_infer(self, roi_list): # 输入原图和box, 返回分类结果
        # print((roi_list))
        # 打印出roi_list的shape
        # print(len(roi_list))
        results = []
        gray_results = []
        for i in roi_list:
            results.extend(self.model_car2.predict(i, conf=self.stage_two_conf, device=0, verbose=False))
            gray_results.extend(self.model_car3.predict(i, device=0, verbose=False))
            # print(results)
        # results = self.model_car2.predict(roi_list, conf = self.stage_two_conf , device = 0 ,verbose = False  )
        if len(results) == 0:  # no detect
            return -1
        # for i in roi_list:
            
        # gray_results = self.model_car3.predict(roi_list , verbose = False)
        label_list = []
        conf_list = []
        for result,gray_result,roi in zip(results,gray_results,roi_list):
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
                    tmp=data[i]
            if len(tmp) != 0:
                x1,y1,x2,y2 = tmp[:4]
                gray = roi[int(y1):int(y2),int(x1):int(x2)]
                if label < 6 and cv2.mean(gray)[0] < 35: # 蓝色阈值可以高些
                    label = self.Blue2Gray[int(label)]
                elif label > 5 and cv2.mean(gray)[0] < 15: # 红色阈值更低
                    label = self.Red2Gray[int(label)]
            else: 
                if len(gray_tmp) != 0: #大概率是灰色了
                    x1,y1,x2,y2 = gray_tmp[:4]
                    gray = roi[int(y1):int(y2),int(x1):int(x2)]
                    if cv2.mean(gray)[0] < 30:
                        label = self.gray2gray[int(gray_label)]
            label_list.append(int(label))
            conf_list.append(float(maxConf))
        return label_list,conf_list
            
                    
    # 总的推理
    def infer(self, frame): # 输入原图，返回推理结果
        if frame is None:
            # print("No frame!")
            return None , None
        results = self.track_infer(frame)
        # 一阶段results判空
        if self.is_results_empty(results):
            # print("No results!")
            return frame , None
        
        exist_armor = [-1] * 12 # armor number映射Track_id
        draw_candidate = [] # 待draw的bbox(track_id,x_left,y_left,x_right,y_right,label)

        confidences, boxes, track_ids = self.parse_results(results) # 可以不要
        zip_results = [] # 最终返回存储追踪器的结果，有box, 分类id, conf

        roi_list = []
        id_list = []
        box_list = []
        
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if self.loop_times % self.life_time == 1: # 0的话无法单张预测
                for i in range(12):
                    self.Track_value[int(track_id)][i] = math.floor(self.Track_value[int(track_id)][i] / 10)

            # 获取二阶段roi_batch
            x,y,w,h = box
            x_left = x - w / 2
            y_left = y - h / 2
            roi = frame[int(y_left): int(y_left + h), int(x_left): int(x_left + w)]
            roi_list.append(roi)
            # 获取track_id的列表
            id_list.append(track_id)
            # 获取boxes列表
            box_list.append(box)
        label_list,conf_list = self.classify_infer(roi_list)
        index = 0
        for i in range(len(roi_list)):

            classify_label = label_list[i]
            conf = conf_list[i]
            track_id = id_list[i]
            box = box_list[i]
            x,y,h,w = box
            status = 0 # normal
            # 二阶段识别次数和置信度的加权
            if classify_label != -1:
                    label = self.Track_value[int(track_id)].index(max(self.Track_value[int(track_id)]))
                    # if track_id == 123:
                    #     print(classify_label)
                    if classify_label > 11: # is Gray
                        status = 1
                        if self.Status[track_id] < 6:
                            self.Status[track_id] += status
                        if label < 6 and self.Gray2Blue[classify_label] == label: # is Blue
                            self.Track_value[int(track_id)][int(float(self.Gray2Blue[classify_label]))] += 0.5 + conf * 0.5
                        elif label > 5 and self.Gray2Red[classify_label] == label : # is Red
                            self.Track_value[int(track_id)][int(float(self.Gray2Red[classify_label]))] += 0.5 + conf * 0.5
                        else:
                            classify_label = -1
                        
                    else :
                        # print(track_id)
                        if self.Status[int(track_id)] > 0:
                            self.Status[int(track_id)] -= 1
                        if self.Status[int(track_id)] < 4 :
                            self.Status[int(track_id)] = 0
                        self.Track_value[int(track_id)][int(float(classify_label))] += 0.5 + conf * 0.5
            label = self.Track_value[int(track_id)].index(max(self.Track_value[int(track_id)])) # 找到计数器最大的类别,即追踪器的分类结果
            '''
                判重:
                    这部分直接举例 ：假设id_1匹配上了B3，并且此时id_3的匹配结果也是B3，那么对其维护的B3装甲板的加权和进行比较，
                    若当前id的加权和较小，则当前维护的加权和清零
                    若当前id的加权和较大，则将old_id维护的加权和清零，并且将draw_candidate[old_id]的label置为NULL
            '''
            if exist_armor[label] != -1:
                old_id = exist_armor[label]
                if self.Track_value[int(track_id)][label] < self.Track_value[int(old_id)][label]:
                    self.Track_value[(int(track_id))][label] = 0
                    label = "NULL"
                else:
                    self.Track_value[(int(old_id))][label] = 0
                    old_id_index = self.id_candidate[old_id]
                    draw_candidate[old_id_index][5] = "NULL"
                    exist_armor[label] = track_id
            else:
                exist_armor[label] = track_id

            pd = self.Track_value[int(track_id)][0] # 获取计数器的第一个值
            # 判断是否所有类别的计数器都一样，如果一样，说明没有识别出来，返回NULL
            same = True
            for j in range(self.class_num - 1):
                if pd != self.Track_value[int(track_id)][j + 1]:
                    same = False
                    break
            if same == False and label != "NULL": # label是分类信息
                label = str(self.labels[label])
            else:
                label = "NULL" 
            # result是预测的类别的字符形式，如果不是NULL, 画上分类结果
            # self.draw_result(frame, box, label, conf)
            # 返回结果,现在是画上了结果的frames
            # 把识别的box，conf和最终的类别组合后返回
            # tracker_results.append((box, label, conf)) # 返回的是一个列表，每个元素是一个元组，包含了box, 分类结果和置信度

            # 画出候选框
            x_left = int(x - w / 2)
            y_left = int(y - h / 2)
            x_right = int(x + w / 2)
            y_right = int(y + h / 2)
            xywh_box = [x,y,w,h]
            xyxy_box = [x_left,y_left,x_right,y_right]
            draw_candidate.append([track_id, x_left, y_left, x_right, y_right, label])

            zip_results.append([xyxy_box, xywh_box , track_id , label ])

            self.id_candidate[track_id] = index

            index = index + 1
        for box in draw_candidate:
            track_id, x_left, y_left, x_right, y_right, label = box
            cv2.rectangle(frame, (x_left, y_left), (x_right, y_right), (255, 128, 0), 3, 8)
            cv2.putText(frame, label, (int(x_left - 10), int(y_right + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 122), 2)
            cv2.putText(frame, str(track_id), (int(x_right + 5), int(y_right + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 122), 2)
        self.loop_times = self.loop_times + 1
        return frame , zip_results

    # 画出result
    def draw_result(self, frame, box, result, conf):
        if result != "NULL":
            # 画上分类结果
            x, y, w, h = box
            cv2.putText(frame, str(result), (int(box[0] - 5), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 122), 2)
            # 画上置信度 debug
            cv2.putText(frame, str(round(conf, 2)), (int(box[0] - 5), int(box[1] - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                          (0, 255, 122), 5)
        return frame

    # 推理循环函数
    def infer_loop(self):
        # time.sleep(0.5)
        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window", 1920, 1080)
        while rclpy.ok():
            now = time.time()
            fps = 1 / (now - self.start_time)
        
            self.start_time = now

            try:
                # 获取缓存的图像
                cv_image = self.getFrame()
                if cv_image is not None:
                    cv_image = cv_image.copy()
                else:
                    continue
                cv_image = cv2.resize(cv_image, (1920, 1080))
                # print(cv_image)
                
                cv2.waitKey(1)
                allRobots = Robots()
                # Inference
                if cv_image is not None:
                    infer_result = self.infer(cv_image)
                
                    if infer_result is not None:
                        result_img, results = infer_result
                        if results is not None:
                            # xyxy_box, xywh_box ,  track_id , label = result
                            for result in results:
                                msg = DetectResult()
                                # 检测输入图像是1920*1080的，由于原图像是3072*2048，将检测到的xy坐标还原到原图像
                                result[0][0] = int(result[0][0] * 3072 / 1920)
                                result[0][1] = int(result[0][1] * 2048 / 1080)
                                result[0][2] = int(result[0][2] * 3072 / 1920)
                                result[0][3] = int(result[0][3] * 2048 / 1080)

                                result[1][0] = int(result[1][0] * 3072 / 1920)
                                result[1][1] = int(result[1][1] * 2048 / 1080)
                                result[1][2] = int(result[1][2] * 3072 / 1920)
                                result[1][3] = int(result[1][3] * 2048 / 1080)

                                msg.xyxy_box = result[0]
                                # print(result[1])
                                msg.xywh_box = [float(x) for x in result[1]]
                                msg.track_id = result[2]
                                msg.label = result[3]
                                allRobots.detect_results.append(msg)

                        # 发布检测可视化图像到 detect_view
                        if result_img is not None:
                            img_msg = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
                            self.pub_res.publish(img_msg)
                            cv2.imshow("Window", result_img)
                        else:
                            img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                            self.pub_res.publish(img_msg)
                            cv2.imshow("Window", cv_image)
                        cv2.waitKey(1)
                else:
                    cv2.imshow("Window", cv_image)

                self.publisher_.publish(allRobots)
        
            except Exception as e:
                self.get_logger().error('Error {0}'.format(e))
    
    def __del__(self):
        # Clean Up
        self.model_car = None
        self.model_car2 = None
        self.tracker_path = None
        self.stage_one_conf = None
        self.stage_two_conf = None
        self.life_time = None
        self.labels = None
        self.class_num = None
        self.Track_value = None
        self.id_candidate = None
        self.loop_times = None
        self.threading = None
        self.working_flag = None
        self.init_flag = None
        self._result_lock = None
        self._results = None
        self.get_logger().info('Detector node is being destroyed.')


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # cv2.namedWindow("Image Window", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image Window", 1800, 1600)
    main()