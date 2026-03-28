"""
utils.py — 相机透视数学与坐标计算工具箱
==========================================================
本模块专为 camera_scheme/hungarian_tracker.py (匈牙利匹配) 提供底层的数学支撑。
严格收口追踪器在计算代价矩阵时所需的边界框运算，避免在核心关联逻辑中掺杂繁琐的坐标变换。

涉及的核心功能模块：
  1. compute_iou: 计算两个边界框的交并比 (IoU)，用于评估卡尔曼预测框与 YOLO 观测框的重合度。
  2. xywh2xyxy: 将 [中心点x, 中心点y, 宽, 高] 转换为 [x1, y1, x2, y2] 的角点格式，为计算 IoU 提供格式适配。
  3. xyxy2xywh: 将角点格式转换为中心点格式，主要用于与底层 BBoxKalmanFilter 的状态向量格式保持对齐。

维护须知：
    请勿在此文件内随意添加与 2D 目标追踪无关的“通用”函数（如网络通信、雷达点云解算等），
    保持该工具箱在视觉感知边界内的纯粹性。
"""
from typing import List
import numpy as np

def compute_iou(bbox1: List[float | int], bbox2: List[float | int]) -> float:
    """
    计算两个边界框的交并比 (Intersection over Union, IoU)。
    
    参数:
        bbox1: 边界框 1，格式为 [x1, y1, x2, y2]
        bbox2: 边界框 2，格式为 [x1, y1, x2, y2]
        
    返回:
        IoU 值，范围 [0.0, 1.0]
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    
    union_area = bbox1_area + bbox2_area - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0

def xywh2xyxy(bbox: List[float | int]) -> List[float]:
    """
    将中心点坐标格式转换为角点坐标格式。
    
    参数:
        bbox: 边界框，格式为 [center_x, center_y, width, height]
        
    返回:
        边界框，格式为 [x1, y1, x2, y2]
    """
    center_x, center_y, width, height = bbox
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return [x1, y1, x2, y2]

def xyxy2xywh(bbox: List[float | int]) -> List[float]:
    """
    将角点坐标格式转换为中心点坐标格式。
    
    参数:
        bbox: 边界框，格式为 [x1, y1, x2, y2]
        
    返回:
        边界框，格式为 [center_x, center_y, width, height]
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    return [center_x, center_y, width, height]

def nms_xywh(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.6) -> list:
    """
    非极大值抑制 (Non-Maximum Suppression)
    用于过滤目标检测中冗余的重叠边界框。
    
    参数:
        boxes (np.ndarray): 边界框数组，形状为 (N, 4)，格式为 [cx, cy, w, h]。
        scores (np.ndarray): 置信度分数数组，形状为 (N,)。
        iou_threshold (float): 交并比 (IoU) 阈值。高于此阈值的重叠框将被抑制。默认值为 0.6。
        
    返回:
        list: 保留下的边界框索引列表。
    """
    if len(boxes) == 0:
        return []

    # 获取按置信度降序排列的索引列表
    order = scores.argsort()[::-1].tolist()
    keep = []

    while len(order) > 0:
        # 提取当前置信度最高的边界框索引
        i = order.pop(0)
        keep.append(i)

        # 将中心点宽高格式 (cx, cy, w, h) 转换为角点格式 (x1, y1, x2, y2) 以便计算 IoU
        box_i_xyxy = xywh2xyxy(boxes[i].tolist())

        rest_order = []
        for j in order:
            box_j_xyxy = xywh2xyxy(boxes[j].tolist())
            
            # 计算当前最高分框与剩余框的 IoU
            iou = compute_iou(box_i_xyxy, box_j_xyxy)
            
            # 仅保留 IoU 小于等于阈值的边界框索引
            if iou <= iou_threshold:
                rest_order.append(j)
                
        # 更新待处理的索引列表
        order = rest_order

    return keep