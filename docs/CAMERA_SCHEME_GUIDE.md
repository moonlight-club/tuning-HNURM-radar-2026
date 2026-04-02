# HNURM Radar 2026 纯相机透视变换方案（README）

## 1. 框架总览

### 1.1 纯相机方案文件结构

```text
HNURM-radar-2026/
├── configs/
│   ├── detector_config.yaml                    # 检测/匹配主参数
│   └── perspective_calib.json                  # 透视标定结果
├── docs/
│   └── CAMERA_SCHEME_GUIDE.md                  # 方案说明文档
└── src/
  ├── hnurm_radar/hnurm_radar/
  │   ├── camera_locator/
  │   │   └── perspective_calibrator.py       # 标定工具
  │   ├── camera_scheme/
  │   │   ├── camera_detector.py              # 主流程：检测/去重/透视/发布
  │   │   ├── hungarian_tracker.py            # 匈牙利匹配与轨迹状态管理
  │   │   └── guess_pts.py                    # GUESSING 长时外推
  │   ├── filters/
  │   │   └── bbox_kalman.py                  # bbox 像素域卡尔曼滤波
  │   └── shared/
  │       ├── type.py                         # 共享数据结构与状态机枚举
  │       └── utils.py                        # IoU/NMS/坐标工具函数
  └── ekf/ekf/
    └── ekf_node.py                         # 坐标级 EKF 平滑
```

### 1.2 数据流图（纯相机主链路）

```mermaid
flowchart TD
  A[相机输入帧/视频输入帧] --> B[camera_detector\nYOLO三阶段检测]
  B --> C[NMS + 简单去重]
  C --> D[hungarian_tracker.update]
  D --> E[bbox_kalman\n短时平滑与预测]
  D --> F[轨迹状态机\nTRACKING/LOST/GUESSING]
  F --> G[guess_pts\n长时外推(可选)]
  E --> H[取滤波框底边中心]
  H --> I[pixel_to_field\nHomography透视变换]
  I --> J[/location 原始坐标发布]
  J --> K[ekf_node 坐标级滤波]
  K --> L[/ekf_location_filtered]
  L --> M[小地图绘制]
```

## 2. 启动方法
### 2.1 步骤 1：透视变换标定

首次使用或更换相机位置后，需要进行透视变换标定：

```bash
source install/setup.bash
ros2 run hnurm_radar perspective_calibrator
```

标定工具（PyQt5 GUI）操作流程：
1. 点击「开始标定」冻结画面
2. 在左侧相机图像上点击 4 个以上地面特征点
3. 在右侧赛场地图上依次点击对应位置
4. （可选）点击「切换到高地层」，标定高地区域的 4+ 个点
5. 点击「保存计算」，标定结果保存到 `configs/perspective_calib.json`

### 2.2 步骤 2：启动系统

分别在不同终端中启动：

```bash
# 终端 1：相机检测 + 透视变换定位
source install/setup.bash
ros2 run hnurm_radar camera_detector

# 终端 2：EKF 坐标滤波
source install/setup.bash
ros2 run ekf ekf_node

# 终端 3：裁判系统通信
source install/setup.bash
ros2 run hnurm_radar judge_messager
```

### 2.3 常用调试命令
```bash
# 相机侧原始坐标输出
ros2 topic echo /location

# EKF 后坐标输出
ros2 topic echo /ekf_location_filtered

# 话题频率
ros2 topic hz /location
ros2 topic hz /ekf_location_filtered
```

---

## 3. 参考与差异说明

本方案参考港科大雷达站透视变换思路，但当前实现中：

- **未引入**港科大 3D 射线求交点坐标模型；
- 采用 **2D Homography**（像素点 → 赛场平面坐标）；
- 框架为 **SORT 风格多目标跟踪**：
  - YOLO 仅用于单帧检测；
  - `bbox_kalman` 用于框平滑与短时预测；
  - `hungarian_tracker` 负责检测-轨迹匹配与 ID 投票；
  - 长时丢失可进入 `GUESSING`（当前分支默认不发布 GUESSING 坐标到小地图，效果待优化）。

---

## 4. 处理链路（按实际代码）

1. `camera_detector.py` 获取图像并执行 YOLO 三阶段检测。  
2. 检测结果经 NMS 和简单去重。  
3. 结果送入 `hungarian_tracker.update(detections, dt)`：  
   - 轨迹预测（基于 `bbox_kalman`）  
   - 匈牙利匹配  
   - 轨迹状态迁移（TRACKING/LOST/GUESSING）  
   - 标签投票与稳定 ID 输出  
4. 对滤波后的 bbox，取底边中心点做 `pixel_to_field()`。  
5. 发布 `/location`（相机原始平面坐标）。  
6. `ekf_node` 订阅 `/location` 并输出 `/ekf_location_filtered`。  
7. 小地图显示发布坐标（GUESSING 发布策略受当前代码开关控制）。

---

## 5. 关键模块职责

- `camera_scheme/camera_detector.py`：推理、去重、发布、可视化入口。  
- `camera_scheme/hungarian_tracker.py`：轨迹管理、匈牙利匹配、投票确认 ID。  
- `filters/bbox_kalman.py`：像素框卡尔曼滤波（短时预测与平滑）。  
- `camera_scheme/guess_pts.py`：GUESSING 状态外推（当前默认不作为主发布源）。  
- `shared/type.py`：统一数据结构与状态枚举。  
- `ekf/ekf_node.py`：坐标级 EKF 平滑与发布。

---

## 6. 配置文件说明（detector_config.yaml）

配置文件路径：`configs/detector_config.yaml`

### 6.1 `path`（模型与配置路径）
- `stage_one_path/stage_two_path/stage_three_path`：三阶段 YOLO 权重路径。  
- `tracker_path`：YOLO 内部 tracker 配置（当前主轨迹维护仍以本项目 `hungarian_tracker` 为主）。

### 6.2 `params`（检测基础参数）
- `labels`：类别列表。  
- `stage_one_conf/stage_two_conf/stage_three_conf`：各阶段置信度阈值。  
- `life_time`：目标生命周期（上层管理相关参数）。

### 6.3 `track`（匈牙利匹配核心参数）
- `iou_thr`：IoU 匹配门控阈值。  
- `dist_thr`：中心距离门控阈值（像素）。  
- `lost_thr`：漏检帧数阈值,进入短时预测LOST阶段的阈值（短时漏检仍维持稳定状态）。  
- `guess_thr`：进入更深丢失GUESSING阶段的阈值。  
- `max_miss`：轨迹最大连续漏检帧，超过后删除。  
- `publish_predict_when_no_det`：无检测时是否允许发布预测轨迹（受当前代码分支控制）。

> 时间换算说明：  
> 若按帧阈值配置，实际时长 = `阈值帧数 / 实际推理FPS`。  
> 例如 `max_miss=54`：  
> - 30 FPS ≈ 1.8s  
> - 40 FPS ≈ 1.35s  

### 6.4 录制参数
- `is_record`：是否保存推理视频。  
- `record_fps`：录制输出帧率。

### 6.5 已弃用配置
- `filter`（物理卡尔曼滤波）原先是对原始透视坐标进行滤波，与ekf重复滤波逻辑重复，目前在纯视觉主链路中已弃用
---

## 7. 调参说明与量化方法

### 7.1 优先级建议
1.  **匹配稳定性**：`track.iou_thr`（重叠度）、`track.dist_thr`（中心点像素位移门控）。
2.  **生命周期**：`lost_thr`（平滑丢失）、`guess_thr`（外推阈值）、`max_miss`（彻底删除）。
3.  **滤波表现**：修改 `bbox_kalman.py` 中的过程噪声 $Q$（响应速度）与观测噪声 $R$（平滑度）。

### 7.2 Kalman 量化调优 (SOP)
为了避免盲目修改，推荐使用**残差分析法 (Innovation Analysis)**：
-   **残差定义**：$y = z - H\hat{x}$（即 YOLO 原始框坐标与 Kalman 预测框坐标之差）。
-   **响应速度不足（滞后）**：若机器人在转弯/加速时，残差 $y$ 持续且显著偏向一侧，说明系统过于依赖匀速模型。应**增大过程噪声 $Q$**（如 `_q_weight_pos`）。
-   **稳定性不足（抖动）**：若机器人静止时，残差 $y$ 剧烈跳变。说明系统过于依赖 YOLO 的瞬时输出。应**增大观测噪声 $R$**（如 `_r_weight_pos`）。
-   **维数均衡原则**：在透视变换方案中，**Y 轴（纵向）对坐标精度极度敏感**。建议给 Y 轴分配比 X 轴高 1.5~2 倍的 $R$ 权重，以换取坐标的垂直稳定性。

### 7.3 关键机制说明
-   **预测阻尼 (Damping)**：`F_k` 矩阵中的阻尼系数（如 0.95~0.99）能防止轨迹在 `LOST` 状态下带速无限飞出，解决“幽灵框”劫持他人 ID 的问题。
-   **尺度自适应 (Scale Factor)**：系统会根据 $w \times h$ 自动计算 $sf$ 因子，针对远距离小目标自动切换为“重滤波”模式。

---

## 8. 当前已知问题

1. 转弯与长时间遮挡场景下，坐标连续性不佳。  
2. GUESSING 长时外推稳定性不足，坐标会飞出，当前默认不作为主发布结果。  
3. 多个机器人检测框交错会发生匈牙利错误匹配轨迹。  
4. 右上角高地发布坐标出现后退现象，可能由于检测框抖动导致机器人坐标透视落在平地与高地交错处，导致坐标抖动