"""
bbox_kalman.py — 纯视觉像素级边界框卡尔曼滤波器
==========================================================
为每个被跟踪的机器人提供底层的数学滤波接口，在二维图像像素平面上
对 YOLO 目标检测框进行时序平滑和短时预测，彻底剥离物理坐标系的转换逻辑。

状态向量: [cx, cy, w, h, vx, vy, vw, vh] (中心坐标 + 宽高 + 各自的变化速度)
观测向量: [cx, cy, w, h]                 (YOLO 转换后的中心点坐标与宽高)

特性:
  - 解耦物理映射: 专职图像平面像素级去噪，从源头切断透视变换的非线性误差放大。
  - 中心坐标转换: 弃用角点追踪，采用质心与尺度分离模型，提升线性匀速假设下的稳定性。
  - 无状态工具类: 契合贫血模型架构，自身不保存状态，仅执行纯粹的矩阵推演。
  - 差分噪声配置: 针对位置（易突变）与尺度（变化缓）配置不同量级的系统噪声与观测噪声协方差。
"""
import numpy as np

class BBoxKalmanFilter(object):
    """
    标准的 8 状态边界框卡尔曼滤波器工具类。
    遵循标准 EKF 的变量命名规范，但保持无状态(Stateless)设计。
    """
    def __init__(self, dt: float = 1.0/30.0):
        """
        初始化系统矩阵。
        参数 dt: 系统的帧间时间间隔。默认假设相机推理帧率为 30Hz。
        """
        self.n = 8  # 状态向量个数n (cx, cy, w, h, vx, vy, vw, vh)
        self.m = 4  # 测量观测值个数m (cx, cy, w, h)
        
        # 预测状态变换矩阵,依据恒定速度运动学方程得出的变换矩阵,维数n x n
        # F_k(n,n) * X^_k-1(n,1) --> x^_k(n,1) 新时刻状态向量
        self.F_k = np.eye(self.n)
        for i in range(self.m):
            self.F_k[i, self.m + i] = 1.0

            # // tunning: ★ 引入预测阻尼（摩擦力），防止预测框因异常速度“飞出去”
            # 对角线元素默认是 1 (v = v)。这里改为 0.90，意味着每推演一帧，像素速度会自动衰减 10%。这在保持预测方向的同时，强行截断了速度爆炸。
            self.F_k[self.m + i, self.m + i] = 0.90

        # 传感器测量值向量与预测值向量之间的线性转换矩阵 (观测矩阵)
        # m x n矩阵, H_k(mxn) * X(nx1) = ZZ_k(mx1)
        self.H_k = np.eye(self.m, self.n)

        # 单位矩阵I, 这里当数字1使用. P_k = (I - K_k*H_k)*P_k
        self.I = np.eye(self.n)

        # 噪声权重参数 (针对雷达站实机画面微调)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._std_weight_scale = 1.0 / 100

    def initiate(self, z: np.ndarray):
        """
        从单个 YOLO 观测值初始化一个新的轨迹状态。
        参数:
            z: 传感器读数 (measurement), [cx, cy, w, h]
        返回:
            x: 初始化的 n维状态向量
            P_result: 初始化的 nxn最优估计协方差矩阵
        """
        # x: 上一时刻(k-1)或当前时刻k的状态向量: n个元素向量
        x = np.r_[z, np.zeros_like(z)]

        # 初始协方差矩阵设定：给速度项分配极大的不确定性
        std = [
            2 * self._std_weight_position * z[2],  # cx 噪声与宽度相关
            2 * self._std_weight_position * z[3],  # cy 噪声与高度相关
            2 * self._std_weight_scale * z[2],     # w 噪声
            2 * self._std_weight_scale * z[3],     # h 噪声
            10 * self._std_weight_velocity * z[2], # vx
            10 * self._std_weight_velocity * z[3], # vy
            10 * self._std_weight_velocity * z[2], # vw
            10 * self._std_weight_velocity * z[3]  # vh
        ]
        # P_result: 最优P_k, 当前时刻最优估计协方差矩阵 (对角阵)
        P_result = np.diag(np.square(std))
        return x, P_result

    def predict(self, x: np.ndarray, P_result: np.ndarray):
        """
        根据系统运动学方程，更新预测状态。
        参数:
            x: 前一时刻(k-1)状态向量
            P_result: 前一时刻最优估计协方差矩阵
        返回:
            x_pred: 新时刻(k)的先验预测状态向量
            P_current: 新时刻的先验预测协方差矩阵
        """
        # Q_k: 各状态变量的预测噪声协方差矩阵 (动态计算)
        std_pos = [
            self._std_weight_position * x[2],
            self._std_weight_position * x[3],
            self._std_weight_scale * x[2],
            self._std_weight_scale * x[3]
        ]
        std_vel = [
            self._std_weight_velocity * x[2],
            self._std_weight_velocity * x[3],
            self._std_weight_velocity * x[2],
            self._std_weight_velocity * x[3]
        ]
        Q_k = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 预测状态方程
        # (1). X_k = F_k * X_k-1
        x_pred = np.dot(self.F_k, x)

        # 预测协方差矩阵
        # (2). P_k = F_k * P_k-1 * F_k^T + Q_k
        P_current = np.dot(self.F_k, np.dot(P_result, self.F_k.T)) + Q_k

        return x_pred, P_current

    def update(self, x: np.ndarray, P_current: np.ndarray, z: np.ndarray):
        """
        观测更新步。结合 YOLO 实际测量值纠正预测状态。
        参数:
            x: 预测状态向量 (X_k)
            P_current: 预测协方差矩阵 (P_k)
            z: 当前时刻 YOLO 传感器读数 [cx, cy, w, h]
        """
        # R_k: 传感器测量噪声协方差矩阵 (动态计算)
        std = [
            self._std_weight_position * z[2],
            self._std_weight_position * z[3],
            self._std_weight_scale * z[2],
            self._std_weight_scale * z[3]
        ]
        R_k = np.diag(np.square(std))

        # 预估测量值向量
        # zz_k = H_k * X_k
        zz_k = np.dot(self.H_k, x)

        # S_k: 创新协方差矩阵 (系统残差的协方差)
        # S_k = H_k * P_k * H_k^T + R_k
        S_k = np.dot(self.H_k, np.dot(P_current, self.H_k.T)) + R_k

        # 卡尔曼增益: K_k
        # (3). K_k = P_k * H_k^T * (H_k * P_k * H_k^T + R_k)^-1
        K_k = np.dot(np.dot(P_current, self.H_k.T), np.linalg.inv(S_k))

        # 最终,最优预测状态向量值
        # (4). X^_k = X_k + K_k * (z_k - zz_k) 
        x_new = x + np.dot(K_k, (z - zz_k))

        # // tunning: 底层状态硬钳制。限制像素速度 (vx, vy, vw, vh) 单帧最大变化量不超过 80 像素
        x_new[4:] = np.clip(x_new[4:], -80, 80)

        # 最后,最优预测协方差矩阵
        # (5). P^_k = P_k - K_k * H_k * P_k = (I - K_k * H_k) * P_k
        P_result = np.dot(self.I - np.dot(K_k, self.H_k), P_current)

        return x_new, P_result