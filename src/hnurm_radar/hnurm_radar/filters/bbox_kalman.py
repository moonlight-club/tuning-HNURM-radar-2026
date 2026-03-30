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
            # 对角线元素默认是 1 (v = v)。这里改为 0.95，意味着每推演一帧，像素速度会自动衰减 1%。这在保持预测方向的同时，强行截断了速度爆炸。
            # [debug] 增加阻尼，防止LOST预测框飞出去
            self.F_k[self.m + i, self.m + i] = 0.95

        # [debug]切断 vw (宽度变化率) 和 vh (高度变化率) 在预测步中对实际宽高的影响。
        # 这样在纯预测 (LOST/GUESSING) 期间，中心点(cx, cy)会按速度(vx, vy)正常滑行，
        # 但框的大小(w, h)将被死死锁住，保持消失前最后一帧的真实大小，面积绝不会变为0！
        # ==========================================
        self.F_k[2, 6] = 0.0  # 锁死宽度的缩放预测
        self.F_k[3, 7] = 0.0  # 锁死高度的缩放预测

        # 传感器测量值向量与预测值向量之间的线性转换矩阵 (观测矩阵)
        # m x n矩阵, H_k(mxn) * X(nx1) = ZZ_k(mx1)
        self.H_k = np.eye(self.m, self.n)

        # 单位矩阵I, 这里当数字1使用. P_k = (I - K_k*H_k)*P_k
        self.I = np.eye(self.n)

       # ==========================================
        # // tunning: 独立配置过程噪声与观测噪声的基础权重
        # ==========================================
        # 1. 过程噪声基准权重 (Q): 极小化，赋予卡尔曼匀速物理模型极高的信任度
        
        # [debug2]放宽位置的过程噪声权重，尝试解决拐弯速度跟不上的问题
        # ==========================================
        # // tunning: 配合阻尼，释放敏捷跟踪能力！
        # ==========================================
        self._q_weight_pos = 1.0 / 4000      # 提高对位置突变的信任，解决拐弯跟不上
        self._q_weight_vel = 1.0 / 400      # 放宽速度变化，有 0.85 的阻尼护航，绝对不会飞出！
        self._q_weight_scale = 1.0 / 20000

        # # [degug1] 尝试根据HIT经验值设置位置与速度的过程噪声权重，进一步强化对匀速假设的信任，同时允许尺度有更大的适应性，提升对远距离小目标的稳定性。
        # # ==========================================
        # # 1. 过程噪声基准权重 (Q): 严格限制绝对量级，防止速度状态发散
        # self._q_weight_pos = 1.0 / 10000    # 中心点位置的过程噪声权重
        # self._q_weight_vel = 1.0 / 1000     # 速度的过程噪声权重 (保持相对位置权重的大比值，但绝对值受限)
        # self._q_weight_scale = 1.0 / 20000  # 尺度的过程噪声权重

        # 2. 观测噪声基准权重 (R): 保持对观测值的适度信任，构建低通屏障
        self._r_weight_pos = 1.0 / 100      # 中心点位置的观测噪声权重 (量级远大于Q)
        self._r_weight_scale = 1.0 / 200    # 尺度的观测噪声权重
        
        # self._q_weight_pos = 1.0 / 200      # 中心点位置的过程噪声权重
        # self._q_weight_vel = 1.0 / 1500     # 速度的过程噪声权重 (极小值，强制收敛静止抖动)
        # self._q_weight_scale = 1.0 / 2000   # 尺度的过程噪声权重

        # # 2. 观测噪声基准权重 (R): 设定为过程噪声的 10 倍，降低对 YOLO 原始高频抖动的信任度
        # self._r_weight_pos = 1.0 / 35       # 中心点位置的观测噪声权重
        # self._r_weight_scale = 1.0 / 200    # 尺度的观测噪声权重


    def _get_adaptive_sf(self, w, h):
        """
        // tunning: 计算尺度自适应因子 (Scale Factor)。
        用于处理远距离小目标检测框抖动剧烈的问题。
        """
        area = w * h
        # 设定基准面积为 3600 px^2 (60x60)
        if area < 3600:
            return 1.0 + (3600 - area) / 1200.0
        return 1.0

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

        # // tunning: 引入尺度因子 sf，针对远距离小目标适度放大初始不确定性
        sf = self._get_adaptive_sf(z[2], z[3])

        # 初始协方差矩阵设定：给速度项分配极大的不确定性
        std = [
            2 * self._r_weight_pos * z[2] * sf,     # cx 初始噪声
            2 * self._r_weight_pos * z[3] * sf,     # cy 初始噪声
            2 * self._r_weight_scale * z[2] * sf,   # w 初始噪声
            2 * self._r_weight_scale * z[3] * sf,   # h 初始噪声
            10 * self._q_weight_vel * z[2] * sf,    # vx 初始分布
            10 * self._q_weight_vel * z[3] * sf,    # vy 初始分布
            10 * self._q_weight_vel * z[2] * sf,    # vw 初始分布
            10 * self._q_weight_vel * z[3] * sf     # vh 初始分布
        ]
        # P_result: 最优P_k, 当前时刻最优估计协方差矩阵 (对角阵)
        P_result = np.diag(np.square(std))
        return x, P_result

    def predict(self, x: np.ndarray, P_result: np.ndarray, dt=0.033):
        """
        根据系统运动学方程，更新预测状态。
        参数:
            x: 前一时刻(k-1)状态向量
            P_result: 前一时刻最优估计协方差矩阵
            dt: 真实时间步长
        返回:
            x_pred: 新时刻(k)的先验预测状态向量
            P_current: 新时刻的先验预测协方差矩阵
        """
        # // tunning: 动态更新状态转移矩阵中的 dt 系数，确保位移量与真实时间步长对齐
        self.F_k[0, 4] = dt
        self.F_k[1, 5] = dt

        # // tunning: 依据当前预测框的宽高计算尺度因子
        sf = self._get_adaptive_sf(x[2], x[3])

        # Q_k: 各状态变量的预测噪声协方差矩阵 (动态计算)
        std_pos = [
            self._q_weight_pos * x[2] * sf,
            self._q_weight_pos * x[3] * sf,
            self._q_weight_scale * x[2] * sf,
            self._q_weight_scale * x[3] * sf
        ]
        std_vel = [
           self._q_weight_vel * x[2] * sf,
            self._q_weight_vel * x[3] * sf,
            self._q_weight_vel * x[2] * sf,
            self._q_weight_vel * x[3] * sf
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

        # // tunning: 依据当前观测值计算尺度因子
        sf = self._get_adaptive_sf(z[2], z[3])

        # R_k: 传感器测量噪声协方差矩阵 (动态计算)
        std = [
            self._r_weight_pos * z[2] * sf,
            self._r_weight_pos * z[3] * sf,
            self._r_weight_scale * z[2] * sf,
            self._r_weight_scale * z[3] * sf
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

        # // tunning: ★ 暴力重置逻辑（Jump Reset）
        # 如果 YOLO 观测点和卡尔曼预测点中心距离超过 100 像素，直接判定预测失败
        # 强行重置位置并清空瞬时速度，防止预测框“起飞”
        dist = np.linalg.norm(x[:2] - z[:2])
        if dist > 150:
            x_new[:4] = z
            x_new[4:] = 0.0
            # [DEBUG]发生突变时，同步重置协方差矩阵 P，强行让卡尔曼“忘掉”历史误差惯性！
            P_current = np.eye(self.n)

        # // tunning: 底层状态硬钳制。限制像素速度 (vx, vy, vw, vh) 单帧最大变化量不超过 50 像素
        x_new[4:] = np.clip(x_new[4:], -50, 50)

        # 最后,最优预测协方差矩阵
        # (5). P^_k = P_k - K_k * H_k * P_k = (I - K_k * H_k) * P_k
        P_result = np.dot(self.I - np.dot(K_k, self.H_k), P_current)

        return x_new, P_result