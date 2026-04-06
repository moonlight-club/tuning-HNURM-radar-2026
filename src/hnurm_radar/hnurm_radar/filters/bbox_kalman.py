"""
bbox_kalman.py — 像素级边界框卡尔曼滤波模块
==========================================================
本模块提供无状态的边界框卡尔曼滤波计算接口，用于在二维图像坐标系内
对检测框进行时序平滑、短时预测与观测校正。模块仅处理像素域运动学，不涉及
物理世界坐标映射与业务发布策略。

状态向量: [cx, cy, w, h, vx, vy, vw, vh]
    - 位置与尺度: 边界框中心点与宽高
    - 速度项: 对应四个量的一阶变化率

观测向量: [cx, cy, w, h]
    - 来自检测器的当前帧边界框中心点与宽高观测

核心逻辑与主要功能：
    - 构建并维护标准线性卡尔曼系统矩阵（F/H/I），支持按真实 `dt` 动态更新状态转移。
    - 基于目标尺度自适应计算过程噪声与观测噪声，改善远距离小目标稳定性。
    - 提供 `initiate/predict/update` 三阶段接口，供上层关联器执行完整滤波循环。
    - 在观测突变场景下执行跳变重置与速度钳制，抑制预测发散与异常“飞框”。
    - 通过位置/速度/尺度差分权重配置，在平滑性与跟随性之间保持可调平衡。
"""
import numpy as np

class BBoxKalmanFilter(object):
    """
    标准的 8 状态边界框卡尔曼滤波器工具类。
    与EKF变量命名统一。
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

        # 预测阻尼：平移速度弱阻尼、尺度速度强阻尼，用于抑制异常外推。
        self.vel_damping_xy = 0.95   # vx, vy
        self.vel_damping_wh = 0.90    # vw, vh

        self.F_k[4, 4] = self.vel_damping_xy
        self.F_k[5, 5] = self.vel_damping_xy
        self.F_k[6, 6] = self.vel_damping_wh
        self.F_k[7, 7] = self.vel_damping_wh  
                
                
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

        # # [debug3]降低对匀速运动模型的过度信任，保证平滑过后的检测框能够跟上机器人实际的快速转向和加速
        # # [debug] 进一步调大 Q_vel (30->10)，提高速度修正的瞬时响应
        # self._q_weight_pos = 1.0 / 150     
        # self._q_weight_vel = 1.0 / 10      
        # self._q_weight_scale = 1.0 / 20000


        # # 2. 观测噪声基准权重 (R): 保持对观测值的适度信任，构建低通屏障
        # # [debug] 调大 R_pos 权重 (
        # # 分母 150->500)，极大提升对 YOLO 观测的信任，消除平滑滞后
        # self._r_weight_pos = 1.0 / 150      # 中心点位置的观测噪声权重 (量级远大于Q)
        # self._r_weight_scale = 1.0 / 200    # 尺度的观测噪声权重

        # [degub3]除了画面消失的机器人，其他的残差波动都比较小
        self._q_weight_pos = 1.0 / 10000    
        self._q_weight_vel = 1.0 / 80      
        self._q_weight_scale = 1.0 / 20000


        # 2. 观测噪声基准权重 (R): 保持对观测值的适度信任，构建低通屏障
        self._r_weight_pos = 1.0 / 100      # 中心点位置的观测噪声权重 (量级远大于Q)
        self._r_weight_scale = 1.0 / 200    # 尺度的观测噪声权重
        
        
        # # [debug2]坐标滤波过后比较平滑，但是出现平滑检测框跟不上实际机器人的情况了
        # 而且预测的效果不佳，预测框基本没有速度，停留在原地
        # self._q_weight_pos = 1.0 / 4000      # 提高对位置突变的信任，解决拐弯跟不上
        # self._q_weight_vel = 1.0 / 400      # 放宽速度变化，有 0.85 的阻尼护航，绝对不会飞出！
        # self._q_weight_scale = 1.0 / 20000


        # # 2. 观测噪声基准权重 (R): 保持对观测值的适度信任，构建低通屏障
        # self._r_weight_pos = 1.0 / 100      # 中心点位置的观测噪声权重 (量级远大于Q)
        # self._r_weight_scale = 1.0 / 200    # 尺度的观测噪声权重
        

        # # [degug1] 尝试根据HIT经验值设置位置与速度的过程噪声权重，进一步强化对匀速假设的信任，同时允许尺度有更大的适应性，提升对远距离小目标的稳定性。
        # # ==========================================
        # # 1. 过程噪声基准权重 (Q): 严格限制绝对量级，防止速度状态发散
        # self._q_weight_pos = 1.0 / 10000    # 中心点位置的过程噪声权重
        # self._q_weight_vel = 1.0 / 1000     # 速度的过程噪声权重 (保持相对位置权重的大比值，但绝对值受限)
        # self._q_weight_scale = 1.0 / 20000  # 尺度的过程噪声权重
        
        # self._q_weight_pos = 1.0 / 200      # 中心点位置的过程噪声权重
        # self._q_weight_vel = 1.0 / 1500     # 速度的过程噪声权重 (极小值，强制收敛静止抖动)
        # self._q_weight_scale = 1.0 / 2000   # 尺度的过程噪声权重



    def _get_adaptive_sf(self, w, h):
        """
        计算尺度自适应因子，用于降低远距离小目标框抖动的影响。
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

        # 引入尺度因子 sf，针对远距离小目标适度放大初始不确定性。
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
        # 动态更新状态转移矩阵中的 dt 系数，使位移与真实时间步长一致。
        self.F_k[0, 4] = dt
        self.F_k[1, 5] = dt

        # 根据当前预测框宽高计算尺度因子。
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
        # X_k = F_k * X_k-1
        x_pred = np.dot(self.F_k, x)

        # 预测协方差矩阵
        # P_k = F_k * P_k-1 * F_k^T + Q_k
        P_current = np.dot(self.F_k, np.dot(P_result, self.F_k.T)) + Q_k

        return x_pred, P_current

    def update(self, x: np.ndarray, P_current: np.ndarray, z: np.ndarray):
        """
        观测更新步。结合 YOLO 实际测量值纠正预测状态。
        参数:
            x: 预测状态向量 (X_k)
            P_current: 预测协方差矩阵 (P_k)
            z: 当前时刻 YOLO 传感器读数 [cx, cy, w, h]
        返回:
            x_new: 更新后的状态向量
            P_result: 更新后的协方差矩阵
            innovation: [debug] 残差 (z - Hx) 用于量化分析
        """

        # 依据当前观测值计算尺度因子
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
        zz_k = np.dot(self.H_k, x)


        # S_k: 创新协方差矩阵 (系统残差的协方差)
        # S_k = H_k * P_k * H_k^T + R_k
        S_k = np.dot(self.H_k, np.dot(P_current, self.H_k.T)) + R_k

        # 卡尔曼增益: K_k
        K_k = np.dot(np.dot(P_current, self.H_k.T), np.linalg.inv(S_k))

        # 最优预测状态向量值
        # X^_k = X_k + K_k * (z_k - zz_k) 
        x_new = x + np.dot(K_k, (z - zz_k))

        P_result = np.dot(self.I - np.dot(K_k, self.H_k), P_current)


        # 如果 YOLO 观测点和卡尔曼预测点中心距离超过 150 像素，直接判定预测失败
        # 强行重置位置并清空瞬时速度，防止预测框“起飞”
        dist = np.linalg.norm(x[:2] - z[:2])
        if dist > 150:
            x_new[:4] = z
            x_new[4:] = 0.0
            
            # 底层状态硬钳制。限制像素速度 (vx, vy, vw, vh) 单帧最大变化量不超过 50 像素
            x_new[4:] = np.clip(x_new[4:], -50, 50)


            # 大残差回退：仅重置位置与尺度，保留平移速度连续性
            x_new[:4] = z

            # 保留并衰减平移速度，避免“速度归零”导致后续几乎不动
            x_new[4:6] = 0.7 * x[4:6]

            # 尺度速度在突变时清零，防止宽高发散
            x_new[6:8] = 0.0

            # 协方差回退但不完全冷启动，降低抖动
            P_reset = P_current.copy()
            P_reset[:4, :4] *= 0.5   # 位置尺度更信观测
            P_reset[4:, 4:] *= 1.2   # 速度保持一定不确定性
        
            return x_new, np.eye(self.n)

            
        return x_new, P_result