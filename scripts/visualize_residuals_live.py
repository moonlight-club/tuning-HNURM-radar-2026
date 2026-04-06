import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import os
import time

# [debug] 动态实时残差监控脚本
# 功能：实时读取日志并绘制“示波器”风格的残差曲线，支持 Label 身份识别

def live_visualization(csv_path):
    if not os.path.exists(csv_path):
        print(f"等待日志文件生成: {csv_path} ...")
        # 创建空文件防止读取失败
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
        with open(csv_path, 'w') as f:
            pass

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 存储不同机器人的曲线对象
    # { 'Label(ID)': { 'line_x': Line2D, 'line_y': Line2D, 'data_x': [], 'data_y': [], 'times': [] } }
    robot_plots = {}
    
    # 最大显示的秒数
    MAX_WINDOW_SEC = 10 
    
    def update(frame):
        try:
            # 实时读取 CSV
            # 格式: timestamp, id, label, dx, dy, dw, dh
            df = pd.read_csv(csv_path, names=['timestamp', 'id', 'label', 'dx', 'dy', 'dw', 'dh'])
        except Exception:
            return []

        if df.empty:
            return []

        curr_time = time.time()
        
        # 获取最近 10 秒的数据
        df = df[df['timestamp'] > curr_time - MAX_WINDOW_SEC]
        
        unique_ids = df['id'].unique()
        
        active_plots = []
        for rid in unique_ids:
            data = df[df['id'] == rid]
            label = data['label'].iloc[-1]
            key = f"{label}({rid})"
            
            # 如果是新机器人，创建曲线
            if key not in robot_plots:
                line_x, = ax_x.plot([], [], label=key, alpha=0.8)
                line_y, = ax_y.plot([], [], label=key, alpha=0.8)
                robot_plots[key] = {'lx': line_x, 'ly': line_y}
                ax_x.legend(loc='upper right', fontsize='small', ncol=3)
                ax_y.legend(loc='upper right', fontsize='small', ncol=3)

            # 更新数据
            rel_times = data['timestamp'] - curr_time
            robot_plots[key]['lx'].set_data(rel_times, data['dx'])
            robot_plots[key]['ly'].set_data(rel_times, data['dy'])
            active_plots.extend([robot_plots[key]['lx'], robot_plots[key]['ly']])

        # 动态调整坐标轴
        ax_x.set_xlim(-MAX_WINDOW_SEC, 0)
        ax_y.set_xlim(-MAX_WINDOW_SEC, 0)
        
        # 自动调整纵轴范围（带缓冲）
        if not df.empty:
            max_err = max(df['dx'].abs().max(), df['dy'].abs().max(), 10)
            ax_x.set_ylim(-max_err * 1.2, max_err * 1.2)
            ax_y.set_ylim(-max_err * 1.2, max_err * 1.2)

        return active_plots

    ax_x.set_title("X-Axis (Center-X) Innovation Residuals")
    ax_x.set_ylabel("Error (pixels)")
    ax_x.grid(True, alpha=0.3)
    ax_x.axhline(0, color='black', lw=1, ls='--')

    ax_y.set_title("Y-Axis (Center-Y) Innovation Residuals")
    ax_y.set_ylabel("Error (pixels)")
    ax_y.set_xlabel("Time (seconds ago)")
    ax_y.grid(True, alpha=0.3)
    ax_y.axhline(0, color='black', lw=1, ls='--')

    ani = FuncAnimation(fig, update, interval=200, blit=False) # 200ms 刷新一次
    plt.show()

if __name__ == "__main__":
    csv_log = "logs/kalman_residuals.csv"
    if len(sys.argv) > 1:
        csv_log = sys.argv[1]
        
    # [debug] 运行前清空旧日志，保证观察的是当前数据
    if os.path.exists(csv_log):
        # os.remove(csv_log)
        # print(f"已清理旧日志: {csv_log}")
        pass

    live_visualization(csv_log)
