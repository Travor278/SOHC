import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import rcParams

# 使用支持中文的字体（SimHei: 黑体）
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# 避免负号显示成方块
rcParams['axes.unicode_minus'] = False
# -----------------------------
# 1. 读取 cy_data.xlsx 数据
# -----------------------------
# 确保 cy_data.xlsx 和此脚本在同一个目录，或者写完整路径
df = pd.read_excel('cy_data.xlsx')

# 提取时间和电压列（跳过表头，pandas 会自动处理）
t_data = df['t_data'].to_numpy()
V_data = df['v_data'].to_numpy()

# 已知参数
R0 = 0.0003        # 已知欧姆内阻(Ω)
I_p = 32.4        # 刚进入弛豫前的恒流脉冲幅值(A)

# -----------------------------
# 2. 将数据插值到 1s 分辨率
# -----------------------------
interp_func = interp1d(t_data, V_data, kind='cubic')  # 可改 'linear' 避免噪声放大
t_interp = np.arange(t_data[0], t_data[-1] + 1, 1)
V_interp = interp_func(t_interp)

# -----------------------------
# 3. 确定稳态电压并指数拟合
# -----------------------------
steady_points = int(0.05 * len(V_interp))  # 末段5%的数据
V_steady = np.mean(V_interp[-steady_points:])

def exp_decay(t, A, tau):
    return V_steady + A * np.exp(-t / tau)

A_init = V_interp[0] - V_steady
tau_init = (t_interp[-1] - t_interp[0]) / 2
popt, _ = curve_fit(exp_decay, t_interp, V_interp, p0=[A_init, tau_init])
A_fit, tau_fit = popt

# -----------------------------
# 4. 计算 R1 和 C1
# -----------------------------
R1 = A_fit / I_p
C1 = tau_fit / R1

# -----------------------------
# 5. 仿真模型响应并验证
# -----------------------------
V_model = exp_decay(t_interp, A_fit, tau_fit)
rmse = np.sqrt(np.mean((V_model - V_interp)**2))

# -----------------------------
# 6. 输出结果
# -----------------------------
print("=== 一阶ECM辨识结果 ===")
print(f"已知 R0: {R0:.6f} Ω")
print(f"拟合 R1: {R1:.6f} Ω")
print(f"拟合 C1: {C1:.6f} F")
print(f"时间常数 τ: {tau_fit:.2f} s")
print(f"电压幅值 A: {A_fit:.6f} V")
print(f"RMSE: {rmse:.6f} V")

# -----------------------------
# 7. 绘制拟合对比曲线
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(t_interp, V_interp, 'b', label='实测(插值)')
plt.plot(t_interp, V_model, 'r--', label='拟合模型')
plt.xlabel("时间 (s)")
plt.ylabel("电压 (V)")
plt.title("一阶等效电路模型拟合")
plt.legend()
plt.grid(True)
plt.show()
