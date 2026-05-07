import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import rcParams

# 字体设置
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_excel('cy_data.xlsx')
t_data = df['t_data'].to_numpy()
V_data = df['v_data'].to_numpy()

# 已知欧姆内阻 & 脉冲电流（正值）
R0 = 0.0003
I_p = 32.4

# 2. 插值到1s分辨率
interp_func = interp1d(t_data, V_data, kind='cubic')
t_interp = np.arange(t_data[0], t_data[-1] + 1, 1)
V_interp = interp_func(t_interp)

# 3. 稳态电压
steady_points = int(0.05 * len(V_interp))
V_steady = np.mean(V_interp[-steady_points:])

# 4. 二阶指数衰减模型
def exp_decay_2(t, A1, tau1, A2, tau2):
    return V_steady + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

# 初始猜测值
A_init1 = (V_interp[0] - V_steady) * 0.6
A_init2 = (V_interp[0] - V_steady) * 0.4
tau_init1 = (t_interp[-1] - t_interp[0]) / 10    # 快支路时间常数
tau_init2 = (t_interp[-1] - t_interp[0]) / 2     # 慢支路时间常数

popt, _ = curve_fit(exp_decay_2, t_interp, V_interp,
                    p0=[A_init1, tau_init1, A_init2, tau_init2],
                    bounds=([-np.inf, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]))

A1_fit, tau1_fit, A2_fit, tau2_fit = popt

# 5. 计算 R1, C1, R2, C2
R1 = A1_fit / I_p
C1 = tau1_fit / R1
R2 = A2_fit / I_p
C2 = tau2_fit / R2

# 6. 模型响应
V_model = exp_decay_2(t_interp, A1_fit, tau1_fit, A2_fit, tau2_fit)
rmse = np.sqrt(np.mean((V_model - V_interp) ** 2))

# 7. 输出结果
print("=== 二阶ECM辨识结果 ===")
print(f"已知 R0: {R0:.6f} Ω")
print(f"拟合 R1: {R1:.6f} Ω, C1: {C1:.2f} F, τ1: {tau1_fit:.2f} s, A1: {A1_fit:.6f} V")
print(f"拟合 R2: {R2:.6f} Ω, C2: {C2:.2f} F, τ2: {tau2_fit:.2f} s, A2: {A2_fit:.6f} V")
print(f"RMSE: {rmse:.6f} V")

# 8. 绘图对比
plt.figure(figsize=(8,5))
plt.plot(t_interp, V_interp, 'b', label='实测(插值)')
plt.plot(t_interp, V_model, 'r--', label='二阶ECM拟合')
plt.xlabel("时间 (s)")
plt.ylabel("电压 (V)")
plt.title("二阶等效电路模型参数辨识")
plt.legend()
plt.grid(True)
plt.show()
