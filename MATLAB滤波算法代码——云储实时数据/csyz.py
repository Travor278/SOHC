import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 字体设置
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_excel('verify_data.xlsx')  # 包含 t, I, SOC, V_meas
t = df['t'].to_numpy()
I_raw = df['I'].to_numpy()
SOC = df['SOC'].to_numpy()
V_meas = df['V_meas'].to_numpy()

print("原始电流前5个点:", I_raw[:5])

# ===== 调整电流符号 =====
# 我假设你的数据是：充电时电流为正，放电时为负
# ECM公式假设放电为正、充电为负，因此我们要翻转符号
I = -I_raw

print("符号调整后电流前5个点:", I[:5])
print("SOC范围: {:.4f} ~ {:.4f}".format(SOC.min(), SOC.max()))

# 2. OCV拟合公式（保持你已有的真实数据表拟合）
SOC_data = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
OCV_data = np.array([2.895, 3.227, 3.277, 3.303, 3.306, 3.307, 3.313, 3.340, 3.341, 3.340, 3.344])

coeffs = np.polyfit(SOC_data, OCV_data, deg=5)
def OCV(soc):
    return np.polyval(coeffs, soc)

print("OCV范围: {:.4f} V ~ {:.4f} V".format(OCV(SOC.min()), OCV(SOC.max())))

# 3. ECM参数
R0 = 0.0003
R1 = 0.002370
C1 = 144115.428234
tau = R1 * C1
print(f"tau = {tau:.3f} 秒")

# 4. 模型计算
V_model = np.zeros_like(t, dtype=float)
Vp = 0.0

V_model[0] = OCV(SOC[0]) - I[0] * R0 - Vp
print(f"初始: SOC0={SOC[0]:.4f}, OCV0={OCV(SOC[0]):.4f}, V_model0={V_model[0]:.4f}")

for k in range(1, len(t)):
    dt = t[k] - t[k-1]
    alpha = np.exp(-dt / tau)
    # 极化支路更新（符号已统一）
    Vp = alpha * Vp + R1 * (1 - alpha) * I[k-1]
    V_model[k] = OCV(SOC[k]) - I[k] * R0 - Vp

# 5. 误差
rmse = np.sqrt(np.mean((V_model - V_meas)**2))
max_err = np.max(np.abs(V_model - V_meas))
print(f"RMSE: {rmse:.6f} V")
print(f"最大绝对误差: {max_err:.6f} V")

# 6. 绘图
plt.figure(figsize=(10,5))
plt.plot(t, V_meas, label='实测端电压', color='blue')
plt.plot(t, V_model, label='模型端电压（修正符号）', color='red', linestyle='--')
plt.xlabel("时间 (s)")
plt.ylabel("端电压 (V)")
plt.title("充电数据符号修正: 实测 vs 模型")
plt.legend()
plt.grid(True)
plt.show()
