import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -------- Matplotlib 中文字体设置 --------
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# ========== 1. 读取验证数据 ==========
df = pd.read_excel('verify_data.xlsx')  # 要有 t, I, SOC, V_meas 四列
t = df['t'].to_numpy()
I_raw = df['I'].to_numpy()
SOC = df['SOC'].to_numpy()
V_meas = df['V_meas'].to_numpy()

# 电流符号修正：放电为正，充电为负
I = -I_raw

print("SOC范围: {:.4f} ~ {:.4f}".format(SOC.min(), SOC.max()))

# ========== 2. OCV-SOC 多项式公式 ==========
# 使用你的真实 OCV 表拟合
SOC_data = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
OCV_data = np.array([2.895, 3.227, 3.277, 3.303, 3.306, 3.307, 3.313, 3.340, 3.341, 3.340, 3.344])

coeffs = np.polyfit(SOC_data, OCV_data, deg=5)
def OCV(soc):
    return np.polyval(coeffs, soc)

print("拟合OCV范围: {:.4f} V ~ {:.4f} V".format(OCV(SOC.min()), OCV(SOC.max())))

# ========== 3. 二阶ECM参数（替换成你的辨识结果） ==========#0.000100000000000000	0.000302189866398078	0.00152546021732605	100000	9985.09709326495
R0 = 0.0001
R1 = 0.000302189866398078
C1 = 100000
R2 = 0.00152546021732605
C2 = 9985.09709326495

tau1 = R1 * C1
tau2 = R2 * C2

print(f"τ1(快过程) = {tau1:.2f} 秒")
print(f"τ2(慢过程) = {tau2:.2f} 秒")

# ========== 4. 二阶ECM仿真端电压 ==========
V_model = np.zeros_like(t, dtype=float)
Vp1 = 0.0
Vp2 = 0.0

# 初值赋值
V_model[0] = OCV(SOC[0]) - I[0]*R0 - Vp1 - Vp2
print(f"初始条件: V_model0={V_model[0]:.6f} V")

for k in range(1, len(t)):
    dt = t[k] - t[k-1]
    alpha1 = np.exp(-dt / tau1)
    alpha2 = np.exp(-dt / tau2)
    Vp1 = alpha1 * Vp1 + R1 * (1 - alpha1) * I[k-1]
    Vp2 = alpha2 * Vp2 + R2 * (1 - alpha2) * I[k-1]
    V_model[k] = OCV(SOC[k]) - I[k]*R0 - Vp1 - Vp2

# ========== 5. 计算误差指标 ==========
rmse = np.sqrt(np.mean((V_model - V_meas)**2))
max_err = np.max(np.abs(V_model - V_meas))

print(f"\nRMSE: {rmse:.6f} V")
print(f"最大绝对误差: {max_err:.6f} V")

# ========== 6. 绘制比较曲线 ==========
plt.figure(figsize=(10,5))
plt.plot(t, V_meas, label='实测端电压', color='blue')
plt.plot(t, V_model, label='二阶模型端电压', color='red', linestyle='--')
plt.xlabel("时间 (s)")
plt.ylabel("端电压 (V)")
plt.title("二阶ECM充电数据验证: 实测 vs 模型")
plt.legend()
plt.grid(True)
plt.show()
