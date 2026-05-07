import matplotlib.pyplot as plt
import numpy as np
import torch
from train import get_batch_data
from model import AdjacentActiveBalancingNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AdjacentActiveBalancingNet().to(device)
model.load_state_dict(torch.load("checkpoint.pth", map_location=device))


def run_closed_loop_sim(model, device):
    print("开始闭环仿真测试...")
    model.eval()

    # --- 1. 初始化仿真环境 ---
    NUM_PACKS = 12
    # 模拟真实容量 (Ah)
    capacities = np.random.normal(10.0, 1.0, NUM_PACKS)
    # 初始不一致的 SOC (0.5 +/- 0.05)
    current_soc = np.random.normal(0.5, 0.05, NUM_PACKS)
    current_soc = np.clip(current_soc, 0.1, 0.9)

    # 硬件参数
    MAX_BALANCE_CURRENT = 5.0  # 假设均衡器最大能力是 5A
    DT = 10 # 仿真步长 1秒
    TOTAL_STEPS = 2000  # 跑10分钟 (600秒)

    # 记录历史
    voltage_history = []
    soc_std_history = []  # 记录不一致性的标准差

    # 初始化 LSTM 的隐藏状态 (如果是有状态预测)
    # 这里简化处理，每次都喂过去50秒的数据，前面补0
    input_buffer = torch.zeros(1, 50, NUM_PACKS, 3).to(device)

    for step in range(TOTAL_STEPS):
        # --- A. 物理环境计算 (Physics) ---
        # 1. 简单的 OCV 模型 (SoC -> Voltage)
        v = 3.0 + 1.0 * current_soc + 0.2 * (current_soc ** 2)
        # 2. 加入随机噪声
        v_noise = v + np.random.normal(0, 0.002, NUM_PACKS)

        # --- B. 构造神经网络输入 ---
        # 更新 Buffer (这就相当于实时 BMS 采样)
        # 这里电流设为0 (静置均衡)，温度设为25
        current_frame = np.stack([v_noise, np.zeros(NUM_PACKS), np.full(NUM_PACKS, 25.0)], axis=1)  # [N, 3]
        current_tensor = torch.FloatTensor(current_frame).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, N, 3]

        # 滚动 Buffer
        input_buffer = torch.cat([input_buffer[:, 1:, :, :], current_tensor], dim=1)

        # 特征处理 (Delta V) - 必须与训练一致
        inputs_processed = input_buffer.clone()
        v_raw = inputs_processed[:, :, :, 0]
        v_mean = torch.mean(v_raw, dim=2, keepdim=True)
        inputs_processed[:, :, :, 0] = (v_raw - v_mean) * 10.0  # 放大特征
        # 归一化其他...

        # --- C. 模型预测 (Control) ---
        with torch.no_grad():
            actions = model(inputs_processed)  # [1, N]
            actions_np = actions[0].cpu().numpy()

        # --- D. 执行动作并更新物理状态 (Feedback) ---
        # Action (-1~1) -> 实际均衡电流 (A)
        # Action < 0: 放电; Action > 0: 充电
        balance_current = actions_np * MAX_BALANCE_CURRENT

        # 物理约束：能量守恒修正 (如果是主动均衡，总和必须为0)
        # 这里的修正模拟了硬件电路的局限性
        balance_current -= np.mean(balance_current)

        # 更新 SOC: dSOC = -I * dt / Capacity (注意符号，输出电流导致SOC下降)
        # balance_current > 0 代表流入电池(充电)，SOC增加
        delta_soc = (balance_current * DT) / (capacities * 3600.0)
        current_soc += delta_soc

        # 记录
        voltage_history.append(v)
        soc_std_history.append(np.std(current_soc))

    # --- 3. 绘图验证 ---
    voltage_history = np.array(voltage_history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 图1: 各单体电压随时间变化
    for i in range(NUM_PACKS):
        ax1.plot(voltage_history[:, i], alpha=0.6)
    ax1.set_title("Validation: Voltage Convergence (Closed-Loop)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True)

    # 图2: 一致性指标 (标准差)
    ax2.plot(soc_std_history, color='red', linewidth=2)
    ax2.set_title("Standard Deviation of SOC (Lower is Better)")
    ax2.set_ylabel("SOC Std Dev")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 运行仿真
# run_closed_loop_sim(model, DEVICE)
if __name__ == '__main__':
    run_closed_loop_sim(model, device)
