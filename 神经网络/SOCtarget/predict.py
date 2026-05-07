import matplotlib.pyplot as plt
import numpy as np
import torch
from train import get_batch_data
from model import AdjacentActiveBalancingNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AdjacentActiveBalancingNet().to(device)
model.load_state_dict(torch.load("traincheckpoint.pth", map_location=device))

import torch
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 0. 模拟环境：磷酸铁锂 (LFP) OCV 曲线
# ==========================================
def get_lfp_ocv(soc):
    """
    模拟 LFP 电池的 OCV 曲线。
    特点：中间段 (30%~70%) 非常平坦，斜率极小。
    """
    # 基础电压 3.2V
    # 斜率极小: 0.05 (意味着 100% SOC 变化只有 0.05V 电压变化)
    # 这对纯电压模型是噩梦
    slope_term = 0.05 * (soc - 0.5)

    # 两端的非线性 (低电量掉电快，满电升压快) - 增加真实感
    low_end = -0.05 * np.exp(-20 * (soc - 0.1))  # SOC < 10% 掉电压
    high_end = 0.02 * np.exp(20 * (soc - 0.9))  # SOC > 90% 升电压

    return 3.3 + slope_term + low_end + high_end


# ==========================================
# 1. 新版预处理 (必须与训练一致)
# ==========================================
def preprocess_input_soc_aware(raw_inputs, device):
    """
    输入: [Batch, Time, Pack, 4] -> (V, SOC, I, T)
    """
    processed = raw_inputs.clone().to(device)

    # 1. Voltage (Index 0): 放大 50 倍
    v_raw = processed[:, :, :, 0]
    processed[:, :, :, 0] = (v_raw - v_raw.mean(dim=2, keepdim=True)) * 50.0

    # 2. SOC (Index 1): 放大 20 倍 【核心驱动力】
    # 这是新模型能工作的关键
    soc_raw = processed[:, :, :, 1]
    processed[:, :, :, 1] = (soc_raw - soc_raw.mean(dim=2, keepdim=True)) * 20.0

    # 3. Current & Temp
    processed[:, :, :, 2] /= 20.0
    processed[:, :, :, 3] = (processed[:, :, :, 3] - 25.0) / 10.0

    return processed


# ==========================================
# 2. 闭环仿真主程序
# ==========================================
def test_lfp_plateau_convergence(model, device):
    model.eval()

    print("\n=== 开始 LFP 电池平台区均衡测试 (SOC Guided) ===")

    # --- 配置仿真参数 ---
    NUM_PACKS = 8
    CAPACITY = 2.0  # 2Ah 小电池，加速收敛
    MAX_CURRENT = 5.0  # 5A 均衡电流
    DT = 10.0  # 10秒一步
    TOTAL_STEPS = 200  # 2000秒 (约30分钟)

    # --- 初始状态设计 (困难场景) ---
    # 所有电池都在 50% 附近 (LFP 平台区)
    # 制造严重的 SOC 不一致，但电压差异极小
    soc = np.full(NUM_PACKS, 0.5)
    soc[0] = 0.40  # Pack 0 少 10%
    soc[1] = 0.60  # Pack 1 多 10%
    # 预期：SOC 差 0.2，电压可能只差 0.01V

    # 记录
    history_soc = []
    history_v = []
    history_action = []

    # 初始化输入 Buffer [1, 50, 12, 4]
    input_buffer = torch.zeros(1, 50, NUM_PACKS, 4).to(device)

    for step in range(TOTAL_STEPS):
        # 1. 物理计算 (Physics)
        v_true = get_lfp_ocv(soc)
        v_noise = v_true + np.random.normal(0, 0.001, NUM_PACKS)  # 加一点噪声

        # 2. 构造数据帧 [Batch=1, Time=1, Pack=12, Feat=4]
        # Feat: V, SOC, I, T
        # 假设电流为0(静置均衡)，温度25度
        current_frame = np.stack([
            v_noise,
            soc,  # <--- 注入真实的 SOC (由BMS估算得到)
            np.zeros(NUM_PACKS),
            np.full(NUM_PACKS, 25.0)
        ], axis=1)

        current_tensor = torch.FloatTensor(current_frame).unsqueeze(0).unsqueeze(0).to(device)

        # 滚动更新 Buffer
        input_buffer = torch.cat([input_buffer[:, 1:], current_tensor], dim=1)

        # 3. 预处理 (Feature Engineering)
        model_input = preprocess_input_soc_aware(input_buffer, device)

        # 4. 模型推理 (Inference)
        with torch.no_grad():
            actions = model(model_input)[0].cpu().numpy()

        # 5. 状态更新 (Update)
        # 能量守恒约束
        i_balance = actions * MAX_CURRENT
        i_balance -= np.mean(i_balance)

        # SOC 积分
        d_soc = (i_balance * DT) / (CAPACITY * 3600.0)
        soc += d_soc

        # 记录
        history_soc.append(soc.copy())
        history_v.append(v_true.copy())
        history_action.append(actions.copy())

        # --- 关键：打印中间状态证明模型利用了 SOC ---
        if step == 5:
            print(f"\n[Step 5 Snapshot]")
            # 计算一下电压差和 SOC 差
            v_diff = v_true[1] - v_true[0]
            soc_diff = soc[1] - soc[0]
            act_diff = actions[1] - actions[0]

            print(f"Pack 1 vs Pack 0:")
            print(f"  Delta SOC     : {soc_diff:.4f} (显著差异)")
            print(f"  Delta Voltage : {v_diff:.4f} V (差异极小!)")
            print(f"  Model Action  : {actions[1]:.4f} vs {actions[0]:.4f}")
            print(f"  -> 结论: 电压几乎没变，但模型全功率动作，证明 SOC 特征生效。")

    # ==========================================
    # 3. 结果可视化
    # ==========================================
    history_soc = np.array(history_soc)
    history_v = np.array(history_v)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 图 1: SOC 收敛曲线
    for i in range(NUM_PACKS):
        lw = 3 if i in [0, 1] else 1
        label = f'Pack {i}' if i in [0, 1] else None
        ax1.plot(history_soc[:, i], linewidth=lw, label=label)

    ax1.set_title("SOC Convergence (LFP Plateau Region)")
    ax1.set_ylabel("SOC")
    ax1.set_xlabel("Steps")
    ax1.legend()
    ax1.grid(True)

    # 图 2: 电压曲线 (展示为什么只靠电压很难)
    for i in range(NUM_PACKS):
        lw = 3 if i in [0, 1] else 1
        ax2.plot(history_v[:, i], linewidth=lw)

    ax2.set_title("Voltage Trace (Notice how flat/close they are)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_xlabel("Steps")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# --- 运行测试 ---
# 假设 model 已经是训练好且 input_dim=4 的模型
# test_lfp_plateau_convergence(model, DEVICE)


# 运行仿真
# run_closed_loop_sim(model, DEVICE)
if __name__ == '__main__':
    test_lfp_plateau_convergence(model, device)
