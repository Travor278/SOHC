import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import AdjacentActiveBalancingNet

def validate_model():
    # ==========================
    # 1. 配置参数
    # ==========================
    DATA_PATH = 'combined_training_data.pt'
    MODEL_PATH = 'linear_model.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    SPLIT_RATIO = 0.8
    NUM_PACKS = 8
    INPUT_DIM = 4  # (V, SOC, I, T)

    print(f"正在使用设备: {DEVICE}")

    # ==========================
    # 2. 准备验证数据
    # ==========================
    print("正在加载数据并切分验证集...")
    full_data = torch.load(DATA_PATH)
    total_samples = len(full_data)
    train_size = int(total_samples * SPLIT_RATIO)

    # 取后 20% 作为验证集 (必须与训练时的切分逻辑一致)
    val_data = full_data[train_size:]
    print(f"验证集样本数: {len(val_data)}")

    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)

    # ==========================
    # 3. 加载模型
    # ==========================
    print("正在加载模型权重...")
    model = AdjacentActiveBalancingNet(input_dim=INPUT_DIM, num_packs=NUM_PACKS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 成功加载 best_model.pth")
    except FileNotFoundError:
        print("❌ 未找到模型文件，请先运行训练脚本！")
        return

    model.eval()

    # ==========================
    # 4. 执行推理与统计
    # ==========================
    all_targets = []  # 真实的 Delta SOC (Input Index 1)
    all_actions = []  # 模型输出

    loss_mse = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_inputs in val_loader:
            inputs = batch_inputs[0].to(DEVICE)

            # 推理
            actions = model(inputs)  # [Batch, 8]

            # 提取 Target (Index 1 是 Delta SOC)
            # 注意：这里的 Target 是预处理过的 (Delta_SOC * 20)，本身就是归一化数值
            targets = inputs[:, -1, :, 1]

            # 计算 Loss
            # 理想情况下，动作应该也是负的 target (Target=-tanh(diff))
            # 但为了简单对比，我们直接看相关性
            loss_mse += criterion(actions, -torch.tanh(targets)).item()

            # 收集数据用于画图
            all_targets.append(targets.cpu().numpy())
            all_actions.append(actions.cpu().numpy())

    avg_mse = loss_mse / len(val_loader)
    print(f"\n验证集平均 MSE Loss: {avg_mse:.6f}")

    # 展平数据用于散点图
    flat_targets = np.concatenate(all_targets).flatten()
    flat_actions = np.concatenate(all_actions).flatten()

    # 计算相关系数
    corr = np.corrcoef(flat_targets, flat_actions)[0, 1]
    print(f"SOC差异与均衡动作的相关系数: {corr:.4f}")
    if corr < -0.9:
        print("✅ 强负相关：逻辑完美符合物理规律。")
    elif corr < -0.5:
        print("⚠️ 弱负相关：逻辑基本正确，但可能不够激进。")
    else:
        print("❌ 正相关或无相关：模型逻辑错误！")

    # ==========================
    # 5. 可视化分析
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图 1: 蝴蝶图 (逻辑验证) ---
    # X轴: 输入的 Delta SOC (特征值), Y轴: 输出的动作
    ax1.scatter(flat_targets, flat_actions, alpha=0.1, s=10, c='blue')
    ax1.set_title(f"Logic Validation (Butterfly Plot)\nCorrelation: {corr:.3f}")
    ax1.set_xlabel("Input Feature: Delta SOC (Weighted)")
    ax1.set_ylabel("Model Output: Balancing Action")
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.axvline(0, color='gray', linestyle='--')
    ax1.grid(True, alpha=0.3)
    # 画一条理想参考线 y = -tanh(x)
    x_ref = np.linspace(flat_targets.min(), flat_targets.max(), 100)
    ax1.plot(x_ref, -np.tanh(x_ref), 'r--', label='Ideal Rule')
    ax1.legend()

    # --- 图 2: 单帧快照 (Snapshot) ---
    # 随机找一个差异比较大的样本看看细节
    # 找 max(abs(target)) 最大的那个样本索引
    sample_idx = np.argmax(np.max(np.abs(all_targets[0]), axis=1))

    snapshot_soc = all_targets[0][sample_idx]  # 取第0个batch的某一行
    snapshot_act = all_actions[0][sample_idx]
    packs = np.arange(1, NUM_PACKS + 1)

    # 双轴图
    ax2_soc = ax2.twinx()

    # 画 SOC 差异 (条形图)
    p1 = ax2.bar(packs - 0.2, snapshot_soc, width=0.4, color='orange', label='Delta SOC (Input)', alpha=0.7)
    ax2.set_xlabel("Battery Pack ID")
    ax2.set_ylabel("Delta SOC Feature", color='orange')
    ax2.set_ylim(-2, 2)

    # 画 动作 (条形图)
    p2 = ax2_soc.bar(packs + 0.2, snapshot_act, width=0.4, color='green', label='Action (Output)', alpha=0.7)
    ax2_soc.set_ylabel("Balancing Weight", color='green')
    ax2_soc.set_ylim(-1.1, 1.1)

    ax2.set_title("Snapshot of a Single Time Step")

    # 图例
    lines = [p1, p2]
    ax2.legend(lines, [l.get_label() for l in lines], loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_model()