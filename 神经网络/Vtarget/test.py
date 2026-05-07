import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ==========================================
# 1. 极简模型 (只保留 MLP)
# ==========================================
class SimpleDebugNet(nn.Module):
    def __init__(self, num_packs=12):
        super(SimpleDebugNet, self).__init__()
        # 输入维度: Num_Packs (直接输入12个电压差值)
        # 既然是均衡，只看当前时刻电压最重要，时序先不管
        self.net = nn.Sequential(
            nn.Linear(num_packs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_packs),
            nn.Tanh()  # 输出 -1 ~ 1
        )

    def forward(self, x):
        # x shape: [Batch, NumPacks] -> 输入的是 Delta V
        return self.net(x)


# ==========================================
# 2. 强力 Loss (MSE 引导)
# ==========================================
def tough_loss(pred_action, delta_v):
    """
    pred_action: 模型输出
    delta_v: 输入的电压差 (已经去均值了)
    """
    # 目标：电压高(>0)，动作就要是负(-1); 电压低(<0)，动作就要是正(+1)
    # 我们构造一个极其激进的 Target
    target = -torch.tanh(delta_v * 100.0)  # 放大100倍，哪怕0.01的差距也变成-0.76

    # 1. MSE Loss (核心)
    loss_mse = nn.MSELoss()(pred_action, target)

    # 2. 能量守恒 Loss (软约束替代代码里的硬减法)
    # 惩罚总和不为0的情况
    loss_sum = torch.mean(torch.sum(pred_action, dim=1) ** 2)

    return loss_mse + 0.1 * loss_sum


# ==========================================
# 3. 调试主程序
# ==========================================
def run_debug():
    # 配置
    DEVICE = torch.device("cpu")  # 调试用 CPU 够了
    NUM_PACKS = 12
    BATCH_SIZE = 4  # 哪怕只有4个样本

    # 初始化
    model = SimpleDebugNet(NUM_PACKS).to(DEVICE)
    # 学习率给大一点，大力出奇迹
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("--- 开始极简模式调试 ---")
    print("目标：验证网络是否具备基本的拟合能力")

    # --- 构造一个固定的“故障”样本 ---
    # 我们不再随机生成，而是固定下来，强迫网络背答案
    # 场景：Pack 0 电压极高，Pack 1 电压极低，其他正常
    raw_v = torch.zeros(BATCH_SIZE, NUM_PACKS).to(DEVICE)
    raw_v[:, 0] = 3.8  # High
    raw_v[:, 1] = 3.2  # Low
    raw_v[:, 2:] = 3.5  # Normal

    # 预处理：计算 Delta V (这是关键特征)
    v_mean = torch.mean(raw_v, dim=1, keepdim=True)
    inputs_delta = raw_v - v_mean  # [Batch, 12]

    # 训练循环
    for i in range(501):
        optimizer.zero_grad()

        # 前向
        actions = model(inputs_delta)

        # Loss
        loss = tough_loss(actions, inputs_delta)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Step {i} | Loss: {loss.item():.6f}")
            # 打印第一个样本的 Pack 0 和 Pack 1 的动作
            act = actions[0].detach().numpy()
            print(f"   -> Pack 0 (High V) Action: {act[0]:.4f} (Expect -1.0)")
            print(f"   -> Pack 1 (Low V)  Action: {act[1]:.4f} (Expect +1.0)")
            print(f"   -> Pack 2 (Norm V) Action: {act[2]:.4f} (Expect 0.0)")

    # 最终验证可视化
    print("\n--- 最终结果 ---")
    final_act = model(inputs_delta)[0].detach().numpy()
    print("输入 Delta V:", np.round(inputs_delta[0].numpy(), 2))
    print("输出 Action :", np.round(final_act, 2))


if __name__ == "__main__":
    run_debug()