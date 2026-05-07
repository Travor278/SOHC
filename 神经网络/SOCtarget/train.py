import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import AdjacentActiveBalancingNet


# def get_batch_data(batch_size, time_steps, num_packs, device):
#     batch_list = []
#     for _ in range(batch_size):
#         # 1. 随机生成初始状态
#         # 模拟真实情况：SOC 不一致 (0.05 的标准差)
#         base_soc = np.random.uniform(0.3, 0.7)
#         soc_init = np.random.normal(base_soc, 0.05, num_packs)
#         soc_init = np.clip(soc_init, 0.0, 1.0)
#
#         # 2. 模拟运行
#         # 这里简化写，不再跑循环，直接生成随机波动数据
#         # 实际项目中应使用完整的 OCV 积分过程
#
#         # [Time, Pack]
#         # 生成随时间变化的 SOC (模拟充放电过程)
#         # 假设电流波动导致 SOC 变化，但包之间的 SOC 差值(Delta)基本保持稳定
#         soc_seq = []
#         current_soc = soc_init.copy()
#         current_profile = np.random.normal(0, 5, time_steps)  # 电流
#
#         voltages = []
#         socs = []
#         currents = []
#         temps = []
#
#         for t in range(time_steps):
#             # 简单模拟：SOC 积分
#             d_soc = current_profile[t] / (50.0 * 3600)  # 假设50Ah
#             current_soc += d_soc
#             current_soc = np.clip(current_soc, 0, 1)
#
#             # OCV 模型: V = f(SOC) + I*R
#             # 加上随机噪声模拟传感器误差
#             v = (3.0 + 1.0 * current_soc) + np.random.normal(0, 0.002, num_packs)
#
#             # 记录数据
#             voltages.append(v)
#             socs.append(current_soc.copy())  # 【新增】记录 SOC
#             currents.append(np.full(num_packs, current_profile[t]))
#             temps.append(np.full(num_packs, 30.0 + np.random.normal(0, 1)))
#
#         # 3. 堆叠数据 [Time, Pack, 4]
#         # Order: Voltage, SOC, Current, Temp
#         sample = np.stack([
#             np.array(voltages),
#             np.array(socs),  # 【新增】
#             np.array(currents),
#             np.array(temps)
#         ], axis=2)
#
#         batch_list.append(sample)
#
#     return torch.FloatTensor(np.array(batch_list)).to(device)
#
#
# def preprocess_enhanced_input(raw_inputs, device):
#     """
#     输入: [Batch, Time, Pack, 4] -> (V, SOC, I, T)
#     输出: [Batch, Time, Pack, 4] -> (Delta_V, Delta_SOC, Norm_I, Norm_T)
#     """
#     processed = raw_inputs.clone().to(device)
#
#     # --- 1. 处理电压 (Index 0) ---
#     v_raw = processed[:, :, :, 0]
#     v_mean = torch.mean(v_raw, dim=2, keepdim=True)
#     # 放大系数 50.0: 0.02V -> 1.0
#     processed[:, :, :, 0] = (v_raw - v_mean) * 50.0
#
#     # --- 2. 处理 SOC (Index 1) 【新增】 ---
#     soc_raw = processed[:, :, :, 1]
#
#     # 计算 Delta SOC
#     # BMS 中 SOC 也是相对的，我们要看谁比平均值高
#     soc_mean = torch.mean(soc_raw, dim=2, keepdim=True)
#     delta_soc = soc_raw - soc_mean
#
#     # 放大系数 10.0 ~ 20.0
#     # 理由: 两个电池差 5% SOC (0.05) 是很大的差异，需要放大到 0.5~1.0 左右
#     processed[:, :, :, 1] = delta_soc * 20.0
#
#     # --- 3. 处理电流 (Index 2) ---
#     processed[:, :, :, 2] = processed[:, :, :, 2] / 20.0
#
#     # --- 4. 处理温度 (Index 3) ---
#     processed[:, :, :, 3] = (processed[:, :, :, 3] - 25.0) / 10.0
#
#     # 同时返回真实的 delta_v 用于 Loss 计算 (Target 生成)
#     # 注意：Target 主要还是参考 Delta V 还是 Delta SOC？
#     # 答案：主要参考 Delta SOC (它是本质)，Delta V 是表象
#     # 但在 OCV 线性区，两者是一致的。建议返回 delta_soc 用于生成 Target
#     return processed, delta_soc
class LinearControlLoss(nn.Module):
    def __init__(self, full_power_threshold=1.5):
        """
        full_power_threshold: 满功率阈值。
        当输入的 Delta_SOC (预处理后的值) 绝对值超过这个数时，目标权重设为 1.0。
        """
        super(LinearControlLoss, self).__init__()
        self.mse = nn.MSELoss()

        # 计算比例系数 k
        # 例如: 阈值是 2.0，那么 k = 0.5。输入 1.0 时，目标 = 1.0 * 0.5 = 0.5
        self.k = 1.0 / full_power_threshold

    def forward(self, pred_actions, delta_soc_input):
        """
        pred_actions: 模型输出的权重 [Batch, 8]
        delta_soc_input: 真实的 SOC 差异特征 [Batch, 8] (预处理放大后的)
        """

        # 1. 计算线性目标 (P-Control)
        # 物理逻辑：SOC 高 -> 需要放电 (负) -> 乘以负系数
        linear_target = -delta_soc_input * self.k

        # 2. 截断/限幅 (Clamp)
        # 超过阈值的部分强制限制在 [-1, 1]
        target = torch.clamp(linear_target, -1.0, 1.0)

        # 3. 能量守恒修正 (Zero-Mean)
        # 这一步非常重要，保证算出来的目标是物理可实现的
        target_mean = torch.mean(target, dim=1, keepdim=True)
        centered_target = target - target_mean

        # 4. 计算 MSE
        return self.mse(pred_actions, centered_target)

class SOCGuidedLoss(nn.Module):
    def __init__(self):
        super(SOCGuidedLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, actions, real_delta_soc):
        # 注意：这里输入变成了 real_delta_soc

        # 1. 生成 Target
        # SOC 高 -> 放电 (-1)
        # 放大倍数 20.0: 5% 的 SOC 差异 (0.05) -> tanh(1.0) ≈ 0.76
        raw_target = -torch.tanh(real_delta_soc * 20.0)

        # 2. 能量守恒 (去均值)
        target_mean = torch.mean(raw_target, dim=1, keepdim=True)
        centered_target = raw_target - target_mean

        # 3. MSE Loss
        loss = self.mse(actions, centered_target) * 1000.0

        return loss


def train_active_balancing():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_PACKS = 8
    TIME_STEPS = 50
    EPOCHS = 300

    full_data_tensor = torch.load('combined_training_data.pt')  # [Total, 50, 8, 4]
    total_samples = len(full_data_tensor)

    # 2. 计算切分点 (例如 8:2 分割)
    split_ratio = 0.8
    train_size = int(total_samples * split_ratio)
    # 剩下的作为验证集
    val_size = total_samples - train_size

    print(f"总样本数: {total_samples}")
    print(f"训练集: {train_size}, 验证集: {val_size}")

    # 3. 手动切片 (保持时间连续性)
    train_tensor = full_data_tensor[:train_size]  # 前 80%
    val_tensor = full_data_tensor[train_size:]  # 后 20%

    # 4. 创建 DataLoader
    # 训练集：需要 shuffle 打乱，让梯度下降更稳定
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 验证集：不需要 shuffle，通常按顺序测试即可
    val_dataset = TensorDataset(val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 实例化
    model = AdjacentActiveBalancingNet(input_dim=4, num_packs=NUM_PACKS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = LinearControlLoss()

    print(f"Training on {DEVICE}...")

    model.train()
    loss_history = []

    best_val_loss = float('inf')  # 用于保存最佳模型

    for epoch in range(EPOCHS):
        # --- A. 训练阶段 (Training) ---
        model.train()  # 启用 Dropout / BatchNorm 更新
        train_loss_sum = 0.0

        for batch_idx, (batch_inputs,) in enumerate(train_loader):
            optimizer.zero_grad()
            actions = model(batch_inputs)

            # 提取 Target: 最后时刻的 Delta SOC (Index 1)
            # 预处理时已经乘过放大系数了，直接用
            target = batch_inputs[:, -1, :, 1]

            loss = criterion(actions, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # --- B. 验证阶段 (Validation) ---
        model.eval()  # 冻结 Dropout / BatchNorm
        val_loss_sum = 0.0

        with torch.no_grad():  # 不计算梯度，节省显存
            for batch_inputs in val_loader:
                # DataLoader 如果 dataset 只有 tensor，出来是 tuple，需要解包
                # 注意: 这里如果不加 (batch_inputs,) = ... 写法，
                # 可以直接 inputs = batch_inputs[0]
                inputs = batch_inputs[0]

                val_actions = model(inputs)
                val_target = inputs[:, -1, :, 1]

                v_loss = criterion(val_actions, val_target)
                val_loss_sum += v_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        # --- C. 打印与保存 ---
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 如果验证集效果变好了，保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "linear_model.pth")
            print("  -> 性能提升，已保存最佳模型 (linear_model.pth)")

            # with torch.no_grad():
            #     # raw_inputs 和 actions 都是当前 Batch 的
            #     # 注意：这里的 raw_inputs 还没有经过特征放大，如果用了增强需要小心
            #     # 但我们只要看 delta_v 和 actions 的关系
            #
            #     # 获取当前 Batch 的 Delta V (last step)
            #     cur_soc= raw_inputs[:, -1, :, 0]
            #     cur_mean = torch.mean(cur_soc, dim=1, keepdim=True)
            #     cur_delta = (cur_soc - cur_mean).cpu().numpy().flatten()
            #     cur_act = actions.cpu().numpy().flatten()
            #
            #     plt.scatter(cur_delta, cur_act)
            #     plt.title(f"Training Batch Snapshot Epoch {epoch}")
            #     plt.show()  # 会弹窗，关闭后继续训练

    # print("训练完成！")
    # torch.save(model.state_dict(), "traincheckpoint.pth")
#     model.eval()
#
#     # 生成测试数据
#     raw_inputs = get_batch_data(batch_size=50, time_steps=50, num_packs=12, device=DEVICE)
#
#     # 预处理
#     model_inputs, real_delta_v = preprocess_input(raw_inputs, DEVICE)
#
#     # 推理
#     with torch.no_grad():
#         actions = model(model_inputs)
#
#     # --- 画图 ---
#     # 取最后一个时刻
#     x_val = real_delta_v[:, -1, :].cpu().numpy().flatten()
#     y_val = actions.cpu().numpy().flatten()
#
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x_val, y_val, alpha=0.6, c='blue', label='Validation Output')
#
#     # 辅助线
#     plt.axhline(0, color='gray', linestyle='--')
#     plt.axvline(0, color='gray', linestyle='--')
#     plt.title(f"Final Verification (Gain={50})")
#     plt.xlabel("Real Delta Voltage (V)")
#     plt.ylabel("Action Weight")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.show()
#
#
#
# def preprocess_input(raw_inputs, device):
#     """
#     验证集专用预处理
#     """
#     # 复制数据
#     processed = raw_inputs.clone().to(device)
#
#     # 1. 提取电压
#     v_raw = processed[:, :, :, 0]
#
#     # 2. 计算均值 (去中心化)
#     # [Batch, Time, 1]
#     v_mean = torch.mean(v_raw, dim=2, keepdim=True)
#
#     # 3. 计算 Delta V
#     delta_v = v_raw - v_mean
#
#     # 4. 【关键修正】特征放大
#     # 没有这一步，S形曲线就会被压缩成一条直线
#     processed[:, :, :, 0] = delta_v * 50
#
#     # 5. 其他特征归一化
#     processed[:, :, :, 1] = processed[:, :, :, 1] / 20.0  # Current
#     processed[:, :, :, 2] = (processed[:, :, :, 2] - 25.0) / 10.0  # Temp
#
#     # 返回处理后的输入 + 真实的 delta_v (用于画图 X 轴)
#     return processed, delta_v


if __name__ == "__main__":
    train_active_balancing()

