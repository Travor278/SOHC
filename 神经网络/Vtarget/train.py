import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import AdjacentActiveBalancingNet


class VirtualBatteryPack:
    """
    模拟一个串联电池包，包含制造差异（容量、内阻）
    """

    def __init__(self, num_packs=12, dt=1.0):
        self.num_packs = num_packs
        self.dt = dt

        # 随机初始化差异参数
        # 容量差异: N(50Ah, 2Ah)
        self.capacities = np.random.normal(50.0, 2.0, num_packs)
        # 内阻差异: N(2mOhm, 0.1mOhm)
        self.r0 = np.random.normal(0.002, 0.0001, num_packs)
        # 初始SOC: 随机分布在 0.2 ~ 0.8
        base_soc = np.random.uniform(0.3, 0.7)
        self.soc = np.random.normal(base_soc, 0.05, num_packs)  # 围绕某个中心值波动
        self.soc = np.clip(self.soc, 0.05, 0.95)

    def ocv_model(self, soc):
        # 简易的三元锂 OCV 曲线拟合
        return 3.0 + 1.0 * soc + 0.2 * (soc ** 2) + 0.1 * np.log(soc + 0.01)

    def generate_cycle(self, steps=50):
        """生成一段时序数据"""
        # 随机生成电流工况 (模拟行车: 随机充放电)
        # 正态分布 + 低频正弦波动
        t = np.linspace(0, 10, steps)
        current_profile = 20 * np.sin(t) + np.random.normal(0, 5, steps)

        voltages = []
        currents = []
        temps = []

        # 初始温度
        temp = np.ones(self.num_packs) * 25.0 + np.random.normal(0, 1.0, self.num_packs)

        for i in range(steps):
            I_load = current_profile[i]  # 串联电流相同

            # 1. 更新 SOC (Ah积分)
            # dSOC = I * dt / Capacity
            self.soc += (I_load * self.dt) / (self.capacities * 3600)
            self.soc = np.clip(self.soc, 0.0, 1.0)

            # 2. 计算电压 V = OCV + I*R
            v = self.ocv_model(self.soc) + I_load * self.r0
            # 加入采样噪声
            v += np.random.normal(0, 0.005, self.num_packs)

            # 3. 简单模拟温度 (焦耳热)
            temp += 0.01 * (I_load ** 2 * self.r0) * self.dt  # 发热
            temp -= 0.005 * (temp - 25.0) * self.dt  # 散热

            voltages.append(v)
            currents.append(np.full(self.num_packs, I_load))
            temps.append(temp.copy())

        # 堆叠数据 [Steps, NumPacks, 3]
        # Data order: Voltage, Current, Temp
        data = np.stack([np.array(voltages), np.array(currents), np.array(temps)], axis=2)
        return data


def get_batch_data(batch_size, time_steps, num_packs, device):
    """
    生成一个 Batch 的训练数据
    返回形状: [Batch, TimeSteps, NumPacks, Features]
    """
    batch_list = []
    for _ in range(batch_size):
        pack = VirtualBatteryPack(num_packs)
        cycle_data = pack.generate_cycle(time_steps)
        batch_list.append(cycle_data)

    # 转换为 Tensor
    tensor_data = torch.FloatTensor(np.array(batch_list)).to(device)
    return tensor_data


class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, actions, real_delta_v):
        # 1. 原始 Target
        # 电压高 -> 需要放电 -> 负值
        raw_target = -torch.tanh(real_delta_v * 50.0)

        # 2. 【核心修正】: 强制 Target 能量守恒
        # 让教师信号自己先满足 "Sum = 0"
        # 这样模型只需要单纯地模仿教师，就不会产生冲突
        target_mean = torch.mean(raw_target, dim=1, keepdim=True)
        centered_target = raw_target - target_mean

        # 3. 只计算 MSE
        # 因为 Target 已经是 centered 的，模型拟合好了自然也是 centered 的
        # 不需要额外的 loss_bias
        loss = self.mse(actions, centered_target) * 1000.0

        return loss


def train_active_balancing():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_PACKS = 12
    TIME_STEPS = 50
    EPOCHS = 300

    # 实例化
    model = AdjacentActiveBalancingNet(num_packs=NUM_PACKS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = PhysicsLoss()

    print(f"Training on {DEVICE}...")

    model.train()
    loss_history = []

    for epoch in range(EPOCHS):
        raw_inputs = get_batch_data(BATCH_SIZE, TIME_STEPS, NUM_PACKS, DEVICE)

        # --- 【关键修改点：特征工程】 ---
        # 我们不能直接把 3.5V 扔进去，必须算出 Delta V
        # 1. 提取电压 [Batch, Time, Pack]
        v_raw = raw_inputs[:, :, :, 0]

        # 2. 计算当前 Batch 内，每个时刻的平均电压 [Batch, Time, 1]
        v_mean = torch.mean(v_raw, dim=2, keepdim=True)

        # 3. 算出 Delta V [Batch, Time, Pack]
        delta_v = v_raw - v_mean

        # 4. 构造网络输入
        # 将原始电压替换为 Delta V (或者放大后的 Delta V)
        model_inputs = raw_inputs.clone()
        model_inputs[:, :, :, 0] = delta_v * 50.0  # 放大特征，让 LSTM 看得更清楚
        # 归一化电流和温度
        model_inputs[:, :, :, 1] /= 20.0
        model_inputs[:, :, :, 2] = (model_inputs[:, :, :, 2] - 25.0) / 10.0

        # --- C. 前向传播 ---
        optimizer.zero_grad()
        actions = model(model_inputs)

        # --- 计算 Loss ---
        # 传入 Loss 的必须是最后一个时刻的真实电压差 (未放大的)
        last_step_delta = delta_v[:, -1, :]
        loss = criterion(actions, last_step_delta)

        # --- 反向传播 ---
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # --- F. 监控 ---
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {loss.item():.6f}")

            # with torch.no_grad():
            #     # raw_inputs 和 actions 都是当前 Batch 的
            #     # 注意：这里的 raw_inputs 还没有经过特征放大，如果用了增强需要小心
            #     # 但我们只要看 delta_v 和 actions 的关系
            #
            #     # 获取当前 Batch 的 Delta V (last step)
            #     cur_v = raw_inputs[:, -1, :, 0]
            #     cur_mean = torch.mean(cur_v, dim=1, keepdim=True)
            #     cur_delta = (cur_v - cur_mean).cpu().numpy().flatten()
            #     cur_act = actions.cpu().numpy().flatten()
            #
            #     plt.scatter(cur_delta, cur_act)
            #     plt.title(f"Training Batch Snapshot Epoch {epoch}")
            #     plt.show()  # 会弹窗，关闭后继续训练

    print("训练完成！")
    torch.save(model.state_dict(), "checkpoint.pth")
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

