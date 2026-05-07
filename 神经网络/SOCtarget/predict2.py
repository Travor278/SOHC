import matplotlib.pyplot as plt
import numpy as np
import torch
from train import get_batch_data,preprocess_enhanced_input
from model import AdjacentActiveBalancingNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AdjacentActiveBalancingNet().to(device)
model.load_state_dict(torch.load("traincheckpoint.pth", map_location=device))


def verify_logic_fixed(model, device):
    model.eval()

    # 1. 生成测试数据
    # 制造一个稍微大一点的差异，方便观察
    raw_inputs = get_batch_data(batch_size=50, time_steps=50, num_packs=8, device=device)

    # 2. 使用统一的预处理 (关键步骤)
    model_inputs, real_delta_soc = preprocess_enhanced_input(raw_inputs, device)

    # 3. 推理
    with torch.no_grad():
        actions = model(model_inputs)  # [50, 12]

    # --- 诊断打印 (Debug Prints) ---
    print("\n--- 诊断信息 (Diagnostic Info) ---")

    # 检查输入特征是否足够大
    input_soc_feat = model_inputs[:, -1, :, 0].cpu().numpy()
    print(f"1. 模型输入的SOC特征范围: [{input_soc_feat.min():.4f}, {input_soc_feat.max():.4f}]")

    # 检查输出动作是否足够大
    act_np = actions.cpu().numpy()
    print(f"2. 模型输出的动作权重范围: [{act_np.min():.4f}, {act_np.max():.4f}]")
    if np.abs(act_np.max()) < 1e-3:
        print("   ⚠️ 警告: 模型输出了全 0 (Dead Model)。")
        print("   建议: 1. 调大训练时的 Loss 权重; 2. 调大 Preprocess 里的放大倍数 (*100); 3. 调大 LR。")
        return  # 不需要画图了，画出来也是一条线

    # 检查相关性
    # 只取最后一个时刻的数据来画图
    flat_delta_v = real_delta_soc[:, -1, :].cpu().numpy().flatten()
    flat_actions = act_np.flatten()

    correlation = np.corrcoef(flat_delta_v, flat_actions)[0, 1]
    print(f"3. soc差与动作的相关系数: {correlation:.4f}")
    if correlation > 0:
        print("   ❌ 错误: 正相关！soc高还在充电。请检查 Loss 函数里的 Target 符号 (-tanh)。")
    elif correlation < -0.8:
        print("   ✅ 通过: 强负相关，逻辑正确。")

    # --- 画图 ---
    plt.figure(figsize=(10, 6))
    plt.scatter(flat_delta_v, flat_actions, alpha=0.5, c='blue', s=20, label='Model Output')

    # 画出理想曲线 (Target) 供对比
    x_ref = np.linspace(flat_delta_v.min(), flat_delta_v.max(), 100)
    y_ref = -np.tanh(x_ref * 10.0)  # 这里的系数要跟训练 Target 一致
    # 注意：如果你的 Target 是 soft 的，画出来可能是曲线；如果是硬的，可能是台阶

    plt.plot(x_ref, y_ref, color='red', linestyle='--', linewidth=2, label='Ideal Target Rule')

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel(f"Real Delta SOC\n(Range: {flat_delta_v.min():.2f} ~ {flat_delta_v.max():.2f})")
    plt.ylabel("Model Action Weight")
    plt.title(f"Logic Verification (Corr: {correlation:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 调用
# verify_logic_fixed(model, torch.device('cpu'))
if __name__ == '__main__':
    verify_logic_fixed(model, device)