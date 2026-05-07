import torch
import torch.nn as nn

# 1. 导入你的模型类 (确保类定义在当前文件中或能被导入)
# 假设你的模型类名是 ActiveBalancingNet
from model import AdjacentActiveBalancingNet


def export_to_onnx():
    # --- 配置参数 ---
    MODEL_PATH = "traincheckpoint.pth"  # 如果你有训练好的权重文件
    ONNX_PATH = "bms_balance_v1.onnx"

    INPUT_DIM = 4  # [Delta_V, Delta_SOC, I, T]
    NUM_PACKS = 8  # 硬件固定的电池串数
    TIME_STEPS = 50  # 模型的时间窗口
    HIDDEN_DIM = 64

    # --- 2. 初始化模型 ---
    device = torch.device("cpu")  # 导出通常在 CPU 上进行
    model = AdjacentActiveBalancingNet(input_dim=INPUT_DIM,
                               hidden_dim=HIDDEN_DIM,
                               num_packs=NUM_PACKS)

    # 如果有训练好的权重，加载它
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"成功加载权重: {MODEL_PATH}")
    except FileNotFoundError:
        print("未找到权重文件，使用随机初始化权重进行演示导出。")

    # !!! 关键步骤 !!!
    # 必须切换到 eval 模式，这会冻结 Dropout 和 BatchNorm，
    # 并确保 LSTM/Attention 的行为是确定性的。
    model.eval()

    # --- 3. 创建虚拟输入 (Dummy Input) ---
    # ONNX 需要通过运行一次模型来追踪计算图，所以需要一个形状正确的假数据
    # Shape: [Batch_Size, Time_Steps, Num_Packs, Features]
    # Batch_Size 设为 1 即可
    dummy_input = torch.randn(1, TIME_STEPS, NUM_PACKS, INPUT_DIM, device=device)

    # --- 4. 执行导出 ---
    print(f"正在导出模型到 {ONNX_PATH} ...")

    torch.onnx.export(
        model,  # 1. 要导出的模型
        dummy_input,  # 2. 虚拟输入
        ONNX_PATH,  # 3. 输出文件名
        export_params=True,  # 4. 是否将权重存储在模型文件中 (必须为True)
        opset_version=18,  # 5. ONNX 算子集版本 (11 或 12 是通用选择，支持 LSTM/Attention)
        do_constant_folding=True,  # 6. 优化常量折叠 (减小模型体积)
        input_names=['input'],  # 7. 输入节点的名称 (部署时会用到)
        output_names=['output'],  # 8. 输出节点的名称

        # 9. 动态轴 (Dynamic Axes) - 非常重要！
        # 我们允许 Batch_Size 是动态的 (可以是 1，也可以是 100)
        # 但 Time_Steps 和 Num_Packs 通常固定，因为嵌入式内存分配是静态的
        dynamic_axes={
            'input': {0: 'batch_size'},  # 第0维(Batch)可变
            'output': {0: 'batch_size'}
        }
    )

    print("导出完成！")


if __name__ == "__main__":
    export_to_onnx()