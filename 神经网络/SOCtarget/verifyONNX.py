
import onnxruntime as ort
import torch
import numpy as np
from model import AdjacentActiveBalancingNet

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def verify_model():
    # 1. 设置路径和参数
    ONNX_PATH = "bms_balance_v1.onnx"
    INPUT_DIM = 4
    NUM_PACKS = 8
    TIME_STEPS = 50

    # 2. 准备 PyTorch 模型 (作为基准)
    torch_model = AdjacentActiveBalancingNet(input_dim=INPUT_DIM, num_packs=NUM_PACKS)
    torch_model.load_state_dict(torch.load("traincheckpoint.pth"))
    torch_model.eval()

    # 3. 准备测试数据
    # 使用随机数据
    dummy_input = torch.randn(1, TIME_STEPS, NUM_PACKS, INPUT_DIM)

    # 4. 获取 PyTorch 输出
    with torch.no_grad():
        torch_out = torch_model(dummy_input)

    # 5. 获取 ONNX Runtime 输出
    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(ONNX_PATH)

    # 构造 ONNX 输入字典 {input_name: numpy_data}
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}

    # 推理
    ort_outs = ort_session.run(None, ort_inputs)

    # 6. 对比结果
    # ort_outs[0] 是第一个输出
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("✅ 验证成功！ONNX 模型输出与 PyTorch 模型一致。")
    print(f"PyTorch Output Shape: {torch_out.shape}")
    print(f"ONNX Output Shape:    {ort_outs[0].shape}")

if __name__ == "__main__":
    verify_model()