import pandas as pd
import numpy as np
import torch
import os

class IncrementalCSVProcessor:
    def __init__(self, num_packs=8, window_size=50):
        self.num_packs = num_packs
        self.window_size = window_size

        # 预处理参数 (保持不变)
        self.GAIN_V = 50.0
        self.GAIN_SOC = 20.0
        self.NORM_I = 20.0
        self.NORM_T_MEAN = 25.0
        self.NORM_T_STD = 10.0

    def _process_single_csv(self, csv_path):
        """
        内部函数：处理单个 CSV 文件，返回对应的 Tensor
        """
        print(f"正在读取并处理: {csv_path} ...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            return None

        # 1. 提取列数据
        v_list, soc_list, i_list, t_list = [], [], [], []
        for k in range(1, self.num_packs + 1):
            # 兼容可能的列名缺失
            if f'V{k}' not in df.columns:
                print(f"❌ 错误: CSV 中缺少列 V{k}")
                return None
            v_list.append(df[f'V{k}'].values)
            soc_list.append(df[f'SOC{k}'].values)
            i_list.append(df[f'I{k}'].values)
            t_list.append(df[f'T{k}'].values)

        # 2. 堆叠成矩阵 [Time, Packs, 4]
        raw_data = np.stack([
            np.column_stack(v_list),
            np.column_stack(soc_list),
            np.column_stack(i_list),
            np.column_stack(t_list)
        ], axis=2)

        # 3. 特征工程 (预处理)
        tensor_data = torch.FloatTensor(raw_data)
        processed = tensor_data.clone()

        # Voltage -> Delta * 50
        v_mean = torch.mean(processed[:, :, 0], dim=1, keepdim=True)
        processed[:, :, 0] = (processed[:, :, 0] - v_mean) * self.GAIN_V

        # SOC -> Delta * 20
        soc_mean = torch.mean(processed[:, :, 1], dim=1, keepdim=True)
        processed[:, :, 1] = (processed[:, :, 1] - soc_mean) * self.GAIN_SOC

        # Current & Temp
        processed[:, :, 2] /= self.NORM_I
        processed[:, :, 3] = (processed[:, :, 3] - self.NORM_T_MEAN) / self.NORM_T_STD

        # 4. 滑动窗口切片
        total_time = processed.shape[0]
        num_samples = total_time - self.window_size + 1

        if num_samples <= 0:
            print(f"⚠️ 警告: 文件 {csv_path} 数据太少 ({total_time}行)，无法生成窗口，已跳过。")
            return None

        windows = []
        for i in range(num_samples):
            windows.append(processed[i: i + self.window_size])

        return torch.stack(windows)  # [Batch, 50, 8, 4]

    def add_csv_to_dataset(self, csv_path, dataset_save_path):
        """
        核心对外接口：处理 CSV 并追加到指定的数据集文件中
        """
        # 1. 处理当前 CSV
        new_tensor = self._process_single_csv(csv_path)
        if new_tensor is None:
            return  # 处理失败或数据为空

        new_count = new_tensor.shape[0]

        # 2. 检查数据集文件是否存在
        if os.path.exists(dataset_save_path):
            print(f"发现已有数据集: {dataset_save_path}，正在加载...")
            try:
                existing_tensor = torch.load(dataset_save_path)

                # 3. 维度检查 (防止把 12电池的数据加到 8电池的数据集里)
                if existing_tensor.shape[1:] != new_tensor.shape[1:]:
                    print(f"❌ 维度不匹配！无法合并。")
                    print(f"  已有数据形状: {existing_tensor.shape[1:]}")
                    print(f"  新数据形状:   {new_tensor.shape[1:]}")
                    return

                # 4. 拼接 (Concatenate)
                # dim=0 表示在 Batch 维度（样本数量）上增加
                combined_tensor = torch.cat([existing_tensor, new_tensor], dim=0)
                print(f"已追加数据。原有样本: {existing_tensor.shape[0]}, 新增: {new_count}")

            except Exception as e:
                print(f"❌ 加载旧数据集失败: {e}。将覆盖创建新文件。")
                combined_tensor = new_tensor
        else:
            print(f"数据集不存在，创建新文件: {dataset_save_path}")
            combined_tensor = new_tensor

        # 5. 保存
        torch.save(combined_tensor, dataset_save_path)
        print(f"✅ 保存成功！当前总样本数: {combined_tensor.shape[0]}")
        print("-" * 50)


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    DATASET_PATH = 'combined_training_data.pt'
    processor = IncrementalCSVProcessor(num_packs=8, window_size=50)

    # 假设你有一堆 CSV 文件
    csv_files = ['battery_data1.csv','battery_data.csv','battery_data2.csv','battery_data3.csv']
    # --- 批量处理并追加 ---

    # 如果你想从头开始，可以先手动删除旧的 .pt 文件
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
        print("已删除旧数据集，重新开始累积...")

    for csv_file in csv_files:
        processor.add_csv_to_dataset(csv_file, DATASET_PATH)

    # 最后验证一下
    if os.path.exists(DATASET_PATH):
        final_data = torch.load(DATASET_PATH)
        print(f"\n最终数据集形状: {final_data.shape}")
