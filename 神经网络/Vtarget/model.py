import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class AdjacentActiveBalancingNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_packs=12, neighbor_radius=1):
        super(AdjacentActiveBalancingNet, self).__init__()

        self.num_packs = num_packs

        # 1. 预先生成拓扑掩码 (Mask)
        # 这是一个 [NumPacks, NumPacks] 的矩阵
        # 对角线及相邻 radius 内为 0 (可见)，其余为 -inf (不可见/被屏蔽)
        self.register_buffer('attn_mask', self._generate_adjacency_mask(num_packs, radius=neighbor_radius))

        # 2. LSTM (处理单体时序特征)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

        # 3. Masked Attention (局部交互)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 4. 决策层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # 输出范围 [-1, 1], 负=放电, 正=充电
        )

    def _generate_adjacency_mask(self, n, radius):
        """生成带状矩阵掩码，模拟相邻连接拓扑"""
        mask = torch.ones(n, n) * float('-inf')  # 初始化全不可见
        for i in range(n):
            # 允许看到自己和前后 radius 个邻居
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
            mask[i, start:end] = 0.0  # 0 表示允许关注
        return mask

    def forward(self, x):
        # x: [Batch, TimeSteps, NumPacks, Features]
        B, T, N, F = x.shape

        # --- LSTM ---
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, F)
        lstm_out, (h_n, c_n) = self.lstm(x_reshaped)
        pack_features = h_n[-1, :, :].view(B, N, -1)  # [B, N, Hidden]

        # --- Masked Attention ---
        # 传入 attn_mask，限制只在相邻电池间交换信息
        # 注意：attn_mask 不需要 batch 维度，PyTorch 会自动广播
        attn_output, _ = self.attention(
            query=pack_features,
            key=pack_features,
            value=pack_features,
            attn_mask=self.attn_mask
        )

        pack_features = pack_features + attn_output

        # --- Output ---
        flat_features = pack_features.view(B * N, -1)
        actions = self.fc_layers(flat_features).view(B, N)  # [B, N]

        # 主动均衡约束：去中心化 (Zero-Mean)，保证能量闭环
        # 在相邻均衡中，虽然能量是局部传递，但总体上也不能凭空产生能量
        actions = actions - torch.mean(actions, dim=1, keepdim=True)

        return actions