# SOH Mamba-Head Ablation

更新日期：2026-05-08

## 目的

补 W4 剩余创新点：在 BatteryML-compatible SOH 流程中加入 W2 Mamba world-model feature/head，对比容量比 oracle、物理统计 Ridge baseline 和 Mamba embedding Ridge head。

## 方法

- 数据：`outputs/world_model_train_data.pt`
- 目标：`SOH = capacity / fresh_capacity`
- 划分：`B0005/B0006/B0007 -> B0018`
- Mamba 模型：`outputs/world_model.pt`
- 防止泄露：提取 Mamba embedding 前将输入张量中的 SOH 通道置为 `1.0`
- head：`StandardScaler + Ridge`

运行命令：

```bash
python -m craic_pipeline.soh_mamba_head \
  --dataset outputs/world_model_train_data.pt \
  --world-model outputs/world_model.pt \
  --soh-baseline outputs/soh_baseline.pt \
  --out outputs/soh_mamba_head.pt \
  --device cuda \
  --batch-size 512
```

## 结果

| 方法 | 输入特征 | 验证划分 | Val RMSE | Val MAE | 说明 |
|---|---|---|---:|---:|---|
| Capacity-ratio oracle | NASA `Capacity` ratio | B0050-B0056 | `~0%` | 未记录 | 软标签 oracle，不代表部署能力 |
| Physical stats Ridge | SOC/V/I/T/action 统计量，SOH 通道置中 | B0018 | `3.18%` | `1.71%` | 无 Mamba 表征 |
| Mamba embedding Ridge | W2 frozen Mamba embedding，SOH 通道置中 | B0018 | `2.87%` | `1.18%` | 相比物理统计 baseline 有增益 |

## 结论

Mamba embedding head 没有达到 `<2% RMSE`，但在不直接读取 SOH 通道的条件下优于物理统计 Ridge baseline，可作为“世界模型表征对 SOH 估计有帮助”的架构创新证据。正式论文中应把 capacity-ratio oracle 与 Mamba-head ablation 分开表述，避免把 oracle 指标误写成部署效果。
