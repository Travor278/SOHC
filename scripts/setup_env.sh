#!/usr/bin/env bash
# Linux/macOS 环境搭建脚本
# 用法：在仓库根目录执行 bash scripts/setup_env.sh
# 推荐 Python 3.10 或 3.11

set -e

echo "[1/4] 创建 venv (.venv_craic)..."
if [ ! -d ".venv_craic" ]; then
    python3 -m venv .venv_craic
else
    echo "  .venv_craic 已存在，跳过"
fi

echo "[2/4] 激活 venv..."
source .venv_craic/bin/activate

echo "[3/4] 升级 pip..."
python -m pip install --upgrade pip

echo "[4/4] 安装依赖（首次约 10-30 分钟）..."
pip install -r requirements.txt

echo ""
echo "完成。下一步："
echo "  1. git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW"
echo "  2. git clone https://github.com/microsoft/BatteryML.git external/BatteryML"
echo "  3. 按 data/README.md 下载数据集"
echo "  4. 按 TODO.md W1 节执行任务"
