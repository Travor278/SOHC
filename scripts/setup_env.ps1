# Windows PowerShell 环境搭建脚本
# 用法：在仓库根目录执行 .\scripts\setup_env.ps1
# 推荐 Python 3.10 或 3.11

$ErrorActionPreference = "Stop"

Write-Host "[1/4] 创建 venv (.venv_craic)..." -ForegroundColor Cyan
if (-Not (Test-Path .venv_craic)) {
    python -m venv .venv_craic
} else {
    Write-Host "  .venv_craic 已存在，跳过" -ForegroundColor Yellow
}

Write-Host "[2/4] 激活 venv..." -ForegroundColor Cyan
& .\.venv_craic\Scripts\Activate.ps1

Write-Host "[3/4] 升级 pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "[4/4] 安装依赖（首次约 10-30 分钟）..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "完成。下一步：" -ForegroundColor Green
Write-Host "  1. git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW"
Write-Host "  2. git clone https://github.com/microsoft/BatteryML.git external/BatteryML"
Write-Host "  3. 按 data/README.md 下载数据集"
Write-Host "  4. 按 TODO.md W1 节执行任务"
