
import pytest
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

def create_dummy_data(daily_path: Path):
    """创建模拟的股票日线数据"""
    daily_path.mkdir(parents=True, exist_ok=True)
    
    # 模拟两只股票的数据，足够长以满足序列长度要求
    dates = pd.date_range(start="20240101", periods=100)
    for code in ["000001", "000002"]:
        df = pd.DataFrame({
            "ts_code": [f"{code}"] * len(dates),
            "trade_date": dates.strftime("%Y%m%d"),
            "open": np.random.uniform(10, 20, len(dates)),
            "high": np.random.uniform(20, 30, len(dates)),
            "low": np.random.uniform(5, 10, len(dates)),
            "close": np.random.uniform(10, 20, len(dates)),
            "change": np.random.uniform(-1, 1, len(dates)),
            "pct_change": np.random.uniform(-0.1, 0.1, len(dates)),
            "vol": np.random.uniform(1000, 5000, len(dates)),
            "amount": np.random.uniform(10000, 50000, len(dates))
        })
        # 写入 CSV
        df.to_csv(daily_path / f"{code}.csv", index=False)

def run_script(script_name: str, args: list, env: dict):
    """运行 scripts 目录下的脚本"""
    script_path = ROOT_DIR / "scripts" / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    cwd = env.get("STOCK_PREDICTION_ROOT", str(ROOT_DIR))
    
    # 打印正在运行的命令，方便调试
    print(f"\n[RUN] {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    
    # Always print output for debugging
    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")
    
    return result


@pytest.fixture
def pipeline_env(tmp_path):
    """配置隔离的运行环境"""
    # 准备目录结构
    stock_daily = tmp_path / "stock_daily"
    pkl_handle = tmp_path / "pkl_handle"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"
    png_dir = tmp_path / "png"
    
    for d in [stock_daily, pkl_handle, models_dir, output_dir, png_dir]:
        d.mkdir(parents=True)
        
    # 创建模拟数据
    create_dummy_data(stock_daily)
    
    # 设置环境变量
    env = os.environ.copy()
    env["STOCK_PREDICTION_ROOT"] = str(tmp_path)
    env["STOCK_DAILY_PATH"] = str(stock_daily)
    env["TRAIN_PKL_PATH"] = str(pkl_handle / "train.pkl")
    env["MODEL_PATH"] = str(models_dir)
    env["PNG_PATH"] = str(png_dir)
    env["PYTHONPATH"] = str(ROOT_DIR / "src") # 确保能导入 src
    env["MPLBACKEND"] = "Agg" # 禁止 GUI 弹窗
    env["OMP_NUM_THREADS"] = "1"
    env["NUM_WORKERS"] = "0" # 避免多进程死锁
    
    return env, tmp_path

def test_full_pipeline_execution(pipeline_env):
    """测试完整流程：预处理 -> 训练 -> 预测"""
    env, tmp_path = pipeline_env
    
    # 1. 运行数据预处理
    print("\n>>> Step 1: Run data_preprocess.py")
    res_prep = run_script("data_preprocess.py", [], env)
    assert res_prep.returncode == 0, "Data preprocessing failed"
    assert (tmp_path / "pkl_handle" / "train.pkl").exists(), "train.pkl not generated"
    
    # 2. 运行训练 (使用 hybrid 模型，少量 epoch)
    print("\n>>> Step 2: Run train.py")
    res_train = run_script("train.py", [
        "--mode", "train",
        "--model", "hybrid",
        "--epoch", "1",
        "--pkl", "1",
        "--cpu", "1" # 在测试环境强制使用 CPU 避免显存问题
    ], env)
    assert res_train.returncode == 0, "Training failed"
    
    # 检查模型文件是否生成 (Models/Generic/HYBRID/...)
    # 注意：Symbol 可能是 Generic，取决于 train.py 逻辑
    # 我们的 dummy 数据含有多个 code，默认行为是训练 Generic 模型
    model_file = list(tmp_path.rglob("*_Model.pkl"))
    assert len(model_file) > 0, "Model file not generated"
    
    # 3. 运行预测
    print("\n>>> Step 3: Run predict.py")
    res_pred = run_script("predict.py", [
        "--model", "hybrid",
        "--test_code", "000001",
        "--pkl", "1",
        "--cpu", "1"
    ], env)
    assert res_pred.returncode == 0, "Prediction failed"
    
    # 检查预测结果
    metrics_files = list((tmp_path / "output").glob("metrics_*.json"))
    assert len(metrics_files) > 0, "Metrics output not generated"

if __name__ == "__main__":
    # 允许直接运行此脚本进行测试
    pytest.main([__file__])
