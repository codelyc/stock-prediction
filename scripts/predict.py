#!/usr/bin/env python
# coding: utf-8
"""
股票预测推理脚本
支持从命令行调用 inference
"""
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_prediction.predict import main

if __name__ == '__main__':
    main()
