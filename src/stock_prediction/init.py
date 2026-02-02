import multiprocessing
import os
import queue
import torch
import threading
import copy
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import glob
import numpy as np
import dill
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

TRAIN_WEIGHT=0.9
SEQ_LEN=5  # 5
LEARNING_RATE=0.00001   # 0.00001
WEIGHT_DECAY=0.05   # 0.05
BATCH_SIZE=8    # for 8 GPUs
EPOCH=10
SAVE_NUM_ITER=10
SAVE_NUM_EPOCH=1
# GET_DATA=True
TEST_NUM=25
SAVE_INTERVAL=300
TEST_INTERVAL=-1  # 100
OUTPUT_DIMENSION=4
INPUT_DIMENSION=30   ## max input dimension
TQDM_NCOLS = 100
NUM_WORKERS = 1
PKL = True
BUFFER_SIZE = 10
D_MODEL = 512
NHEAD = 8
WARMUP_STEPS = 60000
symbol = 'Generic.Data'
# symbol = '000001.SZ'

loss_list=[]
data_list=[]
mean_list=[]
std_list=[]
test_mean_list = []
test_std_list = []
# 用于模型保存的稳定副本（不会被 clear）
saved_mean_list = []
saved_std_list = []
safe_save = False
# data_queue=multiprocessing.Queue()
data_queue=queue.Queue()
test_queue=queue.Queue()
stock_data_queue=queue.Queue()
stock_list_queue = queue.Queue()
csv_queue=queue.Queue()
df_queue=queue.Queue()

NoneDataFrame = pd.DataFrame(columns=["ts_code"])
NoneDataFrame["ts_code"] = ["None"]

## akshare data list (唯一数据源)
name_list = ["open","close","high","low","vol","amount","amplitude","pct_change","change","exchange_rate"]
use_list = [1,1,1,1,0,0,0,0,0,0]
show_list = [1,1,1,1,0,0,0,0,0,0]

OUTPUT_DIMENSION = sum(use_list)
# INPUT_DIMENSION = 20+OUTPUT_DIMENSION

assert OUTPUT_DIMENSION > 0

# 缓存按股票维度的归一化统计
symbol_norm_map = {}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Import configuration module
from .config import (
    config, root_path, train_path, test_path, train_pkl_path, png_path,
    daily_path, handle_path, pkl_path, data_path,
    lstm_path, transformer_path, cnnlstm_path
)

cnname = ""
for item in symbol.split("."):
    cnname += item
save_path = config.get_model_path("LSTM", symbol)
