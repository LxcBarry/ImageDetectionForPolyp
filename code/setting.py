input_path = "../input" #输入文件夹位置
output_path = "../output" #预测结果位置
pic_high = 512
pic_width = 512
input_size = (512,512)

import os
os.chdir(f"{os.getcwd()}/code") #使用vscode的时候需要取消注释
from torch.utils.data import DataLoader
import torch
from  torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from mail import send_to_me

# from inference_config import *
from predict_config import *
# from train_config import *
