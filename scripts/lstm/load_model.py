import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as Data
from torchvision import transforms, datasets
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

# 定义一个类
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
        super(Net,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.fc=nn.Linear(hidden_size,output_size) # 全连接层

    def forward(self,x):
        # batch_size, seq_len = x.size()[0], x.size()[1]    # x.shape=(604,3,3)
        # h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).cuda()
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).cuda()
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0)) # output(5, 30, 64)
        pred = self.fc(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


def load_model(model_path):
    # 参数设置
    seq_length=168 # 时间步长
    input_size=1
    num_layers=6
    hidden_size=12
    batch_size=1
    n_iters=500
    lr=0.001
    output_size=24
    split_ratio=0.9

    moudle=Net(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    moudle.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return moudle

# model = load_model('swz_model.pth')
# print(model)