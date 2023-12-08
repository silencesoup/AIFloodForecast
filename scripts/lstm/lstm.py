from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch
import numpy as np

def normalization_test(data):

    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    data=data.values    # 将pd的系列格式转换为np的数组格式
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    return data,mm_x

def split_data_test(x,split_ratio):

    train_size=int(len(x)*split_ratio)
    test_size=len(x)-train_size

    # x_data=Variable(torch.Tensor(np.array(x))).cuda()
    x_data=Variable(torch.Tensor(np.array(x)))
    x_train=Variable(torch.Tensor(np.array(x[0:train_size])))
    x_test=Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    return x_data,x_train,x_test

def split_windows_test(data,seq_length):

    x=[]
    i = 0
    # for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
    _x=data[i:(i+seq_length),:]
    x.append(_x)
    x=np.array(x)
    print('x.shape=\n',x.shape)
    return x