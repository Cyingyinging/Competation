import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# 读取数据
data = pd.read_csv('./train/A-入库流量(2014-2019).csv')
# 将时间列转换为datetime格式
data['TIME'] = pd.to_datetime(data['TIME'])
# 按时间排序
data = data.sort_values(by='TIME')
# 选择需要的列
data = data[['TIME', 'V']]
# 处理缺失值
data['V'] = data['V'].ffill()
# 将数据设置为时间索引
data.set_index('TIME', inplace=True)

from tsai.all import *
from tsai.basics import *

# 切分数据集
window_len = 30
horizon = 1
x_vars = None
y_vars = None
X, y = apply_sliding_window(data, window_len, horizon=horizon, x_vars=x_vars, y_vars=y_vars)
print(X.shape,y.shape)
# 检查X和y的类型
print("X的类型:", type(X))
print("y的类型:", type(y))

splits = TimeSplitter(valid_size=0.2,show_plot=False)(y)

# dsets = TSDatasets(X, y, splits=splits)
batch_tfms=[TSNormalize(by_sample=True)]
# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, 
#                                 bs=128, num_workers=0, batch_tfms=batch_tfms)
learn = TSRegressor(X, y, splits=splits, batch_tfms=batch_tfms, 
                    arch='MLP', metrics=mae, bs=128, train_metrics=True, device='cuda')
learn.fit(30, 1e-4)