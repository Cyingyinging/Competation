# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:23:31 2025

@author: ZLi27
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tsai.all import *
    
def seed_everything(seed=666):
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# 读取数据
rainfall_data = pd.read_csv('./train/A-雨量水位(2014-2019).csv')
flow_data = pd.read_csv('./train/A-入库流量(2014-2019).csv')

feature_names = ['DAY_OF_YEAR','V_sum', 'V_mean', 'V_max','V_min',
                'AVGV_sum','AVGV_mean', 'AVGV_max', 'AVGV_min', 
                'MAXV_sum','MAXV_mean', 'MAXV_max', 'MAXV_min', 
                'MINV_sum', 'MINV_mean', 'MINV_max', 'MINV_min']

# 数据预处理
def preprocess_data(rainfall_data, flow_data):
    # 将时间列转换为日期时间格式
    rainfall_data['TIME'] = pd.to_datetime(rainfall_data['TIME'])
    rainfall_data = rainfall_data[rainfall_data['TIME'].dt.minute == 0].copy()
    flow_data['TIME'] = pd.to_datetime(flow_data['TIME'])    
    flow_data = flow_data[flow_data['TIME'].dt.minute == 0].copy()
    
    rainfall_data.replace(-10000.0, np.nan, inplace=True)
    rainfall_data = rainfall_data[['TIME','V', 'AVGV', 'MAXV', 'MINV']]
    # 对雨量数据按时间进行聚合（例如求和）
    aggregated_rainfall_data = rainfall_data.groupby('TIME').agg({
        'V': ['sum','mean','max','min'],  # 总雨量
        'AVGV': ['sum','mean','max','min'],  # 平均雨量
        'MAXV': ['sum','mean','max','min'],  # 最大雨量
        'MINV': ['sum','mean','max','min'],  # 最小雨量
    }).reset_index()
    
    aggregated_rainfall_data.columns = ['TIME', 'V_sum', 'V_mean', 'V_max','V_min',
                                    'AVGV_sum','AVGV_mean', 'AVGV_max', 'AVGV_min', 
                                    'MAXV_sum','MAXV_mean', 'MAXV_max', 'MAXV_min', 
                                    'MINV_sum', 'MINV_mean', 'MINV_max', 'MINV_min']

    # 合并数据集
    data = pd.merge(aggregated_rainfall_data, flow_data, on='TIME', suffixes=('_rain', '_flow'))
    # 填充缺失值
    data.fillna(method='ffill', inplace=True)
    
    return data

data = preprocess_data(rainfall_data, flow_data)

# 特征和标签
data['DAY_OF_YEAR'] = data['TIME'].dt.dayofyear
features = data[feature_names]
labels = data['V']

# 标准化特征
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 将数据组织成时间序列数据，以24小时为一个周期
def create_sequences(data, labels, seq_length):
    xs = []
    ys = []
    for i in range(0,len(data) - seq_length+1, 6):
        x = data[i:i+seq_length]
        y = labels[i:i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 72
X, y = create_sequences(features, labels, seq_length)
split_point = int(len(X) * 0.9)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0,2,1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print(X_train.shape)


# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型参数
input_size = X_train.shape[1]
hidden_size = 128
num_layers = 3
output_size = seq_length

# 初始化模型、损失函数和优化器
model = LSTM(input_size,output_size,hidden_size,n_layers=4).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=6e-5, verbose=True)

# 训练模型
num_epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_score=0
best_epoch=0

y_test = y_test.numpy().reshape(-1, seq_length)

# for epoch in range(num_epochs):
#     model.train()
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         # 前向传播
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # 打印每个 epoch 的损失
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#         print("best epoch: ", best_epoch)
#         print("best score: ", best_score)

#     # 评估模型
#     model.eval()
#     with torch.no_grad():
#         y_pred = []
#         for inputs, _ in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             y_pred.extend(outputs.cpu().numpy())
    
#         y_pred = np.array(y_pred).reshape(-1, seq_length)
        
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         score=1/(1+rmse)
#         if score>best_score:
#             best_epoch=epoch
#             torch.save(model.state_dict(), 'lstm_model.pth')
#             best_score=score
#             print("best score: ", score)
#         # print(f'Mean Squared Error: {rmse}')
#         # print(f'Score: {1/(1+rmse)}')
#     scheduler.step(rmse)
    
    
# 读取测试集数据
test_data = pd.read_csv('./test/A-雨量水位（2020-2021）.csv', low_memory=False)
test_data['TIME'] = pd.to_datetime(test_data['TIME'])
test_data = test_data[test_data['TIME'].dt.minute == 0].copy()

test_data.replace(-10000.0, np.nan, inplace=True)
test_data = test_data[['TIME', 'V', 'AVGV', 'MAXV', 'MINV']]

# 对测试集数据按时间进行聚合（例如求和）
aggregated_test_data = test_data.groupby('TIME').agg({
    'V': ['sum', 'mean', 'max', 'min'],  # 总雨量
    'AVGV': ['sum', 'mean', 'max', 'min'],  # 平均雨量
    'MAXV': ['sum', 'mean', 'max', 'min'],  # 最大雨量
    'MINV': ['sum', 'mean', 'max', 'min'],  # 最小雨量
}).reset_index()

aggregated_test_data.columns = ['TIME', 'V_sum', 'V_mean', 'V_max', 'V_min',
                                'AVGV_sum', 'AVGV_mean', 'AVGV_max', 'AVGV_min',
                                'MAXV_sum', 'MAXV_mean', 'MAXV_max', 'MAXV_min',
                                'MINV_sum', 'MINV_mean', 'MINV_max', 'MINV_min']
aggregated_test_data.fillna(method='ffill', inplace=True)
aggregated_test_data['DAY_OF_YEAR'] = aggregated_test_data['TIME'].dt.dayofyear

# 创建测试数据的序列
test_features = aggregated_test_data[feature_names]
test_features = scaler.transform(test_features)

def create_test_sequences(data, seq_length):
    xs = []
    for i in range(0, len(data) - seq_length + 1, seq_length):
        x = data[i:i + seq_length]
        xs.append(x)
    
    # Check if there's a remainder part that needs to be handled
    remainder = len(data) % seq_length
    if remainder != 0:
        last_seq = data[-seq_length:]
        xs.append(last_seq)
    
    return np.array(xs)

def predict_sequences(model, sequences, original_length, seq_length):
    with torch.no_grad():
        y_pred = []
        
        for i in range(X_test_seq.size(0)):
            input_seq = X_test_seq[i].unsqueeze(0).to(device)
            output = model(input_seq)
            y_pred.append(output.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        
        # 计算余数
        remainder = original_length % seq_length
        
        # 保留原始数据长度的预测值
        if remainder != 0:
            # 获取最后一个序列预测值
            last_seq_pred = y_pred[-1]
            # 只保留最后一个序列的 [seq_length-remainder:] 部分
            last_seq_pred = last_seq_pred[(seq_length-remainder):]
            y_pred = np.concatenate([y_pred[:-1].flatten(), last_seq_pred], axis=0)            
        else:
            y_pred = y_pred[:original_length]
            # 将预测结果连接成一个完整的序列
            y_pred = y_pred.flatten()
    
    return y_pred

# 创建测试数据的序列
X_test_seq = create_test_sequences(test_features, seq_length)

# 转换为 PyTorch 张量
X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).permute(0,2,1)

# 进行预测
model = LSTM(input_size,output_size,hidden_size,n_layers=4).cuda()
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()
with torch.no_grad():
    y_pred = []
    for i in range(X_test_seq.size(0)):
        input_seq = X_test_seq[i].unsqueeze(0).to(device)
        output = model(input_seq)
        y_pred.append(output.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    
y_pred = predict_sequences(model, X_test_seq, len(test_features), seq_length)

# 保存预测结果
future_times = aggregated_test_data['TIME']
submission = pd.DataFrame({
    'TIME': future_times,
    'V': y_pred
})

# 保存预测结果
submission.to_csv('submission.csv', index=False)