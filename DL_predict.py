# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 11:19:19 2025

@author: ZLi27
"""
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,TimeSeriesSplit
import torch
from tsai.all import *

# 读取数据
rainfall_data = pd.read_csv('./train./A-雨量水位(2014-2019).csv')
flow_data = pd.read_csv('./train/A-入库流量(2014-2019).csv')

def add_time_features(data):
    """添加时间特征"""
    df_features = data.copy()
    df_features['TIME'] = pd.to_datetime(df_features['TIME'])

    # 时间特征
    df_features['day_of_year'] = df_features['TIME'].dt.dayofyear
    df_features['hour'] = df_features['TIME'].dt.hour
    df_features['day_of_week'] = df_features['TIME'].dt.dayofweek
    df_features['month'] = df_features['TIME'].dt.month

    # 正弦/余弦时间编码
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

    # 周末/夜晚标记
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 5)).astype(int)
    
    # 季节特征
    seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df_features['season'] = df_features['month'].map(seasons)
    
    # 季节的周期性编码
    df_features['season_sin'] = np.sin(2 * np.pi * df_features['season'] / 4)
    df_features['season_cos'] = np.cos(2 * np.pi * df_features['season'] / 4)
    
    # 创建一个字典来存储新的列
    new_columns = {}

    # 添加滞后特征
    # 自定义滞后步数，使用间隔为 2 生成滞后步数
    lags = range(1, 168, 4)  # 从 1 到 29，步长为 2，生成 1, 3, 5, ..., 29
    for lag in lags:
        new_columns[f'lag_{lag}d_V_sum'] = df_features['V_sum'].shift(12 * lag)
        new_columns[f'lag_{lag}d_V_sum_2'] = df_features['V_sum'].shift(24 * lag)
        # new_columns[f'lag_{lag}d_V_sum_3'] = df_features['V_sum'].shift(7 * lag)
        # df_features[f'lag_{lag}d_V_max'] = df_features['V_max'].shift(24 * lag)
        # df_features[f'lag_{lag}d_V_min'] = df_features['V_mean'].shift(24 * lag)

    # 计算差分特征
    # diffs = range(1, 30, 2) 
    # for diff in diffs:
    #     new_columns[f'diff_{diff}d_V_sum'] = df_features['V_sum'].diff(12 * diff) 
    #     new_columns[f'diff_{diff}d_V_sum_2'] = df_features['V_sum'].diff(24 * diff) 
    # 计算滚动均值去趋势
    # new_columns['trend_12h'] = df_features['V_sum'].rolling(window=12, min_periods=1).mean()
    # new_columns['trend_24h'] = df_features['V_sum'].rolling(window=24, min_periods=1).mean()
    
    # df_features['detrended_12h'] = df_features['V_sum'] - df_features['trend_12h']
    # df_features['detrended_24h'] = df_features['V_sum'] - df_features['trend_24h']
    # 填充缺失值
    # 使用 pd.concat 批量将新列添加到原 DataFrame 中
    df_features = pd.concat([df_features, pd.DataFrame(new_columns)], axis=1)   
    df_features = df_features.ffill()

    return df_features.drop(columns=['hour', 'day_of_week', 'month', 'season'])  # 删除不必要的原始时间列

def preprocess_data(rainfall_data, flow_data):
    """合并并预处理数据"""
    rainfall_data['TIME'] = pd.to_datetime(rainfall_data['TIME'])
    flow_data['TIME'] = pd.to_datetime(flow_data['TIME'])
    flow_data = flow_data[['TIME', 'V']]
    
    # 清理无效数据
    rainfall_data.replace(-10000.0, np.nan, inplace=True)
    rainfall_data = rainfall_data[['TIME', 'V', 'AVGV', 'MAXV', 'MINV']]

    # 计算统计特征
    aggregated_rainfall_data = rainfall_data.groupby('TIME').agg({
        'V': ['sum', 'mean', 'max', 'min']
        # 'AVGV': ['sum', 'mean', 'max', 'min'],
    }).reset_index()
    # 重新命名列
    aggregated_rainfall_data.columns = ['TIME', 'V_sum', 'V_mean', 'V_max', 'V_min',
                                        # 'AVGV_sum', 'AVGV_mean', 'AVGV_max', 'AVGV_min',
                                        ]

    # 添加时间特征
    aggregated_rainfall_data = add_time_features(aggregated_rainfall_data)

    # 合并流量数据
    data = pd.merge(aggregated_rainfall_data, flow_data, on='TIME', suffixes=('_rain', '_flow'))

    # 填充缺失值
    data.ffill(inplace=True)

    return data

# 预处理数据
data = preprocess_data(rainfall_data, flow_data)

# 自动获取所有特征列
feature_names = [col for col in data.columns if col not in ['TIME', 'V']]
features = data[feature_names]
labels = data['V']

# 标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

def create_sequences(data, labels, time_step, horizon=1):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(len(data) - time_step+ 1):
        X.append(data[i:i+time_step])
        # 只预测V列，其他特征列仅作为输入
        y.append(labels[i:i+time_step])
    return np.array(X), np.array(y)

X,y = create_sequences(features, labels, 96, 7)
# 按时间顺序进行前后 80/20 分割
split_point = int(len(features) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"训练集: {X_train.shape, y_train.shape}, 验证集: {X_test.shape, y_test.shape}")
    
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 500

c_in, c_out = 97, 96

model = LSTM(c_in, c_out, hidden_size=128, n_layers=3).to(device)

from custom_train import * 
model_dir = './checkpoint'
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
train_model(epochs, train_loader, val_loader, device, model, criterion, optimizer, scheduler=scheduler, output_file=model_dir)
# 预测未来9-24小时入库流量
# def predict_future_flow(model, recent_data):
#     recent_features = scaler.transform(recent_data[feature_names])
#     future_flow = model.predict(recent_features)
#     return future_flow

# # 读取测试集数据
# test_data = pd.read_csv('./test/A-雨量水位(2020-2021).csv',low_memory=False)
# test_data['TIME'] = pd.to_datetime(test_data['TIME'])
# test_data = test_data[test_data['TIME'].dt.minute == 0]
# test_data.replace(-10000.0, np.nan, inplace=True)
# test_data = test_data[['TIME','V', 'AVGV', 'MAXV', 'MINV']]
# # test_data = test_data[test_data['TIME'].dt.minute == 0]

# # 对测试集数据按时间进行聚合（例如求和）
# aggregated_test_data = test_data.groupby('TIME').agg({
#     'V': ['sum','mean','max','min'],  # 总雨量
#     # 'AVGV': ['sum','mean','max','min'],  # 平均雨量
# }).reset_index()

# aggregated_test_data.columns = ['TIME', 'V_sum', 'V_mean', 'V_max','V_min',]
#                                 # 'AVGV_sum','AVGV_mean', 'AVGV_max', 'AVGV_min']
# aggregated_test_data.ffill()
# # 添加前一天同一时间点的值
# aggregated_test_data = add_time_features(aggregated_test_data)

# # 进行预测
# future_flow_predictions = predict_future_flow(model, aggregated_test_data)

# # 保存预测结果
# submission = pd.DataFrame({
#     'TIME': aggregated_test_data['TIME'],
#     'V': future_flow_predictions
# })

# # 保存预测结果
# submission.to_csv('submission.csv', index=False)