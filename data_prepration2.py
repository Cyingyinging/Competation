import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(file_path, is_train=True):
    """
    加载数据并进行预处理，添加时间特征
    参数：
    file_path: 数据文件路径
    is_train: 是否为训练数据，如果为False则生成2020-2021年的时间特征
    """
    if is_train:
        # 加载训练数据
        data = pd.read_csv(file_path)
        data['TIME'] = pd.to_datetime(data['TIME'])
        data = data.sort_values(by='TIME')
    # 提取时间特征
    data['hour'] = data['TIME'].dt.hour
    data['day'] = data['TIME'].dt.day
    data['month'] = data['TIME'].dt.month
    data['dayofweek'] = data['TIME'].dt.dayofweek
    data['quarter'] = data['TIME'].dt.quarter
    data['year'] = data['TIME'].dt.year
    
    # 添加周期性特征
    # 小时特征 (24小时周期)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    
    # 日期特征 (31天周期)
    data['day_sin'] = np.sin(2 * np.pi * data['day']/31)
    data['day_cos'] = np.cos(2 * np.pi * data['day']/31)
    
    # 月份特征 (12月周期)
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    
    # 星期特征 (7天周期)
    data['weekday_sin'] = np.sin(2 * np.pi * data['dayofweek']/7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['dayofweek']/7)
    
    # 年份特征（相对2014年的年数）
    data['year_rel'] = (data['year'] - 2014) / 10.0  # 归一化年份特征
    
    # 特殊日期标记
    # 是否为周末
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    
    # 是否为工作时间（8-18点）
    data['is_worktime'] = ((data['hour'] >= 8) & (data['hour'] <= 18)).astype(int)
    
    # 季节特征
    data['season'] = data['month'].map({
        1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
        7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0
    })  # 0:冬季 1:春季 2:夏季 3:秋季
    data['season_sin'] = np.sin(2 * np.pi * data['season']/4)
    data['season_cos'] = np.cos(2 * np.pi * data['season']/4)
    
    # 处理缺失值
    if is_train:
        data['V'] = data['V'].ffill().bfill()
    
    # 设置时间索引
    data.set_index('TIME', inplace=True)
    
    # 选择特征列
    feature_columns = [
        'V',
        'hour_sin',
        'day_sin', 
        'month_sin', 
        'weekday_sin',
        'season_sin',
        'is_weekend',
        'is_worktime'
    ]
    
    return data[feature_columns]


def normalize_data(data):
    """归一化数据，对每个特征分别进行归一化"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset_multivariate(data, time_step=1, horizon=1):
    """创建多变量数据集"""
    X, y = [], []
    
    # 分离特征和目标变量
    features = data[:, 1:]  # 除了V之外的所有特征
    target = data[:, 0]     # 只取V值
    
    if horizon == 1:
        for i in range(len(data) - time_step):
            X.append(features[i:(i + time_step)])    # 只包含时间特征
            y.append(target[i + time_step])          # 只取V值
    else:
        for i in range(len(data) - time_step - horizon + 1):
            X.append(features[i:(i + time_step)])
            y.append(target[(i + time_step):(i + time_step + horizon)])
    
    return np.array(X), np.array(y)

def prepare_dataloaders(X, y, batch_size=32):
    """准备训练和验证 DataLoader"""
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,drop_last=True)
    return train_loader, val_loader
