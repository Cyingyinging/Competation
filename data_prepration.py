import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

def add_time_features(data):
    """添加时间特征"""
    # 创建新的DataFrame保留原始索引
    df_features = pd.DataFrame(index=data.index)
    df_features['V'] = data['V']  # 保留原始流量值
    
    # 基本时间特征
    df_features['hour'] = data.index.hour
    df_features['day'] = data.index.day
    df_features['day_of_week'] = data.index.dayofweek
    df_features['month'] = data.index.month
    df_features['year'] = data.index.year
    df_features['day_of_year'] = data.index.dayofyear
    
    # 周期性编码（使用正弦和余弦函数捕捉周期性）
    # 小时特征 (24小时周期)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # 星期特征 (7天周期)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # 月份特征 (12个月周期)
    df_features['month_sin'] = np.sin(2 * np.pi * (df_features['month'] - 1) / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * (df_features['month'] - 1) / 12)
    
    # 一年中的天特征 (365天周期)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * (df_features['day_of_year'] - 1) / 365)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * (df_features['day_of_year'] - 1) / 365)
    
    # 特殊时间标记
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 5)).astype(int)
    
    # 季节特征
    seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df_features['season'] = df_features['month'].map(seasons)
    
    # 季节的周期性编码
    df_features['season_sin'] = np.sin(2 * np.pi * df_features['season'] / 4)
    df_features['season_cos'] = np.cos(2 * np.pi * df_features['season'] / 4)
    
    # # 添加滞后特征（前24小时的平均值和标准差）
    # # 注意：这需要确保数据是按时间顺序排列的
    # df_features['lag_24h_mean'] = data['V'].rolling(window=24).mean().shift(1)
    # df_features['lag_24h_std'] = data['V'].rolling(window=24).std().shift(1)
    
    # # 添加前一周同一时间点的值（如果数据足够长）
    # df_features['lag_7d'] = data['V'].shift(24 * 7)
    
    # # 添加前一天同一时间点的值
    # df_features['lag_1d'] = data['V'].shift(24)
    
    # 填充NaN值
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    
    # 删除不需要的原始特征，只保留加工后的特征
    df_features = df_features.drop(['hour', 'day', 'day_of_week', 'month', 'year', 'day_of_year', 'season'], axis=1)

    
    return df_features

def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    data = pd.read_csv(file_path)
    data['TIME'] = pd.to_datetime(data['TIME'])
    data = data.sort_values(by='TIME').set_index('TIME')
    data = data[['V']]  # 单特征版本
    
    # 检查并打印异常值统计
    neg_count = (data['V'] < 0).sum()
    zero_count = (data['V'] == 0).sum()
    print(f"处理前 - 负值数量: {neg_count}, 零值数量: {zero_count}")
    
    # 处理负值和零值 - 使用前后值的平均值替换
    invalid_mask = (data['V'] <= 0)
    
    if invalid_mask.any():
        values = data['V'].values.copy()
        
        for i in range(len(values)):
            if invalid_mask.iloc[i]:
                # 查找前一个有效值
                prev_idx = i - 1
                while prev_idx >= 0 and invalid_mask.iloc[prev_idx]:
                    prev_idx -= 1
                
                # 查找后一个有效值
                next_idx = i + 1
                while next_idx < len(values) and invalid_mask.iloc[next_idx]:
                    next_idx += 1
                
                # 计算前后有效值的平均值
                if prev_idx >= 0 and next_idx < len(values):
                    values[i] = (values[prev_idx] + values[next_idx]) / 2
                elif prev_idx >= 0:
                    values[i] = values[prev_idx]
                elif next_idx < len(values):
                    values[i] = values[next_idx]
                else:
                    values[i] = 1.0
        
        data['V'] = values
    
    # 检查处理后的数据
    neg_count_after = (data['V'] < 0).sum()
    zero_count_after = (data['V'] == 0).sum()
    print(f"处理后 - 负值数量: {neg_count_after}, 零值数量: {zero_count_after}")
    
    # 添加时间特征
    print("添加时间特征...")
    data = add_time_features(data)
    print(f"添加特征后的数据形状: {data.shape}, 特征列: {data.columns.tolist()}")
    
    # 按年份划分数据
    train_val_data = data[data.index.year < 2019]
    test_data = data[data.index.year == 2019]
    
    return train_val_data, test_data

def create_sequences(data, time_step, horizon=1):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(len(data) - time_step - horizon + 1):
        X.append(data[i:i+time_step])
        # 只预测V列，其他特征列仅作为输入
        y.append(data[i+time_step:i+time_step+horizon, 0])
    return np.array(X), np.array(y)

def prepare_data(file_path, time_step, horizon, batch_size=32, val_ratio=0.2):
    """完整的数据准备流程"""
    # 1. 加载和预处理数据
    train_val_data, test_data = load_and_preprocess_data(file_path)
    
    # 2. 只对V列进行标准化
    # 提取V列
    train_val_V = train_val_data[['V']]
    test_V = test_data[['V']]
    
    # 对V列进行标准化
    scaler = StandardScaler().fit(train_val_V)
    train_val_data['V'] = scaler.transform(train_val_V)
    test_data['V'] = scaler.transform(test_V)
    
    # 转换为numpy数组
    train_val_scaled = train_val_data.values
    test_scaled = test_data.values
    
    # 3. 创建序列
    print("创建训练验证序列...")
    X_train_val, y_train_val = create_sequences(train_val_scaled, time_step, horizon)
    print(f"训练验证序列形状: X={X_train_val.shape}, y={y_train_val.shape}")
    
    print("创建测试序列...")
    X_test, y_test = create_sequences(test_scaled, time_step, horizon)
    print(f"测试序列形状: X={X_test.shape}, y={y_test.shape}")
    
    # 4. 随机划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42)
    
    print(f"训练集: {X_train.shape, y_train.shape}, 验证集: {X_val.shape, y_val.shape}, 测试集: {X_test.shape, y_test.shape}")
    
    # 5. 创建DataLoader
    # 调整张量形状为 (batch, features, seq_len)
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 6. 返回数据加载器、scaler和原始数据
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, scaler, {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'test_scaled': test_scaled,
        'feature_names': train_val_data.columns.tolist()  # 保存特征名称
    }

def save_preprocessed_data(raw_data, save_dir="preprocessed"):
    """保存预处理数据"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # 保存数据
    for split in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        np.save(f"{save_dir}/{split}.npy", raw_data[split])
    
    # 保存特征名称
    if 'feature_names' in raw_data:
        with open(f"{save_dir}/feature_names.txt", 'w') as f:
            f.write('\n'.join(raw_data['feature_names']))
    
    print(f"预处理数据已保存到 {save_dir} 目录")

# 使用示例
if __name__ == "__main__":
    # 参数设置
    TIME_STEP = 96  # 使用96小时作为历史窗口
    HORIZON = 16    # 预测未来16小时
    BATCH_SIZE = 64
    
    # 完整数据准备流程
    dataloaders, scaler, raw_data = prepare_data(
        file_path="./train/A-入库流量(2014-2019).csv",
        time_step=TIME_STEP,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        val_ratio=0.2
    )
    
    # 保存预处理数据
    save_preprocessed_data(raw_data)
    
    # 保存scaler
    import joblib
    joblib.dump(scaler, "preprocessed/scaler.joblib")
    print()
    # 打印数据集信息
    print(f"训练集批次: {len(dataloaders['train'])}")
    print(f"验证集批次: {len(dataloaders['val'])}")
    print(f"测试集批次: {len(dataloaders['test'])}")
    print(f"标准化参数：{scaler.mean_}")
    # 检查一个批次的形状
    for x, y in dataloaders['train']:
        print(f"输入批次形状: {x.shape}")  # 应该是 [batch_size, features, time_step]
        print(f"输出批次形状: {y.shape}")  # 应该是 [batch_size, horizon]
        break