import torch
import numpy as np
import pandas as pd

def evaluate_and_predict(model_path, model, test_loader, scaler, device, time_step, horizon, test_data, feature_names=None):
    """评估模型并进行预测，一次性预测horizon长度的序列"""
    # 加载模型
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 1. 首先评估测试集性能
    print("评估测试集性能...")
    test_rmse = 0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            # 保存预测值和真实值
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
            
            # 计算RMSE
            mse = torch.mean((outputs - batch_y) ** 2)
            test_rmse += torch.sqrt(mse).item()
    
    avg_test_rmse = test_rmse / len(test_loader)
    test_score = 1 / (1 + avg_test_rmse)
    
    # 转换回原始尺度 - 只对V列进行逆变换
    test_predictions = np.array(test_predictions).reshape(-1, 1)
    test_targets = np.array(test_targets).reshape(-1, 1)
    
    # 逆变换预测值和目标值
    test_predictions_original = scaler.inverse_transform(test_predictions)
    test_targets_original = scaler.inverse_transform(test_targets)
    
    original_rmse = np.sqrt(np.mean((test_predictions_original - test_targets_original) ** 2))
    original_score = 1 / (1 + original_rmse)
    
    print(f"\n测试集评估结果:")
    print(f"标准化数据 - RMSE: {avg_test_rmse:.6f}, Score: {test_score:.6f}")
    print(f"原始数据 - RMSE: {original_rmse:.6f}, Score: {original_score:.6f}")
    
    # 2. 预测2020-2021年数据
    print("\n开始预测2020-2021年数据...")
    
    # 获取特征数量
    n_features = test_data.shape[1]
    
    # 使用测试集最后time_step个时间点的所有特征作为初始序列
    last_data_points = test_data[-time_step:]
    
    # 创建初始序列张量，保持所有特征
    last_sequence = last_data_points.reshape(1, time_step, n_features)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).permute(0, 2, 1).to(device)
    
    # 生成预测时间索引
    start_date = '2020-01-01 00:00:00'
    end_date = '2021-12-31 23:00:00'
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    n_predictions = len(future_dates)
    
    # 生成未来时间点的时间特征
    future_time_features = generate_future_time_features(future_dates, feature_names)
    
    # 生成预测
    predictions = []
    current_sequence = last_sequence_tensor
    
    with torch.no_grad():
        for i in range(0, n_predictions, horizon):
            # 预测下一个horizon长度的序列
            output = model(current_sequence)
            pred_values = output.cpu().numpy()[0]  # 获取完整的horizon序列
            predictions.extend(pred_values)
            
            # 更新序列 - 滑动窗口并添加时间特征
            new_seq = current_sequence.clone()
            
            # 滑动V值（第一个特征）
            new_seq[0, 0, :-horizon] = current_sequence[0, 0, horizon:].clone()
            new_seq[0, 0, -horizon:] = torch.tensor(pred_values, dtype=torch.float32).to(device)
            
            # 更新时间特征（其他特征）
            for f in range(1, n_features):
                new_seq[0, f, :-horizon] = current_sequence[0, f, horizon:].clone()
                # 获取未来horizon个时间点的特征
                next_time_features = []
                for h in range(horizon):
                    if i + h < len(future_time_features):
                        next_time_features.append(future_time_features[i + h, f-1])
                    else:
                        # 如果超出范围，使用最后一个有效值
                        next_time_features.append(future_time_features[-1, f-1])
                
                new_seq[0, f, -horizon:] = torch.tensor(next_time_features, dtype=torch.float32).to(device)
            
            current_sequence = new_seq
            
            # 打印进度
            if i % (horizon * 10) == 0:
                print(f"已预测 {min(i+horizon, n_predictions)}/{n_predictions} 步")
    
    # 截取所需的预测长度
    predictions = predictions[:n_predictions]
    
    # 转换回原始尺度
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_array)
    
    # 创建结果DataFrame
    future_df = pd.DataFrame({
        'TIME': future_dates[:len(predictions_original)].strftime('%Y-%m-%d %H:%M:%S'),
        'V': predictions_original.flatten()
    })
    
    # 保存预测结果
    future_df.to_csv('submission.csv', index=False)
    print("预测结果已保存到submission.csv")
    
    return future_df

def generate_future_time_features(future_dates, feature_names=None):
    """为未来时间点生成时间特征"""
    # 创建DataFrame
    df_features = pd.DataFrame(index=future_dates)
    
    # 基本时间特征
    df_features['hour'] = df_features.index.hour
    df_features['day'] = df_features.index.day
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    df_features['day_of_year'] = df_features.index.dayofyear
    
    # 周期性编码
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    df_features['month_sin'] = np.sin(2 * np.pi * (df_features['month'] - 1) / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * (df_features['month'] - 1) / 12)
    
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
    
    # 删除中间特征
    df_features = df_features.drop(['hour', 'day', 'day_of_week', 'month', 'year', 'day_of_year', 'season'], axis=1)
    
    # 如果提供了特征名称，确保顺序一致
    if feature_names is not None:
        # 排除V列，因为V列是我们要预测的
        time_features = [f for f in feature_names if f != 'V']
        # 确保所有需要的特征都存在
        for feature in time_features:
            if feature not in df_features.columns:
                print(f"警告: 特征 {feature} 在生成的时间特征中不存在")
        
        # 按照原始特征顺序排列
        df_features = df_features[time_features]
    
    return df_features.values

def predict_future(model_path, model, scaler, device, time_step, horizon, test_data, feature_names):
    """简化版预测函数，只生成未来预测，不评估测试集"""
    # 加载模型
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 获取特征数量
    n_features = test_data.shape[1]
    
    # 使用测试集最后time_step个时间点的所有特征作为初始序列
    last_data_points = test_data[-time_step:]
    
    # 创建初始序列张量，保持所有特征
    last_sequence = last_data_points.reshape(1, time_step, n_features)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).permute(0, 2, 1).to(device)
    
    # 生成预测时间索引
    start_date = '2020-01-01 00:00:00'
    end_date = '2021-12-31 23:00:00'
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    n_predictions = len(future_dates)
    
    # 生成未来时间点的时间特征
    future_time_features = generate_future_time_features(future_dates, feature_names)
    
    # 生成预测
    predictions = []
    current_sequence = last_sequence_tensor
    
    with torch.no_grad():
        for i in range(0, n_predictions, horizon):
            # 预测下一个horizon长度的序列
            output = model(current_sequence)
            pred_values = output.cpu().numpy()[0]  # 获取完整的horizon序列
            predictions.extend(pred_values)
            
            # 更新序列 - 滑动窗口并添加时间特征
            new_seq = current_sequence.clone()
            
            # 滑动V值（第一个特征）
            new_seq[0, 0, :-horizon] = current_sequence[0, 0, horizon:].clone()
            new_seq[0, 0, -horizon:] = torch.tensor(pred_values, dtype=torch.float32).to(device)
            
            # 更新时间特征（其他特征）
            for f in range(1, n_features):
                new_seq[0, f, :-horizon] = current_sequence[0, f, horizon:].clone()
                # 获取未来horizon个时间点的特征
                next_time_features = []
                for h in range(horizon):
                    if i + h < len(future_time_features):
                        next_time_features.append(future_time_features[i + h, f-1])
                    else:
                        # 如果超出范围，使用最后一个有效值
                        next_time_features.append(future_time_features[-1, f-1])
                
                new_seq[0, f, -horizon:] = torch.tensor(next_time_features, dtype=torch.float32).to(device)
            
            current_sequence = new_seq
            
            # 打印进度
            if i % (horizon * 10) == 0:
                print(f"已预测 {min(i+horizon, n_predictions)}/{n_predictions} 步")
    
    # 截取所需的预测长度
    predictions = predictions[:n_predictions]
    
    # 转换回原始尺度
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_array)
    
    # 创建结果DataFrame
    future_df = pd.DataFrame({
        'TIME': future_dates[:len(predictions_original)].strftime('%Y-%m-%d %H:%M:%S'),
        'V': predictions_original.flatten()
    })
    
    # 保存预测结果
    future_df.to_csv('submission.csv', index=False)
    print("预测结果已保存到submission.csv")
    
    return future_df