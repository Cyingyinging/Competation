import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tsai.all import *
from tsai.basics import *
from scipy.stats import zscore

def add_features(data,train=None):
    """添加特征"""
    # 确保时间是索引
    data = data.set_index("TIME")

    # 创建新的 DataFrame 保留原始索引
    df_features = pd.DataFrame(index=data.index)
    
    # df_features["TIME"] = data.index

    df_features["Total_V"] = data["Total_V"]  
    df_features["Mean_V"] = data["Mean_V"]  
    df_features["Max_V"] = data["Max_V"]
    df_features["Min_V"] = data["Min_V"] 
    
    # 提取时间特征
    df_features["hour"] = data.index.hour
    df_features["day"] = data.index.day
    df_features["day_of_week"] = data.index.dayofweek
    df_features["month"] = data.index.month
    df_features["year"] = data.index.year
    df_features["day_of_year"] = data.index.dayofyear

    # 周期性特征
    df_features["hour_sin"] = np.sin(2 * np.pi * df_features["hour"] / 24)
    df_features["hour_cos"] = np.cos(2 * np.pi * df_features["hour"] / 24)

    df_features["day_of_week_sin"] = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
    df_features["day_of_week_cos"] = np.cos(2 * np.pi * df_features["day_of_week"] / 7)

    df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
    df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)

    df_features["day_of_year_sin"] = np.sin(2 * np.pi * df_features["day_of_year"] / 365)
    df_features["day_of_year_cos"] = np.cos(2 * np.pi * df_features["day_of_year"] / 365)

    # 额外特征
    df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)
    df_features["is_night"] = ((df_features["hour"] >= 22) | (df_features["hour"] <= 5)).astype(int)

    # 季节特征
    seasons = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df_features["season"] = df_features["month"].map(seasons)
    df_features["season_sin"] = np.sin(2 * np.pi * df_features["season"] / 4)
    df_features["season_cos"] = np.cos(2 * np.pi * df_features["season"] / 4)

    # 填充缺失值
    df_features.fillna(method="bfill", inplace=True)
    df_features.fillna(method="ffill", inplace=True)
    
    # 删除不需要的原始特征，只保留加工后的特征
    df_features = df_features.drop(['hour', 'day', 'day_of_week', 'month', 'year', 'day_of_year', 'season'], axis=1)
    if train:
        df_features["target"] = data["target"]  # 目标值
        
    return df_features  # 保留时间列

def create_sequences(data,time_step, horizon=1):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(len(data) - time_step - horizon + 1):
        X.append(data[i:i+time_step])  # 获取时间步长的数据
        y.append(data[i+time_step:i + time_step+horizon, -1])  # 预测目标值（Label_V列）
    return np.array(X), np.array(y) 

def create_dataset(time_Step, horizon,val_ratio, batch_size):
    #! 训练数据处理部分
    # 读取 CSV 文件
    df1 = pd.read_csv("./train/A-雨量水位(2014-2019).csv", parse_dates=["TIME"])
    df2 = pd.read_csv("./train/A-入库流量(2014-2019).csv", parse_dates=["TIME"])

    # 确保对齐时间
    df1 = df1[df1["TIME"].isin(df2["TIME"])]
    
    # #! 对异常值进行处理
    # # 统计处理前的异常值数量（V <= 0）
    # num_outliers_before = np.sum(df1['V'] <= 0)
    # print(f"处理前的异常值数量: {num_outliers_before}")
    # # 设置滚动窗口大小
    # window_size = 5
    # # 计算滚动均值
    # rolling_mean = df1['V'].rolling(window=window_size, center=True).mean()
    # # 替换异常值（V <= 0）为滚动均值
    # df1['V'] = df1.apply(lambda row: rolling_mean[row.name] if row['V'] <= 0 else row['V'], axis=1)
    # # 统计处理后的异常值数量（V <= 0）
    # num_outliers_after = np.sum(df1['V'] <= 0)
    # print(f"处理后的异常值数量: {num_outliers_after}")

    agg_df = df1.groupby("TIME")["V"].agg(Total_V="sum", Mean_V="mean", Max_V="max", Min_V="min").reset_index()

    # 组合数据
    data = df2[["TIME"]].merge(agg_df, on="TIME", how="left").fillna(0)
    data["target"] = df2["V"].fillna(0)

    # 添加特征
    data = add_features(data,train=True)
    
    # 标准化
    scale = StandardScaler().fit(data)
    data = scale.transform(data)
    
    # print(data.head())
    X,y = create_sequences(data, time_Step, horizon)

    # 按顺序划分训练集和验证集
    val_size = int(len(X) * val_ratio)
    X_train = X[:-val_size]
    X_val = X[-val_size:]
    y_train = y[:-val_size] 
    y_val = y[-val_size:]

    print(f"训练集: {X_train.shape, y_train.shape}, 验证集: {X_val.shape, y_val.shape}")

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 构建测试集
    df_test_rain = pd.read_csv("./test/A-雨量水位(2020-2021).csv", parse_dates=["TIME"])

    # 生成预测时间段内的完整小时序列
    start_time = '2020-01-01 00:00:00'
    end_time = '2021-12-31 23:00:00'
    full_time = pd.DataFrame({'TIME': pd.date_range(start=start_time, end=end_time, freq='H')})

    # 按小时聚合雨量数据（与训练集相同的统计量）
    agg_test = df_test_rain.groupby("TIME")["V"].agg(
        Total_V="sum",
        Mean_V="mean",
        Max_V="max",
        Min_V="min"
    ).reset_index()

    # 合并到完整时间序列并填充缺失值
    test_data = full_time.merge(agg_test, on="TIME", how="left").fillna(0)

    # 添加与训练集相同的特征
    test_data = add_features(test_data)

    print(type(X_val), type(test_data.to_numpy()))
    return {
        'train': train_loader,
        'val': val_loader,
    }, scale, {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'test_data': test_data
    }
    
def save_preprocessed_data(raw_data, save_dir="preprocessed"):
    """保存预处理数据"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # 保存数据
    for split in ['X_train', 'y_train', 'X_val', 'y_val']:
        np.save(f"{save_dir}/{split}.npy", raw_data[split])
    
    print(f"预处理数据已保存到 {save_dir} 目录")
    
# 使用示例
if __name__ == "__main__":
    # 参数设置
    TIME_STEP = 96  # 使用96小时作为历史窗口
    HORIZON = 16    # 预测未来16小时
    BATCH_SIZE = 64
    
    # 完整数据准备流程
    dataloaders, all_scaler,raw_data = create_dataset(
        time_Step=TIME_STEP,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        val_ratio=0.2)
    
    # 保存预处理数据
    save_preprocessed_data(raw_data)
    
    # 保存scaler
    import joblib
    joblib.dump(all_scaler, "preprocessed/all_scaler.joblib")
    print()
    # 打印数据集信息
    print(f"训练集批次: {len(dataloaders['train'])}")
    print(f"验证集批次: {len(dataloaders['val'])}")
    print(f"标准化参数：{all_scaler.mean_}")
    # 检查一个批次的形状
    for x, y in dataloaders['train']:
        print(f"输入批次形状: {x.shape}")  # 应该是 [batch_size, features, time_step]
        print(f"输出批次形状: {y.shape}")  # 应该是 [batch_size, horizon]
        break
    
    
# #! 测试数据处理部分
# # 读取测试集的雨量数据
# df_test_rain = pd.read_csv("./test/A-雨量水位(2020-2021).csv", parse_dates=["TIME"])

# # 生成预测时间段内的完整小时序列
# start_time = '2020-01-01 00:00:00'
# end_time = '2021-12-31 23:00:00'
# full_time = pd.DataFrame({'TIME': pd.date_range(start=start_time, end=end_time, freq='H')})

# # 按小时聚合雨量数据（与训练集相同的统计量）
# agg_test = df_test_rain.groupby("TIME")["V"].agg(
#     Total_V="sum",
#     Mean_V="mean",
#     Max_V="max",
#     Min_V="min"
# ).reset_index()

# # 合并到完整时间序列并填充缺失值
# test_data = full_time.merge(agg_test, on="TIME", how="left").fillna(0)

# # 添加与训练集相同的特征
# test_data = add_features(test_data)

# print(data.shape , test_data.shape)



# # #! 模型及数据处理部分
# timeStep = 64
# Horizon = 7
# inputC = 17

# X, y = SlidingWindowSplitter(window_len=timeStep, horizon=Horizon, get_y=['target'])(data)
# X, y = X.astype('float32'), y.astype('float32')

# model = LSTM(inputC, Horizon)
# # splits = get_forecasting_splits(data, timeStep, Horizon, stride=1, test_size=0.2, show_plot=False)
# # batch_tfms = [TSStandardize(by_sample=True)]

# # #! 开始训练

# # learn = TSForecaster(X=X, y=y, splits=splits, arch=model, metrics=mae, bs=128,
# #                 opt_func=Adam, lr=0.0001,
# #                 partial_n=.1, train_metrics=True, device=default_device(),
# #                 seed=42, path='.', model_dir='models',)
# # learn.save_all('./tmp', verbose=True)
# # learn.fit_one_cycle(100, 1e-3)
# # learn.plot_metrics()


# # # #! 开始预测
# # # learn = load_learner("./tmp/model.pth",cpu=False)
# # # pred = learn.get_X_preds(test_data, with_decoded=True)
# # weights = torch.load('./tmp/model.pth')
# # model.load_state_dict(weights)
# from predict_2 import *
# a = predict_future(model_path='./tmp/model.pth',model=model,device="cuda:0",time_step=timeStep,test_data=data,future_dates=test_data)