import torch
import numpy as np
import pandas as pd
from datetime import datetime

def predict_future(model_path, model, device, time_step, horizon, test_data, future_dates):
    """
    改进后的预测函数，适配外部生成的future_dates
    
    参数说明：
    - future_dates: 外部生成的预测时间序列（pd.DatetimeIndex）
    """
    # 模型加载与设备设置
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 从future_dates获取预测步数
    n_predictions = len(future_dates)
    print(f"接收预测时间范围：{future_dates.min()} 至 {future_dates.max()}")
    print(f"待预测总步数：{n_predictions}")
    
    # 特征维度验证
    n_features = test_data.shape[1]
    
    # 初始化输入序列（保持与训练时相同的处理逻辑）
    last_data_points = test_data[-time_step:]
    current_sequence = torch.tensor(
        last_data_points.reshape(1, time_step, n_features),
        dtype=torch.float32
    ).permute(0, 2, 1).to(device)  # 形状: [batch, features, time_steps]
    
    predictions = []
    
    with torch.no_grad():
        for i in range(n_predictions):
            # 预测下一个时间步
            output = model(current_sequence)
            pred = output.cpu().numpy().flatten()[0]
            predictions.append(pred)
            
            # 克隆并更新序列
            new_seq = current_sequence.clone()
            
            # 1. 滑动窗口：丢弃最旧时间步，新增当前位置
            # 滑动V值（第一个特征）
            new_seq[0, 0, :-horizon] = current_sequence[0, 0, horizon:].clone()
            new_seq[0, 0, -horizon:] = torch.tensor(pred, dtype=torch.float32).to(device)
            
            # 3. 动态更新时间特征（仅更新最新时间步）
            current_time = future_dates[i]
            
            current_sequence = new_seq
            
            # 进度反馈
            if (i+1) % 1000 == 0:
                print(f"已预测 {i+1}/{n_predictions} 步 ({current_time.strftime('%Y-%m-%d %H:%M')})")

    
    # 构建结果DataFrame
    result_df = pd.DataFrame({
        'TIME': future_dates.strftime('%Y-%m-%d %H:%M:%S'),
        'V': np.round(predictions, 4)  # 保留4位小数
    })
    
    # 结果验证
    assert len(result_df) == n_predictions, "预测结果长度与时间序列不符"
    print(f"生成结果示例：\n{result_df.head(3)}\n...\n{result_df.tail(3)}")
    
    # 保存结果
    result_df.to_csv('submission.csv', index=False)
    print(f"预测结果已保存至 submission.csv")
    
    return result_df
