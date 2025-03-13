import torch
import torch.nn as nn
import torch.optim as optim
from tsai.all import *
import os
import numpy as np
import pandas as pd

def train_model(epochs, train_loader, val_loader, device, model, criterion, optimizer,scheduler, patience=20,output_file=None):
    
    os.makedirs(output_file,exist_ok=True)
    """训练模型"""

    best_score = 0
    best_rmse = float('inf')
    no_improve = 0

    def calculate_rmse(outputs, targets):
        # 计算RMSE
        mse = torch.mean((outputs - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()
    
    def calculate_score(rmse):
        # 根据公式计算得分: 1/(1+RMSE)
        return 1 / (1 + rmse)


    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                val_rmse += calculate_rmse(outputs, batch_y)

        # 计算平均损失和RMSE
        avg_train_loss = train_loss / len(train_loader)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        
        #         
        val_score = calculate_score(avg_val_rmse)
        
        current = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs} | lr: {current:.6f} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | RMSE: {avg_val_rmse:.6f} Score: {val_score:.6f}')

        # 更新学习率并监测验证集上的性能
        scheduler.step(val_loss)

        # 基于验证集RMSE保存最佳模型
        if val_score > best_score:
            best_rmse = avg_val_rmse
            best_score = val_score
            # 保存模型和相关指标
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': best_rmse,
                'score': best_score,
            }
            torch.save(save_dict, os.path.join(output_file, 'best_model.pth'))
            print(f'保存最佳模型 - RMSE: {best_rmse:.6f}, Score: {best_score:.6f}')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停：{patience}个epoch没有改善")
                break

from data_prepration import prepare_data
from predict import *

if __name__ == '__main__':
    # 数据集构建
    # 参数设置
    TIME_STEP = 96 # 使用的窗口长度
    HORIZON = 7 # 预测未来序列的长度
    BATCH_SIZE = 64

    # 数据加载与划分
    # 完整数据准备流程
    dataloaders, scaler, raw_data = prepare_data(
        file_path="./train/A-入库流量(2014-2019).csv",
        time_step=TIME_STEP,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        val_ratio=0.2
    )

    # 保存到模型目录
    model_dir = 'checkpoints'
    os.makedirs(model_dir,exist_ok=True)

    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'] , dataloaders['test']

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 200

    c_in, c_out = 13, HORIZON

    model = MLP(c_in, c_out, TIME_STEP).to(device)
    # seq_len = 60
    # pred_dim = 20

    # arch_config=dict(
    #         n_layers=3,  # number of encoder layers
    #         n_heads=16,  # number of heads
    #         d_model=128,  # dimension of model
    #         d_ff=256,  # dimension of fully connected network (fcn)
    #         attn_dropout=0.,
    #         dropout=0.2,  # dropout applied to all linear layers in the encoder
    #         patch_len=16,  # patch_len
    #         stride=8,  # stride
    #     )

    # model = PatchTST(c_in, c_out, seq_len, pred_dim, **arch_config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # train_model(epochs, train_loader, val_loader, device, model, criterion, optimizer, scheduler=scheduler,output_file=model_dir)
    # 在训练完成后，进行评估和预测

    future_predictions = predict_future(
        model_path=model_dir + '/' + 'best_model.pth',
        model=model,
        scaler=scaler,
        device=device,
        time_step=TIME_STEP,
        horizon=HORIZON,
        test_data = raw_data['test_scaled'],
        feature_names=raw_data['feature_names']
    )
    print(scaler.mean_)