import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


# 自定义Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_window]
        y = self.data[idx + self.input_window:idx + self.input_window + self.output_window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = nn.Embedding(5000, d_model)
        self.encoder = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        src = self.encoder(src) * np.sqrt(src.size(1))
        tgt = self.encoder(tgt) * np.sqrt(tgt.size(1))
        output = self.transformer(src, tgt)
        output = self.decoder(output)
        return output



# 读取数据
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df.values


# 评估指标
def evaluate(pred, true):
    mae = torch.mean(torch.abs(pred - true))
    mse = torch.mean((pred - true) ** 2)
    mape = torch.mean(torch.abs((pred - true) / true)) * 100
    return mae.item(), mse.item(), mape.item()


# 训练模型
def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch, y_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output = model(x_val, y_val)
                val_loss += criterion(output, y_val).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')


# 主函数
def main(train_path, val_path, test_path, input_window, output_window, d_model, nhead, num_encoder_layers,
         num_decoder_layers, dim_feedforward, dropout, num_epochs, learning_rate, batch_size):
    train_data = read_data(train_path)
    val_data = read_data(val_path)
    test_data = read_data(test_path)

    train_dataset = TimeSeriesDataset(train_data, input_window, output_window)
    val_dataset = TimeSeriesDataset(val_data, input_window, output_window)
    test_dataset = TimeSeriesDataset(test_data, input_window, output_window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerModel(train_data.shape[1], d_model, nhead, num_encoder_layers, num_decoder_layers,
                             dim_feedforward, dropout)

    train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    model.eval()
    maes, mses, mapes = [], [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            output = model(x_test, y_test)
            mae, mse, mape = evaluate(output, y_test)
            maes.append(mae)
            mses.append(mse)
            mapes.append(mape)

    print(f'Average MAE: {np.mean(maes):.4f}, Average MSE: {np.mean(mses):.4f}, Average MAPE: {np.mean(mapes):.4f}%')


# 参数配置
if __name__ == '__main__':
    train_path = 'path/to/train.csv'
    val_path = 'path/to/val.csv'
    test_path = 'path/to/test.csv'
    input_window = 24
    output_window = 12
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 32

    main(train_path, val_path, test_path, input_window, output_window, d_model, nhead, num_encoder_layers,
         num_decoder_layers, dim_feedforward, dropout, num_epochs, learning_rate, batch_size)
