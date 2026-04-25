"""
LSTM 模型 — 股價預測
====================
LSTM (Long Short-Term Memory) 是 RNN 的改良版，靠「閘門機制」解決長期依賴問題，
是時間序列預測最經典的深度學習架構。

本檔案定義:
    - LSTMRegressor: PyTorch 模型類別
    - train_lstm: 一次完成訓練 + 預測 + 評估的高階函式 (給 Streamlit 直接呼叫)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ModelResult, compute_metrics


class LSTMRegressor(nn.Module):
    """
    雙層 LSTM + 全連接層
    結構:
        input  → LSTM(hidden, num_layers) → Dropout → Linear → output(1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # 取最後一個時間步的輸出來預測下一天的價格
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


def train_lstm(
    data_dict: dict,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    device: str = "cpu",
    progress_callback=None,
) -> ModelResult:
    """
    訓練 LSTM 模型 + 在測試集上預測

    參數:
        data_dict: prepare_supervised_data() 的輸出
        progress_callback: 給 Streamlit 進度條用的回呼，簽名 (epoch, total, train_loss, val_loss)

    回傳: ModelResult
    """
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    # 轉成 PyTorch Tensor
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,           # 注意: 這裡 shuffle 是 OK 的 — 因為樣本本身已經是時間窗口
        drop_last=False,
    )

    model = LSTMRegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_history, val_history = [], []

    for epoch in range(epochs):
        # ---- 訓練 ----
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        # ---- 驗證 (用測試集當驗證，僅供觀察 loss 趨勢) ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()

        train_history.append(epoch_loss)
        val_history.append(val_loss)

        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, epoch_loss, val_loss)

    # ---- 最終預測 ----
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()

    # 反正規化回原價格尺度
    y_scaler = data_dict["y_scaler"]
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse, mae, dir_acc = compute_metrics(y_true, y_pred)

    return ModelResult(
        model_name="LSTM",
        y_true=y_true,
        y_pred=y_pred,
        train_loss_history=train_history,
        val_loss_history=val_history,
        rmse=rmse,
        mae=mae,
        direction_accuracy=dir_acc,
        test_dates=data_dict["test_dates"],
    )
