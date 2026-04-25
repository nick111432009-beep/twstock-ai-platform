"""
Transformer 模型 — 股價預測
============================
Transformer 用 Self-Attention 機制，可以同時看時間序列上所有位置的關係，
理論上比 LSTM 更能捕捉長期依賴。本檔案使用 PyTorch 內建的
nn.TransformerEncoderLayer 來搭建一個簡單但有效的時序預測模型。

架構:
    Input (B, L, F)
      → Linear 投影到 d_model
      → 加上位置編碼 (Positional Encoding)
      → N 層 Transformer Encoder
      → 取最後位置的向量 → Linear → output(1)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ModelResult, compute_metrics


class PositionalEncoding(nn.Module):
    """
    位置編碼 — 因為 Transformer 沒有遞迴或卷積結構，需要額外把「順序資訊」加進去。
    使用經典的 sin/cos 位置編碼 (Vaswani et al. 2017)。
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    """
    時序預測用的小型 Transformer Encoder
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_len: int = 500,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)        # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)            # → (batch, seq_len, d_model)
        last = x[:, -1, :]             # 取最後 token 預測下一天價格
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


def train_transformer(
    data_dict: dict,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    device: str = "cpu",
    progress_callback=None,
) -> ModelResult:
    """訓練 Transformer 並在測試集上預測"""
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TransformerRegressor(
        input_size=X_train.shape[2],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    # Transformer 用 cosine 退火學習率，比固定 lr 收斂更穩
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_history, val_history = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()

        train_history.append(epoch_loss)
        val_history.append(val_loss)

        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, epoch_loss, val_loss)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()

    y_scaler = data_dict["y_scaler"]
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse, mae, dir_acc = compute_metrics(y_true, y_pred)

    return ModelResult(
        model_name="Transformer",
        y_true=y_true,
        y_pred=y_pred,
        train_loss_history=train_history,
        val_loss_history=val_history,
        rmse=rmse,
        mae=mae,
        direction_accuracy=dir_acc,
        test_dates=data_dict["test_dates"],
    )
