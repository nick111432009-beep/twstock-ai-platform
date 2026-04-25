"""
模型共用工具
============
這個檔案放兩大模型 (LSTM / Transformer) 都會用到的共用元件:

1. ModelResult: 統一的訓練輸出格式 (給前端比較模型用)
2. prepare_supervised_data: 把時間序列轉成監督式學習格式
   - LSTM/Transformer 需要 3D 張量 (samples, time_steps, features)
   - 同時保留 2D 展平版本 (X_train_flat) 供未來擴充樹模型使用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ModelResult:
    """訓練 + 預測結果統一容器"""
    model_name: str                                # "LSTM" / "Transformer"
    y_true: np.ndarray                             # 真實值 (測試集)
    y_pred: np.ndarray                             # 預測值 (測試集)
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    rmse: float = 0.0
    mae: float = 0.0
    direction_accuracy: float = 0.0   # 方向預測準確率 (漲跌方向預測對的比率)
    test_dates: Optional[pd.DatetimeIndex] = None  # 測試集對應日期 (繪圖用)

    def summary(self) -> dict:
        """回傳摘要 dict (給前端表格顯示)"""
        return {
            "Model": self.model_name,
            "RMSE": round(self.rmse, 4),
            "MAE": round(self.mae, 4),
            "方向準確率": f"{self.direction_accuracy * 100:.2f}%",
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """共用評估指標: RMSE / MAE / 方向準確率"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # 方向準確率 = 預測「上漲/下跌方向」與真實一致的比例
    # 注意這裡比較的是相鄰兩日，不是和某基準
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        # 排除「持平」情況 (sign=0)
        valid = (true_direction != 0) & (pred_direction != 0)
        if valid.sum() > 0:
            direction_acc = float((true_direction[valid] == pred_direction[valid]).mean())
        else:
            direction_acc = 0.0
    else:
        direction_acc = 0.0

    return rmse, mae, direction_acc


def prepare_supervised_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Close",
    lookback: int = 30,
    horizon: int = 1,
    test_ratio: float = 0.2,
) -> dict:
    """
    把時間序列 DataFrame 轉成監督式學習格式

    參數:
        df: 已含技術指標的 DataFrame (索引為日期)
        feature_cols: 要當輸入特徵的欄位
        target_col: 要預測的目標欄位 (預設預測收盤價)
        lookback: 用過去幾天的資料當輸入 (例: 30 表示用 30 天歷史)
        horizon: 預測未來第幾天 (1 = 預測明天)
        test_ratio: 測試集比例

    回傳 dict 含:
        X_train, y_train, X_test, y_test            # 給 LSTM/Transformer (3D)
        X_train_flat, X_test_flat                    # 2D 展平版 (備用，給樹模型)
        y_scaler, feature_scaler                     # 反正規化用
        test_dates                                   # 測試集對應日期
    """
    # 移除前面 NaN (技術指標前期會有 NaN)
    # 重要: 用 dict.fromkeys 去除重複欄位，避免 target_col 同時在 feature_cols 時
    #      出現「同名欄位重複」造成 MinMaxScaler 維度錯誤
    unique_cols = list(dict.fromkeys(feature_cols + [target_col]))
    data = df[unique_cols].dropna().copy()

    # 正規化 (時間序列模型不正規化會跑很慢/不收斂)
    feature_scaler = MinMaxScaler()
    feature_scaled = feature_scaler.fit_transform(data[feature_cols].values)

    # 用 .loc 明確取出單欄並轉 2D (n, 1)，避免 DataFrame 有同名欄位時取到多欄
    y_scaler = MinMaxScaler()
    target_values = data[target_col].to_numpy().reshape(-1, 1)
    target_scaled = y_scaler.fit_transform(target_values).flatten()

    # 滑動窗口製作 (X_t = 過去 lookback 天的特徵, y_t = 第 t+horizon 天的價格)
    X, y, dates = [], [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(feature_scaled[i - lookback : i])     # shape: (lookback, n_features)
        y.append(target_scaled[i + horizon - 1])       # 純量
        dates.append(data.index[i + horizon - 1])

    X = np.array(X, dtype=np.float32)                   # (samples, lookback, n_features)
    y = np.array(y, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)

    # 切分訓練/測試 (時間序列用「時序切分」絕不能 shuffle，這是初學者最常犯的錯)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = dates[split_idx:]

    # 2D 展平版本 (samples, lookback × n_features) — 備用，給樹模型擴充用
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_flat": X_train_flat,
        "X_test_flat": X_test_flat,
        "y_scaler": y_scaler,
        "feature_scaler": feature_scaler,
        "test_dates": test_dates,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
