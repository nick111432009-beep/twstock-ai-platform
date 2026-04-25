"""
XGBoost 模型 — 股價預測 (Baseline)
==================================
XGBoost 是 Kaggle 與業界量化交易最常用的梯度提升樹模型。
雖然不是深度學習，但在表格特徵上常常打敗 LSTM/Transformer，
所以一定要當 baseline 比一比，才能在報告中講「為什麼選 X 模型」。

注意:
    - XGBoost 不能直接吃 3D 時間序列張量
    - 我們把過去 lookback 天的特徵展平成一條長向量當輸入 (data_dict 中已準備)
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import ModelResult, compute_metrics


def train_xgboost(
    data_dict: dict,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 30,
) -> ModelResult:
    """
    訓練 XGBoost 迴歸模型，使用展平後的時間序列特徵
    """
    X_train_flat = data_dict["X_train_flat"]
    X_test_flat = data_dict["X_test_flat"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
    )

    model.fit(
        X_train_flat,
        y_train,
        eval_set=[(X_test_flat, y_test)],
        verbose=False,
    )

    y_pred_scaled = model.predict(X_test_flat)

    # 反正規化
    y_scaler = data_dict["y_scaler"]
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse, mae, dir_acc = compute_metrics(y_true, y_pred)

    # XGBoost 沒有 epoch loss history (它是 boosting)，但可以拿 evals_result
    eval_results = model.evals_result()
    val_history = list(eval_results.get("validation_0", {}).get("rmse", []))

    return ModelResult(
        model_name="XGBoost",
        y_true=y_true,
        y_pred=y_pred,
        train_loss_history=[],
        val_loss_history=val_history,
        rmse=rmse,
        mae=mae,
        direction_accuracy=dir_acc,
        test_dates=data_dict["test_dates"],
    )
