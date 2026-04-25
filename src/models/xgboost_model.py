"""
[DEPRECATED] XGBoost 模型 — 已從本專案移除
==========================================
本專案經實驗發現 XGBoost (樹模型) 在股價預測任務上有兩大根本限制:

1. **無法外推 (cannot extrapolate)** — 樹模型只能輸出訓練時看過的數值範圍。
   當股價創歷史新高時，XGBoost 會持續輸出歷史均值附近，造成嚴重預測誤差
   (實驗中 RMSE 高達 382，遠遠輸給 LSTM 的 97)。

2. **時間結構被破壞** — 樹模型只能吃 2D 表格特徵。
   把 30 天時間窗展平成 540 維特徵後，模型不再知道哪個是「昨天」哪個是「一個月前」。

因此本專案聚焦於 LSTM 與 Transformer 兩大主流時序深度學習模型。
本檔案保留為 stub，避免未安裝 xgboost 套件時其他程式碼不小心 import 會直接崩潰。

如需重新啟用，請:
  1. 在 requirements.txt 加回 xgboost==2.1.3
  2. 在 src/models/__init__.py 重新匯入 train_xgboost
  3. 從 git 歷史還原本檔的原始實作 (commit 之前)
"""

from __future__ import annotations


def train_xgboost(*args, **kwargs):
    """已停用 — 呼叫此函式會丟出 NotImplementedError"""
    raise NotImplementedError(
        "XGBoost 模型已從本專案移除。\n"
        "原因:樹模型在股價預測上有外推與時序結構問題 (見本檔頭部說明)。\n"
        "本平台聚焦於 LSTM 與 Transformer 兩大時序模型。"
    )
