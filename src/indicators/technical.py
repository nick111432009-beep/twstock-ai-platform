"""
技術指標計算模組
================
本模組從零實作台股投資者最常用的技術指標 (不依賴 TA-Lib，因為 TA-Lib 在
Streamlit Cloud 部署需要額外編譯 C 套件，會造成失敗)。

實作的指標:
    - 移動平均線: SMA, EMA
    - 動能類:     RSI, MACD, KD (Stochastic)
    - 波動率類:   Bollinger Bands, ATR
    - 量能類:     OBV (On-Balance Volume)

設計原則:
    - 每個函式都是純 pandas 實作，輸入 Series 或 DataFrame，輸出對應 Series/DataFrame
    - 提供 add_all_indicators() 一鍵把所有指標加到 OHLCV DataFrame，方便給 ML 模型當特徵
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# 移動平均
# ============================================================
def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """簡單移動平均 (Simple Moving Average)
    SMA_t = (P_t + P_{t-1} + ... + P_{t-window+1}) / window
    """
    return close.rolling(window=window, min_periods=1).mean()


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    """指數移動平均 (Exponential Moving Average)
    給近期價格較高權重，比 SMA 對價格變動更敏感。
    """
    return close.ewm(span=window, adjust=False, min_periods=1).mean()


# ============================================================
# RSI - 相對強弱指標
# ============================================================
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index) — 動能指標，常用閾值 70 (超買) / 30 (超賣)

    計算邏輯:
        1. 算出每日漲跌 delta
        2. 分別取「平均漲幅 avg_gain」與「平均跌幅 avg_loss」(用 EMA 平滑)
        3. RS = avg_gain / avg_loss
        4. RSI = 100 - 100 / (1 + RS)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder 原始定義使用 EMA 平滑 (alpha = 1/window)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # 避免除零
    rsi_value = 100 - (100 / (1 + rs))
    return rsi_value.fillna(50)  # 初期空值用 50 (中性) 填補


# ============================================================
# MACD - 指數平滑異同移動平均
# ============================================================
def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_window)
    Histogram = MACD - Signal

    回傳 DataFrame，欄位: MACD, MACD_signal, MACD_hist
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_signal": signal_line,
        "MACD_hist": hist,
    })


# ============================================================
# KD (Stochastic Oscillator) - 隨機指標
# ============================================================
def kd_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """
    台股投資人最愛用的 KD 指標 (9 日)

    步驟:
        RSV_t = (C_t - Low_n) / (High_n - Low_n) × 100
        K_t   = 平滑(RSV, k_smooth)  ← 台股慣用 SMA(3)，原始定義用 EMA
        D_t   = 平滑(K, d_smooth)

    回傳 DataFrame，欄位: K, D
    """
    low_n = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()

    # RSV (Raw Stochastic Value)
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((close - low_n) / denom) * 100
    rsv = rsv.fillna(50)  # 初期或盤整無波動時填中性值

    k = rsv.ewm(alpha=1 / k_smooth, adjust=False, min_periods=1).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False, min_periods=1).mean()

    return pd.DataFrame({"K": k, "D": d})


# ============================================================
# Bollinger Bands - 布林通道
# ============================================================
def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    布林通道 — 用平均線 ± n 倍標準差，看價格相對位置與波動率

    回傳 DataFrame，欄位: BB_upper, BB_middle, BB_lower, BB_width
    """
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=1).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid  # 通道寬度，可當波動率代理

    return pd.DataFrame({
        "BB_upper": upper,
        "BB_middle": mid,
        "BB_lower": lower,
        "BB_width": width,
    })


# ============================================================
# ATR - 平均真實區間 (波動率)
# ============================================================
def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Average True Range — 衡量股價波動幅度，常用於設停損

    True Range = max(
        High - Low,
        |High - Close_prev|,
        |Low - Close_prev|
    )
    ATR = EMA(TR, window)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()


# ============================================================
# OBV - 累積能量潮 (量能指標)
# ============================================================
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume — 用收盤漲跌帶有方向地累積成交量
    上漲日 +Volume，下跌日 -Volume，盤整 0
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ============================================================
# 一鍵全餐
# ============================================================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    把以上全部指標都加到 OHLCV DataFrame，產出一張「特徵表」可直接餵給 AI 模型。

    輸入:
        df 必須包含 Open / High / Low / Close / Volume 欄位

    回傳:
        新的 DataFrame (原欄位 + 約 15 個技術指標欄位)
    """
    out = df.copy()

    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    # 多週期 SMA / EMA (給模型挑出短中長期趨勢)
    out["SMA_5"] = sma(close, 5)
    out["SMA_20"] = sma(close, 20)
    out["SMA_60"] = sma(close, 60)
    out["EMA_12"] = ema(close, 12)
    out["EMA_26"] = ema(close, 26)

    # 動能
    out["RSI_14"] = rsi(close, 14)

    macd_df = macd(close)
    out = pd.concat([out, macd_df], axis=1)

    kd_df = kd_stochastic(high, low, close)
    out = pd.concat([out, kd_df], axis=1)

    # 波動
    bb_df = bollinger_bands(close)
    out = pd.concat([out, bb_df], axis=1)
    out["ATR_14"] = atr(high, low, close)

    # 量能
    out["OBV"] = obv(close, volume)

    # 報酬率特徵 (給 ML 模型，相對於價格絕對值穩定很多)
    out["Return_1d"] = close.pct_change(1)
    out["Return_5d"] = close.pct_change(5)

    return out


if __name__ == "__main__":
    # 自我測試
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.data.loader import get_stock_data

    print("[Test] 對 2330 計算所有技術指標 ...")
    df = get_stock_data("2330", period="6mo")
    feat = add_all_indicators(df)
    print(feat.tail())
    print(f"\n欄位: {list(feat.columns)}")
