"""
交易策略集合
============
本檔案定義一系列「策略」物件，每個策略接收一張包含 OHLCV+技術指標的 DataFrame，
回傳一個 Series of {1, 0, -1}:
    1  = 買進信號
    -1 = 賣出信號
    0  = 持平

回測引擎會根據這些信號做模擬交易。

策略設計遵循「單一職責」原則: 信號產生與資金管理分離。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Strategy(ABC):
    """所有策略的抽象基底類別"""

    name: str = "Strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """產生買賣信號 (1 / 0 / -1)"""
        ...


# ============================================================
# 1. 雙均線交叉策略 (黃金交叉/死亡交叉)
# ============================================================
class MovingAverageCrossStrategy(Strategy):
    """
    短均線向上穿越長均線 → 黃金交叉，買進
    短均線向下穿越長均線 → 死亡交叉，賣出
    """

    name = "MA 雙均線交叉"

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short = df["Close"].rolling(self.short_window).mean()
        long = df["Close"].rolling(self.long_window).mean()
        # diff() > 0 表示這一天剛從負轉正 → 黃金交叉
        cross_up = (short > long) & (short.shift(1) <= long.shift(1))
        cross_down = (short < long) & (short.shift(1) >= long.shift(1))

        signals = pd.Series(0, index=df.index)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals


# ============================================================
# 2. RSI 超買超賣策略
# ============================================================
class RSIStrategy(Strategy):
    """RSI < 30 超賣 → 買進；RSI > 70 超買 → 賣出"""

    name = "RSI 超買超賣"

    def __init__(self, oversold: int = 30, overbought: int = 70):
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "RSI_14" not in df.columns:
            raise ValueError("RSIStrategy 需要 RSI_14 欄位，請先呼叫 add_all_indicators()")

        rsi = df["RSI_14"]
        signals = pd.Series(0, index=df.index)
        # 從超賣區突破往上 → 買
        signals[(rsi.shift(1) <= self.oversold) & (rsi > self.oversold)] = 1
        # 從超買區跌破往下 → 賣
        signals[(rsi.shift(1) >= self.overbought) & (rsi < self.overbought)] = -1
        return signals


# ============================================================
# 3. MACD 策略
# ============================================================
class MACDStrategy(Strategy):
    """MACD 線從下往上穿越信號線 → 買；反之 → 賣"""

    name = "MACD 趨勢"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "MACD" not in df.columns:
            raise ValueError("MACDStrategy 需要 MACD 欄位")

        macd_line = df["MACD"]
        signal_line = df["MACD_signal"]
        cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        signals = pd.Series(0, index=df.index)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals


# ============================================================
# 4. 布林通道策略
# ============================================================
class BollingerBandStrategy(Strategy):
    """跌破下軌 → 買 (預期均值回歸)；突破上軌 → 賣"""

    name = "布林通道反轉"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "BB_upper" not in df.columns:
            raise ValueError("BollingerBandStrategy 需要 BB_upper / BB_lower 欄位")

        close = df["Close"]
        signals = pd.Series(0, index=df.index)
        # 從下軌之下回到下軌之上 → 買 (反彈信號)
        signals[(close.shift(1) < df["BB_lower"].shift(1)) & (close >= df["BB_lower"])] = 1
        # 從上軌之上跌回上軌之下 → 賣
        signals[(close.shift(1) > df["BB_upper"].shift(1)) & (close <= df["BB_upper"])] = -1
        return signals


# ============================================================
# 5. AI 信號策略 (用 ML 模型預測值產生信號)
# ============================================================
class AISignalStrategy(Strategy):
    """
    根據 AI 模型對未來 N 天的預測，產生買賣信號:
        預測收盤價 > 今日收盤 × (1 + threshold) → 買
        預測收盤價 < 今日收盤 × (1 - threshold) → 賣

    使用方式:
        把模型對所有交易日的預測整理成 Series，傳給此策略
    """

    name = "AI 模型信號"

    def __init__(self, predictions: pd.Series, threshold: float = 0.005):
        """
        參數:
            predictions: 索引為日期的預測收盤價序列
            threshold: 預測漲跌幅門檻 (預設 0.5%)
        """
        self.predictions = predictions
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 取交集日期 (避免索引不一致)
        common = df.index.intersection(self.predictions.index)
        signals = pd.Series(0, index=df.index)
        if len(common) == 0:
            return signals

        pred_aligned = self.predictions.reindex(df.index)
        ratio = pred_aligned / df["Close"]

        signals[ratio > 1 + self.threshold] = 1
        signals[ratio < 1 - self.threshold] = -1
        return signals.fillna(0).astype(int)


# ============================================================
# 策略註冊表 (給前端下拉選單使用)
# ============================================================
_REGISTRY = {
    "ma_cross": ("MA 雙均線交叉", MovingAverageCrossStrategy),
    "rsi": ("RSI 超買超賣", RSIStrategy),
    "macd": ("MACD 趨勢", MACDStrategy),
    "bollinger": ("布林通道反轉", BollingerBandStrategy),
}


def list_strategies() -> dict[str, str]:
    """回傳 {key: 中文名} 給前端使用"""
    return {k: v[0] for k, v in _REGISTRY.items()}


def get_strategy(key: str, **kwargs) -> Strategy:
    """根據 key 建立策略實例"""
    if key not in _REGISTRY:
        raise ValueError(f"未知策略: {key}，可用: {list(_REGISTRY.keys())}")
    _, cls = _REGISTRY[key]
    return cls(**kwargs)
