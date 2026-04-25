"""
台股資料擷取模組
================
本模組負責從不同來源擷取台股的 OHLCV (開高低收量) 歷史資料。

資料來源策略 (依優先順序):
    1. yfinance: 主要來源，台股代碼後綴 .TW (上市) / .TWO (上櫃)
    2. 本地 parquet 快取: 避免重複請求，加速網頁互動
    3. twstock: 備用來源 (yfinance 失敗時自動切換)

設計原則:
    - 所有對外函式都吃「股票代碼 + 起訖日期」這種統一介面
    - 內建 LRU 快取與本地檔案快取雙層保護
    - 回傳格式統一為 pandas.DataFrame，欄位: Open / High / Low / Close / Volume
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# 第三方資料來源
import yfinance as yf

# ---- 常用台股清單 (給 Streamlit 下拉選單用) ----
# 涵蓋市值前段、ETF、不同產業，方便做 demo 跟回測
POPULAR_STOCKS: dict[str, str] = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2412": "中華電",
    "2882": "國泰金",
    "2881": "富邦金",
    "2891": "中信金",
    "1301": "台塑",
    "1303": "南亞",
    "2002": "中鋼",
    "2303": "聯電",
    "3711": "日月光投控",
    "2603": "長榮",
    "2609": "陽明",
    "2610": "華航",
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "00878": "國泰永續高股息",
    "006208": "富邦台50",
}


def list_popular_stocks() -> dict[str, str]:
    """回傳熱門台股代碼與中文名稱對照表 (給前端 UI 顯示用)"""
    return POPULAR_STOCKS.copy()


# ---- 本地快取資料夾 ----
# Streamlit Cloud 上是暫時檔案系統，每次重啟會清空，但 session 內仍可加速
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StockMeta:
    """股票基本資料容器"""
    symbol: str          # 代碼 (例: 2330)
    name: str            # 中文名 (例: 台積電)
    market: str          # 市場別 (TW=上市, TWO=上櫃)
    yf_ticker: str       # yfinance 用的代號 (例: 2330.TW)


def _resolve_ticker(symbol: str) -> StockMeta:
    """
    把使用者輸入的代碼正規化成 yfinance 認得的格式。

    輸入可以是:
        - "2330" → 自動補 .TW
        - "2330.TW" / "2330.TWO" → 直接使用
        - "AAPL" → 直接當美股處理 (備援，雖然此專案聚焦台股)
    """
    symbol = symbol.strip().upper()
    if "." in symbol:
        # 已經帶後綴，直接用
        ticker = symbol
        market = symbol.split(".")[-1]
    elif symbol.isdigit() or (len(symbol) >= 4 and symbol[:4].isdigit()):
        # 全數字 (或前4碼為數字，例如 00878) → 視為台股，預設上市
        ticker = f"{symbol}.TW"
        market = "TW"
    else:
        ticker = symbol
        market = "US"

    name = POPULAR_STOCKS.get(symbol.split(".")[0], symbol)
    return StockMeta(symbol=symbol, name=name, market=market, yf_ticker=ticker)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """產生快取檔案路徑 (用 ticker + 日期區間當 key)"""
    safe = ticker.replace(".", "_")
    return CACHE_DIR / f"{safe}_{start}_{end}.parquet"


class StockDataLoader:
    """
    台股資料擷取器 (主要對外類別)

    使用範例:
        >>> loader = StockDataLoader()
        >>> df = loader.fetch("2330", start="2023-01-01", end="2024-12-31")
        >>> print(df.head())
    """

    def __init__(self, use_cache: bool = True, retry: int = 3):
        """
        參數:
            use_cache: 是否使用本地 parquet 快取 (預設開啟)
            retry: yfinance 失敗時的重試次數
        """
        self.use_cache = use_cache
        self.retry = retry

    # ------------------------------------------------------------
    # 主要對外方法
    # ------------------------------------------------------------
    def fetch(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        擷取單檔股票的歷史 OHLCV 資料。

        參數:
            symbol: 股票代碼 (例: "2330" 或 "2330.TW")
            start, end: 起訖日期，格式 "YYYY-MM-DD"。若省略則使用 period
            period: yfinance 期間字串，如 "1y", "6mo", "5y"。預設 "2y"

        回傳:
            pandas.DataFrame，索引為日期，欄位:
                Open, High, Low, Close, Volume
        """
        meta = _resolve_ticker(symbol)

        # ---- 處理日期區間 ----
        if start is None and end is None and period is None:
            period = "2y"  # 預設抓近兩年
        if period is not None:
            # period 模式: 自動換算 start/end (這樣才能做快取 key)
            end_dt = datetime.now()
            start_dt = self._period_to_start(period, end_dt)
            start = start_dt.strftime("%Y-%m-%d")
            end = end_dt.strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # ---- 嘗試從快取讀取 ----
        cache_file = _cache_path(meta.yf_ticker, start, end)
        if self.use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    return df
            except Exception:
                # 快取壞了就重撈
                pass

        # ---- 從 yfinance 擷取 (主要來源) ----
        try:
            df = self._fetch_yfinance(meta.yf_ticker, start, end)
        except Exception as yf_err:
            # ---- yfinance 失敗 → 切換到 twstock (台灣本土資料源) ----
            try:
                df = self._fetch_twstock(meta.symbol.split(".")[0], start, end)
                if df.empty:
                    raise ValueError("twstock 也回傳空資料")
            except Exception as tw_err:
                # 兩個來源都失敗才 raise
                raise RuntimeError(
                    f"yfinance 與 twstock 都擷取失敗:\n"
                    f"  yfinance: {yf_err}\n"
                    f"  twstock: {tw_err}"
                )

        # ---- 寫入快取 ----
        if self.use_cache and not df.empty:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass  # 寫入失敗不影響使用，靜默忽略

        return df

    def _fetch_twstock(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """從 twstock 抓資料 (備援來源，直接打台灣證交所)"""
        import twstock

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        stock = twstock.Stock(symbol)
        # twstock 是 fetch_from(年, 月)，會抓那個月之後到現在的資料
        records = stock.fetch_from(start_dt.year, start_dt.month)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "Date": r.date,
                "Open": r.open,
                "High": r.high,
                "Low": r.low,
                "Close": r.close,
                "Volume": r.capacity,
            }
            for r in records
        ]).set_index("Date")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"

        # 截到實際要求的 end 日期
        df = df[df.index <= pd.Timestamp(end_dt)]
        df = df.dropna(how="all")
        return df

    # ------------------------------------------------------------
    # 內部工具方法
    # ------------------------------------------------------------
    def _fetch_yfinance(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """從 yfinance 抓資料，含重試機制"""
        last_err: Optional[Exception] = None
        for attempt in range(self.retry):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,  # 保留原始價格，方便對照看盤軟體
                    threads=False,
                )
                if df.empty:
                    raise ValueError(f"yfinance 回傳空資料: {ticker}")

                # yfinance 1.x 之後欄位變成 MultiIndex，做扁平化處理
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 只保留我們需要的欄位
                wanted = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in wanted if c in df.columns]].copy()

                # 確保索引是 DatetimeIndex 且沒有 timezone (避免後續 plotly 出錯)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"

                # 移除全 NaN 列
                df = df.dropna(how="all")

                return df

            except Exception as e:  # 包含網路錯誤、空資料等
                last_err = e
                time.sleep(1.5 * (attempt + 1))  # 指數退避

        # 全部失敗
        raise RuntimeError(
            f"擷取 {ticker} 失敗 (試了 {self.retry} 次): {last_err}"
        )

    @staticmethod
    def _period_to_start(period: str, end_dt: datetime) -> datetime:
        """把 yfinance 風格的 period 字串轉成起始日期"""
        period = period.strip().lower()
        if period.endswith("y"):
            years = int(period[:-1])
            return end_dt - timedelta(days=365 * years)
        if period.endswith("mo"):
            months = int(period[:-2])
            return end_dt - timedelta(days=30 * months)
        if period.endswith("d"):
            days = int(period[:-1])
            return end_dt - timedelta(days=days)
        # 預設 2 年
        return end_dt - timedelta(days=365 * 2)


# ============================================================
# 便利函式 (給 Streamlit 直接呼叫，搭配 st.cache_data 加速)
# ============================================================
@lru_cache(maxsize=64)
def _cached_fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """LRU 快取版 (給同一個 session 內重複請求用)"""
    loader = StockDataLoader()
    return loader.fetch(symbol, start=start, end=end)


def get_stock_data(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
) -> pd.DataFrame:
    """
    高階便利函式 — 大部分情況下直接呼叫這個就好。

    範例:
        >>> df = get_stock_data("2330", period="1y")
        >>> df = get_stock_data("0050", start="2024-01-01", end="2024-12-31")
    """
    if period is not None:
        loader = StockDataLoader()
        return loader.fetch(symbol, period=period)

    if start is None:
        start = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    return _cached_fetch(symbol, start, end)


if __name__ == "__main__":
    # 自我測試: 直接執行此檔案會撈台積電資料並印出
    print("[Test] 撈取 2330 (台積電) 近一年資料 ...")
    df = get_stock_data("2330", period="1y")
    print(df.tail())
    print(f"\n資料筆數: {len(df)}")
    print(f"日期區間: {df.index.min()} ~ {df.index.max()}")
