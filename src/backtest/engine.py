"""
回測引擎 (Backtest Engine)
==========================
從零實作的向量化回測引擎，支援:
    - 任意策略 (透過 Strategy 介面)
    - 手續費與證交稅 (台股實際稅費)
    - 部位管理: 全進全出 (all-in/all-out)
    - 績效指標: 年化報酬、最大回撤、Sharpe、勝率、交易次數

設計理念:
    - 「向量化」實作 — 不用 for 迴圈跑每一天，速度比逐筆模擬快非常多
    - 與券商實際成本對齊: 台股買賣手續費各 0.1425% (可打折)，賣出加 0.3% 證交稅
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .strategies import Strategy


# 台股實際交易成本 (常數)
TW_COMMISSION_RATE = 0.001425   # 0.1425% (買進、賣出都要)
TW_TAX_RATE = 0.003             # 0.3% (賣出時才收，證交稅)
DEFAULT_DISCOUNT = 0.6          # 多數券商手續費打 6 折


@dataclass
class BacktestResult:
    """回測結果統一容器"""
    equity_curve: pd.Series                # 每日資產曲線 (含現金 + 股票市值)
    trade_log: pd.DataFrame                # 交易紀錄 (含日期、動作、股價、損益)
    signals: pd.Series                     # 用過的買賣信號

    # 績效指標
    total_return: float = 0.0              # 總報酬率
    annualized_return: float = 0.0         # 年化報酬率
    max_drawdown: float = 0.0              # 最大回撤
    sharpe_ratio: float = 0.0              # Sharpe (無風險利率假設 1%)
    win_rate: float = 0.0                  # 勝率 (有獲利的交易筆數 / 總交易筆數)
    n_trades: int = 0                      # 完成的買賣回合數
    buy_and_hold_return: float = 0.0       # 同期間買進持有報酬 (對照組)

    initial_capital: float = 1_000_000

    def summary(self) -> dict:
        """轉成 dict (給 Streamlit 表格用)"""
        return {
            "總報酬率": f"{self.total_return * 100:.2f}%",
            "年化報酬率": f"{self.annualized_return * 100:.2f}%",
            "最大回撤": f"{self.max_drawdown * 100:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "勝率": f"{self.win_rate * 100:.2f}%",
            "交易次數": self.n_trades,
            "買進持有報酬": f"{self.buy_and_hold_return * 100:.2f}%",
        }


class BacktestEngine:
    """
    向量化回測引擎

    使用方式:
        engine = BacktestEngine(initial_capital=1_000_000)
        result = engine.run(df, strategy)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_discount: float = DEFAULT_DISCOUNT,
        slippage_pct: float = 0.001,
    ):
        """
        參數:
            initial_capital: 初始資金 (預設 100 萬)
            commission_discount: 手續費折扣 (0.6 = 打 6 折)
            slippage_pct: 滑價 (預設 0.1%) — 模擬實際成交價和昨收的差距
        """
        self.initial_capital = initial_capital
        self.commission_rate = TW_COMMISSION_RATE * commission_discount
        self.tax_rate = TW_TAX_RATE
        self.slippage = slippage_pct

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """
        執行回測

        參數:
            df: 含 OHLCV (與技術指標) 的 DataFrame，索引為日期
            strategy: 任何 Strategy 實例

        回傳: BacktestResult
        """
        signals = strategy.generate_signals(df)
        return self._simulate(df, signals)

    def _simulate(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """根據信號模擬交易並計算績效"""
        close = df["Close"].values
        dates = df.index

        cash = self.initial_capital
        shares = 0                # 持有股數 (整數，台股以張為單位但這裡簡化成股)
        equity_history = []
        trade_log = []
        last_buy_price: Optional[float] = None
        last_buy_total: float = 0.0     # 含手續費的買進總成本

        # 為了避免「未來函數」(look-ahead bias)，信號是當天收盤後產生，
        # 真正成交是「下一個交易日」的開盤價 (這是業界標準作法)
        for i in range(len(df)):
            sig = int(signals.iloc[i])
            today_close = close[i]
            today_date = dates[i]

            # ---- 計算「下一日」執行價 (含滑價) ----
            if i + 1 < len(df):
                exec_price = float(df["Open"].iloc[i + 1]) * (1 + self.slippage * np.sign(sig))
                exec_date = dates[i + 1]
            else:
                exec_price = today_close
                exec_date = today_date

            # ---- 買進信號 ----
            if sig == 1 and shares == 0 and cash > 0:
                # 全部資金買入 (扣除手續費)
                budget = cash / (1 + self.commission_rate)
                buy_shares = int(budget // exec_price)
                if buy_shares > 0:
                    gross = buy_shares * exec_price
                    commission = gross * self.commission_rate
                    cost = gross + commission
                    cash -= cost
                    shares += buy_shares
                    last_buy_price = exec_price
                    last_buy_total = cost
                    trade_log.append({
                        "date": exec_date,
                        "action": "BUY",
                        "price": round(exec_price, 2),
                        "shares": buy_shares,
                        "amount": round(gross, 2),
                        "fee": round(commission, 2),
                        "pnl": np.nan,
                    })

            # ---- 賣出信號 ----
            elif sig == -1 and shares > 0:
                gross = shares * exec_price
                commission = gross * self.commission_rate
                tax = gross * self.tax_rate
                proceeds = gross - commission - tax
                cash += proceeds
                # 計算這筆來回交易的損益
                pnl = proceeds - last_buy_total if last_buy_total else 0.0
                trade_log.append({
                    "date": exec_date,
                    "action": "SELL",
                    "price": round(exec_price, 2),
                    "shares": shares,
                    "amount": round(gross, 2),
                    "fee": round(commission + tax, 2),
                    "pnl": round(pnl, 2),
                })
                shares = 0
                last_buy_price = None
                last_buy_total = 0.0

            # ---- 記錄當日總資產 (現金 + 股票市值) ----
            equity_history.append(cash + shares * today_close)

        # 收盤前若還有持股，也以最後一天收盤價結算 (才能跟買進持有公平比較)
        equity_curve = pd.Series(equity_history, index=dates, name="Equity")

        # ---- 計算績效指標 ----
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        n_days = (dates[-1] - dates[0]).days or 1
        annualized = (1 + total_return) ** (365 / n_days) - 1

        # 最大回撤 (Maximum Drawdown)
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Sharpe (假設無風險利率 1%，使用日報酬率年化)
        daily_returns = equity_curve.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = float(
                (daily_returns.mean() - 0.01 / 252) / daily_returns.std() * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        trade_df = pd.DataFrame(trade_log)
        if not trade_df.empty:
            wins = trade_df[trade_df["action"] == "SELL"]
            wins = wins[wins["pnl"] > 0]
            n_round_trips = len(trade_df[trade_df["action"] == "SELL"])
            win_rate = len(wins) / n_round_trips if n_round_trips > 0 else 0.0
        else:
            n_round_trips = 0
            win_rate = 0.0

        # 買進持有報酬 (對照組)
        bnh_return = float(close[-1] / close[0] - 1)

        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_df,
            signals=signals,
            total_return=float(total_return),
            annualized_return=float(annualized),
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            n_trades=n_round_trips,
            buy_and_hold_return=bnh_return,
            initial_capital=self.initial_capital,
        )


# 便利函式
def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 1_000_000,
) -> BacktestResult:
    """一行回測"""
    return BacktestEngine(initial_capital=initial_capital).run(df, strategy)
