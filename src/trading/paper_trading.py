"""
虛擬交易模組 (Paper Trading)
============================
用 SQLite 持久化儲存虛擬帳戶、持倉與交易歷史。
讓使用者可以在網頁上根據 AI 信號或自選策略做模擬下單，
完全不涉及真實資金，安全且符合課堂展示需求。

資料庫結構:
    accounts:    帳戶基本資料 (id, name, initial_capital, cash)
    positions:   目前持倉   (account_id, symbol, shares, avg_price)
    trades:      交易歷史   (account_id, symbol, side, price, shares, ts, pnl)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# 資料庫位置:
#   1. 環境變數 PAPER_TRADING_DB_PATH (使用者可覆寫)
#   2. 預設 = 專案下的 data/paper_trading.db
_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "paper_trading.db"
DB_PATH = Path(os.environ.get("PAPER_TRADING_DB_PATH", str(_DEFAULT_DB)))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 沿用 backtest 中的台股實際成本
COMMISSION_RATE = 0.001425 * 0.6   # 打 6 折
TAX_RATE = 0.003


@contextmanager
def _connect():
    """產生 SQLite 連線 (with 區塊保證關閉)"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    initial_capital REAL NOT NULL,
    cash REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    account_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    shares INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (account_id, symbol)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    shares INTEGER NOT NULL,
    fee REAL NOT NULL,
    pnl REAL,
    note TEXT,
    ts TEXT NOT NULL
);
"""


def _init_db():
    """初次執行時建立資料表"""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


_init_db()


class PaperTradingAccount:
    """虛擬交易帳戶 (對應資料庫一個 row)"""

    def __init__(self, account_id: int, name: str, initial_capital: float, cash: float):
        self.id = account_id
        self.name = name
        self.initial_capital = initial_capital
        self.cash = cash

    @classmethod
    def get_or_create(
        cls,
        name: str = "default",
        initial_capital: float = 1_000_000,
    ) -> "PaperTradingAccount":
        """取得或建立指定名稱的帳戶"""
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE name=?", (name,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO accounts (name, initial_capital, cash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (name, initial_capital, initial_capital, datetime.now().isoformat()),
                )
                row = conn.execute(
                    "SELECT * FROM accounts WHERE name=?", (name,)
                ).fetchone()
            return cls(
                account_id=row["id"],
                name=row["name"],
                initial_capital=row["initial_capital"],
                cash=row["cash"],
            )

    def buy(self, symbol: str, price: float, shares: int, note: str = "") -> dict:
        """模擬下單買進"""
        gross = price * shares
        fee = gross * COMMISSION_RATE
        cost = gross + fee
        if cost > self.cash:
            return {"ok": False, "msg": f"資金不足 (需 {cost:,.0f}，現有 {self.cash:,.0f})"}

        with _connect() as conn:
            new_cash = self.cash - cost
            conn.execute("UPDATE accounts SET cash=? WHERE id=?", (new_cash, self.id))
            self.cash = new_cash

            row = conn.execute(
                "SELECT shares, avg_price FROM positions WHERE account_id=? AND symbol=?",
                (self.id, symbol),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO positions (account_id, symbol, shares, avg_price, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.id, symbol, shares, price, datetime.now().isoformat()),
                )
            else:
                old_shares, old_avg = row["shares"], row["avg_price"]
                new_shares = old_shares + shares
                new_avg = (old_shares * old_avg + shares * price) / new_shares
                conn.execute(
                    "UPDATE positions SET shares=?, avg_price=?, updated_at=? "
                    "WHERE account_id=? AND symbol=?",
                    (new_shares, new_avg, datetime.now().isoformat(), self.id, symbol),
                )

            conn.execute(
                "INSERT INTO trades (account_id, symbol, side, price, shares, fee, pnl, note, ts) "
                "VALUES (?, ?, 'BUY', ?, ?, ?, NULL, ?, ?)",
                (self.id, symbol, price, shares, fee, note, datetime.now().isoformat()),
            )

        return {"ok": True, "msg": f"買進 {symbol} {shares} 股 @ {price}", "cost": cost}

    def sell(self, symbol: str, price: float, shares: int, note: str = "") -> dict:
        """模擬下單賣出"""
        with _connect() as conn:
            row = conn.execute(
                "SELECT shares, avg_price FROM positions WHERE account_id=? AND symbol=?",
                (self.id, symbol),
            ).fetchone()
            if row is None or row["shares"] < shares:
                return {"ok": False, "msg": "持股不足"}

            gross = price * shares
            fee = gross * COMMISSION_RATE
            tax = gross * TAX_RATE
            proceeds = gross - fee - tax

            avg = row["avg_price"]
            pnl = proceeds - avg * shares
            new_position_shares = row["shares"] - shares

            new_cash = self.cash + proceeds
            conn.execute("UPDATE accounts SET cash=? WHERE id=?", (new_cash, self.id))
            self.cash = new_cash

            if new_position_shares == 0:
                conn.execute(
                    "DELETE FROM positions WHERE account_id=? AND symbol=?",
                    (self.id, symbol),
                )
            else:
                conn.execute(
                    "UPDATE positions SET shares=?, updated_at=? "
                    "WHERE account_id=? AND symbol=?",
                    (new_position_shares, datetime.now().isoformat(), self.id, symbol),
                )

            conn.execute(
                "INSERT INTO trades (account_id, symbol, side, price, shares, fee, pnl, note, ts) "
                "VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?)",
                (self.id, symbol, price, shares, fee + tax, pnl, note,
                 datetime.now().isoformat()),
            )

        return {
            "ok": True,
            "msg": f"賣出 {symbol} {shares} 股 @ {price} (損益 {pnl:+,.0f})",
            "pnl": pnl,
        }

    def positions(self) -> pd.DataFrame:
        """目前所有持倉"""
        with _connect() as conn:
            return pd.read_sql(
                "SELECT symbol, shares, avg_price, updated_at "
                "FROM positions WHERE account_id=?",
                conn, params=(self.id,),
            )

    def trades(self, limit: int = 100) -> pd.DataFrame:
        """近 N 筆交易紀錄"""
        with _connect() as conn:
            return pd.read_sql(
                "SELECT ts, symbol, side, price, shares, fee, pnl, note "
                "FROM trades WHERE account_id=? "
                "ORDER BY ts DESC LIMIT ?",
                conn, params=(self.id, limit),
            )

    def equity(self, market_prices: dict) -> dict:
        """計算目前帳戶總值"""
        pos = self.positions()
        market_value = 0.0
        for _, row in pos.iterrows():
            mp = market_prices.get(row["symbol"], row["avg_price"])
            market_value += mp * row["shares"]
        total = self.cash + market_value
        pnl = total - self.initial_capital
        pnl_pct = pnl / self.initial_capital
        return {
            "cash": self.cash,
            "market_value": market_value,
            "total": total,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }

    def reset(self):
        """重置帳戶 (給 demo 用，把錢與持倉清空回到初始狀態)"""
        with _connect() as conn:
            conn.execute("UPDATE accounts SET cash=? WHERE id=?",
                         (self.initial_capital, self.id))
            conn.execute("DELETE FROM positions WHERE account_id=?", (self.id,))
            conn.execute("DELETE FROM trades WHERE account_id=?", (self.id,))
        self.cash = self.initial_capital


def get_default_account() -> PaperTradingAccount:
    """取得預設虛擬帳戶 (給 Streamlit 用)"""
    return PaperTradingAccount.get_or_create("default", initial_capital=1_000_000)
