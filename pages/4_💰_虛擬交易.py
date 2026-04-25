"""
頁面 4: 虛擬交易 — Paper Trading 模擬下單與帳戶管理
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.data import get_stock_data, list_popular_stocks
from src.trading import get_default_account
from src.trading.paper_trading import DEFAULT_INITIAL_CAPITAL


st.set_page_config(page_title="虛擬交易", page_icon="💰", layout="wide")
st.title("💰 虛擬交易帳戶")
st.caption("Paper Trading — 安全的模擬下單環境，所有資料持久化儲存於 SQLite")

popular = list_popular_stocks()
account = get_default_account()


# ============================================================
# Sidebar — 帳戶管理
# ============================================================
with st.sidebar:
    st.subheader("💼 帳戶資訊")
    st.metric("初始資金", f"{account.initial_capital:,.0f}")
    st.metric("目前現金", f"{account.cash:,.0f}")

    st.markdown("---")
    st.markdown("**🔧 帳戶調整**")

    if st.button("🔄 重置帳戶", help="清空所有持倉與交易紀錄，回到初始資金"):
        account.reset()
        st.success("帳戶已重置")
        st.rerun()

    # 自訂資金
    st.caption("想練習更大資金的部位配置嗎?")
    new_capital = st.number_input(
        "升級至新初始資金 (元)",
        min_value=100_000,
        max_value=1_000_000_000,
        value=int(DEFAULT_INITIAL_CAPITAL),
        step=1_000_000,
        format="%d",
    )
    if st.button("💎 重置並套用新資金", help="會清空所有交易並把帳戶初始資金升級成上方數字"):
        account.reset(new_initial_capital=float(new_capital))
        st.success(f"帳戶已升級為 {new_capital:,} 元!")
        st.rerun()


# ============================================================
# 取得目前所有持倉的最新市價 (用來計算總資產)
# ============================================================
positions = account.positions()
market_prices = {}
if not positions.empty:
    for sym in positions["symbol"].unique():
        try:
            df = get_stock_data(sym, period="6mo")
            market_prices[sym] = float(df["Close"].iloc[-1])
        except Exception:
            market_prices[sym] = None

equity_info = account.equity({k: v for k, v in market_prices.items() if v is not None})


# ============================================================
# 帳戶總覽
# ============================================================
st.markdown("### 📊 帳戶總覽")
c1, c2, c3, c4 = st.columns(4)
c1.metric("總資產", f"{equity_info['total']:,.0f}",
          delta=f"{equity_info['pnl']:+,.0f} ({equity_info['pnl_pct']*100:+.2f}%)")
c2.metric("現金", f"{equity_info['cash']:,.0f}")
c3.metric("股票市值", f"{equity_info['market_value']:,.0f}")
c4.metric("初始資金", f"{account.initial_capital:,.0f}")

st.markdown("---")

# ============================================================
# 下單面板
# ============================================================
st.markdown("### 📝 模擬下單")
col_form, col_quote = st.columns([2, 1])

with col_form:
    stock_options = [f"{code} {name}" for code, name in popular.items()]
    selected = st.selectbox("選擇股票", options=stock_options, index=0, key="trade_stock")
    custom = st.text_input("或自訂代碼", value="", placeholder="例: 2330")
    symbol = custom.strip() if custom.strip() else selected.split()[0]

# 取得即時價 (其實是最近收盤價)
try:
    df_quote = get_stock_data(symbol, period="6mo")
    market_price = float(df_quote["Close"].iloc[-1])
    last_date = df_quote.index[-1].date()
except Exception as e:
    st.error(f"無法取得 {symbol} 報價: {e}")
    st.stop()

with col_quote:
    st.metric(
        f"{symbol} 最新收盤",
        f"{market_price:.2f}",
        help=f"資料日期: {last_date}",
    )

with col_form:
    c1, c2 = st.columns(2)
    with c1:
        side = st.radio("買 / 賣", ["買進", "賣出"], horizontal=True)
    with c2:
        # 預設 1 張 (1000 股)，但允許零股
        shares = st.number_input("股數", min_value=1, value=1000, step=100)

    note = st.text_input("備註 (選填)", placeholder="例: AI 信號買進、KD 黃金交叉")

    if st.button("✅ 確認下單", type="primary"):
        if side == "買進":
            r = account.buy(symbol, market_price, int(shares), note=note)
        else:
            r = account.sell(symbol, market_price, int(shares), note=note)
        if r["ok"]:
            st.success(r["msg"])
            st.rerun()
        else:
            st.error(r["msg"])


# ============================================================
# 持倉
# ============================================================
st.markdown("---")
st.markdown("### 📦 目前持倉")
if positions.empty:
    st.info("目前沒有持倉")
else:
    rows = []
    for _, row in positions.iterrows():
        sym = row["symbol"]
        mp = market_prices.get(sym, row["avg_price"])
        market_val = mp * row["shares"]
        cost_val = row["avg_price"] * row["shares"]
        pnl = market_val - cost_val
        pnl_pct = pnl / cost_val * 100 if cost_val else 0.0
        rows.append({
            "代碼": sym,
            "中文名稱": popular.get(sym.split(".")[0], sym),
            "持股": int(row["shares"]),
            "成本均價": round(row["avg_price"], 2),
            "現價": round(mp, 2),
            "市值": round(market_val, 0),
            "未實現損益": round(pnl, 0),
            "報酬率(%)": round(pnl_pct, 2),
        })
    import pandas as pd
    pos_df = pd.DataFrame(rows)
    st.dataframe(pos_df, use_container_width=True, hide_index=True)


# ============================================================
# 交易歷史
# ============================================================
st.markdown("---")
st.markdown("### 📜 交易歷史")
trades = account.trades(limit=200)
if trades.empty:
    st.info("尚無交易紀錄")
else:
    # 美化欄位
    trades = trades.rename(columns={
        "ts": "時間", "symbol": "代碼", "side": "動作",
        "price": "成交價", "shares": "股數", "fee": "費用",
        "pnl": "損益", "note": "備註",
    })
    st.dataframe(trades, use_container_width=True, hide_index=True)

    # 損益小結
    sells = trades[trades["動作"] == "SELL"]
    if not sells.empty:
        total_pnl = sells["損益"].sum()
        win_rate = (sells["損益"] > 0).mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("總實現損益", f"{total_pnl:+,.0f}")
        c2.metric("勝率", f"{win_rate:.1f}%")
        c3.metric("交易回合", len(sells))
