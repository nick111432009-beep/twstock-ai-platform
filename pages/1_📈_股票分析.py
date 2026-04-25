"""
頁面 1: 股票分析 — 個股 K 線與技術指標互動分析
"""

import sys
from pathlib import Path

# 把專案根目錄加入 Python 搜尋路徑 (Streamlit 多頁面結構需要)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.data import get_stock_data, list_popular_stocks
from src.indicators import add_all_indicators
from src.utils import candlestick_chart, indicator_chart


st.set_page_config(page_title="股票分析", page_icon="📈", layout="wide")
st.title("📈 個股技術分析")

# ============================================================
# Sidebar 控制項
# ============================================================
popular = list_popular_stocks()

with st.sidebar:
    st.subheader("📊 分析設定")

    # 股票挑選 (下拉 + 手動輸入)
    stock_options = [f"{code} {name}" for code, name in popular.items()]
    selected = st.selectbox("熱門股票", options=stock_options, index=0)
    default_code = selected.split()[0]

    custom_code = st.text_input("自訂代碼 (覆蓋上方)", value="", placeholder="例: 2330")
    symbol = custom_code.strip() if custom_code.strip() else default_code

    period = st.select_slider(
        "資料期間",
        options=["6mo", "1y", "2y", "3y", "5y"],
        value="1y",
    )

    show_overlays = st.multiselect(
        "K 線疊加",
        options=["SMA_5", "SMA_20", "SMA_60", "BB_upper", "BB_lower"],
        default=["SMA_5", "SMA_20"],
    )

    selected_indicators = st.multiselect(
        "下方技術指標",
        options=["RSI", "MACD", "KD"],
        default=["RSI", "MACD"],
    )

# ============================================================
# 主畫面
# ============================================================
@st.cache_data(ttl=3600, show_spinner="正在擷取股價資料...")
def load_data(sym: str, p: str):
    df = get_stock_data(sym, period=p)
    return add_all_indicators(df)

try:
    df = load_data(symbol, period)
except Exception as e:
    st.error(f"無法擷取 {symbol} 的資料: {e}")
    st.info("請檢查股票代碼是否正確，或稍後再試 (yfinance 偶爾會限制請求頻率)。")
    st.stop()

if df.empty:
    st.warning("沒有資料")
    st.stop()

# ---- 摘要卡片 ----
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last
change = last["Close"] - prev["Close"]
change_pct = change / prev["Close"] * 100

stock_name = popular.get(symbol.split(".")[0], symbol)
st.markdown(f"### {symbol} {stock_name}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("收盤價", f"{last['Close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
c2.metric("開盤", f"{last['Open']:.2f}")
c3.metric("最高", f"{last['High']:.2f}")
c4.metric("最低", f"{last['Low']:.2f}")
c5.metric("成交量", f"{int(last['Volume']):,}")

# ---- K 線圖 ----
overlay_dict = {
    "SMA_5": "MA5", "SMA_20": "MA20", "SMA_60": "MA60",
    "BB_upper": "布林上軌", "BB_lower": "布林下軌",
}
overlays = {k: v for k, v in overlay_dict.items() if k in show_overlays}

st.plotly_chart(
    candlestick_chart(df, title=f"{symbol} {stock_name} K 線",
                      show_volume=True, overlays=overlays),
    use_container_width=True,
)

# ---- 技術指標 ----
for ind in selected_indicators:
    st.plotly_chart(indicator_chart(df, ind), use_container_width=True)

# ---- 原始資料 (可展開) ----
with st.expander("查看原始資料"):
    st.dataframe(df.tail(50), use_container_width=True)

# ---- 統計摘要 ----
with st.expander("統計摘要"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**價格統計**")
        st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].describe().round(2))
    with c2:
        st.markdown("**指標目前值**")
        latest = {
            "RSI(14)": round(df["RSI_14"].iloc[-1], 2),
            "MACD": round(df["MACD"].iloc[-1], 4),
            "K": round(df["K"].iloc[-1], 2),
            "D": round(df["D"].iloc[-1], 2),
            "ATR(14)": round(df["ATR_14"].iloc[-1], 2),
        }
        for k, v in latest.items():
            st.metric(k, v)
