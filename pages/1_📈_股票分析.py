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
from src.utils import (
    candlestick_chart, indicator_chart,
    interpret_rsi, interpret_macd, interpret_kd,
    interpret_sma, interpret_bollinger,
)


st.set_page_config(page_title="股票分析", page_icon="📈", layout="wide")
st.title("📈 個股技術分析")
st.caption("🎓 新手友善版 — 每個指標都附詳細白話解讀")


def render_interpretation(result: dict):
    """統一渲染解讀面板 (給技術指標用)"""
    # 狀態徽章 + 一句話結論
    st.markdown(f"### {result['status']}  ·  {result['headline']}")

    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown("**📚 這個指標在算什麼?**")
        st.info(result["what"])
        st.markdown("**📊 目前情況**")
        st.markdown(result["now"])

    with cols[1]:
        st.markdown("**💡 新手該怎麼做?**")
        st.success(result["advice"])
        st.markdown("**⚠️ 但要注意**")
        st.warning(result["warn"])

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

# ============================================================
# 均線解讀 (如果有疊加均線就解讀)
# ============================================================
if "SMA_5" in show_overlays and "SMA_20" in show_overlays:
    with st.expander("🎓 均線交叉解讀(白話版)", expanded=True):
        sma_result = interpret_sma(
            close=float(last["Close"]),
            sma_short=float(last["SMA_5"]),
            sma_long=float(last["SMA_20"]),
            short_label="SMA_5", long_label="SMA_20",
        )
        render_interpretation(sma_result)

# 布林通道解讀
if "BB_upper" in show_overlays and "BB_lower" in show_overlays:
    with st.expander("🎓 布林通道解讀(白話版)", expanded=True):
        bb_result = interpret_bollinger(
            close=float(last["Close"]),
            upper=float(last["BB_upper"]),
            middle=float(last["BB_middle"]),
            lower=float(last["BB_lower"]),
        )
        render_interpretation(bb_result)


# ============================================================
# 下方技術指標 + 即時解讀
# ============================================================
for ind in selected_indicators:
    st.plotly_chart(indicator_chart(df, ind), use_container_width=True)

    # 對應的解讀面板
    if ind == "RSI":
        with st.expander("🎓 RSI 解讀(白話版)", expanded=True):
            rsi_result = interpret_rsi(float(last["RSI_14"]))
            render_interpretation(rsi_result)
    elif ind == "MACD":
        with st.expander("🎓 MACD 解讀(白話版)", expanded=True):
            macd_result = interpret_macd(
                macd=float(last["MACD"]),
                signal=float(last["MACD_signal"]),
                histogram=float(last["MACD_hist"]) if "MACD_hist" in df.columns
                          else float(last["MACD"] - last["MACD_signal"]),
            )
            render_interpretation(macd_result)
    elif ind == "KD":
        with st.expander("🎓 KD 解讀(白話版)", expanded=True):
            kd_result = interpret_kd(
                k=float(last["K"]), d=float(last["D"]),
            )
            render_interpretation(kd_result)


# ============================================================
# 綜合判斷 (把所有解讀結合成總結)
# ============================================================
st.markdown("---")
st.markdown("### 🧭 綜合判斷")
st.caption("把上面所有指標的訊號加總，給你一個整體傾向")

# 計算各指標訊號分數
score = 0
notes = []

# RSI
rsi_val = float(last["RSI_14"])
if rsi_val < 30:
    score += 1; notes.append(f"🟢 RSI({rsi_val:.0f}) 超賣，反彈機率高")
elif rsi_val < 50:
    score += 0; notes.append(f"🟡 RSI({rsi_val:.0f}) 略弱")
elif rsi_val < 70:
    score += 1; notes.append(f"🟢 RSI({rsi_val:.0f}) 健康多頭")
else:
    score -= 1; notes.append(f"🔴 RSI({rsi_val:.0f}) 過熱，注意回檔")

# MACD
macd_val = float(last["MACD"])
sig_val = float(last["MACD_signal"])
if macd_val > sig_val and macd_val > 0:
    score += 1; notes.append("🟢 MACD 強勢多頭(快線在慢線上、零軸之上)")
elif macd_val > sig_val:
    score += 0; notes.append("🟡 MACD 弱多(快線在慢線上但仍在零軸之下)")
elif macd_val < sig_val and macd_val < 0:
    score -= 1; notes.append("🔴 MACD 強勢空頭")
else:
    score += 0; notes.append("🟡 MACD 弱空")

# 均線
close_val = float(last["Close"])
sma5_val = float(last["SMA_5"])
sma20_val = float(last["SMA_20"])
if close_val > sma5_val > sma20_val:
    score += 1; notes.append(f"🟢 多頭排列 (價>{'SMA5'}>{'SMA20'})")
elif close_val < sma5_val < sma20_val:
    score -= 1; notes.append("🔴 空頭排列 (價<SMA5<SMA20)")
else:
    score += 0; notes.append("🟡 均線糾結，趨勢不明")

# 顯示綜合結論
col_a, col_b = st.columns([1, 2])
with col_a:
    if score >= 2:
        st.success(f"### 🟢 偏多 (+{score})")
        st.caption("**新手建議**:可考慮分批進場或續抱")
    elif score >= 1:
        st.info(f"### 🟢 弱多 (+{score})")
        st.caption("**新手建議**:謹慎做多，控制部位")
    elif score >= 0:
        st.warning(f"### 🟡 中性 ({score})")
        st.caption("**新手建議**:觀望等明確訊號")
    else:
        st.error(f"### 🔴 偏空 ({score})")
        st.caption("**新手建議**:不適合買進，已有持股可考慮減碼")

with col_b:
    st.markdown("**訊號明細**")
    for n in notes:
        st.markdown(f"- {n}")
    st.caption("⚠️ 此為機械式評分，僅供參考，實際決策請結合基本面與市場狀況。")

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
