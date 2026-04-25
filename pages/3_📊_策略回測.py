"""
頁面 3: 策略回測 — 把策略丟進歷史資料模擬，看績效報告
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.data import get_stock_data, list_popular_stocks
from src.indicators import add_all_indicators
from src.backtest import (
    run_backtest, list_strategies, get_strategy, AISignalStrategy,
)
from src.utils import equity_curve_chart, drawdown_chart


st.set_page_config(page_title="策略回測", page_icon="📊", layout="wide")
st.title("📊 交易策略回測")
st.caption("含台股實際手續費 (0.1425% × 6 折) 與證交稅 (0.3%)")

popular = list_popular_stocks()
strategies = list_strategies()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.subheader("📊 回測設定")

    stock_options = [f"{code} {name}" for code, name in popular.items()]
    selected = st.selectbox("股票", options=stock_options, index=0)
    symbol = selected.split()[0]

    period = st.select_slider("回測期間",
                              options=["1y", "2y", "3y", "5y"], value="2y")

    # 策略列表加上 AI 信號 (若已訓練)
    strategy_keys = list(strategies.keys())
    strategy_labels = [f"{strategies[k]}" for k in strategy_keys]

    has_ai = "ai_predictions" in st.session_state
    if has_ai:
        for ai_name in st.session_state["ai_predictions"].keys():
            strategy_keys.append(f"ai_{ai_name}")
            strategy_labels.append(f"AI 信號 ({ai_name})")

    chosen_idx = st.selectbox("策略", range(len(strategy_keys)),
                              format_func=lambda i: strategy_labels[i])
    strategy_key = strategy_keys[chosen_idx]

    # 策略參數區
    st.markdown("---")
    st.subheader("⚙️ 策略參數")
    extra_kwargs = {}

    if strategy_key == "ma_cross":
        extra_kwargs["short_window"] = st.slider("短均線", 3, 30, 5)
        extra_kwargs["long_window"] = st.slider("長均線", 10, 120, 20)
    elif strategy_key == "rsi":
        extra_kwargs["oversold"] = st.slider("超賣閾值", 10, 40, 30)
        extra_kwargs["overbought"] = st.slider("超買閾值", 60, 90, 70)
    elif strategy_key.startswith("ai_"):
        ai_threshold = st.slider("漲跌幅門檻 (%)", 0.1, 5.0, 0.5, 0.1) / 100
        extra_kwargs["threshold"] = ai_threshold

    st.markdown("---")
    st.subheader("💼 資金設定")
    initial_capital = st.number_input("初始資金 (元)",
                                       value=1_000_000, step=100_000, min_value=10_000)

    run_btn = st.button("🚀 執行回測", type="primary", use_container_width=True)


# ============================================================
# 主畫面
# ============================================================
if not run_btn:
    st.info("👈 請在左側設定參數後點擊「執行回測」")
    st.markdown("""
    ### 📚 內建策略說明

    | 策略 | 邏輯 | 適合行情 |
    |------|------|---------|
    | **MA 雙均線交叉** | 短均線向上穿越長均線 → 買 | 趨勢明顯的多頭 |
    | **RSI 超買超賣** | RSI 從超賣區突破 → 買 | 區間盤整反轉 |
    | **MACD 趨勢** | MACD 上穿信號線 → 買 | 中期趨勢轉折 |
    | **布林通道反轉** | 跌破下軌反彈 → 買 | 均值回歸 |
    | **AI 信號** | 模型預測明日漲跌幅超過門檻 | 取決於模型準度 |

    ### 🎯 績效指標解讀
    - **總報酬率**: 期末資產 / 初始資產 - 1
    - **年化報酬率**: 把總報酬換算成「每年」的等效報酬率
    - **最大回撤**: 帳戶曾經從高點跌下來最深的幅度 (越小越好)
    - **Sharpe Ratio**: 每承擔一單位風險換到的超額報酬 (>1 算好，>2 很好)
    - **勝率**: 賺錢的交易筆數 / 總交易筆數
    """)
    st.stop()


# ---- 資料準備 ----
@st.cache_data(ttl=3600, show_spinner="擷取資料中...")
def _load(sym, p):
    return add_all_indicators(get_stock_data(sym, period=p))


df = _load(symbol, period)

# ---- 建立策略 ----
if strategy_key.startswith("ai_"):
    ai_name = strategy_key.replace("ai_", "")
    if not has_ai or ai_name not in st.session_state["ai_predictions"]:
        st.error("找不到 AI 預測，請先到「🤖 AI 預測」頁面訓練模型。")
        st.stop()
    if st.session_state.get("ai_symbol") != symbol:
        st.warning(
            f"⚠️ AI 模型是用 {st.session_state.get('ai_symbol')} 訓練的，"
            f"當前回測股票是 {symbol}，預測可能不準。"
        )
    predictions = st.session_state["ai_predictions"][ai_name]
    strategy = AISignalStrategy(predictions=predictions, **extra_kwargs)
    strategy_name = f"AI 信號 ({ai_name})"
else:
    strategy = get_strategy(strategy_key, **extra_kwargs)
    strategy_name = strategies[strategy_key]

# ---- 回測 ----
with st.spinner("回測中..."):
    result = run_backtest(df, strategy, initial_capital=initial_capital)

# ---- 顯示績效 ----
st.markdown(f"### 策略: {strategy_name}")

s = result.summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("總報酬率", s["總報酬率"], delta=s["買進持有報酬"] + " (買進持有)")
c2.metric("年化報酬率", s["年化報酬率"])
c3.metric("最大回撤", s["最大回撤"])
c4.metric("Sharpe Ratio", s["Sharpe Ratio"])

c1, c2, c3 = st.columns(3)
c1.metric("勝率", s["勝率"])
c2.metric("交易次數", s["交易次數"])
c3.metric(
    "期末資產",
    f"{result.equity_curve.iloc[-1]:,.0f}",
    delta=f"{result.equity_curve.iloc[-1] - initial_capital:+,.0f}",
)

# ---- 資金曲線 + 買進持有對照 ----
bnh_curve = (df["Close"] / df["Close"].iloc[0]) * initial_capital
bnh_curve = bnh_curve.reindex(result.equity_curve.index)
st.plotly_chart(
    equity_curve_chart(result.equity_curve, initial_capital, bnh_curve),
    use_container_width=True,
)

# ---- 回撤曲線 ----
st.plotly_chart(drawdown_chart(result.equity_curve), use_container_width=True)

# ---- 交易紀錄 ----
st.markdown("### 📋 交易紀錄")
if result.trade_log.empty:
    st.warning("本次回測期間策略沒有產生任何交易信號。")
else:
    st.dataframe(result.trade_log, use_container_width=True, hide_index=True)
