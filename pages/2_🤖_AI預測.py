"""
頁面 2: AI 預測 — LSTM 與 Transformer 兩大時序模型訓練、預測、白話解讀
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data import get_stock_data, list_popular_stocks
from src.indicators import add_all_indicators
from src.models import (
    prepare_supervised_data, train_lstm, train_transformer,
)
from src.utils import prediction_chart, interpret_ai_prediction


st.set_page_config(page_title="AI 預測", page_icon="🤖", layout="wide")
st.title("🤖 AI 股價預測")
st.caption("LSTM vs Transformer — 兩大主流深度學習時序模型對決")

# ============================================================
# Sidebar
# ============================================================
popular = list_popular_stocks()

with st.sidebar:
    st.subheader("📊 預測設定")
    stock_options = [f"{code} {name}" for code, name in popular.items()]
    selected = st.selectbox("股票", options=stock_options, index=0)
    symbol = selected.split()[0]

    period = st.select_slider("訓練資料長度",
                              options=["1y", "2y", "3y", "5y"], value="2y")

    lookback = st.slider("回看天數 (Lookback)", 10, 60, 30, step=5,
                         help="模型用過去 N 天的資料來預測下一天")

    test_ratio = st.slider("測試集比例", 0.1, 0.4, 0.2, step=0.05)

    st.markdown("---")
    st.subheader("🎯 模型選擇")
    use_lstm = st.checkbox("LSTM", value=True)
    use_transformer = st.checkbox("Transformer", value=True)

    st.markdown("---")
    st.subheader("⚙️ 訓練超參數")
    epochs = st.slider("Epochs (深度模型)", 5, 100, 20, step=5)
    batch_size = st.select_slider("Batch Size", [16, 32, 64, 128], 32)

    train_btn = st.button("🚀 開始訓練", type="primary", use_container_width=True)

# ============================================================
# 主畫面
# ============================================================
@st.cache_data(ttl=3600, show_spinner="擷取資料中...")
def _load(sym, p):
    return add_all_indicators(get_stock_data(sym, period=p))


if not train_btn:
    st.info("👈 請在左側設定參數後點擊「開始訓練」")
    st.markdown("""
    ### 📚 模型介紹

    **LSTM (Long Short-Term Memory)**
    經典時間序列預測模型，透過閘門機制 (Input/Forget/Output Gate) 解決長期依賴問題，
    適合捕捉股價的趨勢與週期。

    **Transformer**
    使用自注意力機制 (Self-Attention)，可同時關注時間序列上所有位置的關係，
    在多項時序任務上表現優於 RNN 系列。

    ### 🎯 評估指標
    - **RMSE**: 均方根誤差 (越小越好)
    - **MAE**: 平均絕對誤差 (越小越好)
    - **方向準確率**: 預測「漲/跌」方向正確的比率 (對交易策略最重要)
    """)
    st.stop()


# ---- 訓練流程 ----
df = _load(symbol, period)

# 選擇特徵欄位 (用所有技術指標)
feature_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_5", "SMA_20", "EMA_12", "EMA_26",
    "RSI_14", "MACD", "MACD_signal", "K", "D",
    "BB_width", "ATR_14", "Return_1d", "Return_5d",
]
feature_cols = [c for c in feature_cols if c in df.columns]

st.write(f"**資料集**: {len(df)} 筆 | **特徵數**: {len(feature_cols)} 個技術指標")

with st.spinner("正在準備訓練資料..."):
    data_dict = prepare_supervised_data(
        df, feature_cols=feature_cols, target_col="Close",
        lookback=lookback, horizon=1, test_ratio=test_ratio,
    )

st.success(
    f"訓練樣本: {len(data_dict['X_train'])} | "
    f"測試樣本: {len(data_dict['X_test'])} | "
    f"輸入形狀: {data_dict['X_train'].shape}"
)

results = {}

# ---- LSTM ----
if use_lstm:
    st.markdown("### 🧠 LSTM 訓練")
    progress_bar = st.progress(0)
    loss_text = st.empty()

    def lstm_callback(epoch, total, train_loss, val_loss):
        progress_bar.progress(epoch / total)
        loss_text.text(f"Epoch {epoch}/{total}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    t0 = time.time()
    results["LSTM"] = train_lstm(
        data_dict, epochs=epochs, batch_size=batch_size,
        progress_callback=lstm_callback,
    )
    st.success(f"LSTM 訓練完成 ({time.time() - t0:.1f}s)")

# ---- Transformer ----
if use_transformer:
    st.markdown("### 🔁 Transformer 訓練")
    progress_bar = st.progress(0)
    loss_text = st.empty()

    def trans_callback(epoch, total, train_loss, val_loss):
        progress_bar.progress(epoch / total)
        loss_text.text(f"Epoch {epoch}/{total}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    t0 = time.time()
    results["Transformer"] = train_transformer(
        data_dict, epochs=epochs, batch_size=batch_size,
        progress_callback=trans_callback,
    )
    st.success(f"Transformer 訓練完成 ({time.time() - t0:.1f}s)")

# ============================================================
# 比較結果
# ============================================================
if results:
    st.markdown("---")
    st.subheader("📊 模型績效比較")

    summary_rows = [r.summary() for r in results.values()]
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 找最佳模型
    best_model = min(results.values(), key=lambda r: r.rmse)
    st.success(f"🏆 RMSE 最低: **{best_model.model_name}** (RMSE = {best_model.rmse:.4f})")

    # ============================================================
    # 🎓 新手白話翻譯 — 把每個模型的最後一筆預測翻成人話
    # ============================================================
    st.markdown("---")
    st.subheader("🎓 AI 預測白話解讀")
    st.caption("把模型的數字輸出翻譯成新手看得懂的中文，並提供操作建議")

    interpret_cols = st.columns(len(results))
    for col, (name, res) in zip(interpret_cols, results.items()):
        with col:
            # 取最後一筆「真實值」當作目前股價，「預測值」當作明日預測
            current_price = float(res.y_true[-1])
            predicted_price = float(res.y_pred[-1])
            interp = interpret_ai_prediction(
                current_price=current_price,
                predicted_price=predicted_price,
                direction_accuracy=res.direction_accuracy,
                model_name=name,
            )

            # 標題與訊號
            st.markdown(f"### {interp['signal']}")
            st.markdown(f"**{interp['headline']}**")

            # 預測數字
            st.metric(
                label=f"明日預測收盤 (vs 目前 {current_price:.2f})",
                value=f"{predicted_price:.2f}",
                delta=f"{interp['change']:+.2f} ({interp['change_pct']:+.2f}%)",
            )

            # 信心度
            st.markdown(
                f"**信心度**: {interp['confidence_emoji']} **{interp['confidence_level']}**  \n"
                f"_({interp['confidence_note']})_  \n"
                f"歷史方向準確率: **{interp['direction_accuracy']*100:.1f}%**"
            )

            st.markdown("**🎯 操作建議**")
            st.success(interp["action_advice"])

    # 共同的安全提醒(只顯示一次)
    st.markdown("---")
    st.markdown("### 🚨 重要提醒(必讀)")
    # 用第一個模型的 general_warn,因為是固定文字
    first_interp = list(results.values())[0]
    sample_warn = interpret_ai_prediction(
        current_price=1.0, predicted_price=1.0,
        direction_accuracy=first_interp.direction_accuracy,
    )
    st.warning(sample_warn["general_warn"])

    # ============================================================
    # 預測 vs 實際對照圖
    # ============================================================
    st.markdown("---")
    st.subheader("📈 預測 vs 實際 走勢圖")
    st.caption("圖中紅線是真實股價、虛線是模型預測。理想狀況兩條線會疊在一起。")

    for name, res in results.items():
        st.plotly_chart(
            prediction_chart(res.test_dates, res.y_true, res.y_pred, name),
            use_container_width=True,
        )

    # 把預測結果存到 session state，給「策略回測」頁面用
    st.session_state["ai_predictions"] = {
        name: pd.Series(res.y_pred, index=res.test_dates)
        for name, res in results.items()
    }
    st.session_state["ai_symbol"] = symbol

    st.info("💡 預測結果已暫存，可以到「📊 策略回測」頁面選擇「AI 信號策略」做回測。")
