"""
台股 AI 量化分析平台 — 主程式入口
=================================
這個檔案是 Streamlit 多頁面應用的「首頁」。
使用 streamlit run app.py 啟動後，左側 sidebar 會自動列出 pages/ 下的子頁面。

子頁面:
    1_📈_股票分析.py   — 個股 K 線、技術指標互動分析
    2_🤖_AI預測.py    — LSTM / Transformer / XGBoost 三模型訓練與比較
    3_📊_策略回測.py   — 多種交易策略回測，看績效報告
    4_💰_虛擬交易.py   — Paper Trading 模擬下單與帳戶管理

作者: 陳彥璋 (碩一下 · 人工智慧期末專案)
"""

import streamlit as st

# ---- 頁面設定 (必須是第一個 streamlit 呼叫) ----
st.set_page_config(
    page_title="台股 AI 量化分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 首頁內容 ----
st.title("📈 台股 AI 量化分析平台")
st.caption("Taiwan Stock AI Quantitative Analysis Platform — 期末專案 Demo")

st.markdown("""
### 專案介紹
本平台整合**深度學習模型**與**量化交易策略**，提供完整的台股投資決策工具鏈，
從資料擷取、技術分析、AI 預測、策略回測到虛擬交易模擬，一站式體驗量化投資流程。

---
""")

# ---- 功能卡片 ----
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📈 個股技術分析
    - 互動式 K 線圖 (含成交量)
    - 移動平均線、布林通道疊加
    - RSI、MACD、KD 動能指標
    - 即時資料 (yfinance 來源)
    """)

    st.markdown("""
    #### 📊 策略回測
    - 5 種內建策略可選
    - 含台股實際手續費與證交稅
    - Sharpe、最大回撤、勝率
    - 與「買進持有」績效對比
    """)

with col2:
    st.markdown("""
    #### 🤖 AI 模型預測
    - **LSTM**: 經典時序模型
    - **Transformer**: 注意力機制
    - **XGBoost**: 梯度提升樹基準
    - 三模型同台比較
    """)

    st.markdown("""
    #### 💰 虛擬交易
    - 100 萬模擬資金
    - SQLite 持久化儲存
    - 損益即時計算
    - 完整交易紀錄
    """)

st.markdown("---")

# ---- 系統架構 ----
st.subheader("🏗️ 系統架構")
st.markdown("""
```
資料層      → yfinance / twstock → 本地 parquet 快取
特徵層      → 技術指標 (SMA, EMA, RSI, MACD, KD, BB, ATR, OBV)
模型層      → LSTM / Transformer / XGBoost (PyTorch + scikit-learn)
策略層      → 5 種內建策略 + AI 信號策略
回測層      → 向量化引擎 (含手續費、證交稅、滑價)
應用層      → Streamlit 多頁面互動式 UI
持久層      → SQLite (虛擬交易帳戶)
```
""")

st.markdown("---")
st.subheader("🚀 使用步驟")
st.markdown("""
1. **左側選擇頁面 → 「📈 股票分析」** 輸入台股代碼 (如 2330)，先看一下 K 線
2. **「🤖 AI 預測」** 訓練三個模型，比較哪一個對該支股票預測最準
3. **「📊 策略回測」** 把策略丟進歷史資料，看歷史報酬與最大回撤
4. **「💰 虛擬交易」** 用 AI 信號或自選策略做模擬下單，追蹤帳戶損益

⚠️ **免責聲明**: 本平台僅供學術研究與教學示範，所有交易均為模擬，
不構成任何投資建議。實際投資請審慎評估風險。
""")

# ---- Sidebar 額外資訊 ----
with st.sidebar:
    st.markdown("### 📚 課程資訊")
    st.caption("**課程名稱**: 人工智慧")
    st.caption("**學期**: 碩一下")
    st.caption("**主題**: 量化金融分析應用")

    st.markdown("---")
    st.markdown("### 🛠️ 技術棧")
    st.caption("- Python 3.10+")
    st.caption("- Streamlit + Plotly")
    st.caption("- PyTorch (LSTM / Transformer)")
    st.caption("- XGBoost / scikit-learn")
    st.caption("- yfinance / pandas")
