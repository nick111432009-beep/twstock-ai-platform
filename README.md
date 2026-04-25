# 📈 台股 AI 量化分析平台

> 碩一下「人工智慧」期末專案 · 陳彥璋
> Taiwan Stock AI Quantitative Analysis Platform

整合 **深度學習** (LSTM / Transformer) 與 **量化交易** 的端到端應用，
提供從資料擷取、技術分析、AI 預測、策略回測到虛擬交易模擬的完整流程。

---

## 🎯 功能總覽

| 模組 | 說明 |
|------|------|
| 📈 個股分析 | 互動式 K 線圖、技術指標 (RSI / MACD / KD / 布林通道) |
| 🤖 AI 預測 | LSTM、Transformer、XGBoost 三模型同台比較 |
| 📊 策略回測 | 5 種內建策略，含台股實際手續費與證交稅 |
| 💰 虛擬交易 | SQLite 持久化模擬下單，安全無風險 |

---

## 🚀 快速開始 (本機跑)

### 1. Clone 專案
```bash
git clone https://github.com/<your-username>/twstock-ai-platform.git
cd twstock-ai-platform
```

### 2. 建立虛擬環境並安裝依賴
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 啟動 Streamlit
```bash
streamlit run app.py
```

瀏覽器自動打開 `http://localhost:8501` 即可使用。

---

## ☁️ 部署到 Streamlit Cloud (零成本上線，用於課堂 DEMO)

Streamlit Community Cloud 提供**完全免費**的 Python 網頁部署服務，
非常適合學術專案展示。

### 步驟 1 — 把專案推上 GitHub

```bash
cd 期末專案_台股量化分析平台
git init
git add .
git commit -m "Initial commit: 台股 AI 量化分析平台"
git branch -M main
git remote add origin https://github.com/<your-username>/twstock-ai-platform.git
git push -u origin main
```

### 步驟 2 — 連到 Streamlit Cloud

1. 用 GitHub 帳號登入 [https://share.streamlit.io](https://share.streamlit.io)
2. 點 **New app** → 選擇你的 repo
3. 設定:
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: 3.10 或 3.11
4. 點 **Deploy** — 大約 5 分鐘後就有公開網址，
   形式為 `https://<your-username>-twstock-ai-platform.streamlit.app/`

### 步驟 3 — 報告時直接給連結

在 PPT 裡放這個 URL，老師就能即時開啟 demo。

---

## 📁 專案結構

```
期末專案_台股量化分析平台/
├── app.py                          # Streamlit 主程式 (首頁)
├── requirements.txt                # 套件清單
├── README.md                       # 本文件
├── .streamlit/config.toml          # Streamlit 主題設定
├── .gitignore
│
├── pages/                          # Streamlit 多頁面
│   ├── 1_📈_股票分析.py            # 個股 K 線分析
│   ├── 2_🤖_AI預測.py              # AI 模型訓練與比較
│   ├── 3_📊_策略回測.py            # 交易策略回測
│   └── 4_💰_虛擬交易.py            # Paper Trading
│
├── src/                            # 核心模組
│   ├── data/                       # 資料擷取 (yfinance)
│   ├── indicators/                 # 技術指標 (RSI/MACD/KD...)
│   ├── models/                     # AI 模型
│   │   ├── base.py                 # 共用工具
│   │   ├── lstm_model.py           # LSTM
│   │   ├── transformer_model.py    # Transformer
│   │   └── xgboost_model.py        # XGBoost
│   ├── backtest/                   # 回測引擎
│   │   ├── engine.py               # 向量化回測
│   │   └── strategies.py           # 5 種策略
│   ├── trading/                    # 虛擬交易
│   │   └── paper_trading.py        # SQLite 帳戶管理
│   └── utils/
│       └── charts.py               # Plotly 圖表
│
├── notebooks/
│   └── test_core.py                # 核心邏輯離線測試
│
└── data/                           # 資料快取與 SQLite (.gitignore)
```

---

## 🧠 技術架構

### 資料流程

```
yfinance API → parquet 快取 → pandas DataFrame
                                    ↓
              add_all_indicators() 加入 ~15 個技術指標
                                    ↓
                ┌─────────────────────────────────┐
                ↓                  ↓               ↓
            前端視覺化          AI 模型訓練     策略回測
            (Plotly K 線)    (LSTM/Trans/XGB)   (信號→績效)
                                    ↓
                             虛擬交易下單 (SQLite)
```

### AI 模型比較

| 模型 | 架構 | 優勢 |
|------|------|------|
| **LSTM** | 雙層 LSTM(64) + Dropout + FC | 經典時序模型，趨勢捕捉好 |
| **Transformer** | 2 層 Encoder + Self-Attention | 長期依賴、平行計算 |
| **XGBoost** | Gradient Boosting Tree | 快速、可解釋、強 baseline |

### 評估指標
- **RMSE / MAE** — 預測誤差
- **方向準確率** — 預測「漲/跌」方向是否正確 (對交易最重要)
- **總報酬率 / 年化報酬率** — 策略獲利能力
- **最大回撤 (MDD)** — 風險指標
- **Sharpe Ratio** — 風險調整後報酬

---

## 💰 台股交易成本 (內建於回測與虛擬交易)

| 費用 | 費率 | 說明 |
|------|------|------|
| 手續費 (買) | 0.1425% × 6 折 = **0.0855%** | 多數券商現行折扣 |
| 手續費 (賣) | 0.1425% × 6 折 = **0.0855%** | |
| 證交稅 (賣) | **0.3%** | 政府收 |
| 滑價 | 0.1% | 模擬實際成交價偏移 |

---

## ⚠️ 免責聲明

本平台僅供**學術研究與教學示範**使用，所有交易功能皆為虛擬模擬，
**不構成任何投資建議**。實際投資請審慎評估風險、諮詢合格理財顧問。

---

## 📚 修課資訊

- **課程**: 人工智慧
- **學期**: 碩一下
- **指導概念**: 監督式學習、時間序列、深度學習 (RNN/Transformer)、評估指標
- **應用領域**: 量化金融分析

## 🙏 致謝

- yfinance — 提供免費股價資料
- Streamlit — 讓 Python 直接生出網頁
- PyTorch — 深度學習框架
- 課程講義中的時間序列、監督式學習章節
