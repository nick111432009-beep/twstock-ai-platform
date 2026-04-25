"""
互動式圖表工具 (Plotly)
=======================
所有頁面共用的繪圖函式，全部使用 Plotly 確保互動性與美觀度。
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 主題色 (與 Streamlit 深色主題搭配)
COLOR_UP = "#26A69A"      # 綠 (台股慣例 — 紅漲綠跌; 但國際慣例綠漲，這裡用國際慣例)
COLOR_DOWN = "#EF5350"
COLOR_PRIMARY = "#FF4B4B"
COLOR_SECONDARY = "#42A5F5"


def candlestick_chart(
    df: pd.DataFrame,
    title: str = "K 線圖",
    show_volume: bool = True,
    overlays: dict[str, str] | None = None,
) -> go.Figure:
    """
    K 線圖 (上方 K 線，下方成交量)

    參數:
        overlays: {欄位名: 顯示名稱} 想疊在 K 線上的線 (例如均線、布林通道)
    """
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights,
        subplot_titles=([title, "成交量"] if show_volume else [title]),
    )

    # K 線
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="K 線",
            increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
        ),
        row=1, col=1,
    )

    # 疊加線
    if overlays:
        for col, label in overlays.items():
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col], name=label,
                        line=dict(width=1.5),
                    ),
                    row=1, col=1,
                )

    # 成交量
    if show_volume:
        colors = [
            COLOR_UP if c >= o else COLOR_DOWN
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="量", marker_color=colors,
                   showlegend=False),
            row=2, col=1,
        )

    fig.update_layout(
        height=600 if show_volume else 450,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def indicator_chart(df: pd.DataFrame, indicator: str) -> go.Figure:
    """
    技術指標子圖 (RSI / KD / MACD 等)
    """
    fig = go.Figure()
    if indicator == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI(14)",
                                 line=dict(color=COLOR_PRIMARY)))
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,80,80,0.5)",
                      annotation_text="超買 70")
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(80,255,80,0.5)",
                      annotation_text="超賣 30")
        fig.update_yaxes(range=[0, 100])
    elif indicator == "MACD":
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                 line=dict(color=COLOR_PRIMARY)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                                 line=dict(color=COLOR_SECONDARY)))
        colors = ["#26A69A" if v >= 0 else "#EF5350" for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist",
                             marker_color=colors))
    elif indicator == "KD":
        fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K",
                                 line=dict(color=COLOR_PRIMARY)))
        fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D",
                                 line=dict(color=COLOR_SECONDARY)))
        fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,80,80,0.4)")
        fig.add_hline(y=20, line_dash="dash", line_color="rgba(80,255,80,0.4)")
        fig.update_yaxes(range=[0, 100])

    fig.update_layout(
        title=indicator,
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def prediction_chart(
    test_dates,
    y_true,
    y_pred,
    model_name: str,
) -> go.Figure:
    """AI 預測 vs 實際 對照圖"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_true, name="實際", line=dict(color=COLOR_PRIMARY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_pred, name=f"{model_name} 預測",
        line=dict(color=COLOR_SECONDARY, width=2, dash="dot"),
    ))
    fig.update_layout(
        title=f"{model_name} - 預測 vs 實際",
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="日期", yaxis_title="股價 (元)",
    )
    return fig


def equity_curve_chart(equity_curve: pd.Series, initial: float, bnh_curve: pd.Series | None = None) -> go.Figure:
    """資金曲線圖 (含買進持有對照)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        name="策略資金", line=dict(color=COLOR_PRIMARY, width=2),
    ))
    if bnh_curve is not None:
        fig.add_trace(go.Scatter(
            x=bnh_curve.index, y=bnh_curve.values,
            name="買進持有", line=dict(color="#888", width=1.5, dash="dash"),
        ))
    fig.add_hline(y=initial, line_color="rgba(255,255,255,0.3)",
                  annotation_text="初始資金")
    fig.update_layout(
        title="資金曲線",
        template="plotly_dark",
        height=400,
        xaxis_title="日期", yaxis_title="資產 (元)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    """回撤曲線 (-%)"""
    rolling_max = equity_curve.cummax()
    dd = (equity_curve - rolling_max) / rolling_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown",
        fill="tozeroy", line=dict(color=COLOR_DOWN),
    ))
    fig.update_layout(
        title="最大回撤 (%)",
        template="plotly_dark",
        height=300,
        xaxis_title="日期", yaxis_title="Drawdown (%)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
