"""
核心邏輯離線測試 (不需網路、不需 yfinance/torch)
================================================
用合成資料驗證技術指標、回測引擎、虛擬交易帳戶都能正常運作
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n_days: int = 250, seed: int = 42) -> pd.DataFrame:
    """製造合成 OHLCV 資料 (隨機漫步 + 趨勢)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    # 起始價 100，每日漲跌服從常態分佈
    rets = rng.normal(0.0005, 0.015, n_days)
    close = 100 * np.exp(rets.cumsum())
    high = close * (1 + rng.uniform(0.001, 0.02, n_days))
    low = close * (1 - rng.uniform(0.001, 0.02, n_days))
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.005, n_days - 1))
    volume = rng.integers(10_000, 100_000, n_days).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_technical_indicators():
    print("\n[Test 1] 技術指標計算")
    from src.indicators import add_all_indicators

    df = make_synthetic_ohlcv(200)
    feat = add_all_indicators(df)

    # 檢查關鍵欄位
    expected = ["SMA_5", "SMA_20", "EMA_12", "RSI_14",
                "MACD", "MACD_signal", "K", "D",
                "BB_upper", "BB_lower", "ATR_14", "OBV"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"缺少欄位: {missing}"

    # 檢查 RSI 範圍
    rsi = feat["RSI_14"].dropna()
    assert rsi.between(0, 100).all(), "RSI 必須在 0~100 之間"

    # 檢查 KD 範圍
    k = feat["K"].dropna()
    assert k.between(0, 100).all(), f"K 必須在 0~100, got [{k.min()}, {k.max()}]"

    # 布林通道: 上 > 中 > 下
    valid = feat[["BB_upper", "BB_middle", "BB_lower"]].dropna()
    assert (valid["BB_upper"] >= valid["BB_middle"]).all()
    assert (valid["BB_middle"] >= valid["BB_lower"]).all()

    print(f"  ✓ 共 {len(feat.columns)} 個欄位，{len(feat)} 筆資料")
    print(f"  ✓ RSI 範圍: [{rsi.min():.2f}, {rsi.max():.2f}]")
    print(f"  ✓ KD 範圍: K∈[{k.min():.2f}, {k.max():.2f}]")
    print("  ✅ 通過")
    return feat


def test_strategies(feat):
    print("\n[Test 2] 交易策略信號產生")
    from src.backtest import (
        MovingAverageCrossStrategy, RSIStrategy, MACDStrategy, BollingerBandStrategy,
    )

    strategies = [
        MovingAverageCrossStrategy(5, 20),
        RSIStrategy(),
        MACDStrategy(),
        BollingerBandStrategy(),
    ]
    for s in strategies:
        sig = s.generate_signals(feat)
        n_buy = (sig == 1).sum()
        n_sell = (sig == -1).sum()
        # 信號值只能是 -1, 0, 1
        assert sig.isin([-1, 0, 1]).all(), f"{s.name} 產生非法信號"
        print(f"  ✓ {s.name:20s}  買 {n_buy:3d}  賣 {n_sell:3d}")
    print("  ✅ 通過")


def test_backtest(feat):
    print("\n[Test 3] 回測引擎")
    from src.backtest import run_backtest, MovingAverageCrossStrategy

    strategy = MovingAverageCrossStrategy(5, 20)
    result = run_backtest(feat, strategy, initial_capital=1_000_000)

    # 檢查資金曲線長度等於資料天數
    assert len(result.equity_curve) == len(feat), \
        f"資金曲線長度不對: {len(result.equity_curve)} vs {len(feat)}"

    # 檢查初始值
    assert abs(result.equity_curve.iloc[0] - 1_000_000) < 1, \
        f"第一天應該是初始資金，got {result.equity_curve.iloc[0]}"

    # 檢查報酬率合理性
    assert -1 <= result.total_return <= 5, "總報酬率不合理"
    # max drawdown 應該 <= 0
    assert result.max_drawdown <= 0, "最大回撤應該是負數或 0"

    print(f"  ✓ 總報酬率: {result.total_return*100:.2f}%")
    print(f"  ✓ 年化報酬: {result.annualized_return*100:.2f}%")
    print(f"  ✓ 最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"  ✓ Sharpe: {result.sharpe_ratio:.3f}")
    print(f"  ✓ 交易次數: {result.n_trades}")
    print(f"  ✓ 勝率: {result.win_rate*100:.2f}%")
    print(f"  ✓ 買進持有對照: {result.buy_and_hold_return*100:.2f}%")
    print("  ✅ 通過")
    return result


def test_paper_trading():
    print("\n[Test 4] 虛擬交易帳戶")
    from src.trading import PaperTradingAccount

    # 用獨立帳戶名稱避免影響其他資料
    acc = PaperTradingAccount.get_or_create("test_account", initial_capital=100_000)
    acc.reset()

    # 買進
    r = acc.buy("2330", price=600, shares=100, note="測試")
    assert r["ok"], r["msg"]
    print(f"  ✓ 買進 100 股 @600 → {r['msg']}")

    # 持倉檢查
    pos = acc.positions()
    assert len(pos) == 1
    assert pos.iloc[0]["shares"] == 100
    print(f"  ✓ 持倉: {pos.iloc[0]['symbol']} {pos.iloc[0]['shares']}股 均價{pos.iloc[0]['avg_price']:.2f}")

    # 賣出
    r2 = acc.sell("2330", price=650, shares=100, note="測試獲利了結")
    assert r2["ok"], r2["msg"]
    assert r2["pnl"] > 0, f"應該獲利 (買 600 賣 650)，但 pnl={r2['pnl']}"
    print(f"  ✓ 賣出 100 股 @650 → 損益 {r2['pnl']:+,.0f}")

    # 持倉應該歸零
    pos = acc.positions()
    assert len(pos) == 0
    print("  ✓ 持倉歸零")

    # 試試資金不足
    r3 = acc.buy("2330", price=600, shares=10000, note="超額")
    assert not r3["ok"]
    print(f"  ✓ 資金不足偵測: {r3['msg']}")

    # 試試持股不足
    r4 = acc.sell("2330", price=600, shares=100)
    assert not r4["ok"]
    print(f"  ✓ 持股不足偵測: {r4['msg']}")

    # 清理測試資料
    acc.reset()
    print("  ✅ 通過")


def test_supervised_data_prep():
    print("\n[Test 5] 監督式資料準備")
    from src.indicators import add_all_indicators
    from src.models.base import prepare_supervised_data

    df = make_synthetic_ohlcv(300)
    feat = add_all_indicators(df)

    feature_cols = ["Open", "High", "Low", "Close", "Volume",
                    "SMA_5", "RSI_14", "MACD"]
    d = prepare_supervised_data(feat, feature_cols=feature_cols,
                                 lookback=20, horizon=1, test_ratio=0.2)

    # 形狀檢查
    assert d["X_train"].ndim == 3
    assert d["X_train"].shape[1] == 20  # lookback
    assert d["X_train"].shape[2] == len(feature_cols)
    assert d["X_train_flat"].shape[1] == 20 * len(feature_cols)
    print(f"  ✓ X_train shape (3D): {d['X_train'].shape}")
    print(f"  ✓ X_train_flat shape (2D): {d['X_train_flat'].shape}")
    print(f"  ✓ 訓練/測試比例: {len(d['X_train'])}/{len(d['X_test'])}")

    # 範圍 (應該都被 normalize 到 0~1)
    assert d["X_train"].min() >= -0.01 and d["X_train"].max() <= 1.01
    print("  ✅ 通過")


if __name__ == "__main__":
    print("=" * 60)
    print("核心邏輯離線測試")
    print("=" * 60)

    feat = test_technical_indicators()
    test_strategies(feat)
    test_backtest(feat)
    test_paper_trading()
    test_supervised_data_prep()

    print("\n" + "=" * 60)
    print("✅ 全部測試通過")
    print("=" * 60)
