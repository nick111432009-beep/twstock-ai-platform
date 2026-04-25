"""
白話解讀引擎
============
把冷冰冰的技術指標數字 / AI 預測結果，翻譯成新手看得懂的中文說明。

每個解讀函式統一回傳 dict，前端可直接顯示:
    {
        "status": "🟢 看多" / "🟡 觀望" / "🔴 警告",
        "headline": 一句話結論 (粗體標題用),
        "what":    "這指標在算什麼" (背景知識),
        "now":     "目前的數字代表什麼" (現況解讀),
        "advice":  "新手該怎麼做" (操作建議),
        "warn":    "需要注意的例外或陷阱" (風險提醒),
    }

設計原則:
    - 完全口語化，不用艱澀術語
    - 每個建議都帶有「但是 / 例外」，避免新手以為指標是聖杯
    - 不給絕對答案 ("一定會漲")，只給機率與情境
"""

from __future__ import annotations

from typing import Optional

# ============================================================
# RSI — 相對強弱指數 (Relative Strength Index)
# ============================================================
def interpret_rsi(value: float) -> dict:
    """RSI 的白話解讀"""
    what = (
        "RSI(相對強弱指數)用 0~100 的分數，"
        "告訴你「最近一段時間裡，買盤跟賣盤誰打贏比較多」。"
        "可以想像成『買vs賣的拳擊得分板』。"
    )

    if value >= 80:
        return {
            "status": "🔴 嚴重過熱",
            "headline": f"RSI = {value:.1f}，買盤打到瘋狂",
            "what": what,
            "now": (
                f"目前 RSI 高達 {value:.1f}，遠超過 70 的『過熱線』。"
                "代表最近股價漲得太兇、太多人在追買，"
                "**短期內回檔修正的機率很高**。"
            ),
            "advice": (
                "**新手強烈建議:不要追高買進**。"
                "如果你已經持有，這是考慮**部分獲利了結**的時候。"
                "等 RSI 回到 50 附近再考慮新進場會比較安全。"
            ),
            "warn": (
                "⚠️ 但要注意：超強勢股(例如某段時期的 NVIDIA、台積電)"
                "可以**連續好幾週維持 RSI > 80**，"
                "如果你看到大趨勢仍向上、新聞利多不斷，光靠 RSI 一個指標斷言『要跌』可能會誤判。"
            ),
        }
    elif value >= 70:
        return {
            "status": "🟡 過熱警告",
            "headline": f"RSI = {value:.1f}，進入過熱區",
            "what": what,
            "now": (
                f"目前 RSI {value:.1f} 已超過 70，"
                "代表股價最近漲勢強勁，買盤略嫌過熱。"
                "歷史上 RSI > 70 後 5 天，股價下跌的機率約 55-60%。"
            ),
            "advice": (
                "**保持警覺，不適合大幅加碼**。"
                "可以等 RSI 回落到 50 附近再進場，或設好停損點再持有。"
            ),
            "warn": (
                "⚠️ 強勢趨勢中 RSI 可能持續偏高，"
                "建議搭配 MACD 看『動能是否還在加速』再決定。"
            ),
        }
    elif value >= 50:
        return {
            "status": "🟢 多方主導",
            "headline": f"RSI = {value:.1f}，買盤稍佔上風",
            "what": what,
            "now": (
                f"目前 RSI {value:.1f} 在 50~70 之間，"
                "屬於**健康的多頭區**。買盤略強、但還沒過熱。"
            ),
            "advice": (
                "**這是相對安全的進場區間**。"
                "如果其他指標(例如 MACD、均線)同向看多，可以考慮買進或續抱。"
            ),
            "warn": "RSI 單一指標不足以決策，最好配合趨勢確認(均線方向)再行動。",
        }
    elif value >= 30:
        return {
            "status": "🟡 空方主導",
            "headline": f"RSI = {value:.1f}，賣盤稍佔上風",
            "what": what,
            "now": (
                f"目前 RSI {value:.1f} 在 30~50 之間，"
                "代表近期賣盤比較強，但還沒到超賣。"
            ),
            "advice": (
                "**新手保守作法:觀望為主**。"
                "等明確的轉折信號出現(RSI 突破 50、MACD 黃金交叉)再進場。"
            ),
            "warn": "如果是長期下跌趨勢中，RSI 可能長時間在 30~50 徘徊不反彈，要小心『跌跌不休』。",
        }
    elif value >= 20:
        return {
            "status": "🟢 超賣機會",
            "headline": f"RSI = {value:.1f}，跌過頭了",
            "what": what,
            "now": (
                f"目前 RSI {value:.1f} 已低於 30 的『超賣線』，"
                "代表股價最近跌得太兇、賣盤太悲觀。"
                "**歷史上 RSI < 30 後反彈的機率較高**。"
            ),
            "advice": (
                "**對逢低布局者來說是觀察機會**。"
                "但**不要急著一次買滿**，可以**分批買進**(例如分 3 次)，降低買在低點還繼續跌的風險。"
            ),
            "warn": (
                "⚠️ 在重大利空(例如財報爆雷、產業崩盤)中，"
                "RSI 可以連續幾週維持 < 30，所謂『鈍化』。"
                "務必確認沒有重大利空，再進場。"
            ),
        }
    else:  # < 20
        return {
            "status": "🔴 嚴重超賣",
            "headline": f"RSI = {value:.1f}，極端恐慌",
            "what": what,
            "now": (
                f"RSI {value:.1f} 是極端低位。"
                "市場處於恐慌賣壓，**通常離反彈不遠**。"
            ),
            "advice": (
                "**有經驗的投資人會在此分批進場**，"
                "但對新手來說，這種極端情況通常伴隨重大利空，**請先搞清楚為什麼跌這麼慘**再決定。"
            ),
            "warn": "極端值往往伴隨重大事件(如金融海嘯、公司倒閉)，光看技術指標撈底很危險。",
        }


# ============================================================
# MACD — 指數平滑異同移動平均線
# ============================================================
def interpret_macd(macd: float, signal: float, histogram: float) -> dict:
    """MACD 的白話解讀"""
    what = (
        "MACD 是股價的『加速度計』。"
        "它由兩條線組成:**快線(DIF)** 和 **慢線(MACD signal)**，"
        "兩條線交叉時就是『動能轉變』的信號。"
        "下方還有柱狀圖代表兩線距離。"
    )

    diff = macd - signal
    above_zero = macd > 0 and signal > 0
    below_zero = macd < 0 and signal < 0

    # 判斷交叉狀態
    if diff > 0 and abs(diff) < abs(macd) * 0.1:
        crossover = "剛剛黃金交叉(快線剛上穿慢線)"
        status = "🟢 強烈看多"
        advice = (
            "**這是教科書級的買進訊號**。"
            "當快線剛剛從下方穿過慢線，代表動能正在從空轉多。"
            "新手可以**小量試單**，並且設好停損點(跌破最近低點就出場)。"
        )
    elif diff > 0:
        crossover = "快線在慢線之上(多頭排列)"
        status = "🟢 偏多" if above_zero else "🟡 弱多"
        advice = (
            "**多頭排列代表動能向上**，可考慮持有或加碼。"
            "如果同時 MACD > 0(在零軸之上)就是更強的訊號。"
        )
    elif diff < 0 and abs(diff) < abs(macd) * 0.1:
        crossover = "剛剛死亡交叉(快線剛下穿慢線)"
        status = "🔴 強烈看空"
        advice = (
            "**這是教科書級的賣出訊號**。"
            "如果你有持股，**現在就是減碼或出場的時機**。"
            "想做空的話可以順勢，但新手不建議放空。"
        )
    else:
        crossover = "快線在慢線之下(空頭排列)"
        status = "🔴 偏空" if below_zero else "🟡 弱空"
        advice = (
            "**動能向下，不適合買進**。"
            "如果有持股應該考慮減碼，等 MACD 翻多再進場。"
        )

    now = (
        f"目前 **MACD = {macd:.3f}**, **Signal = {signal:.3f}**, **柱狀圖 = {histogram:.3f}**。\n\n"
        f"位置: {'零軸之上(整體偏多)' if above_zero else '零軸之下(整體偏空)' if below_zero else '零軸附近(中性盤整)'}\n\n"
        f"狀態: **{crossover}**"
    )

    return {
        "status": status,
        "headline": crossover,
        "what": what,
        "now": now,
        "advice": advice,
        "warn": (
            "⚠️ MACD 屬於『落後指標』，等到交叉發生時，行情通常已經走了一段。"
            "在橫向震盪盤中容易出現『假交叉』反覆騙線，要搭配趨勢判斷使用。"
        ),
    }


# ============================================================
# KD — 隨機指標 (Stochastic Oscillator)
# ============================================================
def interpret_kd(k: float, d: float) -> dict:
    """KD 的白話解讀"""
    what = (
        "KD(隨機指標)跟 RSI 類似，也是 0~100 的分數，"
        "但**反應比 RSI 更快、更敏感**。"
        "由 K 線(快線)和 D 線(慢線)組成，"
        "兩線交叉就是進出場訊號。"
    )

    diff = k - d

    if k > 80 and d > 80:
        status = "🔴 過熱"
        headline = f"K={k:.1f}, D={d:.1f}，雙雙進入超買區"
        advice = (
            "**短期內回檔機率高**，新手**避免追高**。"
            "如果有持股可以考慮**部分獲利了結**。"
        )
    elif k < 20 and d < 20:
        status = "🟢 超賣"
        headline = f"K={k:.1f}, D={d:.1f}，雙雙進入超賣區"
        advice = (
            "**短期反彈機率較高**，可以開始**觀察進場機會**。"
            "等 K 線從下方穿過 D 線(黃金交叉)再進場更安全。"
        )
    elif diff > 0 and k > 50:
        status = "🟢 偏多"
        headline = f"K={k:.1f} > D={d:.1f}，多頭格局"
        advice = "K 線在 D 線之上代表多方主導，可考慮持有。"
    elif diff < 0 and k < 50:
        status = "🔴 偏空"
        headline = f"K={k:.1f} < D={d:.1f}，空頭格局"
        advice = "K 線在 D 線之下，動能向下，新手不建議買進。"
    else:
        status = "🟡 中性"
        headline = f"K={k:.1f}, D={d:.1f}，無明確方向"
        advice = "目前 KD 無明確訊號，等趨勢明朗再行動。"

    return {
        "status": status,
        "headline": headline,
        "what": what,
        "now": (
            f"目前 **K = {k:.1f}**, **D = {d:.1f}**。\n\n"
            f"K-D 差值 = {diff:+.1f} ({'K 在 D 上方' if diff > 0 else 'K 在 D 下方'})"
        ),
        "advice": advice,
        "warn": (
            "⚠️ KD 在強勢趨勢中容易**鈍化**(連續好幾天 > 80 或 < 20)。"
            "鈍化時 KD 訊號失效，要回去看均線方向。"
        ),
    }


# ============================================================
# 均線 (SMA) — 移動平均線
# ============================================================
def interpret_sma(close: float, sma_short: float, sma_long: float,
                  short_label: str = "SMA_5", long_label: str = "SMA_20") -> dict:
    """均線交叉的白話解讀"""
    what = (
        f"移動平均線 = 過去 N 天收盤價的平均值，能濾掉每日雜訊看真正趨勢。"
        f"**{short_label}**(短期均線)反應快，**{long_label}**(長期均線)反應慢。"
        "兩線交叉就是經典買賣訊號。"
    )

    above_short = close > sma_short
    above_long = close > sma_long
    short_above_long = sma_short > sma_long

    if above_short and above_long and short_above_long:
        status = "🟢 多頭排列"
        headline = "完美多頭結構"
        now_status = (
            f"✅ 股價({close:.2f}) > {short_label}({sma_short:.2f}) > {long_label}({sma_long:.2f})\n\n"
            "三者形成『**多頭排列**』，是教科書最理想的多頭格局。"
        )
        advice = (
            "**強勢上漲趨勢**，可以**順勢操作**(買進或續抱)。"
            "停損可以設在 {} 附近 — 跌破就出場。".format(short_label)
        )
    elif not above_short and not above_long and not short_above_long:
        status = "🔴 空頭排列"
        headline = "完美空頭結構"
        now_status = (
            f"❌ 股價({close:.2f}) < {short_label}({sma_short:.2f}) < {long_label}({sma_long:.2f})\n\n"
            "三者形成『**空頭排列**』，趨勢明確向下。"
        )
        advice = (
            "**不適合買進!** 即使股價看起來很便宜，下跌趨勢中接刀子很危險。"
            "等 {} 突破 {}(黃金交叉)再考慮進場。".format(short_label, long_label)
        )
    elif short_above_long and not above_short:
        status = "🟡 多頭回檔"
        headline = "多頭趨勢中的拉回"
        now_status = (
            f"短期均線雖然還在長期均線之上，但**股價跌破 {short_label}**。\n\n"
            "屬於**多頭趨勢中的回檔**，要觀察會不會持續走弱。"
        )
        advice = (
            "**不急著買也不急著賣**。"
            "如果跌到 {} 附近止跌反彈，可以是進場機會;"
            "若連 {} 也跌破，就要小心趨勢反轉。".format(long_label, long_label)
        )
    else:
        status = "🟡 整理盤"
        headline = "盤整或趨勢不明"
        now_status = (
            f"股價({close:.2f}), {short_label}({sma_short:.2f}), {long_label}({sma_long:.2f}) "
            "互相糾纏，沒有明確方向。"
        )
        advice = (
            "**整理盤對新手是最危險的**，容易左右挨打。"
            "建議**等趨勢明朗**(均線打開、股價站上長期均線)再進場。"
        )

    return {
        "status": status,
        "headline": headline,
        "what": what,
        "now": now_status,
        "advice": advice,
        "warn": (
            "⚠️ 均線是『**落後指標**』 — 等到黃金交叉發生，行情通常已經走了 1~2 週。"
            "適合做波段不適合短線，且在橫盤震盪期容易頻繁假訊號。"
        ),
    }


# ============================================================
# 布林通道
# ============================================================
def interpret_bollinger(close: float, upper: float, middle: float, lower: float) -> dict:
    """布林通道的白話解讀"""
    what = (
        "布林通道由 3 條線組成：**上軌 / 中軌 / 下軌**。"
        "上下軌道是中軌的 ±2 個標準差，"
        "**統計上股價有 95% 機率會落在通道內**。"
        "碰到上軌通常是『超買』，碰到下軌通常是『超賣』。"
    )
    width = upper - lower
    pct_position = (close - lower) / width if width > 0 else 0.5

    if close >= upper:
        status = "🔴 觸及上軌"
        headline = "股價貼到布林通道上軌，超買"
        advice = (
            "**短期回檔機率高**，新手不建議追高。"
            "如果有持股，可以考慮獲利了結部分。"
        )
    elif close <= lower:
        status = "🟢 觸及下軌"
        headline = "股價貼到布林通道下軌，超賣"
        advice = (
            "**短期反彈機率較高**，可以分批布局。"
            "但要先確認是『正常回檔』還是『重大利空』再進場。"
        )
    elif close > middle:
        status = "🟢 偏多區"
        headline = f"股價位於通道上半部 ({pct_position*100:.0f}% 位置)"
        advice = "在中軌之上是相對安全的多頭格局，可考慮持有。"
    else:
        status = "🟡 偏空區"
        headline = f"股價位於通道下半部 ({pct_position*100:.0f}% 位置)"
        advice = "在中軌之下動能偏弱，等突破中軌再進場較安全。"

    return {
        "status": status,
        "headline": headline,
        "what": what,
        "now": (
            f"上軌 {upper:.2f} | 中軌 {middle:.2f} | 下軌 {lower:.2f} | 現價 {close:.2f}\n\n"
            f"位置: 通道內 {pct_position*100:.0f}% (越接近 100% 越靠近上軌)"
        ),
        "advice": advice,
        "warn": (
            "⚠️ 布林通道在**強勢趨勢中**會持續沿著上軌或下軌走，"
            "這時候『碰上軌就賣』反而會錯過大波段，要結合趨勢判斷。"
        ),
    }


# ============================================================
# AI 預測解讀 (給頁面 2 用)
# ============================================================
def interpret_ai_prediction(
    current_price: float,
    predicted_price: float,
    direction_accuracy: float,
    model_name: str = "AI 模型",
) -> dict:
    """AI 預測值的白話翻譯"""
    change = predicted_price - current_price
    change_pct = change / current_price * 100

    # 信心等級
    if direction_accuracy >= 0.62:
        confidence_level = "中高"
        confidence_emoji = "💪"
        confidence_note = "在歷史測試中表現良好，可以參考"
    elif direction_accuracy >= 0.55:
        confidence_level = "中等"
        confidence_emoji = "👌"
        confidence_note = "略勝亂猜，僅供參考"
    else:
        confidence_level = "偏低"
        confidence_emoji = "🤷"
        confidence_note = "預測能力跟亂猜差不多，**不建議據此交易**"

    # 變化幅度
    abs_pct = abs(change_pct)
    if abs_pct < 0.5:
        magnitude = "幾乎不動"
        signal = "🟡 觀望"
        action_advice = "AI 認為明天波動很小，**不適合短線進出**(交易成本就要 0.57%)。"
    elif abs_pct < 1.5:
        magnitude = "小幅" + ("上漲" if change > 0 else "下跌")
        signal = "🟢 偏多" if change > 0 else "🔴 偏空"
        action_advice = (
            f"AI 預期**小幅{'上漲' if change > 0 else '下跌'}**，"
            f"變化幅度 {abs_pct:.2f}% 不大，**不適合單獨作為短線進出依據**。"
            "建議搭配技術指標(RSI、MACD)同向確認再行動。"
        )
    elif abs_pct < 3.0:
        magnitude = "中度" + ("上漲" if change > 0 else "下跌")
        signal = "🟢 看多" if change > 0 else "🔴 看空"
        action_advice = (
            f"AI 預期**中度{'上漲' if change > 0 else '下跌'}**，幅度 {abs_pct:.2f}%。"
            "可以**將此作為方向參考**，但**不要 all-in**(資金分批進場)。"
        )
    else:
        magnitude = "大幅" + ("上漲" if change > 0 else "下跌")
        signal = "🟢 強烈看多" if change > 0 else "🔴 強烈看空"
        action_advice = (
            f"AI 預期**大幅{'上漲' if change > 0 else '下跌'}** (>3%)。"
            "⚠️ 這種大幅預測通常**不夠準確**(歷史上 AI 對極端值的預測誤差最大)，"
            "請務必搭配新聞、基本面確認再決策。"
        )

    return {
        "signal": signal,
        "headline": f"{model_name} 預期明日{magnitude}",
        "predicted_price": predicted_price,
        "current_price": current_price,
        "change": change,
        "change_pct": change_pct,
        "magnitude": magnitude,
        "confidence_level": confidence_level,
        "confidence_emoji": confidence_emoji,
        "confidence_note": confidence_note,
        "direction_accuracy": direction_accuracy,
        "action_advice": action_advice,
        "general_warn": (
            "⚠️ **重要提醒**:\n\n"
            "1. **AI 預測 ≠ 一定準確**。即使方向準確率 60%，仍有 40% 會錯。\n\n"
            "2. **不要 all-in**。把 AI 當『會看歷史的學長』，給你方向參考，"
            "不要替你決策。\n\n"
            "3. **過去績效不代表未來**。模型在 2024 年表現好，"
            "不保證 2026 年也好(市場結構可能改變)。\n\n"
            "4. **永遠設停損**。即使再有信心，也要事先決定『跌多少就出場』。"
        ),
    }
