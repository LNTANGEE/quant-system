"""Risk filters and risk level scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_MIN_AMOUNT_FOR_SHORTLIST
from models.indicators import enrich_indicators, rolling_return
from utils.helpers import clamp, safe_div, to_float


def risk_level(score: float) -> str:
    if score >= 75:
        return "极高"
    if score >= 55:
        return "高"
    if score >= 30:
        return "中"
    return "低"


def assess_risk(
    quote: dict | None,
    daily_df: pd.DataFrame | None,
    fundamentals: dict | None = None,
) -> dict:
    quote = quote or {}
    fundamentals = fundamentals or {}
    name = str(quote.get("name") or fundamentals.get("name") or "")
    amount = to_float(quote.get("amount"), 0)
    price = to_float(quote.get("price"), np.nan)
    pct_chg = to_float(quote.get("pct_chg"), 0)
    volume_ratio = to_float(quote.get("volume_ratio"), 1)
    pe = to_float(quote.get("pe"), np.nan)
    score = 8.0
    flags: list[str] = []

    if "ST" in name.upper():
        score += 55
        flags.append("ST或退市风险标记")
    if pd.isna(price) or price <= 0:
        score += 50
        flags.append("停牌或实时价格缺失")
    if amount and amount < DEFAULT_MIN_AMOUNT_FOR_SHORTLIST:
        score += 12
        flags.append("成交额低于短线榜默认门槛")
    if amount and amount < 100_000_000:
        score += 10
        flags.append("流动性偏弱")
    if abs(pct_chg) >= 9.5 and volume_ratio > 1.8:
        score += 8
        flags.append("涨跌停附近且量能放大")

    if daily_df is not None and not daily_df.empty:
        enriched = enrich_indicators(daily_df)
        row = enriched.iloc[-1]
        close = to_float(row.get("close"), price if not pd.isna(price) else 0)
        ma20 = to_float(row.get("ma20"), close)
        ma60 = to_float(row.get("ma60"), close)
        vol_ma20 = to_float(row.get("vol_ma20"), 0)
        latest_volume = to_float(row.get("volume"), 0)
        five_day_return = rolling_return(enriched, 5)
        twenty_day_return = rolling_return(enriched, 20)
        if five_day_return > 25 and latest_volume > vol_ma20 * 1.3:
            score += 20
            flags.append("近5日涨幅较大且放量，标记高位波动风险")
        if twenty_day_return > 45:
            score += 12
            flags.append("近20日涨幅偏大，追高风险上升")
        if ma20 > 0 and close > ma20 * 1.18:
            score += 12
            flags.append("股价短期偏离MA20较大")
        if ma60 > 0 and close < ma60 and to_float(row.get("ma5"), close) < ma20:
            score += 10
            flags.append("跌破中期趋势且承接不足")
        if close < to_float(row.get("bb_lower"), close * 0.95) and pct_chg < 0:
            score += 6
            flags.append("跌破布林下轨，短期波动风险较高")

    debt_ratio = to_float(fundamentals.get("debt_ratio"), np.nan)
    net_profit_yoy = to_float(fundamentals.get("net_profit_yoy"), np.nan)
    if pd.notna(debt_ratio) and debt_ratio > 70:
        score += 10
        flags.append("负债率偏高")
    if pd.notna(net_profit_yoy) and net_profit_yoy < -30:
        score += 12
        flags.append("净利润同比下滑较大")
    if pd.notna(pe) and (pe < 0 or pe > 120):
        score += 8
        flags.append("估值指标异常或较高")

    final_score = clamp(score)
    level = risk_level(final_score)
    return {
        "risk_score": round(final_score, 1),
        "risk_level": level,
        "flags": flags or ["未触发明显高风险过滤项"],
        "filter_out_shortlist": level in {"高", "极高"} or "ST或退市风险标记" in flags,
        "forbid_t": level == "极高" or "停牌或实时价格缺失" in flags,
        "explain": "风险模型覆盖ST/停牌、流动性、短期过热、趋势破位、估值和基础财务异常等公开数据项。",
    }
