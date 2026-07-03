"""Proxy main-force behavior score.

Public AKShare data does not provide reliable real-time large-order flow in
the MVP, so this model uses volume, VWAP, intraday structure and relative
strength as a proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.indicators import add_intraday_vwap, calculate_intraday_vwap
from utils.helpers import clamp, safe_div, to_float


def _label(score: float) -> str:
    if score >= 75:
        return "强承接"
    if score >= 60:
        return "偏强"
    if score >= 45:
        return "震荡"
    if score >= 30:
        return "偏弱"
    return "风险较大"


def score_main_force(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None,
    quote: dict | None,
    sector_context: dict | None = None,
) -> dict:
    quote = quote or {}
    sector_context = sector_context or {}
    if daily_df is None or daily_df.empty:
        return {
            "score": 45,
            "label": "数据不足",
            "absorption_score": 45,
            "distribution_risk": 45,
            "tail_risk": 45,
            "explain": "K线不足，主力行为仅按中性偏谨慎处理。",
        }
    latest = daily_df.iloc[-1]
    prev = daily_df.iloc[-2] if len(daily_df) >= 2 else latest
    price = to_float(quote.get("price"), to_float(latest.get("close")))
    day_low = to_float(quote.get("low"), to_float(latest.get("low")))
    day_high = to_float(quote.get("high"), to_float(latest.get("high")))
    day_open = to_float(quote.get("open"), to_float(latest.get("open")))
    prev_low = to_float(prev.get("low"), day_low)
    pct_chg = to_float(quote.get("pct_chg"), to_float(latest.get("pct_chg"), 0))
    volume_ratio = to_float(quote.get("volume_ratio"), 1)
    vwap = calculate_intraday_vwap(minute_df) if minute_df is not None else np.nan
    relative_strength = to_float(sector_context.get("relative_strength"), 0)
    sector_pct = to_float(sector_context.get("sector_pct_chg"), 0)

    intraday_range = max(day_high - day_low, 0.01)
    price_position = safe_div(price - day_low, intraday_range)

    absorption = 35.0
    absorption += 15 if pd.notna(vwap) and price >= vwap else 0
    absorption += 12 if day_low <= prev_low and price > prev_low else 0
    absorption += 10 if price_position >= 0.55 else 4 if price_position >= 0.35 else 0
    absorption += 10 if 1 <= volume_ratio <= 2.5 and pct_chg >= -1.5 else 4 if volume_ratio > 0.8 else 0
    absorption += 10 if sector_pct < 0 and pct_chg > sector_pct else 0
    absorption += 8 if relative_strength > 1 else 0
    absorption = clamp(absorption)

    distribution = 15.0
    distribution += 18 if safe_div(day_high - price, max(day_high, 0.01)) * 100 > 2.5 else 0
    distribution += 16 if pd.notna(vwap) and price < vwap else 0
    distribution += 14 if volume_ratio > 2 and pct_chg < 1 else 0
    distribution += 12 if price < day_open and pct_chg < 0 else 0
    distribution += 10 if relative_strength < -1 else 0
    distribution += 8 if price_position < 0.25 else 0
    distribution = clamp(distribution)

    tail_risk = 10.0
    if minute_df is not None and not minute_df.empty and "datetime" in minute_df.columns:
        minute = add_intraday_vwap(minute_df)
        tail = minute[minute["datetime"].dt.time >= pd.to_datetime("14:30").time()]
        if len(tail) >= 3:
            tail_return = safe_div(tail["close"].iloc[-1] - tail["close"].iloc[0], tail["close"].iloc[0]) * 100
            tail_volume = to_float(tail["volume"].sum())
            avg_tail_volume = to_float(minute["volume"].tail(len(tail) * 2).mean() * len(tail), tail_volume)
            tail_risk += 22 if tail_return < -1 else 10 if tail_return < -0.4 else 0
            tail_risk += 12 if tail_volume > avg_tail_volume * 1.2 and tail_return < 0 else 0
            tail_risk += 10 if pd.notna(vwap) and tail["close"].iloc[-1] < vwap else 0
    tail_risk += 8 if sector_pct < -1.5 and pct_chg < 0 else 0
    tail_risk = clamp(tail_risk)

    score = clamp(55 + absorption * 0.35 - distribution * 0.30 - tail_risk * 0.15)
    return {
        "score": round(score, 1),
        "label": _label(score),
        "absorption_score": round(absorption, 1),
        "distribution_risk": round(distribution, 1),
        "tail_risk": round(tail_risk, 1),
        "explain": "基础版使用承接、VWAP、放量结构、相对板块强弱和尾盘风险的代理评分。",
    }
