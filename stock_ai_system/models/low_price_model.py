"""Low-price zone estimation model.

The output is an interval and probability estimate, not a prediction of an
absolute low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.indicators import calculate_intraday_vwap, enrich_indicators, support_resistance
from models.statistical_range_model import estimate_statistical_extremes
from utils.helpers import clamp, safe_div, to_float


def _valid(values: list[float]) -> list[float]:
    return [to_float(value, np.nan) for value in values if pd.notna(to_float(value, np.nan)) and value > 0]


def _median(values: list[float], fallback: float) -> float:
    values = _valid(values)
    return float(np.median(values)) if values else fallback


def _zone(center: float, width: float) -> dict[str, float]:
    center = to_float(center)
    width = abs(to_float(width))
    return {
        "low": round(max(center - width, 0), 2),
        "high": round(max(center + width, 0), 2),
        "center": round(max(center, 0), 2),
    }


def _distance_to_zone(price: float, zone: dict[str, float]) -> float:
    if price <= 0:
        return 0.0
    if zone["low"] <= price <= zone["high"]:
        return 0.0
    if price > zone["high"]:
        return safe_div(price - zone["high"], price) * 100
    return -safe_div(zone["low"] - price, price) * 100


def _scenario_label(probability: float, confidence: float) -> str:
    if confidence < 45:
        return "数据置信度偏低，仅作观察区间"
    if probability >= 68:
        return "弱势回踩概率较高，低吸需分批"
    if probability >= 48:
        return "震荡回踩概率中等，等待触达区间"
    return "当前低点已较充分，追低性价比一般"


def estimate_low_price_zones(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None = None,
    quote: dict | None = None,
    market_risk_bias: float = 0.0,
) -> dict:
    quote = quote or {}
    enriched = enrich_indicators(daily_df)
    if enriched.empty:
        return {
            "first_low_zone": {"low": 0, "high": 0, "center": 0},
            "second_low_zone": {"low": 0, "high": 0, "center": 0},
            "extreme_low_zone": {"low": 0, "high": 0, "center": 0},
            "estimated_low_zone": {"low": 0, "high": 0, "center": 0},
            "base_case_low": 0,
            "weak_case_low": 0,
            "panic_case_low": 0,
            "estimated_low_confidence": 20,
            "estimated_low_reach_probability": 50,
            "estimated_low_label": "数据不足",
            "distance_to_first_pct": 0,
            "distance_to_second_pct": 0,
            "new_low_probability": 50,
            "break_first_probability": 50,
            "break_second_probability": 35,
            "tomorrow_gap_down_probability": 35,
            "explain": "历史K线不足，区间为保守占位。",
        }

    latest = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) >= 2 else latest
    stat_model = estimate_statistical_extremes(daily_df, minute_df, quote, market_risk_bias)
    stat_low = stat_model.get("low", {}) if stat_model.get("available") else {}
    price = to_float(quote.get("price"), to_float(latest.get("close")))
    current_low = to_float(quote.get("low"), to_float(latest.get("low")))
    current_high = to_float(quote.get("high"), to_float(latest.get("high")))
    prev_close = to_float(quote.get("pre_close"), to_float(prev.get("close")))
    atr14 = max(to_float(latest.get("atr14")), price * 0.012)
    width = max(atr14 * 0.14, price * 0.003)
    vwap = calculate_intraday_vwap(minute_df) if minute_df is not None else np.nan
    minute_support = (
        to_float(minute_df["low"].tail(48).min(), np.nan)
        if minute_df is not None and not minute_df.empty and "low" in minute_df.columns
        else np.nan
    )
    sr = support_resistance(enriched)
    low_3 = to_float(enriched["low"].tail(3).min(), to_float(prev.get("low")))
    low_20 = to_float(enriched["low"].tail(20).min(), to_float(latest.get("low")))
    limit_down = to_float(quote.get("limit_down"), prev_close * 0.9)

    first_center = _median(
        [
            to_float(prev.get("low")),
            vwap,
            to_float(latest.get("ma5")),
            minute_support,
            prev_close - 0.5 * atr14,
            sr.get("support", np.nan),
        ],
        price - 0.4 * atr14,
    )
    second_center = _median(
        [
            low_3,
            to_float(latest.get("ma10")),
            to_float(latest.get("bb_lower")),
            prev_close - 0.8 * atr14,
            sr.get("support", np.nan) - 0.25 * atr14,
        ],
        first_center - 0.5 * atr14,
    )
    extreme_center = _median(
        [
            to_float(latest.get("ma20")),
            low_20,
            prev_close - 1.2 * atr14,
            limit_down,
        ],
        second_center - 0.8 * atr14,
    )

    if stat_low:
        first_center = first_center * 0.58 + to_float(stat_low.get("base"), first_center) * 0.42
        second_center = second_center * 0.58 + to_float(stat_low.get("weak"), second_center) * 0.42
        extreme_center = extreme_center * 0.62 + to_float(stat_low.get("panic"), extreme_center) * 0.38
        width = max(width, to_float(stat_low.get("zone_width"), width) * 0.70)

    weakness_shift = atr14 * clamp(market_risk_bias * 2, 0, 0.4)
    first_center -= weakness_shift * 0.25
    second_center -= weakness_shift * 0.45
    extreme_center -= weakness_shift * 0.65

    second_center = min(second_center, first_center - max(width * 1.2, atr14 * 0.12))
    extreme_center = min(extreme_center, second_center - max(width * 1.4, atr14 * 0.18))
    extreme_center = max(extreme_center, limit_down * 0.99)

    first_zone = _zone(first_center, width)
    second_zone = _zone(second_center, width * 1.1)
    extreme_zone = _zone(extreme_center, width * 1.25)

    intraday_position = safe_div(price - current_low, max(current_high - current_low, 0.01))
    pct_chg = to_float(quote.get("pct_chg"), safe_div(price - prev_close, prev_close) * 100)
    volume_ratio = to_float(quote.get("volume_ratio"), 1.0)
    low_break_pressure = 0
    low_break_pressure += 12 if pct_chg < -1 else 0
    low_break_pressure += 10 if intraday_position < 0.3 else 0
    low_break_pressure += 8 if volume_ratio > 1.5 and pct_chg < 0 else 0
    low_break_pressure += market_risk_bias * 100
    low_break_pressure += to_float(stat_low.get("pressure_score"), 0) * 0.35

    new_low_prob = clamp(35 + low_break_pressure, 10, 88)
    if stat_low:
        new_low_prob = clamp(
            new_low_prob * 0.55 + to_float(stat_low.get("new_extreme_probability"), new_low_prob) * 0.45,
            8,
            90,
        )
    break_first_prob = clamp(28 + low_break_pressure + max(0, _distance_to_zone(price, first_zone)) * 0.4, 8, 85)
    break_second_prob = clamp(16 + low_break_pressure * 0.75, 5, 72)
    tomorrow_gap_down_prob = clamp(
        22 + (18 if intraday_position < 0.25 else 0) + (12 if pct_chg < -2 else 0) + market_risk_bias * 80,
        5,
        80,
    )
    volume_confidence = 8 if volume_ratio >= 0.8 else 3
    minute_confidence = 12 if minute_df is not None and len(minute_df) >= 30 else 4
    history_confidence = 18 if len(enriched) >= 120 else 12 if len(enriched) >= 60 else 7
    structure_confidence = 12 if sr.get("support", 0) > 0 and atr14 > 0 else 6
    confidence = clamp(35 + volume_confidence + minute_confidence + history_confidence + structure_confidence, 20, 88)
    if stat_low:
        confidence = clamp(
            confidence * 0.58 + to_float(stat_model.get("confidence"), confidence) * 0.42,
            20,
            93,
        )

    base_case_low = min(current_low if current_low > 0 else first_zone["center"], first_zone["center"])
    weak_case_low = min(second_zone["center"], base_case_low - atr14 * 0.15)
    panic_case_low = min(extreme_zone["center"], weak_case_low - atr14 * 0.25)
    probability_weight = clamp((new_low_prob - 35) / 45, 0, 1)
    estimated_center = (
        first_zone["center"] * (0.62 - probability_weight * 0.22)
        + second_zone["center"] * (0.28 + probability_weight * 0.16)
        + extreme_zone["center"] * (0.10 + probability_weight * 0.06)
    )
    if stat_low:
        estimated_center = estimated_center * 0.55 + to_float(stat_low.get("estimated"), estimated_center) * 0.45
    if current_low > 0 and current_low < estimated_center:
        estimated_center = current_low * 0.55 + estimated_center * 0.45
    estimated_width = max(width * 0.75, atr14 * (0.08 + probability_weight * 0.04))
    if stat_low:
        estimated_width = max(estimated_width, to_float(stat_low.get("zone_width"), estimated_width) * 0.72)
    estimated_low_zone = _zone(estimated_center, estimated_width)
    reach_probability = clamp(
        42
        + low_break_pressure * 0.55
        + max(0, _distance_to_zone(price, estimated_low_zone)) * 0.35
        - (10 if price <= estimated_low_zone["high"] else 0),
        12,
        86,
    )
    if stat_low:
        reach_probability = clamp(
            reach_probability * 0.55 + to_float(stat_low.get("touch_probability"), reach_probability) * 0.45,
            10,
            92,
        )
    estimated_label = _scenario_label(reach_probability, confidence)

    return {
        "first_low_zone": first_zone,
        "second_low_zone": second_zone,
        "extreme_low_zone": extreme_zone,
        "estimated_low_zone": estimated_low_zone,
        "base_case_low": round(max(base_case_low, 0), 2),
        "weak_case_low": round(max(weak_case_low, 0), 2),
        "panic_case_low": round(max(panic_case_low, 0), 2),
        "estimated_low_confidence": round(confidence, 1),
        "estimated_low_reach_probability": round(reach_probability, 1),
        "estimated_low_label": estimated_label,
        "distance_to_first_pct": round(_distance_to_zone(price, first_zone), 2),
        "distance_to_second_pct": round(_distance_to_zone(price, second_zone), 2),
        "new_low_probability": round(new_low_prob, 1),
        "break_first_probability": round(break_first_prob, 1),
        "break_second_probability": round(break_second_prob, 1),
        "tomorrow_gap_down_probability": round(tomorrow_gap_down_prob, 1),
        "statistical_low_zone": stat_low.get("zone", {}),
        "statistical_model": {
            "available": bool(stat_model.get("available")),
            "model_version": stat_model.get("model_version", ""),
            "sample_size": stat_model.get("sample_size", 0),
            "effective_sample_size": stat_model.get("effective_sample_size", 0),
            "confidence": stat_model.get("confidence", 0),
            "day_progress": stat_model.get("day_progress", 0),
            "features": stat_model.get("features", {}),
            "quantiles_pct": stat_low.get("quantiles_pct", {}),
        },
        "reference": {
            "vwap": round(to_float(vwap, 0), 2),
            "atr14": round(atr14, 2),
            "support": round(to_float(sr.get("support"), 0), 2),
            "resistance": round(to_float(sr.get("resistance"), 0), 2),
        },
        "explain": (
            "基于昨日低点、VWAP、均线、ATR、布林下轨、支撑位、大盘/板块弱势修正，"
            "并叠加历史相似日分布校准生成的概率区间。"
        ),
    }
