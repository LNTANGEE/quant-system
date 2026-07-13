"""Intraday high-price zone estimation model.

The model outputs probability zones and confidence levels. It does not predict
an absolute high or guarantee that a price will be reached.
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


def _zone(center: float, width: float, cap: float = 0.0) -> dict[str, float]:
    center = max(to_float(center), 0)
    width = abs(to_float(width))
    low = max(center - width, 0)
    high = max(center + width, 0)
    if cap > 0:
        low = min(low, cap)
        high = min(high, cap)
        center = min(center, cap)
    return {"low": round(low, 2), "high": round(high, 2), "center": round(center, 2)}


def _distance_to_zone(price: float, zone: dict[str, float]) -> float:
    if price <= 0:
        return 0.0
    if zone["low"] <= price <= zone["high"]:
        return 0.0
    if price < zone["low"]:
        return safe_div(zone["low"] - price, price) * 100
    return -safe_div(price - zone["high"], price) * 100


def _scenario_label(probability: float, confidence: float) -> str:
    if confidence < 45:
        return "数据置信度偏低，最高价区间仅作观察"
    if probability >= 68:
        return "冲高触达概率较高，适合重点盯高抛区但不追高"
    if probability >= 48:
        return "冲高触达概率中等，等待量能和VWAP确认"
    return "当前位置上攻条件一般，高位区间以风控观察为主"


def estimate_high_price_zones(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None = None,
    quote: dict | None = None,
    market_risk_bias: float = 0.0,
) -> dict:
    quote = quote or {}
    enriched = enrich_indicators(daily_df)
    if enriched.empty:
        return {
            "first_high_zone": {"low": 0, "high": 0, "center": 0},
            "second_high_zone": {"low": 0, "high": 0, "center": 0},
            "extreme_high_zone": {"low": 0, "high": 0, "center": 0},
            "estimated_high_zone": {"low": 0, "high": 0, "center": 0},
            "base_case_high": 0,
            "strong_case_high": 0,
            "spike_case_high": 0,
            "estimated_high_confidence": 20,
            "estimated_high_reach_probability": 45,
            "estimated_high_label": "数据不足",
            "distance_to_first_high_pct": 0,
            "distance_to_second_high_pct": 0,
            "new_high_probability": 45,
            "break_first_high_probability": 35,
            "break_second_high_probability": 22,
            "tomorrow_gap_up_probability": 25,
            "explain": "历史K线不足，高位区间为保守占位。",
        }

    latest = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) >= 2 else latest
    stat_model = estimate_statistical_extremes(daily_df, minute_df, quote, market_risk_bias)
    stat_high = stat_model.get("high", {}) if stat_model.get("available") else {}
    price = to_float(quote.get("price"), to_float(latest.get("close")))
    current_high = to_float(quote.get("high"), to_float(latest.get("high")))
    current_low = to_float(quote.get("low"), to_float(latest.get("low")))
    prev_close = to_float(quote.get("pre_close"), to_float(prev.get("close")))
    atr14 = max(to_float(latest.get("atr14")), price * 0.012)
    width = max(atr14 * 0.16, price * 0.0035)
    vwap = calculate_intraday_vwap(minute_df) if minute_df is not None else np.nan
    minute_resistance = (
        to_float(minute_df["high"].tail(48).max(), np.nan)
        if minute_df is not None and not minute_df.empty and "high" in minute_df.columns
        else np.nan
    )
    sr = support_resistance(enriched)
    high_3 = to_float(enriched["high"].tail(3).max(), to_float(prev.get("high")))
    high_20 = to_float(enriched["high"].tail(20).max(), to_float(latest.get("high")))
    limit_up = to_float(quote.get("limit_up"), prev_close * 1.1)
    if limit_up <= 0:
        limit_up = price * 1.1

    first_center = _median(
        [
            to_float(prev.get("high")),
            vwap,
            to_float(latest.get("ma5")),
            minute_resistance,
            prev_close + 0.50 * atr14,
            sr.get("resistance", np.nan),
        ],
        price + 0.45 * atr14,
    )
    second_center = _median(
        [
            high_3,
            to_float(latest.get("ma10")),
            to_float(latest.get("bb_upper")),
            prev_close + 0.85 * atr14,
            sr.get("resistance", np.nan) + 0.25 * atr14,
        ],
        first_center + 0.55 * atr14,
    )
    extreme_center = _median(
        [
            to_float(latest.get("ma20")),
            high_20,
            prev_close + 1.25 * atr14,
            limit_up,
        ],
        second_center + 0.85 * atr14,
    )

    if stat_high:
        first_center = first_center * 0.58 + to_float(stat_high.get("base"), first_center) * 0.42
        second_center = second_center * 0.58 + to_float(stat_high.get("strong"), second_center) * 0.42
        extreme_center = extreme_center * 0.62 + to_float(stat_high.get("spike"), extreme_center) * 0.38
        width = max(width, to_float(stat_high.get("zone_width"), width) * 0.70)

    weak_market_shift = atr14 * clamp(market_risk_bias * 2, 0, 0.45)
    first_center -= weak_market_shift * 0.20
    second_center -= weak_market_shift * 0.35
    extreme_center -= weak_market_shift * 0.50

    first_center = max(first_center, price + max(width * 0.8, atr14 * 0.08))
    second_center = max(second_center, first_center + max(width * 1.2, atr14 * 0.12))
    extreme_center = max(extreme_center, second_center + max(width * 1.4, atr14 * 0.18))
    first_center = min(first_center, limit_up * 0.998)
    second_center = min(second_center, limit_up * 0.999)
    extreme_center = min(extreme_center, limit_up)

    first_zone = _zone(first_center, width, limit_up)
    second_zone = _zone(second_center, width * 1.1, limit_up)
    extreme_zone = _zone(extreme_center, width * 1.25, limit_up)

    intraday_range = max(current_high - current_low, 0.01)
    intraday_position = safe_div(price - current_low, intraday_range)
    pct_chg = to_float(quote.get("pct_chg"), safe_div(price - prev_close, prev_close) * 100)
    volume_ratio = to_float(quote.get("volume_ratio"), 1.0)
    above_vwap = price > 0 and vwap > 0 and price >= vwap
    ma5 = to_float(latest.get("ma5"))
    ma10 = to_float(latest.get("ma10"))
    rsi14 = to_float(latest.get("rsi14"), 50)
    macd_hist = to_float(latest.get("macd_hist"), 0)

    upside_pressure = 0.0
    upside_pressure += 14 if pct_chg > 1 else 7 if pct_chg > 0 else -6 if pct_chg < -1 else 0
    upside_pressure += 12 if intraday_position > 0.70 else 6 if intraday_position > 0.52 else -5
    upside_pressure += 10 if 1.1 <= volume_ratio <= 2.8 and pct_chg >= 0 else 5 if volume_ratio > 0.8 else -4
    upside_pressure += 8 if above_vwap else -5
    upside_pressure += 6 if price >= ma5 >= ma10 else 0
    upside_pressure += 5 if macd_hist >= 0 else -3
    upside_pressure += 4 if 48 <= rsi14 <= 74 else -5 if rsi14 > 82 else 0
    upside_pressure -= market_risk_bias * 85
    upside_pressure += to_float(stat_high.get("pressure_score"), 0) * 0.35

    new_high_prob = clamp(36 + upside_pressure, 8, 88)
    if stat_high:
        new_high_prob = clamp(
            new_high_prob * 0.55 + to_float(stat_high.get("new_extreme_probability"), new_high_prob) * 0.45,
            8,
            90,
        )
    break_first_prob = clamp(
        28 + upside_pressure * 0.85 + max(0, _distance_to_zone(price, first_zone)) * 0.25,
        6,
        84,
    )
    break_second_prob = clamp(18 + upside_pressure * 0.62, 4, 72)
    tomorrow_gap_up_prob = clamp(
        20 + (14 if pct_chg > 2 else 6 if pct_chg > 0 else 0) + (8 if intraday_position > 0.72 else 0)
        - market_risk_bias * 65,
        4,
        78,
    )

    volume_confidence = 10 if volume_ratio >= 0.9 else 4
    minute_confidence = 12 if minute_df is not None and len(minute_df) >= 30 else 4
    history_confidence = 18 if len(enriched) >= 120 else 12 if len(enriched) >= 60 else 7
    structure_confidence = 12 if sr.get("resistance", 0) > 0 and atr14 > 0 else 6
    confidence = clamp(34 + volume_confidence + minute_confidence + history_confidence + structure_confidence, 20, 90)
    if stat_high:
        confidence = clamp(
            confidence * 0.58 + to_float(stat_model.get("confidence"), confidence) * 0.42,
            20,
            94,
        )

    base_case_high = max(current_high if current_high > 0 else first_zone["center"], first_zone["center"])
    strong_case_high = max(second_zone["center"], base_case_high + atr14 * 0.15)
    spike_case_high = max(extreme_zone["center"], strong_case_high + atr14 * 0.20)
    spike_case_high = min(spike_case_high, limit_up)
    momentum_weight = clamp((new_high_prob - 35) / 45, 0, 1)
    estimated_center = (
        first_zone["center"] * (0.60 - momentum_weight * 0.18)
        + second_zone["center"] * (0.30 + momentum_weight * 0.12)
        + extreme_zone["center"] * (0.10 + momentum_weight * 0.06)
    )
    if stat_high:
        estimated_center = estimated_center * 0.55 + to_float(stat_high.get("estimated"), estimated_center) * 0.45
    if current_high > 0 and current_high > estimated_center:
        estimated_center = current_high * 0.58 + estimated_center * 0.42
    estimated_center = min(max(estimated_center, price), limit_up)
    estimated_width = max(width * 0.8, atr14 * (0.08 + momentum_weight * 0.05))
    if stat_high:
        estimated_width = max(estimated_width, to_float(stat_high.get("zone_width"), estimated_width) * 0.72)
    estimated_high_zone = _zone(estimated_center, estimated_width, limit_up)
    reach_probability = clamp(
        40
        + upside_pressure * 0.55
        + max(0, _distance_to_zone(price, estimated_high_zone)) * 0.25
        - (8 if price >= estimated_high_zone["low"] else 0),
        10,
        86,
    )
    if stat_high:
        reach_probability = clamp(
            reach_probability * 0.55 + to_float(stat_high.get("touch_probability"), reach_probability) * 0.45,
            10,
            92,
        )

    return {
        "first_high_zone": first_zone,
        "second_high_zone": second_zone,
        "extreme_high_zone": extreme_zone,
        "estimated_high_zone": estimated_high_zone,
        "base_case_high": round(max(base_case_high, 0), 2),
        "strong_case_high": round(max(strong_case_high, 0), 2),
        "spike_case_high": round(max(spike_case_high, 0), 2),
        "estimated_high_confidence": round(confidence, 1),
        "estimated_high_reach_probability": round(reach_probability, 1),
        "estimated_high_label": _scenario_label(reach_probability, confidence),
        "distance_to_first_high_pct": round(_distance_to_zone(price, first_zone), 2),
        "distance_to_second_high_pct": round(_distance_to_zone(price, second_zone), 2),
        "new_high_probability": round(new_high_prob, 1),
        "break_first_high_probability": round(break_first_prob, 1),
        "break_second_high_probability": round(break_second_prob, 1),
        "tomorrow_gap_up_probability": round(tomorrow_gap_up_prob, 1),
        "statistical_high_zone": stat_high.get("zone", {}),
        "statistical_model": {
            "available": bool(stat_model.get("available")),
            "model_version": stat_model.get("model_version", ""),
            "sample_size": stat_model.get("sample_size", 0),
            "effective_sample_size": stat_model.get("effective_sample_size", 0),
            "confidence": stat_model.get("confidence", 0),
            "day_progress": stat_model.get("day_progress", 0),
            "features": stat_model.get("features", {}),
            "quantiles_pct": stat_high.get("quantiles_pct", {}),
        },
        "reference": {
            "vwap": round(to_float(vwap, 0), 2),
            "atr14": round(atr14, 2),
            "resistance": round(to_float(sr.get("resistance"), 0), 2),
            "limit_up": round(limit_up, 2),
        },
        "explain": (
            "基于昨日高点、VWAP、MA5/MA10、ATR、布林带上轨、近期平台压力、"
            "量比、分时位置、市场强弱修正，并叠加历史相似日分布校准生成的概率区间。"
        ),
    }
