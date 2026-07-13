"""Statistical intraday high/low range model.

This module adds a professional-style calibration layer on top of structural
technical zones. It compares the current day with historically similar days and
returns probability ranges for the full-day high and low. The output is always
an interval/probability estimate, never a deterministic price call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.indicators import calculate_intraday_vwap, enrich_indicators
from utils.helpers import clamp, safe_div, to_float


def _zone(center: float, width: float, floor: float = 0.0, cap: float = 0.0) -> dict[str, float]:
    center = max(to_float(center), floor)
    width = abs(to_float(width))
    low = max(center - width, floor)
    high = max(center + width, floor)
    if cap > 0:
        center = min(center, cap)
        low = min(low, cap)
        high = min(high, cap)
    return {"low": round(low, 2), "high": round(high, 2), "center": round(center, 2)}


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "weight": weights})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["weight"] > 0]
    if data.empty:
        return 0.0
    data = data.sort_values("value")
    cumulative = data["weight"].cumsum()
    cutoff = cumulative.iloc[-1] * clamp(quantile, 0.0, 1.0)
    return float(data.loc[cumulative >= cutoff, "value"].iloc[0])


def _weighted_probability(mask: pd.Series, weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    total = float(weights.sum())
    if total <= 0:
        return 50.0
    return float(weights[mask.fillna(False)].sum() / total * 100)


def _similarity_weights(samples: pd.DataFrame, features: dict[str, float]) -> pd.Series:
    atr_scale = max(abs(features.get("atr_pct", 1.8)), 0.8)
    distance = (
        (samples["gap_pct"] - features["gap_pct"]).abs() / max(atr_scale, 0.8) * 1.15
        + (samples["prev_ret_pct"] - features["prev_ret_pct"]).abs() / 3.8
        + (samples["atr_pct"] - features["atr_pct"]).abs() / 1.8
        + (samples["trend_pct"] - features["trend_pct"]).abs() / 3.2
        + (samples["rsi14"] - features["rsi14"]).abs() / 28.0
    )
    recency = np.linspace(0.72, 1.28, len(samples))
    weights = np.exp(-distance.clip(lower=0, upper=8)) * recency
    return pd.Series(weights, index=samples.index).replace([np.inf, -np.inf], np.nan).fillna(0)


def _trading_progress(minute_df: pd.DataFrame | None) -> float:
    if minute_df is None or minute_df.empty:
        return 0.18
    rows = len(minute_df)
    expected_rows = 240 if rows > 80 else 48
    return clamp(rows / expected_rows, 0.08, 0.96)


def _fallback(price: float = 0.0) -> dict:
    width = max(price * 0.006, 0.01)
    return {
        "available": False,
        "model_version": "statistical_range_v2",
        "sample_size": 0,
        "effective_sample_size": 0,
        "confidence": 22.0,
        "day_progress": 0.18,
        "low": {
            "estimated": round(price, 2),
            "base": round(max(price - width, 0), 2),
            "weak": round(max(price - width * 2, 0), 2),
            "panic": round(max(price - width * 3, 0), 2),
            "zone": _zone(price, width),
            "touch_probability": 50.0,
            "new_extreme_probability": 50.0,
            "zone_width": round(width, 2),
        },
        "high": {
            "estimated": round(price, 2),
            "base": round(price + width, 2),
            "strong": round(price + width * 2, 2),
            "spike": round(price + width * 3, 2),
            "zone": _zone(price, width),
            "touch_probability": 50.0,
            "new_extreme_probability": 50.0,
            "zone_width": round(width, 2),
        },
        "features": {},
        "explain": "历史样本不足，统计分布模型未启用。",
    }


def estimate_statistical_extremes(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None = None,
    quote: dict | None = None,
    market_risk_bias: float = 0.0,
) -> dict:
    quote = quote or {}
    enriched = enrich_indicators(daily_df)
    price = to_float(quote.get("price"), 0)
    if enriched.empty:
        return _fallback(price)

    latest = enriched.iloc[-1]
    prev_close = to_float(quote.get("pre_close"), to_float(latest.get("close"), price))
    if prev_close <= 0:
        prev_close = price or to_float(latest.get("close"), 0)
    if prev_close <= 0:
        return _fallback(price)

    open_price = to_float(quote.get("open"), to_float(latest.get("open"), prev_close))
    current_high = to_float(quote.get("high"), max(price, open_price))
    current_low = to_float(quote.get("low"), min(price, open_price))
    if price <= 0:
        price = to_float(latest.get("close"), prev_close)

    sample = enriched.copy()
    sample["prev_close"] = pd.to_numeric(sample["close"], errors="coerce").shift(1)
    sample["gap_pct"] = (pd.to_numeric(sample["open"], errors="coerce") / sample["prev_close"] - 1) * 100
    sample["low_ext_pct"] = (pd.to_numeric(sample["low"], errors="coerce") / sample["prev_close"] - 1) * 100
    sample["high_ext_pct"] = (pd.to_numeric(sample["high"], errors="coerce") / sample["prev_close"] - 1) * 100
    sample["prev_ret_pct"] = pd.to_numeric(sample["daily_return"], errors="coerce").shift(1).fillna(0)
    sample["atr_pct"] = pd.to_numeric(sample["atr14"], errors="coerce").shift(1) / sample["prev_close"] * 100
    sample["trend_pct"] = (
        (pd.to_numeric(sample["ma5"], errors="coerce").shift(1) - pd.to_numeric(sample["ma20"], errors="coerce").shift(1))
        / sample["prev_close"]
        * 100
    )
    sample["rsi14"] = pd.to_numeric(sample["rsi14"], errors="coerce").shift(1)
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prev_close", "gap_pct", "low_ext_pct", "high_ext_pct", "atr_pct", "trend_pct", "rsi14"]
    )
    sample = sample[(sample["prev_close"] > 0) & (sample["low_ext_pct"] > -35) & (sample["high_ext_pct"] < 35)]
    sample = sample.tail(720)
    if len(sample) < 45:
        return _fallback(price)

    vwap = calculate_intraday_vwap(minute_df) if minute_df is not None else np.nan
    current_gap_pct = safe_div(open_price - prev_close, prev_close) * 100
    current_pct = safe_div(price - prev_close, prev_close) * 100
    features = {
        "gap_pct": current_gap_pct,
        "prev_ret_pct": to_float(latest.get("daily_return"), 0.0),
        "atr_pct": max(safe_div(to_float(latest.get("atr14")), prev_close) * 100, 0.8),
        "trend_pct": safe_div(to_float(latest.get("ma5")) - to_float(latest.get("ma20")), prev_close) * 100,
        "rsi14": to_float(latest.get("rsi14"), 50.0),
    }
    weights = _similarity_weights(sample, features)
    if weights.sum() <= 0:
        weights = pd.Series(np.ones(len(sample)), index=sample.index)

    effective_n = float(weights.sum() ** 2 / max(float((weights**2).sum()), 1e-9))
    day_progress = _trading_progress(minute_df)
    volume_ratio = to_float(quote.get("volume_ratio"), 1.0)
    intraday_range = max(current_high - current_low, 0.01)
    intraday_position = clamp(safe_div(price - current_low, intraday_range), 0, 1)
    above_vwap = bool(price > 0 and vwap > 0 and price >= vwap)

    downside_score = 0.0
    downside_score += 8 if current_pct < 0 else -4
    downside_score += 7 if intraday_position < 0.35 else -4 if intraday_position > 0.72 else 0
    downside_score += 6 if volume_ratio >= 1.2 and current_pct < 0 else -2 if volume_ratio < 0.7 else 0
    downside_score += -5 if above_vwap else 5
    downside_score += market_risk_bias * 70

    upside_score = 0.0
    upside_score += 8 if current_pct > 0 else -4
    upside_score += 7 if intraday_position > 0.65 else -4 if intraday_position < 0.28 else 0
    upside_score += 6 if volume_ratio >= 1.05 and current_pct >= 0 else -2 if volume_ratio < 0.7 else 0
    upside_score += 5 if above_vwap else -5
    upside_score -= market_risk_bias * 65

    low_base_pct = _weighted_quantile(sample["low_ext_pct"], weights, 0.42)
    low_weak_pct = _weighted_quantile(sample["low_ext_pct"], weights, 0.24)
    low_panic_pct = _weighted_quantile(sample["low_ext_pct"], weights, 0.11)
    high_base_pct = _weighted_quantile(sample["high_ext_pct"], weights, 0.58)
    high_strong_pct = _weighted_quantile(sample["high_ext_pct"], weights, 0.76)
    high_spike_pct = _weighted_quantile(sample["high_ext_pct"], weights, 0.89)

    downside_weight = clamp((downside_score + 14) / 42, 0, 1)
    upside_weight = clamp((upside_score + 14) / 42, 0, 1)
    estimated_low_pct = (
        low_base_pct * (0.68 - downside_weight * 0.22)
        + low_weak_pct * (0.24 + downside_weight * 0.16)
        + low_panic_pct * (0.08 + downside_weight * 0.06)
    )
    estimated_high_pct = (
        high_base_pct * (0.66 - upside_weight * 0.20)
        + high_strong_pct * (0.26 + upside_weight * 0.14)
        + high_spike_pct * (0.08 + upside_weight * 0.06)
    )

    limit_up = to_float(quote.get("limit_up"), prev_close * 1.1)
    limit_down = to_float(quote.get("limit_down"), prev_close * 0.9)
    low_est = max(prev_close * (1 + estimated_low_pct / 100), limit_down * 0.995)
    high_est = min(prev_close * (1 + estimated_high_pct / 100), limit_up)
    if current_low > 0:
        low_est = min(low_est * (1 - day_progress * 0.10) + current_low * day_progress * 0.10, current_low)
    if current_high > 0:
        high_est = max(high_est * (1 - day_progress * 0.10) + current_high * day_progress * 0.10, current_high)

    low_base = max(prev_close * (1 + low_base_pct / 100), limit_down * 0.995)
    low_weak = max(prev_close * (1 + low_weak_pct / 100), limit_down * 0.995)
    low_panic = max(prev_close * (1 + low_panic_pct / 100), limit_down * 0.995)
    high_base = min(prev_close * (1 + high_base_pct / 100), limit_up)
    high_strong = min(prev_close * (1 + high_strong_pct / 100), limit_up)
    high_spike = min(prev_close * (1 + high_spike_pct / 100), limit_up)

    atr_price = max(to_float(latest.get("atr14")), price * 0.012)
    low_width = max(abs(low_base - low_weak) * 0.32, atr_price * 0.09, price * 0.0025)
    high_width = max(abs(high_strong - high_base) * 0.32, atr_price * 0.09, price * 0.0025)
    low_zone = _zone(low_est, low_width, floor=max(limit_down * 0.99, 0))
    high_zone = _zone(high_est, high_width, floor=0.0, cap=limit_up)

    low_zone_high_pct = safe_div(low_zone["high"] - prev_close, prev_close) * 100
    high_zone_low_pct = safe_div(high_zone["low"] - prev_close, prev_close) * 100
    low_touch_prob = _weighted_probability(sample["low_ext_pct"] <= low_zone_high_pct, weights)
    high_touch_prob = _weighted_probability(sample["high_ext_pct"] >= high_zone_low_pct, weights)
    low_touch_prob = clamp(low_touch_prob + downside_score * 0.55 + day_progress * 8, 8, 92)
    high_touch_prob = clamp(high_touch_prob + upside_score * 0.55 + day_progress * 8, 8, 92)
    if current_low <= low_zone["high"]:
        low_touch_prob = clamp(max(low_touch_prob, 68 + day_progress * 16), 8, 96)
    if current_high >= high_zone["low"]:
        high_touch_prob = clamp(max(high_touch_prob, 68 + day_progress * 16), 8, 96)

    new_low_prob = clamp(
        _weighted_probability(sample["low_ext_pct"] <= safe_div(current_low - prev_close, prev_close) * 100, weights)
        + downside_score * 0.5
        - day_progress * 16,
        5,
        88,
    )
    new_high_prob = clamp(
        _weighted_probability(sample["high_ext_pct"] >= safe_div(current_high - prev_close, prev_close) * 100, weights)
        + upside_score * 0.5
        - day_progress * 16,
        5,
        88,
    )

    confidence = clamp(
        35
        + min(effective_n, 260) / 260 * 24
        + min(len(sample), 720) / 720 * 12
        + (14 if minute_df is not None and len(minute_df) >= 30 else 4)
        + (8 if volume_ratio >= 0.75 else 2),
        25,
        92,
    )

    return {
        "available": True,
        "model_version": "statistical_range_v2",
        "sample_size": int(len(sample)),
        "effective_sample_size": round(effective_n, 1),
        "confidence": round(confidence, 1),
        "day_progress": round(day_progress, 2),
        "low": {
            "estimated": round(low_est, 2),
            "base": round(low_base, 2),
            "weak": round(low_weak, 2),
            "panic": round(low_panic, 2),
            "zone": low_zone,
            "touch_probability": round(low_touch_prob, 1),
            "new_extreme_probability": round(new_low_prob, 1),
            "zone_width": round(low_width, 2),
            "quantiles_pct": {
                "base_q42": round(low_base_pct, 2),
                "weak_q24": round(low_weak_pct, 2),
                "panic_q11": round(low_panic_pct, 2),
            },
            "pressure_score": round(downside_score, 1),
        },
        "high": {
            "estimated": round(high_est, 2),
            "base": round(high_base, 2),
            "strong": round(high_strong, 2),
            "spike": round(high_spike, 2),
            "zone": high_zone,
            "touch_probability": round(high_touch_prob, 1),
            "new_extreme_probability": round(new_high_prob, 1),
            "zone_width": round(high_width, 2),
            "quantiles_pct": {
                "base_q58": round(high_base_pct, 2),
                "strong_q76": round(high_strong_pct, 2),
                "spike_q89": round(high_spike_pct, 2),
            },
            "pressure_score": round(upside_score, 1),
        },
        "features": {
            "gap_pct": round(current_gap_pct, 2),
            "current_pct": round(current_pct, 2),
            "atr_pct": round(features["atr_pct"], 2),
            "trend_pct": round(features["trend_pct"], 2),
            "rsi14": round(features["rsi14"], 1),
            "above_vwap": above_vwap,
            "intraday_position": round(intraday_position, 2),
        },
        "explain": "基于近年历史相似日、缺口、ATR波动、趋势、RSI、盘中位置、VWAP和量比的统计分布校准。",
    }
