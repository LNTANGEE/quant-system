"""Forecast the highest price over the next trading hour.

The existing high-price model estimates the *full-day* high range.  This module
uses only information available at the latest minute bar and targets the
maximum high in the following 60 trading minutes.  The lunch break is skipped,
and the horizon is shortened when fewer than 60 trading minutes remain before
the close.

The output is a calibrated probability interval, not a guaranteed price.
"""

from __future__ import annotations

from math import ceil, erf, exp, sqrt
from typing import Any

import numpy as np
import pandas as pd

from models.indicators import enrich_indicators
from utils.helpers import clamp, safe_div, to_float


_MORNING_START = 9 * 60 + 30
_MORNING_END = 11 * 60 + 30
_AFTERNOON_START = 13 * 60
_AFTERNOON_END = 15 * 60
_TRADING_MINUTES_PER_DAY = 240
_FEATURE_COLUMNS = (
    "ret_15",
    "ret_30",
    "horizon_volatility",
    "recent_range",
    "price_position",
    "vwap_gap",
    "volume_impulse",
    "time_index",
)


def _trading_index(value: pd.Timestamp) -> int | None:
    """Return elapsed A-share trading minutes, excluding the lunch break."""

    if pd.isna(value):
        return None
    minute = int(value.hour) * 60 + int(value.minute)
    if _MORNING_START <= minute <= _MORNING_END:
        return minute - _MORNING_START
    if _AFTERNOON_START <= minute <= _AFTERNOON_END:
        return 120 + minute - _AFTERNOON_START
    return None


def _advance_trading_minutes(value: pd.Timestamp, minutes: int) -> pd.Timestamp:
    start_index = _trading_index(value)
    if start_index is None:
        return value
    target = int(clamp(start_index + max(int(minutes), 0), 0, _TRADING_MINUTES_PER_DAY))
    date = value.normalize()
    if target <= 120:
        return date + pd.Timedelta(minutes=_MORNING_START + target)
    return date + pd.Timedelta(minutes=_AFTERNOON_START + target - 120)


def _is_session_time(value: pd.Timestamp) -> bool:
    if _trading_index(value) is None:
        return False
    minute = int(value.hour) * 60 + int(value.minute)
    # 09:30/13:00 are auction or session-open snapshots in several free feeds,
    # not completed one-minute bars.
    return minute not in {_MORNING_START, _AFTERNOON_START}


def _clean_minute_data(minute_df: pd.DataFrame | None) -> pd.DataFrame:
    required = {"datetime", "close", "high"}
    if minute_df is None or minute_df.empty or not required.issubset(minute_df.columns):
        return pd.DataFrame()
    out = minute_df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    for column in ("open", "close", "high", "low", "volume", "amount"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["datetime", "close", "high"])
    out = out[(out["close"] > 0) & (out["high"] > 0)]
    out = out[out["datetime"].map(_is_session_time)]
    return (
        out.sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )


def _local_naive_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp


def _truncate_at(data: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.DataFrame:
    if data.empty or cutoff is None:
        return data
    datetimes = data["datetime"]
    series_tz = getattr(datetimes.dt, "tz", None)
    comparable_cutoff = cutoff
    if series_tz is not None:
        comparable_cutoff = cutoff.tz_localize("Asia/Shanghai").tz_convert(series_tz)
    return data[datetimes <= comparable_cutoff].copy()


def _infer_bar_minutes(data: pd.DataFrame) -> int:
    if data.empty or "datetime" not in data.columns:
        return 5
    dates = data["datetime"].dt.date
    diffs = data["datetime"].diff().dt.total_seconds().div(60)
    same_day = dates.eq(dates.shift(1))
    valid = diffs[same_day & diffs.between(0.5, 60)]
    if valid.empty:
        return 5
    observed = float(valid.median())
    allowed = (1, 5, 15, 30, 60)
    return min(allowed, key=lambda item: abs(item - observed))


def _steps_for(minutes: int, bar_minutes: int) -> int:
    return max(1, int(ceil(max(minutes, 1) / max(bar_minutes, 1))))


def _return_over(close: pd.Series, minutes: int, bar_minutes: int) -> float:
    if close.empty:
        return 0.0
    steps = min(_steps_for(minutes, bar_minutes), len(close) - 1)
    if steps <= 0:
        return 0.0
    start = to_float(close.iloc[-steps - 1], 0)
    end = to_float(close.iloc[-1], 0)
    return safe_div(end - start, start) * 100


def _feature_snapshot(
    session_prefix: pd.DataFrame,
    horizon_minutes: int,
    bar_minutes: int,
) -> dict[str, float]:
    if session_prefix.empty:
        return {key: 0.0 for key in _FEATURE_COLUMNS}

    lookback_bars = max(4, _steps_for(60, bar_minutes))
    recent = session_prefix.tail(lookback_bars + 1).copy()
    close = pd.to_numeric(recent["close"], errors="coerce").dropna()
    price = to_float(close.iloc[-1] if not close.empty else 0, 0)

    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    target_steps = _steps_for(horizon_minutes, bar_minutes)
    horizon_volatility = (
        float(log_returns.tail(lookback_bars).std(ddof=0)) * sqrt(target_steps) * 100
        if len(log_returns) >= 3
        else 0.0
    )

    recent_high = to_float(pd.to_numeric(recent["high"], errors="coerce").max(), price)
    if "low" in recent.columns:
        recent_low = to_float(pd.to_numeric(recent["low"], errors="coerce").min(), price)
    else:
        recent_low = to_float(close.min() if not close.empty else price, price)
    recent_range = safe_div(recent_high - recent_low, price) * 100
    price_position = clamp(safe_div(price - recent_low, max(recent_high - recent_low, 1e-9)), 0, 1)

    volume = pd.to_numeric(recent.get("volume", pd.Series(index=recent.index, dtype=float)), errors="coerce")
    volume = volume.fillna(0).clip(lower=0)
    short_bars = max(1, _steps_for(15, bar_minutes))
    short_volume = to_float(volume.tail(short_bars).mean(), 0)
    baseline_volume = to_float(volume.head(max(len(volume) - short_bars, 1)).mean(), 0)
    if baseline_volume <= 0:
        baseline_volume = to_float(volume.mean(), 0)
    volume_impulse = clamp(safe_div(short_volume, baseline_volume), 0, 5) if baseline_volume > 0 else 1.0

    amount = pd.to_numeric(recent.get("amount", pd.Series(index=recent.index, dtype=float)), errors="coerce")
    amount = amount.fillna(0).clip(lower=0)
    share_volume = volume * 100
    if amount.sum() > 0 and share_volume.sum() > 0:
        vwap = to_float(amount.sum() / share_volume.sum(), price)
    elif volume.sum() > 0:
        aligned_close = pd.to_numeric(recent["close"], errors="coerce").fillna(price)
        vwap = to_float((aligned_close * volume).sum() / volume.sum(), price)
    else:
        vwap = price

    last_time = pd.to_datetime(session_prefix["datetime"].iloc[-1], errors="coerce")
    trading_index = _trading_index(last_time) or 0
    return {
        "ret_15": round(_return_over(close, 15, bar_minutes), 6),
        "ret_30": round(_return_over(close, 30, bar_minutes), 6),
        "horizon_volatility": round(max(horizon_volatility, 0), 6),
        "recent_range": round(max(recent_range, 0), 6),
        "price_position": round(price_position, 6),
        "vwap_gap": round(safe_div(price - vwap, vwap) * 100, 6),
        "volume_impulse": round(volume_impulse, 6),
        "time_index": round(trading_index / _TRADING_MINUTES_PER_DAY, 6),
    }


def _build_historical_samples(
    data: pd.DataFrame,
    horizon_minutes: int,
    bar_minutes: int,
    target_time_index: int,
) -> pd.DataFrame:
    """Build one leakage-free, time-matched sample per trading day."""

    if data.empty:
        return pd.DataFrame()
    target_bars = _steps_for(horizon_minutes, bar_minutes)
    context_bars = max(4, _steps_for(20, bar_minutes))
    rows: list[dict[str, Any]] = []

    for _, session in data.groupby(data["datetime"].dt.date, sort=True):
        session = session.sort_values("datetime").reset_index(drop=True)
        if len(session) < context_bars + target_bars + 1:
            continue
        candidates: list[tuple[int, int]] = []
        for anchor in range(context_bars - 1, len(session) - target_bars):
            anchor_index = _trading_index(pd.to_datetime(session["datetime"].iloc[anchor], errors="coerce"))
            if anchor_index is not None:
                candidates.append((abs(anchor_index - target_time_index), anchor))
        if not candidates:
            continue
        # One anchor per date prevents overlapping windows from creating an
        # artificially large sample/effective-sample count.
        for _, anchor in [min(candidates, key=lambda item: item[0])]:
            prefix = session.iloc[: anchor + 1]
            future = session.iloc[anchor + 1 : anchor + 1 + target_bars]
            if len(future) < target_bars:
                continue
            anchor_time = pd.to_datetime(prefix["datetime"].iloc[-1], errors="coerce")
            future_end = pd.to_datetime(future["datetime"].iloc[-1], errors="coerce")
            anchor_index = _trading_index(anchor_time)
            future_index = _trading_index(future_end)
            if anchor_index is None or future_index is None:
                continue
            if future_index - anchor_index < horizon_minutes * 0.72:
                continue

            anchor_price = to_float(prefix["close"].iloc[-1], 0)
            future_high = to_float(pd.to_numeric(future["high"], errors="coerce").max(), 0)
            if anchor_price <= 0 or future_high <= 0:
                continue
            target_pct = max(safe_div(future_high - anchor_price, anchor_price) * 100, 0.0)
            if target_pct > 35:
                continue
            known_high = to_float(pd.to_numeric(prefix["high"], errors="coerce").max(), anchor_price)
            feature = _feature_snapshot(prefix, horizon_minutes, bar_minutes)
            rows.append(
                {
                    **feature,
                    "target_pct": target_pct,
                    "break_day_high": future_high > known_high + max(anchor_price * 0.0001, 0.005),
                    "anchor_time": anchor_time,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).tail(1200).reset_index(drop=True)


def _similarity_weights(samples: pd.DataFrame, current: dict[str, float]) -> pd.Series:
    scales = {
        "ret_15": 0.9,
        "ret_30": 1.3,
        "horizon_volatility": 0.65,
        "recent_range": 1.0,
        "price_position": 0.28,
        "vwap_gap": 0.8,
        "volume_impulse": 0.8,
        "time_index": 0.20,
    }
    importance = {
        "ret_15": 1.20,
        "ret_30": 1.05,
        "horizon_volatility": 1.15,
        "recent_range": 0.85,
        "price_position": 1.00,
        "vwap_gap": 0.95,
        "volume_impulse": 0.65,
        "time_index": 1.25,
    }
    distance = pd.Series(0.0, index=samples.index)
    for column in _FEATURE_COLUMNS:
        values = pd.to_numeric(samples[column], errors="coerce")
        distance += (
            (values - to_float(current.get(column), 0)).abs()
            / scales[column]
            * importance[column]
        )
    recency = pd.Series(np.linspace(0.72, 1.28, len(samples)), index=samples.index)
    weights = distance.clip(lower=0, upper=14).map(lambda value: exp(-value)) * recency
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0)


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    frame = frame[frame["weight"] > 0].sort_values("value")
    if frame.empty:
        return 0.0
    cumulative = frame["weight"].cumsum()
    cutoff = cumulative.iloc[-1] * clamp(quantile, 0, 1)
    return float(frame.loc[cumulative >= cutoff, "value"].iloc[0])


def _weighted_probability(mask: pd.Series, weights: pd.Series) -> float:
    valid_weights = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    total = to_float(valid_weights.sum(), 0)
    if total <= 0:
        return 50.0
    return float(valid_weights[mask.fillna(False)].sum() / total * 100)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + erf(value / sqrt(2)))


def _daily_atr_pct(daily_df: pd.DataFrame, price: float) -> float:
    if daily_df is None or daily_df.empty or price <= 0:
        return 1.8
    enriched = enrich_indicators(daily_df)
    if enriched.empty:
        return 1.8
    atr = to_float(enriched.iloc[-1].get("atr14"), 0)
    if atr <= 0:
        high = to_float(enriched.iloc[-1].get("high"), price)
        low = to_float(enriched.iloc[-1].get("low"), price)
        atr = max(high - low, price * 0.018)
    return clamp(safe_div(atr, price) * 100, 0.35, 12)


def _fallback_quantiles(
    feature: dict[str, float],
    daily_df: pd.DataFrame,
    quote: dict[str, Any],
    price: float,
    horizon_minutes: int,
    market_risk_bias: float,
) -> dict[str, float]:
    atr_pct = _daily_atr_pct(daily_df, price)
    atr_component = atr_pct * sqrt(max(horizon_minutes, 1) / _TRADING_MINUTES_PER_DAY) * 0.55
    volatility_component = to_float(feature.get("horizon_volatility"), 0) * 0.82
    range_component = to_float(feature.get("recent_range"), 0) * 0.34
    scale = max(0.10, atr_component, volatility_component, range_component)

    volume_ratio = to_float(quote.get("volume_ratio"), 1.0)
    volume_impulse = to_float(feature.get("volume_impulse"), 1.0)
    momentum = (
        to_float(feature.get("ret_15"), 0) * 0.34
        + to_float(feature.get("ret_30"), 0) * 0.20
        + to_float(feature.get("vwap_gap"), 0) * 0.18
        + (0.10 * scale if volume_impulse >= 1.25 or volume_ratio >= 1.2 else 0)
        - max(market_risk_bias, 0) * scale * 1.8
    )
    median = max(scale * 0.74 + clamp(momentum, -scale * 0.45, scale * 0.75) * 0.42, 0.0)
    return {
        "q35": max(median * 0.62, 0.0),
        "q50": max(median, 0.0),
        "q75": max(median * 1.42 + scale * 0.06, 0.02),
        "q90": max(median * 1.92 + scale * 0.10, 0.04),
    }


def _unavailable(
    status: str,
    label: str,
    price: float = 0.0,
    last_time: pd.Timestamp | None = None,
    source: str = "",
) -> dict[str, Any]:
    time_text = last_time.strftime("%Y-%m-%d %H:%M") if last_time is not None and pd.notna(last_time) else ""
    return {
        "available": False,
        "status": status,
        "requested_horizon_minutes": 60,
        "horizon_minutes": 0,
        "window_start": time_text,
        "window_end": "",
        "predicted_high": None,
        "predicted_high_zone": {},
        "expected_upside_pct": None,
        "break_day_high_probability": None,
        "confidence": None,
        "sample_size": 0,
        "effective_sample_size": 0.0,
        "method": "unavailable",
        "bar_minutes": 0,
        "source": source,
        "quantiles_pct": {},
        "label": label,
        "explain": label,
    }


def estimate_next_hour_high(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None = None,
    quote: dict | None = None,
    market_risk_bias: float = 0.0,
    horizon_minutes: int = 60,
    historical_minute_df: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Estimate the maximum price in the next N A-share trading minutes.

    Historical samples are constructed at earlier minute-bar anchors.  Every
    feature uses only the prefix available at that anchor, while the label uses
    the following N trading minutes.  If too few samples exist, a conservative
    ATR + realized-volatility fallback is used.
    """

    quote = quote or {}
    requested_horizon = int(clamp(int(horizon_minutes or 60), 5, 120))
    cutoff = _local_naive_timestamp(as_of)
    data = _truncate_at(_clean_minute_data(minute_df), cutoff)
    source = str(getattr(minute_df, "attrs", {}).get("source", "") if minute_df is not None else "")
    quote_price = to_float(quote.get("price"), 0)
    if data.empty:
        return _unavailable(
            "minute_data_unavailable",
            "分钟行情暂不可用，无法生成未来一小时高点预测。",
            quote_price,
            source=source,
        )

    last_time = pd.to_datetime(data["datetime"].iloc[-1], errors="coerce")
    price = quote_price or to_float(data["close"].iloc[-1], 0)
    if price <= 0:
        return _unavailable(
            "price_unavailable",
            "当前价格不可用，暂不生成短时预测。",
            last_time=last_time,
            source=source,
        )

    if cutoff is not None and last_time.date() < cutoff.date():
        return _unavailable(
            "minute_data_stale",
            "分钟行情不是当前交易日数据，暂不生成未来一小时预测。",
            price,
            last_time,
            source,
        )
    if cutoff is not None and last_time.date() == cutoff.date():
        cutoff_minute = cutoff.hour * 60 + cutoff.minute
        if cutoff_minute >= _AFTERNOON_END:
            return _unavailable(
                "market_closed",
                "当日已收盘，请在下一交易日开盘后刷新未来一小时预测。",
                price,
                last_time,
                source,
            )

    start_index = _trading_index(last_time)
    if start_index is None:
        return _unavailable(
            "outside_trading_session",
            "最后一根分钟线不在A股连续交易时段内。",
            price,
            last_time,
            source,
        )
    remaining_minutes = max(_TRADING_MINUTES_PER_DAY - start_index, 0)
    if remaining_minutes <= 0:
        return _unavailable(
            "market_closed",
            "当日已收盘，请在下一交易日开盘后刷新未来一小时预测。",
            price,
            last_time,
            source,
        )

    actual_horizon = min(requested_horizon, remaining_minutes)
    window_end = _advance_trading_minutes(last_time, actual_horizon)
    latest_date = last_time.date()
    current_session = data[data["datetime"].dt.date == latest_date].copy()
    bar_minutes = _infer_bar_minutes(data)
    feature = _feature_snapshot(current_session, actual_horizon, bar_minutes)
    history_data = _truncate_at(_clean_minute_data(historical_minute_df), cutoff)
    history_source = str(
        getattr(historical_minute_df, "attrs", {}).get("source", "")
        if historical_minute_df is not None
        else ""
    )
    history_is_synthetic = bool(
        getattr(historical_minute_df, "attrs", {}).get("is_synthetic_ohlc", False)
        if historical_minute_df is not None
        else False
    )
    if history_data.empty:
        history_data = data
        history_source = source
        history_is_synthetic = bool(
            getattr(minute_df, "attrs", {}).get("is_synthetic_ohlc", False)
            if minute_df is not None
            else False
        )
    history_bar_minutes = _infer_bar_minutes(history_data)
    if history_is_synthetic or "腾讯" in history_source or "Tencent" in history_source:
        # Tencent's free minute fallback synthesizes high/low from adjacent
        # prices.  It is useful for current momentum, but not as a high target.
        samples = pd.DataFrame()
    else:
        samples = _build_historical_samples(
            history_data,
            actual_horizon,
            history_bar_minutes,
            start_index,
        )
    weights = pd.Series(dtype=float)
    effective_n = 0.0

    fallback = _fallback_quantiles(
        feature,
        daily_df,
        quote,
        price,
        actual_horizon,
        market_risk_bias,
    )
    quantiles = fallback.copy()
    method = "atr_realized_volatility_fallback"

    if len(samples) >= 8:
        weights = _similarity_weights(samples, feature)
        if to_float(weights.sum(), 0) > 0:
            effective_n = float(weights.sum() ** 2 / max(to_float((weights**2).sum(), 0), 1e-9))
        if effective_n >= 4:
            empirical = {
                "q35": _weighted_quantile(samples["target_pct"], weights, 0.35),
                "q50": _weighted_quantile(samples["target_pct"], weights, 0.50),
                "q75": _weighted_quantile(samples["target_pct"], weights, 0.75),
                "q90": _weighted_quantile(samples["target_pct"], weights, 0.90),
            }
            empirical_weight = clamp(0.42 + effective_n / 120, 0.42, 0.78)
            quantiles = {
                key: empirical[key] * empirical_weight + fallback[key] * (1 - empirical_weight)
                for key in fallback
            }
            method = (
                "historical_similarity"
                if len(samples) >= 25 and effective_n >= 12
                else "time_slot_empirical"
            )

    limit_up = to_float(quote.get("limit_up"), 0)
    if limit_up <= 0:
        prev_close = to_float(quote.get("pre_close"), price)
        limit_up = prev_close * 1.1 if prev_close > 0 else price * 1.1
    room_pct = max(safe_div(limit_up - price, price) * 100, 0.0)

    q35 = clamp(to_float(quantiles.get("q35"), 0), 0, room_pct)
    q50 = clamp(max(to_float(quantiles.get("q50"), 0), q35), 0, room_pct)
    q75 = clamp(max(to_float(quantiles.get("q75"), 0), q50), 0, room_pct)
    q90 = clamp(max(to_float(quantiles.get("q90"), 0), q75), 0, room_pct)

    zone_low = min(price * (1 + q35 / 100), limit_up)
    zone_center = min(price * (1 + q50 / 100), limit_up)
    zone_high = min(price * (1 + q75 / 100), limit_up)
    zone = {
        "low": round(max(zone_low, price), 2),
        "high": round(max(zone_high, zone_low, price), 2),
        "center": round(max(zone_center, zone_low, price), 2),
    }

    day_high = to_float(
        quote.get("high"),
        to_float(pd.to_numeric(current_session["high"], errors="coerce").max(), price),
    )
    day_high = max(day_high, price)
    threshold_pct = max(safe_div(day_high - price, price) * 100, 0)
    sigma = max((q75 - q35) / 1.349, q50 * 0.35, 0.08)
    fallback_break_probability = (1 - _normal_cdf((threshold_pct - q50) / sigma)) * 100
    if method in {"historical_similarity", "time_slot_empirical"} and not weights.empty:
        empirical_break = _weighted_probability(samples["break_day_high"].astype(bool), weights)
        break_probability = empirical_break * 0.68 + fallback_break_probability * 0.32
    else:
        break_probability = fallback_break_probability
    break_probability = clamp(break_probability, 3, 95)

    days = int(history_data["datetime"].dt.date.nunique())
    current_minutes = len(current_session) * bar_minutes
    current_data_score = min(current_minutes / 60, 1) * 15
    day_score = min(days / 12, 1) * 11
    sample_score = (
        min(effective_n / 70, 1) * 25
        if method in {"historical_similarity", "time_slot_empirical"}
        else 0
    )
    daily_score = 9 if daily_df is not None and len(daily_df) >= 60 else 4
    period_score = 8 if bar_minutes <= 5 else 3
    source_penalty = 8 if ("腾讯" in source or "Tencent" in source) else 0
    shortened_penalty = (1 - actual_horizon / requested_horizon) * 10
    confidence = clamp(
        (46 if method == "historical_similarity" else 34 if method == "time_slot_empirical" else 27)
        + current_data_score
        + day_score
        + sample_score
        + daily_score
        + period_score
        - source_penalty
        - shortened_penalty,
        18,
        90,
    )
    if method == "time_slot_empirical":
        confidence = min(confidence, 58)

    status = "ok" if actual_horizon == requested_horizon else "reduced_horizon"
    if actual_horizon < requested_horizon:
        label = f"距收盘仅剩{actual_horizon}个交易分钟，预测窗口已相应缩短。"
    elif confidence < 42:
        label = "分钟样本较少，短时最高价区间仅作低置信度观察。"
    elif break_probability >= 65:
        label = "未来一小时再冲高概率偏高，仍需结合量能与VWAP确认。"
    elif break_probability >= 42:
        label = "未来一小时冲高概率中等，重点观察预测区间下沿。"
    else:
        label = "未来一小时突破今日高点概率偏低，避免把区间当成必达目标。"

    return {
        "available": True,
        "status": status,
        "requested_horizon_minutes": requested_horizon,
        "horizon_minutes": int(actual_horizon),
        "window_start": last_time.strftime("%Y-%m-%d %H:%M"),
        "window_end": window_end.strftime("%Y-%m-%d %H:%M"),
        "predicted_high": zone["center"],
        "predicted_high_zone": zone,
        "expected_upside_pct": round(q50, 2),
        "break_day_high_probability": round(break_probability, 1),
        "confidence": round(confidence, 1),
        "sample_size": int(len(samples)),
        "effective_sample_size": round(effective_n, 1),
        "method": method,
        "bar_minutes": int(bar_minutes),
        "source": source,
        "history_source": history_source,
        "quantiles_pct": {
            "q35": round(q35, 3),
            "q50": round(q50, 3),
            "q75": round(q75, 3),
            "q90": round(q90, 3),
        },
        "features": {key: round(to_float(value), 3) for key, value in feature.items()},
        "label": label,
        "explain": (
            f"使用截至{last_time.strftime('%H:%M')}的分钟线，预测其后{actual_horizon}个交易分钟内的最高价；"
            f"方法为{method}，历史样本{len(samples)}个、有效相似样本{effective_n:.1f}个。"
            "模型特征包含15/30分钟动量、实现波动率、近期振幅、区间位置、VWAP偏离、量能和时段；"
            "午间休市不计入60分钟，且不会使用预测窗口内的未来数据。"
        ),
    }
