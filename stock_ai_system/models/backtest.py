"""Simple backtest utilities for the MVP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from models.indicators import enrich_indicators
from models.hourly_high_model import (
    _clean_minute_data,
    _infer_bar_minutes,
    _steps_for,
    _trading_index,
    estimate_next_hour_high,
)
from models.low_price_model import estimate_low_price_zones
from models.potential_score import score_short_potential
from models.risk_model import assess_risk
from utils.helpers import safe_div, to_float


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = (1 + pd.Series(returns) / 100).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    return round(float(drawdown.min()), 2)


def _max_consecutive_losses(returns: list[float]) -> int:
    max_losses = 0
    current = 0
    for item in returns:
        if item < 0:
            current += 1
            max_losses = max(max_losses, current)
        else:
            current = 0
    return max_losses


def summarize_returns(returns: list[float]) -> dict[str, Any]:
    wins = [item for item in returns if item > 0]
    losses = [item for item in returns if item <= 0]
    return {
        "样本数": len(returns),
        "胜率": round(len(wins) / max(len(returns), 1) * 100, 2),
        "平均收益": round(float(np.mean(returns)) if returns else 0, 2),
        "最大回撤": _max_drawdown(returns),
        "盈亏比": round(abs(np.mean(wins) / np.mean(losses)), 2) if wins and losses and np.mean(losses) != 0 else 0,
        "连续失败次数": _max_consecutive_losses(returns),
    }


def run_low_zone_backtest(daily_df: pd.DataFrame, lookback: int = 80) -> pd.DataFrame:
    data = enrich_indicators(daily_df)
    if len(data) < 80:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    start = max(60, len(data) - lookback)
    for idx in range(start, len(data)):
        past = data.iloc[:idx].copy()
        day = data.iloc[idx]
        prev = past.iloc[-1]
        quote = {
            "price": prev["close"],
            "pre_close": prev["close"],
            "low": prev["low"],
            "high": prev["high"],
            "limit_down": prev["close"] * 0.9,
        }
        zones = estimate_low_price_zones(past, None, quote)
        first = zones["first_low_zone"]
        second = zones["second_low_zone"]
        extreme = zones["extreme_low_zone"]
        day_low = to_float(day["low"])
        day_high = to_float(day["high"])
        rows.append(
            {
                "date": day["date"],
                "actual_low": day_low,
                "first_zone": f"{first['low']:.2f}-{first['high']:.2f}",
                "second_zone": f"{second['low']:.2f}-{second['high']:.2f}",
                "extreme_zone": f"{extreme['low']:.2f}-{extreme['high']:.2f}",
                "hit_first": day_low <= first["high"] and day_high >= first["low"],
                "hit_second": day_low <= second["high"] and day_high >= second["low"],
                "hit_extreme": day_low <= extreme["high"] and day_high >= extreme["low"],
            }
        )
    return pd.DataFrame(rows)


def run_t_strategy_backtest(daily_df: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    data = enrich_indicators(daily_df)
    if len(data) < 80:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    start = max(60, len(data) - lookback)
    for idx in range(start, len(data)):
        past = data.iloc[:idx].copy()
        day = data.iloc[idx]
        prev = past.iloc[-1]
        zones = estimate_low_price_zones(
            past,
            None,
            {
                "price": prev["close"],
                "pre_close": prev["close"],
                "low": prev["low"],
                "high": prev["high"],
                "limit_down": prev["close"] * 0.9,
            },
        )
        first = zones["first_low_zone"]
        atr14 = to_float(past.iloc[-1].get("atr14"), prev["close"] * 0.015)
        buy_price = first["center"]
        target_price = buy_price + atr14 * 0.65
        stop_price = zones["second_low_zone"]["low"] * 0.99
        touched_buy = to_float(day["low"]) <= first["high"] and to_float(day["high"]) >= first["low"]
        if not touched_buy:
            continue
        if to_float(day["high"]) >= target_price:
            result = safe_div(target_price - buy_price, buy_price) * 100
        elif to_float(day["low"]) <= stop_price:
            result = safe_div(stop_price - buy_price, buy_price) * 100
        else:
            result = safe_div(to_float(day["close"]) - buy_price, buy_price) * 100
        rows.append(
            {
                "date": day["date"],
                "buy_zone": f"{first['low']:.2f}-{first['high']:.2f}",
                "target_price": round(target_price, 2),
                "stop_price": round(stop_price, 2),
                "return_pct": round(result, 2),
                "win": result > 0,
            }
        )
    return pd.DataFrame(rows)


def _quote_from_daily_row(row: pd.Series) -> dict[str, Any]:
    return {
        "price": to_float(row.get("close")),
        "pre_close": to_float(row.get("close")),
        "open": to_float(row.get("open")),
        "high": to_float(row.get("high")),
        "low": to_float(row.get("low")),
        "pct_chg": to_float(row.get("pct_chg"), to_float(row.get("daily_return"), 0)),
        "volume": to_float(row.get("volume")),
        "amount": to_float(row.get("amount")),
        "volume_ratio": 1,
    }


def run_potential_backtest(histories: dict[str, pd.DataFrame], lookback: int = 120, top_n: int = 10) -> dict[str, Any]:
    scored_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for code, daily in histories.items():
        data = enrich_indicators(daily)
        if len(data) < 90:
            continue
        start = max(60, len(data) - lookback - 6)
        for idx in range(start, len(data) - 6):
            window = data.iloc[: idx + 1].copy()
            row = data.iloc[idx]
            quote = _quote_from_daily_row(row)
            risk = assess_risk(quote, window, {})
            short = score_short_potential(
                window,
                None,
                quote,
                {"sector_strength": "中性", "sector_pct_chg": 0, "relative_strength": 0},
                {"score": 50, "tail_risk": 35},
                risk,
            )
            next1 = safe_div(data.iloc[idx + 1]["close"] - row["close"], row["close"]) * 100
            next3 = safe_div(data.iloc[idx + 3]["close"] - row["close"], row["close"]) * 100
            next5 = safe_div(data.iloc[idx + 5]["close"] - row["close"], row["close"]) * 100
            scored_by_date[pd.to_datetime(row["date"])].append(
                {
                    "date": row["date"],
                    "code": code,
                    "short_score": short["score"],
                    "next1_return": round(next1, 2),
                    "next3_return": round(next3, 2),
                    "next5_return": round(next5, 2),
                }
            )

    picks: list[dict[str, Any]] = []
    for _, rows in scored_by_date.items():
        top = sorted(rows, key=lambda item: item["short_score"], reverse=True)[:top_n]
        picks.extend(top)
    picks_df = pd.DataFrame(picks)
    if picks_df.empty:
        return {"picks": picks_df, "summary": {}, "threshold_summary": pd.DataFrame()}

    summary = {
        "次日": summarize_returns(picks_df["next1_return"].tolist()),
        "未来3日": summarize_returns(picks_df["next3_return"].tolist()),
        "未来5日": summarize_returns(picks_df["next5_return"].tolist()),
    }
    threshold_rows = []
    for threshold in [70, 75, 80]:
        subset = picks_df[picks_df["short_score"] >= threshold]
        threshold_rows.append(
            {
                "评分阈值": threshold,
                **{f"次日{key}": value for key, value in summarize_returns(subset["next1_return"].tolist()).items()},
                **{f"5日{key}": value for key, value in summarize_returns(subset["next5_return"].tolist()).items()},
            }
        )
    return {
        "picks": picks_df,
        "summary": summary,
        "threshold_summary": pd.DataFrame(threshold_rows),
    }


def run_next_hour_high_backtest(
    minute_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
    horizon_minutes: int = 60,
    max_anchors: int = 60,
) -> pd.DataFrame:
    """Walk-forward validation for the next-trading-hour high forecast.

    Each prediction receives only minute rows at or before its anchor.  Daily
    indicators are restricted to completed dates strictly before the anchor's
    session, preventing current-day and future-day leakage.
    """

    data = _clean_minute_data(minute_df)
    if data.empty:
        return pd.DataFrame()
    source_attrs = dict(getattr(minute_df, "attrs", {}))
    bar_minutes = _infer_bar_minutes(data)
    horizon_bars = _steps_for(horizon_minutes, bar_minutes)
    context_bars = max(4, _steps_for(30, bar_minutes))
    anchor_stride = max(1, _steps_for(30, bar_minutes))
    candidates: list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []

    for _, session in data.groupby(data["datetime"].dt.date, sort=True):
        session = session.sort_values("datetime").reset_index(drop=True)
        for anchor in range(context_bars - 1, len(session) - horizon_bars, anchor_stride):
            prefix = session.iloc[: anchor + 1].copy()
            future = session.iloc[anchor + 1 : anchor + 1 + horizon_bars].copy()
            if len(future) < horizon_bars:
                continue
            start_index = _trading_index(pd.to_datetime(prefix["datetime"].iloc[-1]))
            end_index = _trading_index(pd.to_datetime(future["datetime"].iloc[-1]))
            if start_index is None or end_index is None:
                continue
            if end_index - start_index < horizon_minutes * 0.72:
                continue
            candidates.append((pd.to_datetime(prefix["datetime"].iloc[-1]), prefix, future))

    if not candidates:
        return pd.DataFrame()
    candidates = candidates[-max(int(max_anchors), 1) :]
    rows: list[dict[str, Any]] = []
    daily = daily_df.copy() if daily_df is not None else pd.DataFrame()
    if not daily.empty and "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    for anchor_time, session_prefix, future in candidates:
        history = data[data["datetime"] <= anchor_time].copy()
        history.attrs.update(source_attrs)
        session_prefix.attrs.update(source_attrs)
        if daily.empty or "date" not in daily.columns:
            daily_past = pd.DataFrame()
        else:
            daily_past = daily[daily["date"].dt.date < anchor_time.date()].copy()

        price = to_float(session_prefix["close"].iloc[-1], 0)
        session_open = to_float(
            session_prefix["open"].iloc[0] if "open" in session_prefix.columns else price,
            price,
        )
        known_high = to_float(pd.to_numeric(session_prefix["high"], errors="coerce").max(), price)
        known_low = to_float(
            pd.to_numeric(
                session_prefix.get("low", session_prefix["close"]),
                errors="coerce",
            ).min(),
            price,
        )
        if not daily_past.empty:
            prev_close = to_float(daily_past.iloc[-1].get("close"), session_open)
        else:
            prev_close = session_open
        volume = pd.to_numeric(
            session_prefix.get("volume", pd.Series(index=session_prefix.index, dtype=float)),
            errors="coerce",
        ).fillna(0)
        volume_ratio = safe_div(to_float(volume.tail(3).mean(), 0), to_float(volume.mean(), 0))
        quote = {
            "price": price,
            "open": session_open,
            "high": known_high,
            "low": known_low,
            "pre_close": prev_close,
            "pct_chg": safe_div(price - prev_close, prev_close) * 100,
            "volume_ratio": volume_ratio or 1.0,
            "limit_up": prev_close * 1.1 if prev_close > 0 else price * 1.1,
        }
        prediction = estimate_next_hour_high(
            daily_past,
            session_prefix,
            quote,
            horizon_minutes=horizon_minutes,
            historical_minute_df=history,
            as_of=anchor_time,
        )
        if not prediction.get("available"):
            continue

        actual_high = to_float(pd.to_numeric(future["high"], errors="coerce").max(), 0)
        predicted_high = to_float(prediction.get("predicted_high"), 0)
        zone = prediction.get("predicted_high_zone", {})
        zone_low = to_float(zone.get("low"), 0)
        zone_high = to_float(zone.get("high"), 0)
        actual_break = actual_high > known_high + max(price * 0.0001, 0.005)
        predicted_break_probability = to_float(prediction.get("break_day_high_probability"), 0) / 100
        error_pct = safe_div(predicted_high - actual_high, actual_high) * 100
        rows.append(
            {
                "anchor_time": anchor_time,
                "window_end": prediction.get("window_end", ""),
                "actual_high": round(actual_high, 2),
                "predicted_high": round(predicted_high, 2),
                "zone_low": round(zone_low, 2),
                "zone_high": round(zone_high, 2),
                "interval_hit": zone_low <= actual_high <= zone_high,
                "lower_bound_reached": actual_high >= zone_low,
                "error_pct": round(error_pct, 3),
                "absolute_error_pct": round(abs(error_pct), 3),
                "actual_break_day_high": actual_break,
                "predicted_break_probability": round(predicted_break_probability * 100, 1),
                "brier_score": round(
                    (predicted_break_probability - float(actual_break)) ** 2,
                    4,
                ),
                "confidence": prediction.get("confidence", 0),
                "method": prediction.get("method", ""),
                "sample_size": prediction.get("sample_size", 0),
                "effective_sample_size": prediction.get("effective_sample_size", 0),
            }
        )
    return pd.DataFrame(rows)


def summarize_next_hour_high_backtest(result: pd.DataFrame) -> dict[str, Any]:
    if result is None or result.empty:
        return {
            "样本数": 0,
            "区间覆盖率": 0.0,
            "区间下沿触达率": 0.0,
            "中位绝对误差": 0.0,
            "平均预测偏差": 0.0,
            "突破概率Brier分数": 0.0,
            "平均置信度": 0.0,
        }
    return {
        "样本数": int(len(result)),
        "区间覆盖率": round(float(result["interval_hit"].mean()) * 100, 2),
        "区间下沿触达率": round(float(result["lower_bound_reached"].mean()) * 100, 2),
        "中位绝对误差": round(float(result["absolute_error_pct"].median()), 3),
        "平均预测偏差": round(float(result["error_pct"].mean()), 3),
        "突破概率Brier分数": round(float(result["brier_score"].mean()), 4),
        "平均置信度": round(float(pd.to_numeric(result["confidence"], errors="coerce").mean()), 2),
    }
"""Simple backtest utilities for the MVP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from models.indicators import enrich_indicators
from models.low_price_model import estimate_low_price_zones
from models.potential_score import score_short_potential
from models.risk_model import assess_risk
from utils.helpers import safe_div, to_float


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = (1 + pd.Series(returns) / 100).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    return round(float(drawdown.min()), 2)


def _max_consecutive_losses(returns: list[float]) -> int:
    max_losses = 0
    current = 0
    for item in returns:
        if item < 0:
            current += 1
            max_losses = max(max_losses, current)
        else:
            current = 0
    return max_losses


def summarize_returns(returns: list[float]) -> dict[str, Any]:
    wins = [item for item in returns if item > 0]
    losses = [item for item in returns if item <= 0]
    return {
        "样本数": len(returns),
        "胜率": round(len(wins) / max(len(returns), 1) * 100, 2),
        "平均收益": round(float(np.mean(returns)) if returns else 0, 2),
        "最大回撤": _max_drawdown(returns),
        "盈亏比": round(abs(np.mean(wins) / np.mean(losses)), 2) if wins and losses and np.mean(losses) != 0 else 0,
        "连续失败次数": _max_consecutive_losses(returns),
    }


def run_low_zone_backtest(daily_df: pd.DataFrame, lookback: int = 80) -> pd.DataFrame:
    data = enrich_indicators(daily_df)
    if len(data) < 80:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    start = max(60, len(data) - lookback)
    for idx in range(start, len(data)):
        past = data.iloc[:idx].copy()
        day = data.iloc[idx]
        prev = past.iloc[-1]
        quote = {
            "price": prev["close"],
            "pre_close": prev["close"],
            "low": prev["low"],
            "high": prev["high"],
            "limit_down": prev["close"] * 0.9,
        }
        zones = estimate_low_price_zones(past, None, quote)
        first = zones["first_low_zone"]
        second = zones["second_low_zone"]
        extreme = zones["extreme_low_zone"]
        day_low = to_float(day["low"])
        day_high = to_float(day["high"])
        rows.append(
            {
                "date": day["date"],
                "actual_low": day_low,
                "first_zone": f"{first['low']:.2f}-{first['high']:.2f}",
                "second_zone": f"{second['low']:.2f}-{second['high']:.2f}",
                "extreme_zone": f"{extreme['low']:.2f}-{extreme['high']:.2f}",
                "hit_first": day_low <= first["high"] and day_high >= first["low"],
                "hit_second": day_low <= second["high"] and day_high >= second["low"],
                "hit_extreme": day_low <= extreme["high"] and day_high >= extreme["low"],
            }
        )
    return pd.DataFrame(rows)


def run_t_strategy_backtest(daily_df: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    data = enrich_indicators(daily_df)
    if len(data) < 80:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    start = max(60, len(data) - lookback)
    for idx in range(start, len(data)):
        past = data.iloc[:idx].copy()
        day = data.iloc[idx]
        prev = past.iloc[-1]
        zones = estimate_low_price_zones(
            past,
            None,
            {
                "price": prev["close"],
                "pre_close": prev["close"],
                "low": prev["low"],
                "high": prev["high"],
                "limit_down": prev["close"] * 0.9,
            },
        )
        first = zones["first_low_zone"]
        atr14 = to_float(past.iloc[-1].get("atr14"), prev["close"] * 0.015)
        buy_price = first["center"]
        target_price = buy_price + atr14 * 0.65
        stop_price = zones["second_low_zone"]["low"] * 0.99
        touched_buy = to_float(day["low"]) <= first["high"] and to_float(day["high"]) >= first["low"]
        if not touched_buy:
            continue
        if to_float(day["high"]) >= target_price:
            result = safe_div(target_price - buy_price, buy_price) * 100
        elif to_float(day["low"]) <= stop_price:
            result = safe_div(stop_price - buy_price, buy_price) * 100
        else:
            result = safe_div(to_float(day["close"]) - buy_price, buy_price) * 100
        rows.append(
            {
                "date": day["date"],
                "buy_zone": f"{first['low']:.2f}-{first['high']:.2f}",
                "target_price": round(target_price, 2),
                "stop_price": round(stop_price, 2),
                "return_pct": round(result, 2),
                "win": result > 0,
            }
        )
    return pd.DataFrame(rows)


def _quote_from_daily_row(row: pd.Series) -> dict[str, Any]:
    return {
        "price": to_float(row.get("close")),
        "pre_close": to_float(row.get("close")),
        "open": to_float(row.get("open")),
        "high": to_float(row.get("high")),
        "low": to_float(row.get("low")),
        "pct_chg": to_float(row.get("pct_chg"), to_float(row.get("daily_return"), 0)),
        "volume": to_float(row.get("volume")),
        "amount": to_float(row.get("amount")),
        "volume_ratio": 1,
    }


def run_potential_backtest(histories: dict[str, pd.DataFrame], lookback: int = 120, top_n: int = 10) -> dict[str, Any]:
    scored_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for code, daily in histories.items():
        data = enrich_indicators(daily)
        if len(data) < 90:
            continue
        start = max(60, len(data) - lookback - 6)
        for idx in range(start, len(data) - 6):
            window = data.iloc[: idx + 1].copy()
            row = data.iloc[idx]
            quote = _quote_from_daily_row(row)
            risk = assess_risk(quote, window, {})
            short = score_short_potential(
                window,
                None,
                quote,
                {"sector_strength": "中性", "sector_pct_chg": 0, "relative_strength": 0},
                {"score": 50, "tail_risk": 35},
                risk,
            )
            next1 = safe_div(data.iloc[idx + 1]["close"] - row["close"], row["close"]) * 100
            next3 = safe_div(data.iloc[idx + 3]["close"] - row["close"], row["close"]) * 100
            next5 = safe_div(data.iloc[idx + 5]["close"] - row["close"], row["close"]) * 100
            scored_by_date[pd.to_datetime(row["date"])].append(
                {
                    "date": row["date"],
                    "code": code,
                    "short_score": short["score"],
                    "next1_return": round(next1, 2),
                    "next3_return": round(next3, 2),
                    "next5_return": round(next5, 2),
                }
            )

    picks: list[dict[str, Any]] = []
    for _, rows in scored_by_date.items():
        top = sorted(rows, key=lambda item: item["short_score"], reverse=True)[:top_n]
        picks.extend(top)
    picks_df = pd.DataFrame(picks)
    if picks_df.empty:
        return {"picks": picks_df, "summary": {}, "threshold_summary": pd.DataFrame()}

    summary = {
        "次日": summarize_returns(picks_df["next1_return"].tolist()),
        "未来3日": summarize_returns(picks_df["next3_return"].tolist()),
        "未来5日": summarize_returns(picks_df["next5_return"].tolist()),
    }
    threshold_rows = []
    for threshold in [70, 75, 80]:
        subset = picks_df[picks_df["short_score"] >= threshold]
        threshold_rows.append(
            {
                "评分阈值": threshold,
                **{f"次日{key}": value for key, value in summarize_returns(subset["next1_return"].tolist()).items()},
                **{f"5日{key}": value for key, value in summarize_returns(subset["next5_return"].tolist()).items()},
            }
        )
    return {
        "picks": picks_df,
        "summary": summary,
        "threshold_summary": pd.DataFrame(threshold_rows),
    }
