"""Technical indicators used by the MVP models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.helpers import safe_div, to_float


def sma(series: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=1).mean()


def bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    mid = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std().fillna(0)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + num_std * std,
            "bb_lower": mid - num_std * std,
        }
    )


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(series: pd.Series) -> pd.DataFrame:
    close = pd.to_numeric(series, errors="coerce")
    dif = ema(close, 12) - ema(close, 26)
    dea = ema(dif, 9)
    hist = (dif - dea) * 2
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": hist})


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    for window in [5, 10, 20, 30, 60, 120, 250]:
        out[f"ma{window}"] = sma(close, window)
    out["atr14"] = atr(out, 14)
    out["rsi6"] = rsi(close, 6)
    out["rsi14"] = rsi(close, 14)
    out = pd.concat([out, bollinger(out), macd(close)], axis=1)
    out["vol_ma5"] = sma(out["volume"], 5) if "volume" in out.columns else np.nan
    out["vol_ma20"] = sma(out["volume"], 20) if "volume" in out.columns else np.nan
    out["amount_ma20"] = sma(out["amount"], 20) if "amount" in out.columns else np.nan
    out["daily_return"] = close.pct_change() * 100
    return out


def calculate_intraday_vwap(minute_df: pd.DataFrame) -> float:
    if minute_df is None or minute_df.empty:
        return np.nan
    amount = pd.to_numeric(minute_df.get("amount"), errors="coerce").fillna(0)
    volume = pd.to_numeric(minute_df.get("volume"), errors="coerce").fillna(0) * 100
    total_volume = volume.sum()
    if total_volume <= 0:
        return np.nan
    return float(amount.sum() / total_volume)


def add_intraday_vwap(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()
    out = minute_df.copy()
    amount = pd.to_numeric(out.get("amount"), errors="coerce").fillna(0)
    volume = pd.to_numeric(out.get("volume"), errors="coerce").fillna(0) * 100
    cumulative_volume = volume.cumsum().replace(0, np.nan)
    out["vwap"] = amount.cumsum() / cumulative_volume
    return out


def support_resistance(df: pd.DataFrame, window: int = 60) -> dict[str, float]:
    data = enrich_indicators(df).tail(window)
    if data.empty:
        return {"support": np.nan, "resistance": np.nan}
    latest = data.iloc[-1]
    supports = [
        data["low"].tail(5).min(),
        data["low"].tail(10).min(),
        data["low"].tail(20).min(),
        latest.get("ma5"),
        latest.get("ma10"),
        latest.get("ma20"),
        latest.get("bb_lower"),
    ]
    resistances = [
        data["high"].tail(5).max(),
        data["high"].tail(10).max(),
        data["high"].tail(20).max(),
        latest.get("ma5"),
        latest.get("ma10"),
        latest.get("ma20"),
        latest.get("bb_upper"),
    ]
    supports = [to_float(x, np.nan) for x in supports if pd.notna(to_float(x, np.nan))]
    resistances = [to_float(x, np.nan) for x in resistances if pd.notna(to_float(x, np.nan))]
    close = to_float(latest.get("close"), 0)
    support_candidates = [value for value in supports if value <= close * 1.01]
    resistance_candidates = [value for value in resistances if value >= close * 0.99]
    support = max(support_candidates) if support_candidates else (min(supports) if supports else np.nan)
    resistance = (
        min(resistance_candidates) if resistance_candidates else (max(resistances) if resistances else np.nan)
    )
    return {"support": support, "resistance": resistance}


def latest_snapshot(df: pd.DataFrame) -> dict[str, float]:
    enriched = enrich_indicators(df)
    if enriched.empty:
        return {}
    row = enriched.iloc[-1]
    keys = [
        "close",
        "high",
        "low",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "atr14",
        "bb_upper",
        "bb_lower",
        "rsi14",
        "macd_hist",
        "vol_ma20",
        "amount_ma20",
    ]
    return {key: to_float(row.get(key), np.nan) for key in keys}


def trend_strength(df: pd.DataFrame) -> float:
    enriched = enrich_indicators(df)
    if enriched.empty:
        return 50.0
    row = enriched.iloc[-1]
    close = to_float(row.get("close"))
    points = 0.0
    points += 15 if close >= to_float(row.get("ma5")) else 5
    points += 15 if to_float(row.get("ma5")) >= to_float(row.get("ma10")) else 5
    points += 15 if to_float(row.get("ma10")) >= to_float(row.get("ma20")) else 5
    points += 15 if close >= to_float(row.get("ma20")) else 5
    if len(enriched) >= 20:
        ma20_slope = to_float(row.get("ma20")) - to_float(enriched.iloc[-10].get("ma20"))
        points += 15 if ma20_slope > 0 else 5
    else:
        points += 8
    rsi14 = to_float(row.get("rsi14"), 50)
    points += 15 if 45 <= rsi14 <= 72 else 8 if rsi14 < 80 else 3
    points += 10 if to_float(row.get("macd_hist")) >= 0 else 4
    return float(min(100, points))


def rolling_return(df: pd.DataFrame, days: int) -> float:
    if df is None or len(df) <= days:
        return 0.0
    close = pd.to_numeric(df["close"], errors="coerce")
    return safe_div(close.iloc[-1] - close.iloc[-days - 1], close.iloc[-days - 1]) * 100
