from __future__ import annotations

import unittest
import sys
import types

import numpy as np
import pandas as pd

# The production image installs Streamlit.  The bundled validation runtime does
# not need it, so provide the smallest import stub required by utils.helpers.
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.SimpleNamespace()

from models.backtest import run_next_hour_high_backtest
from models.hourly_high_model import estimate_next_hour_high


def _daily_data(periods: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-10-01", periods=periods)
    base = 10 + np.linspace(0, 1.5, periods) + np.sin(np.arange(periods) / 8) * 0.15
    close = base + np.sin(np.arange(periods) / 3) * 0.03
    return pd.DataFrame(
        {
            "date": dates,
            "open": base - 0.02,
            "close": close,
            "high": np.maximum(base, close) + 0.12,
            "low": np.minimum(base, close) - 0.12,
            "volume": 100_000 + np.arange(periods) * 100,
            "amount": close * (100_000 + np.arange(periods) * 100) * 100,
            "pct_chg": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
        }
    )


def _session_times(date: pd.Timestamp) -> pd.DatetimeIndex:
    morning = pd.date_range(
        date.normalize() + pd.Timedelta(hours=9, minutes=35),
        date.normalize() + pd.Timedelta(hours=11, minutes=30),
        freq="5min",
    )
    afternoon = pd.date_range(
        date.normalize() + pd.Timedelta(hours=13, minutes=5),
        date.normalize() + pd.Timedelta(hours=15),
        freq="5min",
    )
    return morning.append(afternoon)


def _minute_data(days: int = 35, latest_end: str | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", periods=days)
    frames = []
    prior_close = 11.0
    for day_no, date in enumerate(dates):
        times = _session_times(date)
        phase = np.arange(len(times))
        close = (
            prior_close
            + np.sin(phase / 5 + day_no / 4) * 0.035
            + phase * 0.0015
            + day_no * 0.002
        )
        frame = pd.DataFrame(
            {
                "datetime": times,
                "open": np.r_[prior_close, close[:-1]],
                "close": close,
                "high": np.maximum(np.r_[prior_close, close[:-1]], close) + 0.015,
                "low": np.minimum(np.r_[prior_close, close[:-1]], close) - 0.015,
                "volume": 800 + (phase % 8) * 40 + day_no,
            }
        )
        frame["amount"] = frame["close"] * frame["volume"] * 100
        frames.append(frame)
        prior_close = float(close[-1])
    result = pd.concat(frames, ignore_index=True)
    if latest_end:
        latest_date = dates[-1].date()
        cutoff = pd.Timestamp(f"{latest_date} {latest_end}")
        result = result[
            (result["datetime"].dt.date < latest_date) | (result["datetime"] <= cutoff)
        ].copy()
    result.attrs["source"] = "AKShare 5分钟线"
    result.attrs["bar_minutes"] = 5
    result.attrs["is_synthetic_ohlc"] = False
    return result.reset_index(drop=True)


def _quote(minute: pd.DataFrame) -> dict:
    latest_date = minute["datetime"].iloc[-1].date()
    session = minute[minute["datetime"].dt.date == latest_date]
    price = float(session["close"].iloc[-1])
    pre_close = float(minute[minute["datetime"].dt.date < latest_date]["close"].iloc[-1])
    return {
        "price": price,
        "open": float(session["open"].iloc[0]),
        "high": float(session["high"].max()),
        "low": float(session["low"].min()),
        "pre_close": pre_close,
        "pct_chg": (price / pre_close - 1) * 100,
        "volume_ratio": 1.15,
        "limit_up": pre_close * 1.1,
    }


class HourlyHighModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.daily = _daily_data()

    def test_crosses_lunch_break_and_preserves_interval_order(self) -> None:
        minute = _minute_data(latest_end="11:15")
        result = estimate_next_hour_high(
            self.daily,
            minute,
            _quote(minute),
            historical_minute_df=minute,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["horizon_minutes"], 60)
        self.assertTrue(result["window_end"].endswith("13:45"))
        zone = result["predicted_high_zone"]
        self.assertLessEqual(zone["low"], zone["center"])
        self.assertLessEqual(zone["center"], zone["high"])
        self.assertGreaterEqual(zone["low"], _quote(minute)["price"])
        self.assertLessEqual(zone["high"], _quote(minute)["limit_up"])
        self.assertLessEqual(result["sample_size"], minute["datetime"].dt.date.nunique())

    def test_horizon_is_shortened_near_close(self) -> None:
        minute = _minute_data(latest_end="14:30")
        result = estimate_next_hour_high(
            self.daily,
            minute,
            _quote(minute),
            historical_minute_df=minute,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["status"], "reduced_horizon")
        self.assertEqual(result["horizon_minutes"], 30)
        self.assertTrue(result["window_end"].endswith("15:00"))

    def test_market_close_and_missing_minute_data_are_unavailable(self) -> None:
        closed = _minute_data(latest_end="15:00")
        closed_result = estimate_next_hour_high(
            self.daily,
            closed,
            _quote(closed),
            historical_minute_df=closed,
        )
        self.assertFalse(closed_result["available"])
        self.assertEqual(closed_result["status"], "market_closed")

        missing = estimate_next_hour_high(self.daily, pd.DataFrame(), {"price": 10.0})
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["predicted_high"])
        self.assertEqual(missing["predicted_high_zone"], {})

    def test_synthetic_high_low_is_not_used_as_training_target(self) -> None:
        minute = _minute_data(latest_end="10:30")
        synthetic_history = minute.copy()
        synthetic_history.attrs["source"] = "腾讯分钟线备用源"
        synthetic_history.attrs["is_synthetic_ohlc"] = True
        result = estimate_next_hour_high(
            self.daily,
            minute,
            _quote(minute),
            historical_minute_df=synthetic_history,
        )
        self.assertEqual(result["method"], "atr_realized_volatility_fallback")
        self.assertEqual(result["sample_size"], 0)

    def test_as_of_truncates_current_session_and_history(self) -> None:
        minute = _minute_data(days=12, latest_end="15:00")
        latest_date = minute["datetime"].dt.date.max()
        cutoff = pd.Timestamp.combine(latest_date, pd.Timestamp("10:30").time())
        prefix = minute[minute["datetime"] <= cutoff].copy()
        first = estimate_next_hour_high(
            self.daily,
            minute,
            _quote(prefix),
            historical_minute_df=minute,
            as_of=cutoff,
        )

        changed = minute.copy()
        changed.attrs.update(minute.attrs)
        future_mask = changed["datetime"] > cutoff
        changed.loc[future_mask, ["open", "close", "high", "low"]] *= 1.08
        second = estimate_next_hour_high(
            self.daily,
            changed,
            _quote(prefix),
            historical_minute_df=changed,
            as_of=cutoff,
        )

        self.assertTrue(first["available"])
        self.assertEqual(first["predicted_high"], second["predicted_high"])
        self.assertEqual(first["predicted_high_zone"], second["predicted_high_zone"])
        self.assertEqual(first["break_day_high_probability"], second["break_day_high_probability"])

    def test_walk_forward_prediction_does_not_use_future_window(self) -> None:
        minute = _minute_data(days=12)
        original = run_next_hour_high_backtest(
            minute,
            self.daily,
            horizon_minutes=60,
            max_anchors=1,
        )
        self.assertFalse(original.empty)

        changed = minute.copy()
        changed.attrs.update(minute.attrs)
        latest_date = changed["datetime"].dt.date.max()
        future_mask = (
            (changed["datetime"].dt.date == latest_date)
            & (changed["datetime"].dt.time > pd.Timestamp("14:00").time())
        )
        changed.loc[future_mask, ["open", "close", "high", "low"]] *= 1.08
        mutated = run_next_hour_high_backtest(
            changed,
            self.daily,
            horizon_minutes=60,
            max_anchors=1,
        )
        self.assertFalse(mutated.empty)
        self.assertEqual(original.iloc[0]["predicted_high"], mutated.iloc[0]["predicted_high"])
        self.assertNotEqual(original.iloc[0]["actual_high"], mutated.iloc[0]["actual_high"])


if __name__ == "__main__":
    unittest.main()
