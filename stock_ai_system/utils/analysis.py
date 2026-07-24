"""High-level analysis orchestration used by Streamlit pages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data.fundamental_data import get_basic_fundamentals
from data.sector_data import get_market_status, get_sector_context
from data.stock_data import get_hist_daily, get_minute_kline, get_quote, get_realtime_quotes, get_stock_name
from models.high_price_model import estimate_high_price_zones
from models.hourly_high_model import estimate_next_hour_high
from models.low_price_model import estimate_low_price_zones
from models.main_force_score import score_main_force
from models.potential_score import build_potential_record
from models.risk_model import assess_risk
from models.t_strategy import build_t_strategy
from utils.helpers import normalize_code, to_float


def _fallback_sector() -> dict[str, Any]:
    return {
        "industry": "待AKShare恢复",
        "sector": "待AKShare恢复",
        "sector_pct_chg": 0.0,
        "sector_rank": None,
        "sector_count": 0,
        "sector_strength": "数据不足",
        "relative_strength": 0.0,
        "leader": "",
    }


def _fallback_fundamentals(code: str, quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": code,
        "name": quote.get("name", ""),
        "industry": "待AKShare恢复",
        "concepts": "待AKShare恢复",
        "total_mv": quote.get("total_mv"),
        "circ_mv": quote.get("circ_mv"),
        "pe": quote.get("pe"),
        "pb": quote.get("pb"),
        "ps": None,
        "peg": None,
        "revenue_yoy": None,
        "net_profit_yoy": None,
        "gross_margin": None,
        "net_margin": None,
        "roe": None,
        "roa": None,
        "debt_ratio": None,
        "data_quality": "当前使用新浪/腾讯备用源，AKShare深度基本面暂不可用",
    }


def _neutral_market() -> dict[str, Any]:
    return {
        "status": "个股快速模式",
        "up_count": 0,
        "down_count": 0,
        "flat_count": 0,
        "median_pct": 0.0,
        "up_ratio": 0.0,
        "amount": 0.0,
        "risk_bias": 0.0,
    }


def analyze_stock(
    code: str,
    watch: dict[str, Any] | None = None,
    spot_df: pd.DataFrame | None = None,
    adjust: str = "qfq",
    minute_period: str = "5",
    include_minute: bool = True,
) -> dict[str, Any]:
    code = normalize_code(code)
    quote = get_quote(code, spot_df)
    if not quote.get("name"):
        quote["name"] = get_stock_name(code, spot_df)
    manual_price = to_float((watch or {}).get("manual_price"), 0)
    if manual_price > 0 and to_float(quote.get("price"), 0) <= 0:
        prev_close = to_float(quote.get("pre_close"), manual_price)
        quote.update(
            {
                "price": manual_price,
                "open": to_float(quote.get("open"), manual_price),
                "high": max(to_float(quote.get("high"), manual_price), manual_price),
                "low": min(to_float(quote.get("low"), manual_price), manual_price),
                "pre_close": prev_close,
                "pct_chg": (manual_price - prev_close) / prev_close * 100 if prev_close else 0,
                "data_source": "手动现价兜底",
            }
        )
    daily = get_hist_daily(code, adjust=adjust)
    minute = get_minute_kline(code, period=minute_period) if include_minute else pd.DataFrame()
    minute_history = pd.DataFrame()
    analysis_time = pd.Timestamp.now(tz="Asia/Shanghai")
    if include_minute:
        minute_history = get_minute_kline(
            code,
            period="5",
            start_date=(analysis_time - pd.Timedelta(days=180)).strftime("%Y-%m-%d 09:30:00"),
            end_date=analysis_time.strftime("%Y-%m-%d 15:00:00"),
            ttl=1800,
            use_realtime_fallback=False,
        )
    market = get_market_status(spot_df) if spot_df is not None and not spot_df.empty else _neutral_market()
    using_fast_fallback = bool(quote.get("data_source")) or "腾讯" in str(daily.attrs.get("source", ""))
    if using_fast_fallback:
        sector = _fallback_sector()
        fundamentals = _fallback_fundamentals(code, quote)
    else:
        sector = get_sector_context(code, quote)
        fundamentals = get_basic_fundamentals(code, quote)
    risk = assess_risk(quote, daily, fundamentals)
    low_model = estimate_low_price_zones(
        daily,
        minute,
        quote,
        market_risk_bias=market.get("risk_bias", 0.0),
    )
    high_model = estimate_high_price_zones(
        daily,
        minute,
        quote,
        market_risk_bias=market.get("risk_bias", 0.0),
    )
    next_hour_high = estimate_next_hour_high(
        daily,
        minute,
        quote,
        market_risk_bias=market.get("risk_bias", 0.0),
        horizon_minutes=60,
        historical_minute_df=minute_history,
        as_of=analysis_time,
    )
    main_force = score_main_force(daily, minute, quote, sector)
    t_strategy = build_t_strategy(
        quote,
        daily,
        minute,
        low_model,
        main_force,
        risk,
        watch,
        sector,
        high_model=high_model,
    )
    potential = build_potential_record(
        code,
        quote,
        daily,
        minute,
        fundamentals,
        sector,
        low_model,
        t_strategy,
        main_force,
        risk,
    )
    return {
        "code": code,
        "quote": quote,
        "daily": daily,
        "minute": minute,
        "market": market,
        "sector": sector,
        "fundamentals": fundamentals,
        "risk": risk,
        "low_model": low_model,
        "high_model": high_model,
        "next_hour_high": next_hour_high,
        "main_force": main_force,
        "t_strategy": t_strategy,
        "potential": potential,
    }


def analyze_many(
    items: list[dict[str, Any]] | list[str],
    include_minute: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    codes = [normalize_code(item["code"] if isinstance(item, dict) else item) for item in items]
    if limit:
        codes = codes[:limit]
    spot_df = get_realtime_quotes(codes)
    results: list[dict[str, Any]] = []
    item_by_code = {
        normalize_code(item["code"]): item for item in items if isinstance(item, dict) and item.get("code")
    }
    for code in codes:
        watch = item_by_code.get(code, {})
        try:
            results.append(analyze_stock(code, watch=watch, spot_df=spot_df, include_minute=include_minute))
        except Exception as exc:
            results.append(
                {
                    "code": code,
                    "error": str(exc),
                    "quote": {"code": code, "name": watch.get("name", "") if isinstance(watch, dict) else ""},
                }
            )
    return results
"""High-level analysis orchestration used by Streamlit pages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data.fundamental_data import get_basic_fundamentals
from data.sector_data import get_market_status, get_sector_context
from data.stock_data import get_hist_daily, get_minute_kline, get_quote, get_realtime_quotes, get_stock_name
from models.high_price_model import estimate_high_price_zones
from models.low_price_model import estimate_low_price_zones
from models.main_force_score import score_main_force
from models.potential_score import build_potential_record
from models.risk_model import assess_risk
from models.t_strategy import build_t_strategy
from utils.helpers import normalize_code, to_float


def _fallback_sector() -> dict[str, Any]:
    return {
        "industry": "待AKShare恢复",
        "sector": "待AKShare恢复",
        "sector_pct_chg": 0.0,
        "sector_rank": None,
        "sector_count": 0,
        "sector_strength": "数据不足",
        "relative_strength": 0.0,
        "leader": "",
    }


def _fallback_fundamentals(code: str, quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": code,
        "name": quote.get("name", ""),
        "industry": "待AKShare恢复",
        "concepts": "待AKShare恢复",
        "total_mv": quote.get("total_mv"),
        "circ_mv": quote.get("circ_mv"),
        "pe": quote.get("pe"),
        "pb": quote.get("pb"),
        "ps": None,
        "peg": None,
        "revenue_yoy": None,
        "net_profit_yoy": None,
        "gross_margin": None,
        "net_margin": None,
        "roe": None,
        "roa": None,
        "debt_ratio": None,
        "data_quality": "当前使用新浪/腾讯备用源，AKShare深度基本面暂不可用",
    }


def _neutral_market() -> dict[str, Any]:
    return {
        "status": "个股快速模式",
        "up_count": 0,
        "down_count": 0,
        "flat_count": 0,
        "median_pct": 0.0,
        "up_ratio": 0.0,
        "amount": 0.0,
        "risk_bias": 0.0,
    }


def analyze_stock(
    code: str,
    watch: dict[str, Any] | None = None,
    spot_df: pd.DataFrame | None = None,
    adjust: str = "qfq",
    minute_period: str = "5",
    include_minute: bool = True,
) -> dict[str, Any]:
    code = normalize_code(code)
    quote = get_quote(code, spot_df)
    if not quote.get("name"):
        quote["name"] = get_stock_name(code, spot_df)
    manual_price = to_float((watch or {}).get("manual_price"), 0)
    if manual_price > 0 and to_float(quote.get("price"), 0) <= 0:
        prev_close = to_float(quote.get("pre_close"), manual_price)
        quote.update(
            {
                "price": manual_price,
                "open": to_float(quote.get("open"), manual_price),
                "high": max(to_float(quote.get("high"), manual_price), manual_price),
                "low": min(to_float(quote.get("low"), manual_price), manual_price),
                "pre_close": prev_close,
                "pct_chg": (manual_price - prev_close) / prev_close * 100 if prev_close else 0,
                "data_source": "手动现价兜底",
            }
        )
    daily = get_hist_daily(code, adjust=adjust)
    minute = get_minute_kline(code, period=minute_period) if include_minute else pd.DataFrame()
    market = get_market_status(spot_df) if spot_df is not None and not spot_df.empty else _neutral_market()
    using_fast_fallback = bool(quote.get("data_source")) or "腾讯" in str(daily.attrs.get("source", ""))
    if using_fast_fallback:
        sector = _fallback_sector()
        fundamentals = _fallback_fundamentals(code, quote)
    else:
        sector = get_sector_context(code, quote)
        fundamentals = get_basic_fundamentals(code, quote)
    risk = assess_risk(quote, daily, fundamentals)
    low_model = estimate_low_price_zones(
        daily,
        minute,
        quote,
        market_risk_bias=market.get("risk_bias", 0.0),
    )
    high_model = estimate_high_price_zones(
        daily,
        minute,
        quote,
        market_risk_bias=market.get("risk_bias", 0.0),
    )
    main_force = score_main_force(daily, minute, quote, sector)
    t_strategy = build_t_strategy(
        quote,
        daily,
        minute,
        low_model,
        main_force,
        risk,
        watch,
        sector,
        high_model=high_model,
    )
    potential = build_potential_record(
        code,
        quote,
        daily,
        minute,
        fundamentals,
        sector,
        low_model,
        t_strategy,
        main_force,
        risk,
    )
    return {
        "code": code,
        "quote": quote,
        "daily": daily,
        "minute": minute,
        "market": market,
        "sector": sector,
        "fundamentals": fundamentals,
        "risk": risk,
        "low_model": low_model,
        "high_model": high_model,
        "main_force": main_force,
        "t_strategy": t_strategy,
        "potential": potential,
    }


def analyze_many(
    items: list[dict[str, Any]] | list[str],
    include_minute: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    codes = [normalize_code(item["code"] if isinstance(item, dict) else item) for item in items]
    if limit:
        codes = codes[:limit]
    spot_df = get_realtime_quotes(codes)
    results: list[dict[str, Any]] = []
    item_by_code = {
        normalize_code(item["code"]): item for item in items if isinstance(item, dict) and item.get("code")
    }
    for code in codes:
        watch = item_by_code.get(code, {})
        try:
            results.append(analyze_stock(code, watch=watch, spot_df=spot_df, include_minute=include_minute))
        except Exception as exc:
            results.append(
                {
                    "code": code,
                    "error": str(exc),
                    "quote": {"code": code, "name": watch.get("name", "") if isinstance(watch, dict) else ""},
                }
            )
    return results
