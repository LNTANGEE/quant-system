"""Company profile and fundamental data adapters."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from data.stock_data import _run_with_timeout, get_quote
from utils.helpers import normalize_code, to_float
from utils.logger import get_logger


logger = get_logger(__name__)


def _akshare():
    try:
        import akshare as ak  # type: ignore

        return ak
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("未安装或无法导入 AKShare，请先执行 pip install -r requirements.txt") from exc


def _bucket(seconds: int) -> int:
    return int(time.time() // max(seconds, 1))


def _find_value(row: pd.Series, candidates: list[str], default: float | str | None = None) -> Any:
    for keyword in candidates:
        for col in row.index:
            if keyword in str(col):
                value = row[col]
                if pd.notna(value):
                    return value
    return default


@lru_cache(maxsize=512)
def _cached_profile(code: str, cache_bucket: int) -> dict[str, Any]:
    del cache_bucket
    try:
        ak = _akshare()
        df = _run_with_timeout(ak.stock_individual_info_em, symbol=normalize_code(code))
        if not df.empty and {"item", "value"}.issubset(df.columns):
            return {str(k): v for k, v in zip(df["item"], df["value"])}
    except Exception:
        logger.info("company profile failed: %s", code)
    return {}


def get_company_profile(code: str, ttl: int = 3600) -> dict[str, Any]:
    return dict(_cached_profile(normalize_code(code), _bucket(ttl)))


@lru_cache(maxsize=512)
def _cached_financial_indicators(code: str, start_year: str, cache_bucket: int) -> dict[str, Any]:
    del cache_bucket
    try:
        ak = _akshare()
        df = _run_with_timeout(
            ak.stock_financial_analysis_indicator,
            symbol=normalize_code(code),
            start_year=start_year,
        )
        if df.empty:
            return {}
        date_col = next((col for col in df.columns if "日期" in str(col) or "报告" in str(col)), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col)
        latest = df.iloc[-1]
        return {
            "report_date": str(latest.get(date_col, ""))[:10] if date_col else "",
            "revenue_yoy": to_float(
                _find_value(latest, ["营业总收入同比", "主营业务收入增长率", "营业收入增长率"], np.nan),
                np.nan,
            ),
            "net_profit_yoy": to_float(
                _find_value(latest, ["净利润同比", "净利润增长率"], np.nan),
                np.nan,
            ),
            "deducted_profit_yoy": to_float(
                _find_value(latest, ["扣非", "扣除非经常性损益后的净利润增长率"], np.nan),
                np.nan,
            ),
            "gross_margin": to_float(_find_value(latest, ["销售毛利率", "毛利率"], np.nan), np.nan),
            "net_margin": to_float(_find_value(latest, ["销售净利率", "净利率"], np.nan), np.nan),
            "roe": to_float(_find_value(latest, ["净资产收益率", "ROE"], np.nan), np.nan),
            "roa": to_float(_find_value(latest, ["总资产净利率", "总资产净利润率", "ROA"], np.nan), np.nan),
            "debt_ratio": to_float(_find_value(latest, ["资产负债率"], np.nan), np.nan),
            "cashflow_quality": to_float(
                _find_value(latest, ["经营现金流量净额", "经营现金净流量"], np.nan),
                np.nan,
            ),
            "raw_columns": list(map(str, df.columns)),
        }
    except Exception:
        logger.info("financial indicators failed: %s", code)
    return {}


def get_financial_indicators(code: str, start_year: str = "2020", ttl: int = 86400) -> dict[str, Any]:
    return dict(_cached_financial_indicators(normalize_code(code), start_year, _bucket(ttl)))


def get_basic_fundamentals(code: str, quote: dict[str, Any] | None = None) -> dict[str, Any]:
    code = normalize_code(code)
    quote = quote or get_quote(code)
    profile = get_company_profile(code)
    indicators = get_financial_indicators(code)
    total_mv = to_float(quote.get("total_mv"), to_float(profile.get("总市值"), np.nan))
    circ_mv = to_float(quote.get("circ_mv"), to_float(profile.get("流通市值"), np.nan))
    result = {
        "code": code,
        "name": quote.get("name") or profile.get("股票简称") or profile.get("简称") or "",
        "industry": profile.get("行业", ""),
        "concepts": "预留接口",
        "total_mv": total_mv,
        "circ_mv": circ_mv,
        "total_share": profile.get("总股本", ""),
        "float_share": profile.get("流通股", ""),
        "pe": to_float(quote.get("pe"), np.nan),
        "pb": to_float(quote.get("pb"), np.nan),
        "ps": np.nan,
        "peg": np.nan,
        "valuation_percentile": np.nan,
        "revenue": np.nan,
        "revenue_yoy": indicators.get("revenue_yoy", np.nan),
        "net_profit": np.nan,
        "net_profit_yoy": indicators.get("net_profit_yoy", np.nan),
        "deducted_profit": np.nan,
        "deducted_profit_yoy": indicators.get("deducted_profit_yoy", np.nan),
        "gross_margin": indicators.get("gross_margin", np.nan),
        "net_margin": indicators.get("net_margin", np.nan),
        "roe": indicators.get("roe", np.nan),
        "roa": indicators.get("roa", np.nan),
        "debt_ratio": indicators.get("debt_ratio", np.nan),
        "operating_cashflow": indicators.get("cashflow_quality", np.nan),
        "rd_expense": np.nan,
        "rd_ratio": np.nan,
        "shareholder_change": "预留接口",
        "institution_holding_change": "预留接口",
        "fund_holding_change": "预留接口",
        "report_date": indicators.get("report_date", ""),
        "data_quality": "基础版：AKShare公开数据，部分财务字段预留",
    }
    return result
