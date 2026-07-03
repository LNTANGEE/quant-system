"""Industry and market breadth adapters."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import pandas as pd
import requests

from config import AKSHARE_TIMEOUT_SECONDS
from data.stock_data import _run_with_timeout, get_realtime_quotes
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


def _empty(error: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["error"] = error
    return df


def normalize_sector_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "排名": "rank",
        "板块名称": "sector",
        "板块代码": "sector_code",
        "最新价": "price",
        "涨跌额": "change",
        "涨跌幅": "pct_chg",
        "成交额": "amount",
        "总市值": "total_mv",
        "换手率": "turnover",
        "上涨家数": "up_count",
        "下跌家数": "down_count",
        "领涨股票": "leader",
        "领涨股票-涨跌幅": "leader_pct_chg",
    }
    out = df.rename(columns=mapping).copy()
    for col in [
        "rank",
        "price",
        "change",
        "pct_chg",
        "amount",
        "total_mv",
        "turnover",
        "up_count",
        "down_count",
        "leader_pct_chg",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("pct_chg", ascending=False, na_position="last").reset_index(drop=True)


def _eastmoney_sector_fallback() -> pd.DataFrame:
    params = {
        "pn": 1,
        "pz": 500,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:90+t:2",
        "fields": "f12,f14,f3,f6,f104,f105,f128,f140",
    }
    try:
        response = requests.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params=params,
            timeout=AKSHARE_TIMEOUT_SECONDS + 4,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        rows = response.json().get("data", {}).get("diff", [])
        if not rows:
            return _empty("Eastmoney sector fallback returned no rows")
        df = pd.DataFrame(rows).rename(
            columns={
                "f12": "sector_code",
                "f14": "sector",
                "f3": "pct_chg",
                "f6": "amount",
                "f104": "up_count",
                "f105": "down_count",
                "f128": "leader",
                "f140": "leader_code",
            }
        )
        df["rank"] = range(1, len(df) + 1)
        for col in ["pct_chg", "amount", "up_count", "down_count"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.attrs["source"] = "东方财富板块直连备用源"
        return df.sort_values("pct_chg", ascending=False, na_position="last").reset_index(drop=True)
    except Exception as exc:
        return _empty(f"Eastmoney sector fallback failed: {exc}")


@lru_cache(maxsize=8)
def _cached_industry_snapshot(cache_bucket: int) -> pd.DataFrame:
    del cache_bucket
    try:
        ak = _akshare()
        df = _run_with_timeout(ak.stock_board_industry_name_em)
        return normalize_sector_columns(df)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.warning("fetch sector snapshot failed: %s", exc)
        fallback = _eastmoney_sector_fallback()
        if not fallback.empty:
            return fallback
        return _empty(str(exc))


def get_industry_snapshot(ttl: int = 300) -> pd.DataFrame:
    df = _cached_industry_snapshot(_bucket(ttl)).copy()
    if df.empty:
        fallback = _eastmoney_sector_fallback()
        if not fallback.empty:
            return fallback
    return df


def get_strong_weak_sectors(n: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_industry_snapshot()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    strong = df.head(n).copy()
    weak = df.tail(n).sort_values("pct_chg", ascending=True).copy()
    return strong, weak


@lru_cache(maxsize=512)
def get_stock_industry(code: str) -> str:
    code = normalize_code(code)
    try:
        ak = _akshare()
        info = _run_with_timeout(ak.stock_individual_info_em, symbol=code)
        if not info.empty and {"item", "value"}.issubset(info.columns):
            mapped = dict(zip(info["item"], info["value"]))
            return str(mapped.get("行业", "") or mapped.get("所属行业", "") or "")
    except Exception:
        logger.info("industry lookup failed: %s", code)
    return ""


def get_sector_context(code: str, quote: dict[str, Any] | None = None) -> dict[str, Any]:
    industry = get_stock_industry(code)
    snapshot = get_industry_snapshot()
    context = {
        "industry": industry or "未知",
        "sector": industry or "未知",
        "sector_pct_chg": 0.0,
        "sector_rank": None,
        "sector_count": int(len(snapshot)) if not snapshot.empty else 0,
        "sector_strength": "数据不足",
        "relative_strength": 0.0,
        "leader": "",
    }
    if snapshot.empty or not industry:
        return context
    match = snapshot[snapshot["sector"].astype(str).str.contains(industry, regex=False, na=False)]
    if match.empty:
        match = snapshot[snapshot["sector"].astype(str) == industry]
    if match.empty:
        return context
    row = match.iloc[0]
    pct = to_float(row.get("pct_chg"))
    rank = int(to_float(row.get("rank"), 0)) or int(match.index[0] + 1)
    count = max(len(snapshot), 1)
    if rank <= 5:
        label = "强势前排"
    elif rank <= max(10, count * 0.25):
        label = "偏强"
    elif rank >= count * 0.75:
        label = "偏弱"
    else:
        label = "中性"
    stock_pct = to_float((quote or {}).get("pct_chg"), 0.0)
    context.update(
        {
            "sector": str(row.get("sector", industry)),
            "sector_pct_chg": pct,
            "sector_rank": rank,
            "sector_count": count,
            "sector_strength": label,
            "relative_strength": stock_pct - pct,
            "leader": str(row.get("leader", "")),
        }
    )
    return context


def get_market_status(spot_df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = spot_df if spot_df is not None else get_realtime_quotes()
    if df is None or df.empty or "pct_chg" not in df.columns:
        return {
            "status": "数据不足",
            "up_count": 0,
            "down_count": 0,
            "flat_count": 0,
            "median_pct": 0.0,
            "up_ratio": 0.0,
            "amount": 0.0,
            "risk_bias": 0.0,
        }
    pct = pd.to_numeric(df["pct_chg"], errors="coerce").dropna()
    total = max(len(pct), 1)
    up_count = int((pct > 0).sum())
    down_count = int((pct < 0).sum())
    median_pct = float(pct.median()) if len(pct) else 0.0
    up_ratio = up_count / total
    amount = float(pd.to_numeric(df.get("amount"), errors="coerce").sum())
    if median_pct >= 0.7 and up_ratio >= 0.58:
        status = "偏强"
        risk_bias = -0.08
    elif median_pct >= 0.15 or up_ratio >= 0.52:
        status = "震荡偏强"
        risk_bias = -0.03
    elif median_pct <= -0.7 and up_ratio <= 0.38:
        status = "偏弱"
        risk_bias = 0.12
    elif median_pct <= -0.15 or up_ratio <= 0.45:
        status = "震荡偏弱"
        risk_bias = 0.06
    else:
        status = "震荡"
        risk_bias = 0.0
    return {
        "status": status,
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": int((pct == 0).sum()),
        "median_pct": median_pct,
        "up_ratio": up_ratio,
        "amount": amount,
        "risk_bias": risk_bias,
    }


def get_industry_constituents(sector_name: str) -> pd.DataFrame:
    try:
        ak = _akshare()
        df = _run_with_timeout(ak.stock_board_industry_cons_em, symbol=sector_name)
        return df
    except Exception as exc:  # pragma: no cover - external dependency
        logger.info("industry constituents failed: %s", sector_name)
        return _empty(str(exc))
