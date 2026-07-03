"""AKShare market data adapters.

The rest of the application works with normalized English field names while
keeping the raw AKShare columns available when useful.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import AKSHARE_TIMEOUT_SECONDS, FAST_FALLBACK_FIRST
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


def _empty_with_error(error: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["error"] = error
    return df


def _run_with_timeout(func, *args, timeout: int | None = None, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout or AKSHARE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"AKShare request timed out after {timeout or AKSHARE_TIMEOUT_SECONDS}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_limit_ratio(code: str, name: str = "") -> float:
    code = normalize_code(code)
    name_upper = str(name).upper()
    if "ST" in name_upper or "*ST" in name_upper:
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4", "920")):
        return 0.30
    return 0.10


def add_limit_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    ratios = [
        get_limit_ratio(row.get("code", ""), row.get("name", ""))
        for _, row in df.iterrows()
    ]
    prev_close = pd.to_numeric(df.get("pre_close"), errors="coerce")
    df["limit_up"] = (prev_close * (1 + pd.Series(ratios, index=df.index))).round(2)
    df["limit_down"] = (prev_close * (1 - pd.Series(ratios, index=df.index))).round(2)
    return df


def _market_code(code: str) -> str:
    code = normalize_code(code)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8", "920")):
        return f"bj{code}"
    return f"sz{code}"


def _request_text(url: str, encoding: str = "utf-8") -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn/",
    }
    response = requests.get(url, timeout=AKSHARE_TIMEOUT_SECONDS, headers=headers)
    response.raise_for_status()
    response.encoding = encoding
    return response.text


def _fallback_quote_sina(code: str) -> dict[str, Any] | None:
    code = normalize_code(code)
    symbol = _market_code(code)
    url = f"https://hq.sinajs.cn/list={symbol}"
    text = _request_text(url, encoding="gbk")
    if '=""' in text or '"' not in text:
        return None
    payload = text.split('"', 1)[1].rsplit('"', 1)[0]
    fields = payload.split(",")
    if len(fields) < 32 or not fields[0]:
        return None
    name = fields[0]
    open_price = to_float(fields[1], 0)
    pre_close = to_float(fields[2], 0)
    price = to_float(fields[3], 0)
    high = to_float(fields[4], 0)
    low = to_float(fields[5], 0)
    volume = to_float(fields[8], 0)
    amount = to_float(fields[9], 0)
    change = price - pre_close if price and pre_close else 0
    pct_chg = change / pre_close * 100 if pre_close else 0
    row = {
        "code": code,
        "name": name,
        "price": price,
        "pct_chg": pct_chg,
        "change": change,
        "volume": volume,
        "amount": amount,
        "amplitude": (high - low) / pre_close * 100 if pre_close else 0,
        "high": high,
        "low": low,
        "open": open_price,
        "pre_close": pre_close,
        "volume_ratio": 1.0,
        "turnover": np.nan,
        "pe": np.nan,
        "pb": np.nan,
        "total_mv": np.nan,
        "circ_mv": np.nan,
        "data_source": "新浪实时行情备用源",
    }
    return row


def _fallback_quote_tencent(code: str) -> dict[str, Any] | None:
    code = normalize_code(code)
    symbol = _market_code(code)
    url = f"https://qt.gtimg.cn/q={symbol}"
    text = _request_text(url, encoding="gbk")
    if '=""' in text or '"' not in text:
        return None
    payload = text.split('"', 1)[1].rsplit('"', 1)[0]
    fields = payload.split("~")
    if len(fields) < 40:
        return None
    price = to_float(fields[3], 0)
    pre_close = to_float(fields[4], 0)
    open_price = to_float(fields[5], 0)
    volume = to_float(fields[6], 0)
    amount = to_float(fields[37], 0) * 10000 if len(fields) > 37 else 0
    high = to_float(fields[33], 0) if len(fields) > 33 else max(price, open_price)
    low = to_float(fields[34], 0) if len(fields) > 34 else min(price, open_price)
    pct_chg = to_float(fields[32], 0) if len(fields) > 32 else 0
    change = price - pre_close if price and pre_close else 0
    row = {
        "code": code,
        "name": fields[1],
        "price": price,
        "pct_chg": pct_chg,
        "change": change,
        "volume": volume,
        "amount": amount,
        "amplitude": (high - low) / pre_close * 100 if pre_close else 0,
        "high": high,
        "low": low,
        "open": open_price,
        "pre_close": pre_close,
        "volume_ratio": 1.0,
        "turnover": np.nan,
        "pe": np.nan,
        "pb": np.nan,
        "total_mv": np.nan,
        "circ_mv": np.nan,
        "data_source": "腾讯实时行情备用源",
    }
    return row


def _fallback_quotes(codes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for code in codes:
        code = normalize_code(code)
        if not code:
            continue
        row = None
        for fetcher in (_fallback_quote_sina, _fallback_quote_tencent):
            try:
                row = fetcher(code)
                if row:
                    break
            except Exception as exc:
                errors.append(f"{code}: {exc}")
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_limit_prices(df)
        df.attrs["source"] = "fallback_realtime"
    elif errors:
        df.attrs["error"] = "; ".join(errors[:3])
    return df


def _eastmoney_spot_fallback() -> pd.DataFrame:
    fields = (
        "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f14,f15,f16,f17,f18,"
        "f20,f21,f23,f24,f25,f22,f115"
    )
    params = {
        "pn": 1,
        "pz": 6000,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f12",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": fields,
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
            return _empty_with_error("Eastmoney spot fallback returned no rows")
        df = pd.DataFrame(rows).rename(
            columns={
                "f12": "code",
                "f14": "name",
                "f2": "price",
                "f3": "pct_chg",
                "f4": "change",
                "f5": "volume",
                "f6": "amount",
                "f7": "amplitude",
                "f8": "turnover",
                "f9": "pe",
                "f10": "volume_ratio",
                "f15": "high",
                "f16": "low",
                "f17": "open",
                "f18": "pre_close",
                "f20": "total_mv",
                "f21": "circ_mv",
                "f23": "pb",
                "f24": "pct_chg_60d",
                "f25": "pct_chg_ytd",
                "f22": "speed",
                "f115": "pe_dynamic",
            }
        )
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
        df = _numeric(
            df,
            [
                "price",
                "pct_chg",
                "change",
                "volume",
                "amount",
                "amplitude",
                "turnover",
                "pe",
                "volume_ratio",
                "high",
                "low",
                "open",
                "pre_close",
                "total_mv",
                "circ_mv",
                "pb",
                "pct_chg_60d",
                "pct_chg_ytd",
                "speed",
            ],
        )
        df["data_source"] = "东方财富直连备用源"
        df = add_limit_prices(df)
        df.attrs["source"] = "eastmoney_spot_fallback"
        return df.reset_index(drop=True)
    except Exception as exc:
        return _empty_with_error(f"Eastmoney spot fallback failed: {exc}")


def _tencent_daily_fallback(code: str, adjust: str = "qfq", count: int = 1200) -> pd.DataFrame:
    code = normalize_code(code)
    symbol = _market_code(code)
    if adjust == "qfq":
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{count},qfq"
        key = "qfqday"
    elif adjust == "hfq":
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{count},hfq"
        key = "hfqday"
    else:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={symbol},day,,,{count}"
        key = "day"
    try:
        data = requests.get(url, timeout=AKSHARE_TIMEOUT_SECONDS).json()
        rows = data.get("data", {}).get(symbol, {}).get(key, [])
        if not rows:
            return _empty_with_error("Tencent daily fallback returned no rows")
        df = pd.DataFrame(rows)
        df = df.iloc[:, :6]
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        df["code"] = code
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["amount"] = df["close"] * df["volume"] * 100
        df["change"] = df["close"].diff().fillna(0)
        df["pct_chg"] = df["close"].pct_change().fillna(0) * 100
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan) * 100
        df["turnover"] = np.nan
        df.attrs["source"] = "腾讯日K备用源"
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception as exc:
        return _empty_with_error(f"Tencent daily fallback failed: {exc}")


def _tencent_minute_fallback(code: str) -> pd.DataFrame:
    code = normalize_code(code)
    symbol = _market_code(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={symbol}"
    try:
        data = requests.get(url, timeout=AKSHARE_TIMEOUT_SECONDS).json()
        stock_data = data.get("data", {}).get(symbol, {}).get("data", {})
        rows = stock_data.get("data", [])
        trade_date = stock_data.get("date") or datetime.now().strftime("%Y%m%d")
        if not rows:
            return _empty_with_error("Tencent minute fallback returned no rows")
        parsed = []
        prev_price = None
        prev_volume = 0.0
        prev_amount = 0.0
        for item in rows:
            parts = item.split()
            if len(parts) < 4:
                continue
            hm, price_text, volume_text, amount_text = parts[:4]
            price = to_float(price_text, 0)
            cumulative_volume = to_float(volume_text, 0)
            cumulative_amount = to_float(amount_text, 0)
            bar_volume = max(cumulative_volume - prev_volume, 0)
            bar_amount = max(cumulative_amount - prev_amount, 0)
            open_price = prev_price if prev_price is not None else price
            dt = pd.to_datetime(f"{trade_date} {hm[:2]}:{hm[2:]}", format="%Y%m%d %H:%M", errors="coerce")
            parsed.append(
                {
                    "datetime": dt,
                    "open": open_price,
                    "close": price,
                    "high": max(open_price, price),
                    "low": min(open_price, price),
                    "volume": bar_volume,
                    "amount": bar_amount,
                    "avg_price": cumulative_amount / (cumulative_volume * 100) if cumulative_volume else np.nan,
                }
            )
            prev_price = price
            prev_volume = cumulative_volume
            prev_amount = cumulative_amount
        df = pd.DataFrame(parsed)
        df.attrs["source"] = "腾讯分钟线备用源"
        return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    except Exception as exc:
        return _empty_with_error(f"Tencent minute fallback failed: {exc}")


def normalize_spot_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "昨收": "pre_close",
        "量比": "volume_ratio",
        "换手率": "turnover",
        "市盈率-动态": "pe",
        "市净率": "pb",
        "总市值": "total_mv",
        "流通市值": "circ_mv",
        "涨速": "speed",
        "5分钟涨跌": "pct_chg_5m",
        "60日涨跌幅": "pct_chg_60d",
        "年初至今涨跌幅": "pct_chg_ytd",
    }
    out = df.rename(columns=mapping).copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.zfill(6)
    numeric_cols = [
        "price",
        "pct_chg",
        "change",
        "volume",
        "amount",
        "amplitude",
        "high",
        "low",
        "open",
        "pre_close",
        "volume_ratio",
        "turnover",
        "pe",
        "pb",
        "total_mv",
        "circ_mv",
        "speed",
        "pct_chg_5m",
        "pct_chg_60d",
        "pct_chg_ytd",
    ]
    out = _numeric(out, numeric_cols)
    out = add_limit_prices(out)
    return out


@lru_cache(maxsize=12)
def _cached_spot_data(cache_bucket: int) -> pd.DataFrame:
    del cache_bucket
    try:
        ak = _akshare()
        df = _run_with_timeout(ak.stock_zh_a_spot_em)
        return normalize_spot_columns(df)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.warning("fetch spot data failed: %s", exc)
        return _empty_with_error(str(exc))


def get_realtime_quotes(codes: list[str] | None = None, ttl: int = 60) -> pd.DataFrame:
    if codes and FAST_FALLBACK_FIRST and len(codes) <= 50:
        fallback = _fallback_quotes(codes)
        if not fallback.empty:
            return fallback.reset_index(drop=True)
    df = _cached_spot_data(_bucket(ttl)).copy()
    if codes and not df.empty and "code" in df.columns:
        normalized = [normalize_code(code) for code in codes]
        df = df[df["code"].isin(normalized)].copy()
    if codes and (df.empty or "code" not in df.columns):
        fallback = _fallback_quotes(codes)
        if not fallback.empty:
            return fallback.reset_index(drop=True)
    if not codes and (df.empty or "code" not in df.columns):
        fallback = _eastmoney_spot_fallback()
        if not fallback.empty:
            return fallback.reset_index(drop=True)
    return df.reset_index(drop=True)


def get_quote(code: str, spot_df: pd.DataFrame | None = None) -> dict[str, Any]:
    code = normalize_code(code)
    df = spot_df if spot_df is not None else get_realtime_quotes([code])
    if df is None or df.empty or "code" not in df.columns:
        return {"code": code, "name": "", "data_error": getattr(df, "attrs", {}).get("error", "")}
    row = df[df["code"] == code]
    if row.empty:
        return {"code": code, "name": "", "data_error": "实时行情中未找到该股票"}
    return row.iloc[0].to_dict()


def normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    out = df.rename(columns=mapping).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.zfill(6)
    out = _numeric(
        out,
        [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "change",
            "turnover",
        ],
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=256)
def _cached_daily(
    code: str,
    start_date: str,
    end_date: str,
    adjust: str,
    cache_bucket: int,
) -> pd.DataFrame:
    del cache_bucket
    try:
        ak = _akshare()
        df = _run_with_timeout(
            ak.stock_zh_a_hist,
            symbol=normalize_code(code),
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        return normalize_daily_columns(df)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.warning("fetch daily kline failed: %s, %s", code, exc)
        return _empty_with_error(str(exc))


def get_hist_daily(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "qfq",
    ttl: int = 3600,
) -> pd.DataFrame:
    today = datetime.now()
    if end_date is None:
        end_date = today.strftime("%Y%m%d")
    if start_date is None:
        start_date = (today - timedelta(days=365 * 5 + 30)).strftime("%Y%m%d")
    if FAST_FALLBACK_FIRST:
        fallback = _tencent_daily_fallback(code, adjust=adjust)
        if not fallback.empty:
            return fallback
    df = _cached_daily(
        normalize_code(code),
        start_date.replace("-", ""),
        end_date.replace("-", ""),
        adjust,
        _bucket(ttl),
    ).copy()
    if df.empty:
        fallback = _tencent_daily_fallback(code, adjust=adjust)
        if not fallback.empty:
            return fallback
    return df


def normalize_minute_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "均价": "avg_price",
    }
    out = df.rename(columns=mapping).copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = _numeric(out, ["open", "close", "high", "low", "volume", "amount", "avg_price"])
    if "amount" in out.columns and "volume" in out.columns:
        volume_in_shares = out["volume"].replace(0, np.nan) * 100
        out["vwap_bar"] = out["amount"] / volume_in_shares
    return out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


@lru_cache(maxsize=512)
def _cached_minute(
    code: str,
    period: str,
    start_date: str,
    end_date: str,
    adjust: str,
    cache_bucket: int,
) -> pd.DataFrame:
    del cache_bucket
    try:
        ak = _akshare()
        kwargs: dict[str, Any] = {
            "symbol": normalize_code(code),
            "period": str(period),
            "start_date": start_date,
            "end_date": end_date,
        }
        if str(period) != "1":
            kwargs["adjust"] = adjust
        df = _run_with_timeout(ak.stock_zh_a_hist_min_em, **kwargs)
        return normalize_minute_columns(df)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.warning("fetch minute kline failed: %s, %s", code, exc)
        return _empty_with_error(str(exc))


def get_minute_kline(
    code: str,
    period: str = "5",
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "",
    ttl: int = 60,
) -> pd.DataFrame:
    now = datetime.now()
    if end_date is None:
        end_date = now.strftime("%Y-%m-%d %H:%M:%S")
    if start_date is None:
        start_date = (now - timedelta(days=10)).strftime("%Y-%m-%d 09:30:00")
    if FAST_FALLBACK_FIRST and str(period) in {"1", "5"}:
        fallback = _tencent_minute_fallback(code)
        if not fallback.empty:
            return fallback
    df = _cached_minute(
        normalize_code(code),
        str(period),
        start_date,
        end_date,
        adjust,
        _bucket(ttl),
    ).copy()
    if df.empty and str(period) in {"1", "5"}:
        fallback = _tencent_minute_fallback(code)
        if not fallback.empty:
            return fallback
    return df


def get_stock_name(code: str, spot_df: pd.DataFrame | None = None) -> str:
    quote = get_quote(code, spot_df)
    name = str(quote.get("name", "") or "")
    if name:
        return name
    try:
        ak = _akshare()
        info = _run_with_timeout(ak.stock_individual_info_em, symbol=normalize_code(code))
        if not info.empty and {"item", "value"}.issubset(info.columns):
            mapped = dict(zip(info["item"], info["value"]))
            return str(mapped.get("股票简称", "") or mapped.get("简称", ""))
    except Exception:
        logger.info("stock name lookup failed: %s", code)
    return ""


def quote_to_frame(quote: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([quote]) if quote else pd.DataFrame()


def market_status_from_spot(spot_df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = spot_df if spot_df is not None else get_realtime_quotes()
    if df is None or df.empty or "pct_chg" not in df.columns:
        return {
            "status": "数据不足",
            "up_count": 0,
            "down_count": 0,
            "flat_count": 0,
            "median_pct": 0.0,
            "amount": 0.0,
            "risk_bias": 0.0,
        }
    pct = pd.to_numeric(df["pct_chg"], errors="coerce").dropna()
    up_count = int((pct > 0).sum())
    down_count = int((pct < 0).sum())
    flat_count = int((pct == 0).sum())
    total = max(len(pct), 1)
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
        "flat_count": flat_count,
        "median_pct": median_pct,
        "up_ratio": up_ratio,
        "amount": amount,
        "risk_bias": risk_bias,
    }


def current_price_from(quote: dict[str, Any], daily: pd.DataFrame | None = None) -> float:
    price = to_float(quote.get("price") if quote else None, np.nan)
    if not np.isnan(price) and price > 0:
        return price
    if daily is not None and not daily.empty:
        return to_float(daily.iloc[-1].get("close"), 0.0)
    return 0.0
