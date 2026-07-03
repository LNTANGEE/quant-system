"""Application configuration for the A-share analysis MVP."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = BASE_DIR / "local_data"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = LOCAL_DATA_DIR / "stock_ai.sqlite3"
LOG_PATH = LOCAL_DATA_DIR / "app.log"

APP_NAME = "量化系统"
PAGE_ICON = "📈"

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
AKSHARE_TIMEOUT_SECONDS = int(os.getenv("AKSHARE_TIMEOUT_SECONDS", "2"))
FAST_FALLBACK_FIRST = os.getenv("FAST_FALLBACK_FIRST", "1") == "1"

RISK_DISCLOSURE = (
    "本系统只做量化概率分析，不构成投资建议，股市有风险，操作由用户自行承担。"
)

DEFAULT_WATCHLIST_SETTINGS = {
    "shares": 0,
    "cost_price": 0.0,
    "manual_price": 0.0,
    "sellable_quantity": 0,
    "max_trade_amount": 5000.0,
    "max_t_trades_per_day": 1,
    "alert_threshold": 2.0,
}

DEFAULT_MIN_AMOUNT_FOR_SHORTLIST = 300_000_000
DEFAULT_LOOKBACK_DAYS = 260

LOW_ZONE_LABELS = {
    "first": "第一低位区",
    "second": "第二低位区",
    "extreme": "极限低位区",
}
