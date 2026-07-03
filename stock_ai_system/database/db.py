"""SQLite persistence helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from config import DB_PATH, DEFAULT_WATCHLIST_SETTINGS


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def normalize_code(code: str) -> str:
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else ""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _ensure_columns(conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()
    }
    if "manual_price" not in existing:
        conn.execute("ALTER TABLE watchlist ADD COLUMN manual_price REAL DEFAULT 0")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def add_watch_stock(code: str, name: str = "") -> bool:
    code = normalize_code(code)
    if not code:
        return False
    defaults = DEFAULT_WATCHLIST_SETTINGS
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO watchlist (
                code, name, shares, cost_price, manual_price, sellable_quantity,
                max_trade_amount, max_t_trades_per_day, alert_threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name=COALESCE(NULLIF(excluded.name, ''), watchlist.name),
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                code,
                name,
                defaults["shares"],
                defaults["cost_price"],
                defaults.get("manual_price", 0.0),
                defaults["sellable_quantity"],
                defaults["max_trade_amount"],
                defaults["max_t_trades_per_day"],
                defaults["alert_threshold"],
            ),
        )
    return True


def delete_watch_stock(code: str) -> None:
    code = normalize_code(code)
    with get_connection() as conn:
        conn.execute("DELETE FROM watchlist WHERE code = ?", (code,))


def update_watch_stock(code: str, **fields: Any) -> None:
    code = normalize_code(code)
    allowed = {
        "name",
        "shares",
        "cost_price",
        "manual_price",
        "sellable_quantity",
        "max_trade_amount",
        "max_t_trades_per_day",
        "alert_threshold",
    }
    payload = {key: value for key, value in fields.items() if key in allowed}
    if not code or not payload:
        return
    assignments = ", ".join(f"{key} = ?" for key in payload)
    values = list(payload.values()) + [code]
    with get_connection() as conn:
        conn.execute(
            f"UPDATE watchlist SET {assignments}, updated_at=CURRENT_TIMESTAMP WHERE code = ?",
            values,
        )


def get_watchlist() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY created_at DESC").fetchall()
    return rows_to_dicts(rows)


def get_watch_stock(code: str) -> dict[str, Any] | None:
    code = normalize_code(code)
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM watchlist WHERE code = ?", (code,)).fetchone()
    return row_to_dict(row)


def upsert_pool_stock(code: str, name: str = "", source: str = "manual") -> bool:
    code = normalize_code(code)
    if not code:
        return False
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO stock_pool (code, name, source)
            VALUES (?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name=COALESCE(NULLIF(excluded.name, ''), stock_pool.name),
                source=excluded.source,
                updated_at=CURRENT_TIMESTAMP
            """,
            (code, name, source),
        )
    return True


def remove_pool_stock(code: str) -> None:
    code = normalize_code(code)
    with get_connection() as conn:
        conn.execute("DELETE FROM stock_pool WHERE code = ?", (code,))


def get_manual_pool() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM stock_pool ORDER BY created_at DESC").fetchall()
    return rows_to_dicts(rows)


def get_stock_pool(include_watchlist: bool = True) -> list[dict[str, Any]]:
    pool = get_manual_pool()
    if include_watchlist:
        seen = {item["code"] for item in pool}
        for item in get_watchlist():
            if item["code"] not in seen:
                pool.append(
                    {
                        "code": item["code"],
                        "name": item.get("name", ""),
                        "source": "watchlist",
                    }
                )
                seen.add(item["code"])
    return pool


def add_sold_position(
    code: str,
    name: str,
    sell_price: float,
    sell_amount: float,
    sell_date: str,
    note: str = "",
) -> None:
    code = normalize_code(code)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sold_positions (code, name, sell_price, sell_amount, sell_date, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (code, name, sell_price, sell_amount, sell_date, note),
        )


def get_sold_positions(active_only: bool = False) -> list[dict[str, Any]]:
    sql = "SELECT * FROM sold_positions"
    params: tuple[Any, ...] = ()
    if active_only:
        sql += " WHERE status = ?"
        params = ("active",)
    sql += " ORDER BY sell_date DESC, created_at DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return rows_to_dicts(rows)


def update_sold_position_status(record_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE sold_positions SET status = ?, updated_at=CURRENT_TIMESTAMP WHERE id = ?",
            (status, record_id),
        )


def get_today_operation_count(code: str) -> int:
    code = normalize_code(code)
    today = date.today().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT operation_count FROM t_operations WHERE code = ? AND trade_date = ?",
            (code, today),
        ).fetchone()
    return int(row["operation_count"]) if row else 0


def increment_t_operation(code: str, amount: int = 1) -> None:
    code = normalize_code(code)
    today = date.today().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO t_operations (code, trade_date, operation_count)
            VALUES (?, ?, ?)
            ON CONFLICT(code, trade_date) DO UPDATE SET
                operation_count = operation_count + excluded.operation_count,
                updated_at=CURRENT_TIMESTAMP
            """,
            (code, today, amount),
        )


def add_alert(code: str, alert_type: str, message: str, trigger_price: float = 0) -> None:
    code = normalize_code(code)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO alerts (code, alert_type, message, trigger_price)
            VALUES (?, ?, ?, ?)
            """,
            (code, alert_type, message, trigger_price),
        )


def get_recent_alerts(limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows_to_dicts(rows)


def set_setting(key: str, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
            """,
            (key, text),
        )


def get_setting(key: str, default: Any = None) -> Any:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]
