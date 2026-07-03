CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT DEFAULT '',
    shares INTEGER DEFAULT 0,
    cost_price REAL DEFAULT 0,
    manual_price REAL DEFAULT 0,
    sellable_quantity INTEGER DEFAULT 0,
    max_trade_amount REAL DEFAULT 5000,
    max_t_trades_per_day INTEGER DEFAULT 1,
    alert_threshold REAL DEFAULT 2,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_pool (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT DEFAULT '',
    source TEXT DEFAULT 'manual',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sold_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    name TEXT DEFAULT '',
    sell_price REAL NOT NULL,
    sell_amount REAL DEFAULT 0,
    sell_date TEXT NOT NULL,
    note TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS t_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    operation_count INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(code, trade_date)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    trigger_price REAL DEFAULT 0,
    status TEXT DEFAULT 'new',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT DEFAULT '',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
