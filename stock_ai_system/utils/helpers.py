"""UI and numeric helper functions."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import RISK_DISCLOSURE


def normalize_code(code: str) -> str:
    digits = "".join(re.findall(r"\d+", str(code)))
    return digits.zfill(6)[-6:] if digits else ""


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
            if value in {"", "-", "--", "None", "nan"}:
                return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    denominator = to_float(denominator)
    if abs(denominator) < 1e-12:
        return default
    return to_float(numerator) / denominator


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, to_float(value)))


def format_price(value: Any, digits: int = 2) -> str:
    number = to_float(value, np.nan)
    if pd.isna(number):
        return "--"
    return f"{number:.{digits}f}"


def format_pct(value: Any, digits: int = 2) -> str:
    number = to_float(value, np.nan)
    if pd.isna(number):
        return "--"
    return f"{number:.{digits}f}%"


def format_amount(value: Any) -> str:
    number = to_float(value, np.nan)
    if pd.isna(number):
        return "--"
    if abs(number) >= 100_000_000:
        return f"{number / 100_000_000:.2f}亿"
    if abs(number) >= 10_000:
        return f"{number / 10_000:.2f}万"
    return f"{number:.0f}"


def format_zone(zone: dict[str, Any] | None) -> str:
    if not zone:
        return "--"
    return f"{format_price(zone.get('low'))}-{format_price(zone.get('high'))}"


def score_text(score: Any) -> str:
    score = to_float(score)
    return f"{score:.0f}"


def risk_color(level: str) -> str:
    return {
        "低": "green",
        "中": "orange",
        "高": "red",
        "极高": "red",
    }.get(level, "gray")


def label_by_score(score: float, thresholds: list[tuple[float, str]]) -> str:
    score = to_float(score)
    for min_score, label in thresholds:
        if score >= min_score:
            return label
    return thresholds[-1][1] if thresholds else ""


def render_risk_footer() -> None:
    st.divider()
    st.caption(RISK_DISCLOSURE)


def render_mobile_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.6rem;
            max-width: 1180px;
        }
        h1 {
            font-size: clamp(1.65rem, 4vw, 2.35rem) !important;
            line-height: 1.18 !important;
        }
        h2, h3 {
            line-height: 1.25 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.92);
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 8px;
            padding: 0.75rem;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.84rem;
            white-space: normal;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
            line-height: 1.2;
        }
        .stDataFrame, div[data-testid="stDataEditor"] {
            overflow-x: auto;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.65rem;
        }
        .stButton button, .stDownloadButton button {
            min-height: 2.5rem;
            border-radius: 8px;
        }
        div[data-testid="stTabs"] button {
            min-width: max-content;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.55rem;
                padding-right: 0.55rem;
            }
            div[data-testid="column"] {
                min-width: 0 !important;
            }
            div[data-testid="stMetric"] {
                padding: 0.58rem;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.05rem;
            }
            div[data-testid="stMetricDelta"] {
                font-size: 0.78rem;
            }
            .stAlert {
                padding: 0.65rem;
            }
            p, li, label, span {
                word-break: break-word;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dataframe_with_formats(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({np.nan: None})


def parse_code_lines(text: str) -> list[str]:
    codes: list[str] = []
    for part in re.split(r"[\s,，;；]+", text.strip()):
        code = normalize_code(part)
        if code:
            codes.append(code)
    return list(dict.fromkeys(codes))
