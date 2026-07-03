"""Short, medium and long horizon potential scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_MIN_AMOUNT_FOR_SHORTLIST
from models.indicators import calculate_intraday_vwap, enrich_indicators, trend_strength
from utils.helpers import clamp, safe_div, to_float


def _short_label(score: float) -> str:
    if score >= 80:
        return "短线强势候选"
    if score >= 70:
        return "可重点观察"
    if score >= 60:
        return "有潜力但等低吸"
    if score >= 50:
        return "普通观察"
    return "不建议短线参与"


def _mid_label(score: float) -> str:
    if score >= 80:
        return "中期重点候选"
    if score >= 70:
        return "中期可跟踪"
    if score >= 60:
        return "等待回调"
    if score >= 50:
        return "一般"
    return "暂不考虑"


def _long_label(score: float) -> str:
    if score >= 85:
        return "长期核心候选"
    if score >= 75:
        return "长期重点观察"
    if score >= 65:
        return "可小仓配置"
    if score >= 55:
        return "估值或业绩需要验证"
    return "不适合长期持有"


def score_short_potential(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None,
    quote: dict | None,
    sector_context: dict | None,
    main_force: dict | None,
    risk: dict | None,
) -> dict:
    quote = quote or {}
    sector_context = sector_context or {}
    main_force = main_force or {}
    risk = risk or {}
    enriched = enrich_indicators(daily_df)
    if enriched.empty:
        return {"score": 45, "label": "数据不足", "factors": {}}
    row = enriched.iloc[-1]
    price = to_float(quote.get("price"), to_float(row.get("close")))
    sector_rank = sector_context.get("sector_rank")
    sector_count = max(int(to_float(sector_context.get("sector_count"), 0)), 1)
    sector_pct = to_float(sector_context.get("sector_pct_chg"), 0)
    relative_strength = to_float(sector_context.get("relative_strength"), 0)

    if sector_rank:
        rank_ratio = to_float(sector_rank) / sector_count
        sector_points = 20 if rank_ratio <= 0.08 else 15 if rank_ratio <= 0.25 else 10 if rank_ratio <= 0.6 else 5
    else:
        sector_points = 10 + clamp(sector_pct, -3, 3) / 3 * 4
    sector_points += 2 if sector_context.get("leader") else 0
    sector_points = clamp(sector_points, 0, 20)

    relative_points = 7
    relative_points += 5 if relative_strength > 1 else 3 if relative_strength > 0 else 0
    relative_points += 3 if to_float(quote.get("pct_chg"), 0) > 0 else 0
    relative_points = clamp(relative_points, 0, 15)

    recent_high = to_float(enriched["high"].tail(20).max(), price)
    vol_ma20 = to_float(row.get("vol_ma20"), 0)
    latest_volume = to_float(row.get("volume"), 0)
    volume_multiple = safe_div(latest_volume, vol_ma20, 1)
    volume_price_points = 5
    volume_price_points += 5 if price >= recent_high * 0.995 and volume_multiple >= 1.1 else 0
    volume_price_points += 4 if volume_multiple >= 1.2 and to_float(row.get("daily_return"), 0) > 0 else 0
    volume_price_points += 3 if volume_multiple < 0.85 and price >= to_float(row.get("ma10"), price) else 0
    if volume_multiple > 2.5 and safe_div(to_float(row.get("high")) - price, price) * 100 > 2:
        volume_price_points -= 5
    volume_price_points = clamp(volume_price_points, 0, 15)

    trend_points = trend_strength(enriched) * 0.15

    funds_points = clamp(to_float(main_force.get("score"), 50) * 0.11, 0, 11)
    funds_points += 4 if to_float(quote.get("volume_ratio"), 1) >= 1.1 else 0
    funds_points = clamp(funds_points, 0, 15)

    risk_score = to_float(risk.get("risk_score"), 50)
    risk_points = clamp(10 * (1 - risk_score / 100), 0, 10)
    amount = to_float(quote.get("amount"), 0)
    if amount and amount < DEFAULT_MIN_AMOUNT_FOR_SHORTLIST:
        risk_points = min(risk_points, 4)

    vwap = calculate_intraday_vwap(minute_df) if minute_df is not None else np.nan
    intraday_points = 5
    intraday_points += 3 if pd.notna(vwap) and price >= vwap else 0
    intraday_points += 2 if to_float(main_force.get("tail_risk"), 50) < 25 else 0
    if safe_div(to_float(quote.get("high"), price) - price, price) * 100 > 3:
        intraday_points -= 3
    intraday_points = clamp(intraday_points, 0, 10)

    total = (
        sector_points
        + relative_points
        + volume_price_points
        + trend_points
        + funds_points
        + risk_points
        + intraday_points
    )
    total = clamp(total)
    return {
        "score": round(total, 1),
        "label": _short_label(total),
        "factors": {
            "板块强度": round(sector_points, 1),
            "个股相对强度": round(relative_points, 1),
            "量价结构": round(volume_price_points, 1),
            "趋势结构": round(trend_points, 1),
            "资金行为代理": round(funds_points, 1),
            "风险控制": round(risk_points, 1),
            "盘口/分时强弱": round(intraday_points, 1),
        },
    }


def score_mid_potential(
    daily_df: pd.DataFrame,
    fundamentals: dict | None,
    sector_context: dict | None,
    risk: dict | None,
) -> dict:
    fundamentals = fundamentals or {}
    sector_context = sector_context or {}
    risk = risk or {}
    enriched = enrich_indicators(daily_df)
    industry_points = 10
    if sector_context.get("sector_strength") == "强势前排":
        industry_points = 18
    elif sector_context.get("sector_strength") == "偏强":
        industry_points = 15
    elif sector_context.get("sector_strength") == "偏弱":
        industry_points = 7

    revenue_yoy = to_float(fundamentals.get("revenue_yoy"), np.nan)
    profit_yoy = to_float(fundamentals.get("net_profit_yoy"), np.nan)
    growth_points = 8
    if pd.notna(revenue_yoy):
        growth_points += 6 if revenue_yoy > 20 else 4 if revenue_yoy > 8 else 1 if revenue_yoy > 0 else -2
    if pd.notna(profit_yoy):
        growth_points += 6 if profit_yoy > 25 else 4 if profit_yoy > 10 else 1 if profit_yoy > 0 else -3
    growth_points = clamp(growth_points, 0, 20)

    pe = to_float(fundamentals.get("pe"), np.nan)
    pb = to_float(fundamentals.get("pb"), np.nan)
    valuation_points = 8
    if pd.notna(pe):
        valuation_points += 4 if 0 < pe < 35 else 2 if 35 <= pe < 70 else -3 if pe < 0 or pe > 120 else 0
    if pd.notna(pb):
        valuation_points += 3 if 0 < pb < 5 else -2 if pb > 10 else 0
    valuation_points = clamp(valuation_points, 0, 15)

    tech_points = trend_strength(enriched) * 0.15 if not enriched.empty else 7
    institution_points = 5
    fund_trend_points = 5
    if not enriched.empty and "amount" in enriched.columns:
        amount20 = to_float(enriched["amount"].tail(20).mean(), 0)
        amount60 = to_float(enriched["amount"].tail(60).mean(), amount20)
        fund_trend_points += 5 if amount20 > amount60 * 1.08 else 2 if amount20 > amount60 else 0
    fund_trend_points = clamp(fund_trend_points, 0, 10)
    risk_points = clamp(10 * (1 - to_float(risk.get("risk_score"), 50) / 100), 0, 10)

    total = clamp(
        industry_points
        + growth_points
        + valuation_points
        + tech_points
        + institution_points
        + fund_trend_points
        + risk_points
    )
    return {
        "score": round(total, 1),
        "label": _mid_label(total),
        "factors": {
            "行业景气度代理": round(industry_points, 1),
            "业绩增长": round(growth_points, 1),
            "估值合理性": round(valuation_points, 1),
            "技术趋势": round(tech_points, 1),
            "机构关注度预留": round(institution_points, 1),
            "资金趋势代理": round(fund_trend_points, 1),
            "风险项": round(risk_points, 1),
        },
    }


def score_long_potential(
    daily_df: pd.DataFrame,
    fundamentals: dict | None,
    risk: dict | None,
) -> dict:
    fundamentals = fundamentals or {}
    risk = risk or {}
    total_mv = to_float(fundamentals.get("total_mv"), np.nan)
    industry_position = 9
    if pd.notna(total_mv):
        industry_position = 18 if total_mv >= 100_000_000_000 else 15 if total_mv >= 40_000_000_000 else 11

    revenue_yoy = to_float(fundamentals.get("revenue_yoy"), np.nan)
    profit_yoy = to_float(fundamentals.get("net_profit_yoy"), np.nan)
    growth_points = 8
    if pd.notna(revenue_yoy):
        growth_points += 5 if revenue_yoy > 18 else 3 if revenue_yoy > 5 else 0
    if pd.notna(profit_yoy):
        growth_points += 7 if profit_yoy > 20 else 4 if profit_yoy > 8 else 0 if profit_yoy > -10 else -3
    growth_points = clamp(growth_points, 0, 20)

    roe = to_float(fundamentals.get("roe"), np.nan)
    gross_margin = to_float(fundamentals.get("gross_margin"), np.nan)
    net_margin = to_float(fundamentals.get("net_margin"), np.nan)
    debt_ratio = to_float(fundamentals.get("debt_ratio"), np.nan)
    quality_points = 5
    if pd.notna(roe):
        quality_points += 5 if roe >= 15 else 3 if roe >= 8 else 0
    if pd.notna(gross_margin):
        quality_points += 3 if gross_margin >= 30 else 1 if gross_margin >= 15 else 0
    if pd.notna(net_margin):
        quality_points += 2 if net_margin >= 10 else 1 if net_margin >= 5 else 0
    if pd.notna(debt_ratio) and debt_ratio <= 55:
        quality_points += 2
    quality_points = clamp(quality_points, 0, 15)

    rd_points = 8
    rd_ratio = to_float(fundamentals.get("rd_ratio"), np.nan)
    if pd.notna(rd_ratio):
        rd_points = 14 if rd_ratio >= 8 else 11 if rd_ratio >= 4 else 7

    pe = to_float(fundamentals.get("pe"), np.nan)
    pb = to_float(fundamentals.get("pb"), np.nan)
    valuation_points = 8
    if pd.notna(pe):
        valuation_points += 4 if 0 < pe < 45 else 1 if 45 <= pe < 80 else -3 if pe < 0 or pe > 150 else 0
    if pd.notna(pb):
        valuation_points += 3 if 0 < pb < 6 else -2 if pb > 12 else 0
    valuation_points = clamp(valuation_points, 0, 15)

    shareholder_points = 5
    risk_deduction = clamp(to_float(risk.get("risk_score"), 50) / 100 * 5, 0, 5)
    total = clamp(
        industry_position
        + growth_points
        + quality_points
        + rd_points
        + valuation_points
        + shareholder_points
        + (5 - risk_deduction)
    )
    return {
        "score": round(total, 1),
        "label": _long_label(total),
        "factors": {
            "公司行业地位代理": round(industry_position, 1),
            "成长性": round(growth_points, 1),
            "盈利质量": round(quality_points, 1),
            "研发和技术预留": round(rd_points, 1),
            "估值安全边际": round(valuation_points, 1),
            "股东和机构结构预留": round(shareholder_points, 1),
            "风险扣分后得分": round(5 - risk_deduction, 1),
        },
    }


def comprehensive_score(short_score: float, mid_score: float, long_score: float) -> float:
    return round(short_score * 0.30 + mid_score * 0.35 + long_score * 0.35, 1)


def classify_stock(short_score: float, mid_score: float, long_score: float, risk: dict | None) -> str:
    risk = risk or {}
    risk_score = to_float(risk.get("risk_score"), 50)
    comp = comprehensive_score(short_score, mid_score, long_score)
    if risk_score >= 65:
        return "高风险题材型"
    if comp < 55:
        return "不建议参与型"
    if short_score >= 75 and mid_score < 65 and long_score < 65:
        return "短线机会型"
    if mid_score >= 75 and long_score >= 65:
        return "中期趋势型"
    if long_score >= 75 and mid_score >= 65:
        return "长期成长型"
    if short_score >= 60 and comp >= 58:
        return "低吸观察型"
    return "不建议参与型"


def operation_suggestion(
    short_score: float,
    mid_score: float,
    long_score: float,
    risk: dict | None,
    price_position: str = "",
) -> str:
    risk = risk or {}
    risk_score = to_float(risk.get("risk_score"), 50)
    comp = comprehensive_score(short_score, mid_score, long_score)
    if risk_score >= 65:
        return "风险评分偏高，暂不参与，仅保留观察。"
    if comp >= 80 and short_score >= 75:
        return "重点关注，等待回踩低吸区，不追高。"
    if comp >= 75 and (mid_score >= 75 or long_score >= 75):
        return "中长期观察，可按区间分批评估，等待估值或价格进入更合理区。"
    if short_score >= 75 and long_score < 60:
        return "只适合短线观察，需严格止损，不适合长期拿。"
    if long_score >= 75 and short_score < 60:
        return "好公司概率较高但短线偏弱，不急买，等待趋势修复。"
    if short_score >= 60:
        return "有潜力但等待低吸区，控制仓位和失败成本。"
    return "评分普通，保持观察。"


def build_potential_record(
    code: str,
    quote: dict,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None,
    fundamentals: dict,
    sector_context: dict,
    low_model: dict,
    t_strategy: dict,
    main_force: dict,
    risk: dict,
) -> dict:
    short = score_short_potential(daily_df, minute_df, quote, sector_context, main_force, risk)
    mid = score_mid_potential(daily_df, fundamentals, sector_context, risk)
    long = score_long_potential(daily_df, fundamentals, risk)
    comp = comprehensive_score(short["score"], mid["score"], long["score"])
    stock_type = classify_stock(short["score"], mid["score"], long["score"], risk)
    suggestion = operation_suggestion(short["score"], mid["score"], long["score"], risk)
    return {
        "code": code,
        "name": quote.get("name") or fundamentals.get("name") or "",
        "industry": sector_context.get("industry") or fundamentals.get("industry") or "未知",
        "sector": sector_context.get("sector") or "未知",
        "price": to_float(quote.get("price"), np.nan),
        "pct_chg": to_float(quote.get("pct_chg"), 0),
        "amount": to_float(quote.get("amount"), 0),
        "short_score": short["score"],
        "short_label": short["label"],
        "mid_score": mid["score"],
        "mid_label": mid["label"],
        "long_score": long["score"],
        "long_label": long["label"],
        "comprehensive_score": comp,
        "stock_type": stock_type,
        "risk_score": risk.get("risk_score"),
        "risk_level": risk.get("risk_level"),
        "low_buy_zone": low_model.get("first_low_zone"),
        "high_sell_zone": t_strategy.get("high_sell_zone"),
        "suggest_observe_price": low_model.get("first_low_zone", {}).get("high"),
        "suggest_low_zone": low_model.get("second_low_zone"),
        "operation_suggestion": suggestion,
        "short_factors": short.get("factors", {}),
        "mid_factors": mid.get("factors", {}),
        "long_factors": long.get("factors", {}),
        "fundamental": fundamentals,
        "risk_flags": risk.get("flags", []),
    }
