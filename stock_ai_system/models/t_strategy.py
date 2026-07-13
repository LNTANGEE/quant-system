"""T-trading interval and action model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.indicators import calculate_intraday_vwap, enrich_indicators, support_resistance
from utils.helpers import clamp, safe_div, to_float


def _zone(center: float, width: float) -> dict[str, float]:
    return {
        "low": round(max(center - width, 0), 2),
        "high": round(max(center + width, 0), 2),
        "center": round(max(center, 0), 2),
    }


def estimate_high_sell_zone(daily_df: pd.DataFrame, quote: dict | None = None) -> dict[str, float]:
    quote = quote or {}
    enriched = enrich_indicators(daily_df)
    if enriched.empty:
        price = to_float(quote.get("price"), 0)
        return _zone(price * 1.02, max(price * 0.005, 0.01))
    row = enriched.iloc[-1]
    price = to_float(quote.get("price"), to_float(row.get("close")))
    atr14 = max(to_float(row.get("atr14")), price * 0.012)
    sr = support_resistance(enriched)
    prev_high = to_float(enriched["high"].tail(3).max(), price)
    center = np.median(
        [
            value
            for value in [
                prev_high,
                to_float(row.get("bb_upper"), price + atr14),
                to_float(sr.get("resistance"), price + atr14),
                price + 0.65 * atr14,
            ]
            if pd.notna(value) and value > 0
        ]
    )
    center = max(center, price * 1.008)
    return _zone(float(center), max(atr14 * 0.16, price * 0.004))


def _inside(price: float, zone: dict[str, float]) -> bool:
    return zone.get("low", 0) <= price <= zone.get("high", 0)


def _near(price: float, zone: dict[str, float], pct: float = 0.008) -> bool:
    if price <= 0:
        return False
    return abs(price - zone.get("center", 0)) / price <= pct or _inside(price, zone)


def _valid_zone(zone: dict | None) -> bool:
    return bool(zone and to_float(zone.get("low")) > 0 and to_float(zone.get("high")) >= to_float(zone.get("low")))


def _blend_zone(primary: dict[str, float], secondary: dict[str, float], secondary_weight: float = 0.55) -> dict[str, float]:
    if not _valid_zone(primary):
        return secondary
    if not _valid_zone(secondary):
        return primary
    weight = clamp(secondary_weight, 0, 1)
    primary_width = max(to_float(primary.get("high")) - to_float(primary.get("low")), 0.01)
    secondary_width = max(to_float(secondary.get("high")) - to_float(secondary.get("low")), 0.01)
    center = to_float(primary.get("center")) * (1 - weight) + to_float(secondary.get("center")) * weight
    width = max(primary_width * 0.5, secondary_width * 0.5) * 0.72 + abs(to_float(primary.get("center")) - to_float(secondary.get("center"))) * 0.18
    return _zone(center, max(width, center * 0.0025))


def _lot_floor(shares: float) -> int:
    return max(int(shares // 100) * 100, 0)


def _amount_to_lot_shares(amount: float, price: float) -> int:
    if amount <= 0 or price <= 0:
        return 0
    return _lot_floor(amount / price)


def _one_lot_amount(price: float) -> float:
    return price * 100 if price > 0 else 0.0


def t_score_label(score: float) -> str:
    if score >= 75:
        return "可执行做T"
    if score >= 60:
        return "小仓试探"
    if score >= 45:
        return "观察"
    if score >= 30:
        return "不建议做T"
    return "禁止做T"


def build_t_strategy(
    quote: dict | None,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame | None,
    low_model: dict,
    main_force: dict,
    risk: dict,
    watch: dict | None = None,
    sector_context: dict | None = None,
    high_model: dict | None = None,
) -> dict:
    quote = quote or {}
    watch = watch or {}
    sector_context = sector_context or {}
    high_model = high_model or {}
    price = to_float(quote.get("price"), to_float(quote.get("pre_close"), 0))
    low_zone = low_model.get("first_low_zone", {"low": 0, "high": 0, "center": 0})
    second_zone = low_model.get("second_low_zone", {"low": 0, "high": 0, "center": 0})
    model_low_zone = low_model.get("estimated_low_zone")
    low_reach_probability = to_float(low_model.get("estimated_low_reach_probability"), 50)
    break_first_probability = to_float(low_model.get("break_first_probability"), 45)
    if _valid_zone(model_low_zone):
        low_weight = 0.64 if low_reach_probability >= 60 or break_first_probability >= 55 else 0.40
        low_zone = _blend_zone(low_zone, model_low_zone, low_weight)
    high_zone = estimate_high_sell_zone(daily_df, quote)
    model_high_zone = high_model.get("estimated_high_zone") or high_model.get("first_high_zone")
    if _valid_zone(model_high_zone):
        high_zone = _blend_zone(high_zone, model_high_zone, 0.58)
    vwap = calculate_intraday_vwap(minute_df) if minute_df is not None else np.nan
    shares = int(to_float(watch.get("shares"), 0))
    sellable = int(to_float(watch.get("sellable_quantity"), 0))
    max_amount = to_float(watch.get("max_trade_amount"), 0)
    max_count = int(to_float(watch.get("max_t_trades_per_day"), 1))
    risk_score = to_float(risk.get("risk_score"), 50)
    main_score = to_float(main_force.get("score"), 50)
    high_reach_probability = to_float(high_model.get("estimated_high_reach_probability"), 50)
    low_confidence = to_float(low_model.get("estimated_low_confidence"), 45)
    high_confidence = to_float(high_model.get("estimated_high_confidence"), 45)

    low_opportunity = 0.0
    if _inside(price, low_zone):
        low_opportunity = 25
    elif _near(price, low_zone, 0.015):
        low_opportunity = 18
    elif price < low_zone.get("low", 0):
        low_opportunity = 12
    else:
        low_opportunity = max(0, 12 - max(0, price - low_zone.get("high", price)) / max(price, 0.01) * 800)
    low_opportunity *= 0.82 + clamp(low_reach_probability, 10, 90) / 250
    low_opportunity *= 0.86 + clamp(low_confidence, 20, 95) / 300

    high_opportunity = 0.0
    if _inside(price, high_zone):
        high_opportunity = 25
    elif _near(price, high_zone, 0.015):
        high_opportunity = 16
    elif price > high_zone.get("high", 0):
        high_opportunity = 18
    high_opportunity *= 0.82 + clamp(high_reach_probability, 10, 90) / 250
    high_opportunity *= 0.86 + clamp(high_confidence, 20, 95) / 300

    price_position_points = max(low_opportunity, high_opportunity)
    trend_points = clamp(main_score, 0, 100) * 0.20
    volume_ratio = to_float(quote.get("volume_ratio"), 1)
    volume_points = 15 if 0.8 <= volume_ratio <= 2.5 else 9 if volume_ratio < 3.5 else 5
    sector_label = sector_context.get("sector_strength", "")
    sector_points = 15 if sector_label in {"强势前排", "偏强"} else 9 if sector_label in {"中性", ""} else 5
    risk_points = max(0, 15 * (1 - risk_score / 100))
    confidence_points = clamp((min(low_confidence, high_confidence) - 35) / 50 * 6, 0, 6)
    stop_loss = second_zone.get("low", 0) * 0.99
    take_profit = high_zone.get("low", 0)
    gain_space = safe_div(take_profit - price, price) * 100
    loss_space = safe_div(price - stop_loss, price) * 100 if stop_loss > 0 else 5
    payoff_ratio = safe_div(max(gain_space, 0.1), max(loss_space, 0.1))
    payoff_points = clamp(payoff_ratio / 2 * 10, 0, 10)
    score = clamp(
        price_position_points
        + trend_points
        + volume_points
        + sector_points
        + risk_points
        + confidence_points
        + payoff_points
    )

    reasons: list[str] = []
    if risk.get("forbid_t"):
        reasons.append("风险等级过高或价格数据缺失")
    if shares <= 0:
        reasons.append("无底仓，普通A股T+1下不构成做T条件")
    if sellable <= 0:
        reasons.append("可卖数量为0，不能卖出已有底仓")
    if max_count <= 0:
        reasons.append("单日最大做T次数设置为0")
    if score < 30:
        reasons.append("做T评分低于30")
    if min(low_confidence, high_confidence) < 35:
        reasons.append("高低点模型置信度过低")

    forbid = bool(reasons)
    ratio = 0.0
    if not forbid:
        base_ratio = 0.25 if score >= 75 else 0.15 if score >= 60 else 0.08 if score >= 45 else 0.0
        confidence_scale = clamp(min(low_confidence, high_confidence) / 75, 0.45, 1.12)
        probability_scale = clamp(max(low_reach_probability, high_reach_probability) / 70, 0.50, 1.08)
        ratio = min(0.30, base_ratio * confidence_scale * probability_scale)
    buy_budget = min(max_amount, max_amount * ratio) if low_opportunity >= high_opportunity else 0
    sell_budget = min(max_amount, sellable * price * ratio) if high_opportunity >= low_opportunity else 0
    suggested_buy_price = low_zone.get("center", price)
    suggested_sell_price = high_zone.get("center", price)
    suggested_buy_shares = _amount_to_lot_shares(buy_budget, suggested_buy_price)
    suggested_sell_shares = min(_lot_floor(sellable * ratio), _amount_to_lot_shares(sell_budget, suggested_sell_price))
    one_lot_buy_amount = _one_lot_amount(suggested_buy_price)
    one_lot_sell_amount = _one_lot_amount(suggested_sell_price)
    if (
        not forbid
        and suggested_buy_shares == 0
        and ratio > 0
        and low_opportunity >= high_opportunity
        and max_amount >= one_lot_buy_amount > 0
    ):
        suggested_buy_shares = 100
    if (
        not forbid
        and suggested_sell_shares == 0
        and ratio > 0
        and high_opportunity >= low_opportunity
        and sellable >= 100
        and max_amount >= one_lot_sell_amount > 0
    ):
        suggested_sell_shares = 100
    suggested_buy_amount = suggested_buy_shares * suggested_buy_price
    suggested_sell_amount = suggested_sell_shares * suggested_sell_price
    observe_ratio = 0.18 if score >= 75 else 0.12 if score >= 60 else 0.06 if score >= 45 else 0.0
    observe_buy_budget = min(max_amount, max_amount * observe_ratio) if risk_score < 65 else 0
    observe_buy_shares = _amount_to_lot_shares(observe_buy_budget, suggested_buy_price)
    if observe_buy_shares == 0 and observe_ratio > 0 and risk_score < 65 and max_amount >= one_lot_buy_amount > 0:
        observe_buy_shares = 100
    observe_buy_amount = observe_buy_shares * suggested_buy_price

    if forbid:
        positive_t = "禁止或暂不适合做T：" + "；".join(reasons)
        reverse_t = positive_t
        current_action = "仅观察"
    elif score >= 75 and low_opportunity >= high_opportunity:
        positive_t = "正T条件较好：低吸区内小比例买入，反弹到高抛区后用已有可卖底仓卖出。"
        reverse_t = "反T条件一般，优先等待反弹到高抛区再评估。"
        current_action = "偏正T"
    elif score >= 75:
        positive_t = "正T等待低吸区。"
        reverse_t = "反T条件较好：高抛区内可分批卖出可卖底仓，回落到低吸区再接回。"
        current_action = "偏反T"
    elif score >= 60:
        positive_t = "可小仓试探，需控制单次金额和失败成本。"
        reverse_t = "可小仓观察，触发高抛或低吸区后再执行。"
        current_action = "小仓观察"
    elif score >= 45:
        positive_t = "评分处于观察区，等待价格进入更明确的低吸或高抛区间。"
        reverse_t = positive_t
        current_action = "观察"
    else:
        positive_t = "评分偏低，不建议做T。"
        reverse_t = positive_t
        current_action = "不建议做T"

    if suggested_buy_shares > 0:
        buy_plan = f"低吸观察价约{suggested_buy_price:.2f}，可买{suggested_buy_shares}股，约{suggested_buy_amount:.0f}元"
    elif max_amount < one_lot_buy_amount:
        buy_plan = f"低吸观察价约{suggested_buy_price:.2f}，最大单次金额不足一手，至少约{one_lot_buy_amount:.0f}元才可买100股"
    else:
        buy_plan = f"低吸观察价约{suggested_buy_price:.2f}，当前价格位置或评分暂不触发买入数量"

    if suggested_sell_shares > 0:
        sell_plan = f"高抛观察价约{suggested_sell_price:.2f}，可卖{suggested_sell_shares}股，约{suggested_sell_amount:.0f}元"
    elif sellable < 100:
        sell_plan = f"高抛观察价约{suggested_sell_price:.2f}，可卖数量不足一手，暂不生成卖出数量"
    else:
        sell_plan = f"高抛观察价约{suggested_sell_price:.2f}，当前价格位置或评分暂不触发卖出数量"

    if observe_buy_shares > 0:
        observe_plan = (
            f"非做T低吸观察：若不考虑T+1卖出闭环，仅按最大单次金额和评分，观察买入约{observe_buy_shares}股，"
            f"约{observe_buy_amount:.0f}元，参考价{suggested_buy_price:.2f}。"
        )
    elif max_amount < one_lot_buy_amount:
        observe_plan = (
            f"非做T低吸观察：最大单次操作金额约{max_amount:.0f}元不足一手，"
            f"按参考价{suggested_buy_price:.2f}计算至少约{one_lot_buy_amount:.0f}元才可买100股。"
        )
    else:
        observe_plan = "非做T低吸观察：当前评分/风险/价格位置条件不足，暂不生成观察买入数量。"

    return {
        "low_buy_zone": low_zone,
        "second_low_buy_zone": second_zone,
        "high_sell_zone": high_zone,
        "low_reach_probability": round(low_reach_probability, 1),
        "high_reach_probability": round(high_reach_probability, 1),
        "low_confidence": round(low_confidence, 1),
        "high_confidence": round(high_confidence, 1),
        "positive_t": positive_t,
        "reverse_t": reverse_t,
        "suggested_ratio": round(ratio, 2),
        "suggested_buy_price": round(suggested_buy_price, 2),
        "suggested_sell_price": round(suggested_sell_price, 2),
        "suggested_buy_shares": suggested_buy_shares,
        "suggested_sell_shares": suggested_sell_shares,
        "suggested_buy_amount": round(suggested_buy_amount, 2),
        "suggested_sell_amount": round(suggested_sell_amount, 2),
        "observe_buy_shares": observe_buy_shares,
        "observe_buy_amount": round(observe_buy_amount, 2),
        "observe_buy_price": round(suggested_buy_price, 2),
        "one_lot_buy_amount": round(one_lot_buy_amount, 2),
        "one_lot_sell_amount": round(one_lot_sell_amount, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "t_score": round(score, 1),
        "t_score_label": t_score_label(score),
        "forbid_t": forbid,
        "forbid_reasons": reasons,
        "current_action": current_action,
        "payoff_ratio": round(payoff_ratio, 2),
        "vwap": round(to_float(vwap, 0), 2),
        "execution_plan": (
            f"{buy_plan}；{sell_plan}。"
            if not forbid
            else "当前不生成买卖数量，先满足底仓、可卖数量、风险和评分条件。"
        ),
        "observe_plan": observe_plan,
        "note": "做T仅基于已有底仓和可卖数量，今日新买入数量不计入可卖数量。",
    }
