"""Private intraday steward signals.

The steward converts model outputs into structured, probability-based intraday
instructions. It never marks a trade as certain; it only describes trigger
zones, position sizes, risk controls, and whether conditions are currently met.
"""

from __future__ import annotations

from typing import Any

from utils.helpers import format_amount, format_price, format_zone, safe_div, to_float


def _in_zone(price: float, zone: dict[str, Any] | None) -> bool:
    if not zone or price <= 0:
        return False
    return to_float(zone.get("low")) <= price <= to_float(zone.get("high"))


def _above_zone(price: float, zone: dict[str, Any] | None) -> bool:
    return bool(zone and price > 0 and price >= to_float(zone.get("low")))


def _below_zone(price: float, zone: dict[str, Any] | None) -> bool:
    return bool(zone and price > 0 and price <= to_float(zone.get("high")))


def _distance_pct(price: float, target: float) -> float:
    if price <= 0 or target <= 0:
        return 0.0
    return safe_div(target - price, price) * 100


def build_steward_advice(
    result: dict[str, Any],
    watch: dict[str, Any] | None = None,
    today_count: int = 0,
) -> dict[str, Any]:
    watch = watch or {}
    quote = result.get("quote", {})
    low_model = result.get("low_model", {})
    high_model = result.get("high_model", {})
    next_hour = result.get("next_hour_high", {})
    next_hour_available = bool(next_hour.get("available"))
    low_stat = low_model.get("statistical_model", {})
    high_stat = high_model.get("statistical_model", {})
    strategy = result.get("t_strategy", {})
    risk = result.get("risk", {})
    potential = result.get("potential", {})
    main_force = result.get("main_force", {})

    price = to_float(quote.get("price"))
    cost_price = to_float(watch.get("cost_price"))
    holding_shares = int(to_float(watch.get("shares")))
    low_zone = strategy.get("low_buy_zone") or low_model.get("first_low_zone")
    second_low_zone = strategy.get("second_low_buy_zone") or low_model.get("second_low_zone")
    high_zone = strategy.get("high_sell_zone")
    stop_loss = to_float(strategy.get("stop_loss"))
    take_profit = to_float(strategy.get("take_profit"))
    buy_price = to_float(strategy.get("suggested_buy_price"))
    sell_price = to_float(strategy.get("suggested_sell_price"))
    buy_shares = int(to_float(strategy.get("suggested_buy_shares")))
    sell_shares = int(to_float(strategy.get("suggested_sell_shares")))
    buy_amount = to_float(strategy.get("suggested_buy_amount"))
    sell_amount = to_float(strategy.get("suggested_sell_amount"))
    max_count = int(to_float(watch.get("max_t_trades_per_day"), 1))
    remaining_count = max(max_count - today_count, 0)
    score = to_float(strategy.get("t_score"))
    risk_score = to_float(risk.get("risk_score"), 50)
    reach_probability = to_float(low_model.get("estimated_low_reach_probability"))
    confidence = to_float(low_model.get("estimated_low_confidence"))
    high_reach_probability = to_float(high_model.get("estimated_high_reach_probability"))
    high_confidence = to_float(high_model.get("estimated_high_confidence"))
    next_hour_horizon = int(to_float(next_hour.get("horizon_minutes"), 0))
    next_hour_probability = to_float(next_hour.get("break_day_high_probability"))
    next_hour_confidence = to_float(next_hour.get("confidence"))
    position_profit_pct = safe_div(price - cost_price, cost_price) * 100 if cost_price > 0 and price > 0 else 0.0
    position_value = price * holding_shares if price > 0 and holding_shares > 0 else 0.0

    buy_triggered = _in_zone(price, low_zone) or (buy_price > 0 and price <= buy_price)
    sell_triggered = _in_zone(price, high_zone) or (sell_price > 0 and price >= sell_price)
    weak_break = _below_zone(price, second_low_zone)
    stop_triggered = stop_loss > 0 and price <= stop_loss
    take_profit_triggered = take_profit > 0 and price >= take_profit
    forbid = bool(strategy.get("forbid_t")) or remaining_count <= 0 or risk_score >= 80

    if forbid:
        signal = "仅观察"
        signal_level = "禁止做T"
    elif stop_triggered:
        signal = "风控优先"
        signal_level = "高风险"
    elif sell_triggered and sell_shares > 0 and score >= 60:
        signal = "高抛区触发"
        signal_level = "可评估反T/减仓"
    elif buy_triggered and buy_shares > 0 and score >= 60:
        signal = "低吸区触发"
        signal_level = "可评估正T"
    elif score >= 75:
        signal = "强观察"
        signal_level = "等待触发"
    elif score >= 60:
        signal = "小仓观察"
        signal_level = "等待触发"
    else:
        signal = "耐心观察"
        signal_level = "不急操作"

    if forbid:
        current_advice = "当前不生成做T执行数量，先检查底仓、可卖数量、单次金额、风险评分和今日次数。"
    elif stop_triggered:
        current_advice = "风控优先：价格接近止损观察线，降低操作频率，避免把做T变成补亏。"
    elif buy_triggered and buy_shares > 0:
        current_advice = f"可评估低吸正T：参考{format_price(buy_price)}附近，约{buy_shares}股，金额约{format_amount(buy_amount)}。"
    elif sell_triggered and sell_shares > 0:
        current_advice = f"可评估高抛/反T：参考{format_price(sell_price)}附近，约{sell_shares}股，金额约{format_amount(sell_amount)}。"
    elif price > 0 and buy_price > 0 and price > buy_price:
        current_advice = "未到低吸观察价，不追高；等待价格回落到低吸区或分时承接信号更明确。"
    else:
        current_advice = "当前信号不够集中，先观察VWAP、成交量和低吸/高抛区触发情况。"

    realtime_alerts: list[str] = []
    if buy_triggered and buy_shares > 0 and not forbid:
        realtime_alerts.append(
            f"低吸提醒：价格进入低吸观察区{format_zone(low_zone)}，可按{buy_shares}股以内评估，仍需确认不放量破位。"
        )
    if sell_triggered and sell_shares > 0 and not forbid:
        realtime_alerts.append(
            f"高抛提醒：价格进入高抛观察区{format_zone(high_zone)}，可按{sell_shares}股以内评估，避免追涨。"
        )
    if weak_break:
        realtime_alerts.append(
            f"弱势提醒：价格接近第二低位区{format_zone(second_low_zone)}，短线波动风险升高，低吸需分批并降低金额。"
        )
    if stop_triggered:
        realtime_alerts.append(
            f"风控提醒：价格接近止损观察线{format_price(stop_loss)}，优先控制失败成本。"
        )
    if take_profit_triggered:
        realtime_alerts.append(
            f"止盈提醒：价格接近止盈观察线{format_price(take_profit)}，优先评估高抛区，不追高。"
        )
    if not realtime_alerts:
        realtime_alerts.append("暂无强触发提醒：继续盯低吸区、高抛区、VWAP和成交量变化。")

    if buy_triggered and buy_shares > 0 and not forbid:
        entry_timing = (
            f"当前价已接近/进入低吸触发区{format_zone(low_zone)}，若分时不继续放量破位，"
            f"可按小仓观察价{format_price(buy_price)}、约{buy_shares}股、{format_amount(buy_amount)}评估正T。"
        )
    elif weak_break:
        entry_timing = (
            f"当前价接近第二低位区{format_zone(second_low_zone)}，弱势概率升高，优先等企稳和承接评分修复。"
        )
    else:
        distance_to_buy = _distance_pct(price, buy_price)
        entry_timing = (
            f"等待回落到低吸观察价{format_price(buy_price)}附近，约还差{distance_to_buy:.2f}%；"
            f"今日预估最低区间{format_zone(low_model.get('estimated_low_zone'))}，触达概率约{reach_probability:.1f}%。"
        )

    if sell_triggered and sell_shares > 0 and not forbid:
        exit_timing = (
            f"当前价已接近/进入高抛触发区{format_zone(high_zone)}，可按观察价{format_price(sell_price)}、"
            f"约{sell_shares}股、{format_amount(sell_amount)}评估高抛或反T；"
            f"今日预估最高区间{format_zone(high_model.get('estimated_high_zone'))}。"
        )
    else:
        distance_to_sell = _distance_pct(price, sell_price)
        exit_timing = (
            f"等待反弹到高抛观察价{format_price(sell_price)}附近，约还差{distance_to_sell:.2f}%；"
            f"今日预估最高区间{format_zone(high_model.get('estimated_high_zone'))}，"
            f"触达概率约{high_reach_probability:.1f}%，止盈观察线{format_price(take_profit)}。"
        )
    if next_hour_available:
        exit_timing += (
            f" 短时参考：未来{next_hour_horizon}个交易分钟预估最高区间"
            f"{format_zone(next_hour.get('predicted_high_zone'))}，"
            f"突破今日高点概率约{next_hour_probability:.1f}%。"
        )

    if stop_triggered:
        risk_guard = f"当前价接近/跌破止损观察线{format_price(stop_loss)}，优先降低做T频率，避免扩大失败成本。"
    elif take_profit_triggered:
        risk_guard = f"当前价接近/超过止盈观察线{format_price(take_profit)}，不追高，优先评估高抛区。"
    elif remaining_count <= 0:
        risk_guard = "今日做T次数已用完，后续只观察，不再生成执行数量。"
    else:
        risk_guard = (
            f"止损观察线{format_price(stop_loss)}，止盈观察线{format_price(take_profit)}；"
            f"今日剩余可记录做T次数{remaining_count}次。"
        )

    if buy_shares > 0 and sell_shares > 0:
        trade_size_text = (
            f"低吸约{buy_shares}股/{format_amount(buy_amount)}；"
            f"高抛约{sell_shares}股/{format_amount(sell_amount)}。"
        )
    elif buy_shares > 0:
        trade_size_text = f"当前仅生成低吸数量：约{buy_shares}股/{format_amount(buy_amount)}。"
    elif sell_shares > 0:
        trade_size_text = f"当前仅生成高抛数量：约{sell_shares}股/{format_amount(sell_amount)}。"
    else:
        trade_size_text = strategy.get("execution_plan", "当前未生成买卖数量。")

    low_judgement = (
        f"今日最低价概率判断：预估区间{format_zone(low_model.get('estimated_low_zone'))}，"
        f"基准/弱势/恐慌低点分别为{format_price(low_model.get('base_case_low'))}/"
        f"{format_price(low_model.get('weak_case_low'))}/{format_price(low_model.get('panic_case_low'))}，"
        f"触达概率约{reach_probability:.1f}%，模型置信度约{confidence:.1f}%。"
        f"统计校准样本{low_stat.get('sample_size', 0)}个，有效相似样本{low_stat.get('effective_sample_size', 0)}个。"
        "该区间综合昨日低点、VWAP、MA5/MA10/MA20、ATR、布林下轨、支撑压力、市场弱势和历史相似日分布生成。"
    )
    high_judgement = (
        f"今日最高价概率判断：预估区间{format_zone(high_model.get('estimated_high_zone'))}，"
        f"基准/强势/冲高回落高点分别为{format_price(high_model.get('base_case_high'))}/"
        f"{format_price(high_model.get('strong_case_high'))}/{format_price(high_model.get('spike_case_high'))}，"
        f"触达概率约{high_reach_probability:.1f}%，模型置信度约{high_confidence:.1f}%。"
        f"统计校准样本{high_stat.get('sample_size', 0)}个，有效相似样本{high_stat.get('effective_sample_size', 0)}个。"
        "该区间综合昨日高点、VWAP、MA5/MA10、ATR、布林上轨、平台压力、量比、市场强弱和历史相似日分布生成。"
    )
    if next_hour_available:
        next_hour_high_judgement = (
            f"未来{next_hour_horizon}个交易分钟最高价判断：预估最高价"
            f"{format_price(next_hour.get('predicted_high'))}，区间"
            f"{format_zone(next_hour.get('predicted_high_zone'))}，"
            f"较现价预期上行{to_float(next_hour.get('expected_upside_pct')):.2f}%，"
            f"突破今日高点概率{next_hour_probability:.1f}%，模型置信度{next_hour_confidence:.1f}%；"
            f"预测窗口{next_hour.get('window_start', '--')}至{next_hour.get('window_end', '--')}，"
            f"样本/有效样本{next_hour.get('sample_size', 0)}/"
            f"{next_hour.get('effective_sample_size', 0)}。"
        )
    else:
        next_hour_high_judgement = (
            "未来一小时最高价判断暂不可用："
            f"{next_hour.get('label') or next_hour.get('explain') or '分钟行情不足'}"
        )

    return {
        "signal": signal,
        "signal_level": signal_level,
        "current_price": price,
        "can_consider_t": (not forbid) and score >= 60,
        "buy_triggered": buy_triggered,
        "sell_triggered": sell_triggered,
        "stop_triggered": stop_triggered,
        "buy_price": round(buy_price, 2),
        "sell_price": round(sell_price, 2),
        "buy_shares": buy_shares,
        "sell_shares": sell_shares,
        "buy_amount": round(buy_amount, 2),
        "sell_amount": round(sell_amount, 2),
        "cost_price": round(cost_price, 3),
        "holding_shares": holding_shares,
        "position_value": round(position_value, 2),
        "position_profit_pct": round(position_profit_pct, 2),
        "current_advice": current_advice,
        "realtime_alerts": realtime_alerts,
        "entry_timing": entry_timing,
        "exit_timing": exit_timing,
        "risk_guard": risk_guard,
        "low_judgement": low_judgement,
        "high_judgement": high_judgement,
        "next_hour_high_judgement": next_hour_high_judgement,
        "trade_size_text": trade_size_text,
        "summary": (
            f"{signal_level}：做T评分{score:.1f}，主力评分{to_float(main_force.get('score')):.1f}，"
            f"短线评分{to_float(potential.get('short_score')):.1f}，风险评分{risk_score:.1f}，"
            f"最低区间置信度{confidence:.1f}%，最高区间置信度{high_confidence:.1f}%。"
        ),
        "next_watch": (
            f"重点盯{format_zone(low_zone)}低吸区、{format_zone(high_zone)}高抛区、"
            f"{format_zone(high_model.get('estimated_high_zone'))}今日预估最高区"
            + (
                f"、{format_zone(next_hour.get('predicted_high_zone'))}"
                f"未来{next_hour_horizon}交易分钟最高区"
                if next_hour_available
                else ""
            )
            + "；"
            f"价格进入区间后再结合VWAP、放量和风险等级确认。"
        ),
    }
