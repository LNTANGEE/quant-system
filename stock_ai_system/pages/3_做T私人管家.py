from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from database.db import get_recent_alerts, get_today_operation_count, get_watchlist, increment_t_operation, init_db
from models.private_steward import build_steward_advice
from utils.analysis import analyze_many, analyze_stock
from utils.helpers import format_amount, format_pct, format_price, format_zone, render_mobile_style, render_risk_footer


def _schedule_refresh(enabled: bool, seconds: int) -> None:
    if not enabled or seconds <= 0:
        return
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        height=0,
    )


def _steward_row(result: dict, watch: dict, today_count: int) -> dict:
    if result.get("error"):
        return {
            "代码": result.get("code", ""),
            "名称": result.get("quote", {}).get("name", watch.get("name", "")),
            "当前价": "--",
            "管家信号": "数据不足",
            "做T判断": "等待行情恢复",
            "低吸区": "--",
            "高抛区": "--",
            "今日预估最高区": "--",
            "可买": "--",
            "可卖": "--",
            "何时进": result.get("error", ""),
            "何时出": "--",
            "风控": "--",
        }
    steward = build_steward_advice(result, watch, today_count)
    quote = result["quote"]
    strategy = result["t_strategy"]
    high_model = result.get("high_model", {})
    return {
        "代码": result["code"],
        "名称": quote.get("name", watch.get("name", "")),
        "当前价": format_price(quote.get("price")),
        "管家信号": steward["signal"],
        "做T判断": steward["signal_level"],
        "低吸区": format_zone(strategy.get("low_buy_zone")),
        "高抛区": format_zone(strategy.get("high_sell_zone")),
        "今日预估最高区": format_zone(high_model.get("estimated_high_zone")),
        "可买": f"{steward['buy_shares']}股 / {format_amount(steward['buy_amount'])}",
        "可卖": f"{steward['sell_shares']}股 / {format_amount(steward['sell_amount'])}",
        "何时进": steward["entry_timing"],
        "何时出": steward["exit_timing"],
        "风控": steward["risk_guard"],
    }


init_db()
render_mobile_style()

st.title("做T私人管家")
st.caption("多因子概率量化框架：自动刷新自选股，输出是否适合做T、何时进、何时出、可买可卖股数和金额。")

with st.sidebar:
    st.header("私人管家")
    mode = st.radio("盯盘模式", ["单股精细盯盘", "全自选快扫"], horizontal=False)
    auto_refresh = st.toggle("自动刷新", value=True)
    refresh_seconds = st.slider("刷新间隔(秒)", 15, 180, 30, step=15)
    include_minute = st.checkbox("单股启用分钟K精细计算", value=True)
    st.caption("免费公开行情有延迟和限流，刷新过快可能导致接口暂时不可用。")

_schedule_refresh(auto_refresh, refresh_seconds)

watchlist = get_watchlist()
if not watchlist:
    st.info("请先在自选股管理页添加持仓股票，并设置持仓数量、可卖数量和最大单次操作金额。")
    render_risk_footer()
    st.stop()

st.caption(f"最近刷新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if mode == "全自选快扫":
    with st.spinner("私人管家正在扫描全部自选股..."):
        results = analyze_many(watchlist, include_minute=False)
    watch_by_code = {item["code"]: item for item in watchlist}
    rows = []
    for result in results:
        code = result.get("code", "")
        watch = watch_by_code.get(code, {})
        rows.append(_steward_row(result, watch, get_today_operation_count(code)))
    st.subheader("全自选私人管家总览")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("全自选快扫用于盯盘排序；需要分时图、VWAP和更细做T解释时，切换到“单股精细盯盘”。")
    render_risk_footer()
    st.stop()

labels = [f"{item['code']} {item.get('name', '')}" for item in watchlist]
selected = st.selectbox("选择盯盘股票", labels)
watch = watchlist[labels.index(selected)]
code = watch["code"]

with st.spinner("私人管家正在计算分时、低吸区、高抛区和做T数量..."):
    result = analyze_stock(code, watch=watch, include_minute=include_minute)

quote = result["quote"]
t_strategy = result["t_strategy"]
low_model = result["low_model"]
high_model = result.get("high_model", {})
next_hour = result.get("next_hour_high", {})
low_stat = low_model.get("statistical_model", {})
high_stat = high_model.get("statistical_model", {})
main_force = result["main_force"]
risk = result["risk"]
potential = result["potential"]
today_count = get_today_operation_count(code)
steward = build_steward_advice(result, watch, today_count)

st.subheader(f"{code} {quote.get('name', watch.get('name', ''))}")
top_cols = st.columns(5)
top_cols[0].metric("当前价", format_price(quote.get("price")))
top_cols[1].metric("管家信号", steward["signal"])
top_cols[2].metric("做T判断", steward["signal_level"])
top_cols[3].metric("做T评分", f"{t_strategy.get('t_score')} / {t_strategy.get('t_score_label')}")
top_cols[4].metric("风险等级", f"{risk.get('risk_level')}({risk.get('risk_score')})")

position_cols = st.columns(4)
position_cols[0].metric("持仓数量", f"{steward['holding_shares']}股")
position_cols[1].metric("成本价", format_price(steward["cost_price"], 3))
position_cols[2].metric("持仓市值", format_amount(steward["position_value"]))
position_cols[3].metric("相对成本", f"{steward['position_profit_pct']:.2f}%")

trade_cols = st.columns(5)
trade_cols[0].metric("可低吸股数/金额", f"{steward['buy_shares']}股 / {format_amount(steward['buy_amount'])}")
trade_cols[1].metric("可高抛股数/金额", f"{steward['sell_shares']}股 / {format_amount(steward['sell_amount'])}")
trade_cols[2].metric("低点触达概率", format_pct(low_model.get("estimated_low_reach_probability")))
if next_hour.get("available"):
    hour_horizon = int(next_hour.get("horizon_minutes") or 60)
    trade_cols[3].metric(
        f"未来{hour_horizon}交易分钟预估最高价",
        format_price(next_hour.get("predicted_high")),
    )
    trade_cols[4].metric(
        "突破今日高点概率",
        format_pct(next_hour.get("break_day_high_probability")),
    )
    st.caption(
        f"短时最高区间 {format_zone(next_hour.get('predicted_high_zone'))} · "
        f"较现价预期上行 {format_pct(next_hour.get('expected_upside_pct'))} · "
        f"置信度 {format_pct(next_hour.get('confidence'))} · "
        f"窗口 {next_hour.get('window_start', '--')} 至 {next_hour.get('window_end', '--')}"
    )
else:
    hour_horizon = 60
    trade_cols[3].metric("未来一小时预估最高价", "--")
    trade_cols[4].metric("突破今日高点概率", "--")
    if include_minute:
        st.warning(next_hour.get("label") or "分钟行情不足，短时预测暂不可用。")
    else:
        st.info("请开启侧边栏“单股启用分钟K精细计算”后查看未来一小时预测。")

st.success(steward["current_advice"])
st.info(steward["low_judgement"])
st.info(steward.get("high_judgement", ""))
st.info(steward.get("next_hour_high_judgement", ""))
st.subheader("实时提醒")
for alert in steward["realtime_alerts"]:
    st.warning(alert) if "风控" in alert or "弱势" in alert else st.info(alert)
st.info(steward["summary"])
st.write(steward["trade_size_text"])

entry_col, exit_col = st.columns(2)
entry_col.success(steward["entry_timing"])
exit_col.warning(steward["exit_timing"])
st.caption(steward["risk_guard"])
st.caption(steward["next_watch"])

st.subheader("管家执行清单")
st.dataframe(
    pd.DataFrame(
        [
            {"项目": "今日预估最低价区间", "内容": format_zone(low_model.get("estimated_low_zone"))},
            {"项目": "统计校准低位区", "内容": format_zone(low_model.get("statistical_low_zone"))},
            {"项目": "基准/弱势/恐慌低点", "内容": f"{format_price(low_model.get('base_case_low'))} / {format_price(low_model.get('weak_case_low'))} / {format_price(low_model.get('panic_case_low'))}"},
            {"项目": "今日预估最高价区间", "内容": format_zone(high_model.get("estimated_high_zone"))},
            {"项目": "统计校准高位区", "内容": format_zone(high_model.get("statistical_high_zone"))},
            {"项目": "基准/强势/冲高高点", "内容": f"{format_price(high_model.get('base_case_high'))} / {format_price(high_model.get('strong_case_high'))} / {format_price(high_model.get('spike_case_high'))}"},
            {
                "项目": f"未来{hour_horizon}交易分钟预估最高价/区间",
                "内容": (
                    f"{format_price(next_hour.get('predicted_high'))} / "
                    f"{format_zone(next_hour.get('predicted_high_zone'))}"
                    if next_hour.get("available")
                    else next_hour.get("label", "暂不可用")
                ),
            },
            {
                "项目": "短时突破今日高点概率/预测置信度",
                "内容": (
                    f"{format_pct(next_hour.get('break_day_high_probability'))} / "
                    f"{format_pct(next_hour.get('confidence'))}"
                    if next_hour.get("available")
                    else "--"
                ),
            },
            {
                "项目": "短时预测窗口/样本/方法",
                "内容": (
                    f"{next_hour.get('window_start', '--')} 至 {next_hour.get('window_end', '--')} / "
                    f"{next_hour.get('sample_size', 0)}/{next_hour.get('effective_sample_size', 0)} / "
                    f"{next_hour.get('method', '--')}"
                    if next_hour.get("available")
                    else "--"
                ),
            },
            {"项目": "统计样本/有效样本", "内容": f"{low_stat.get('sample_size', 0)} / {low_stat.get('effective_sample_size', 0)}"},
            {"项目": "盘中进度/统计置信度", "内容": f"{low_stat.get('day_progress', 0) * 100:.0f}% / {low_stat.get('confidence', 0)}%"},
            {"项目": "最高区间触达概率", "内容": format_pct(high_model.get("estimated_high_reach_probability"))},
            {"项目": "今日再创新高概率", "内容": format_pct(high_model.get("new_high_probability"))},
            {"项目": "今日低吸区", "内容": format_zone(t_strategy.get("low_buy_zone"))},
            {"项目": "今日高抛区", "内容": format_zone(t_strategy.get("high_sell_zone"))},
            {"项目": "正T建议", "内容": t_strategy.get("positive_t")},
            {"项目": "反T建议", "内容": t_strategy.get("reverse_t")},
            {"项目": "当前是否适合操作", "内容": t_strategy.get("current_action")},
            {"项目": "建议做T比例", "内容": f"{t_strategy.get('suggested_ratio', 0) * 100:.0f}%"},
            {"项目": "建议低吸价格", "内容": format_price(t_strategy.get("suggested_buy_price"))},
            {"项目": "建议低吸股数", "内容": f"{t_strategy.get('suggested_buy_shares', 0)}股"},
            {"项目": "建议买入金额", "内容": format_amount(t_strategy.get("suggested_buy_amount"))},
            {"项目": "买一手所需金额", "内容": format_amount(t_strategy.get("one_lot_buy_amount"))},
            {"项目": "非做T观察买入", "内容": f"{t_strategy.get('observe_buy_shares', 0)}股 / {format_amount(t_strategy.get('observe_buy_amount'))}"},
            {"项目": "建议高抛价格", "内容": format_price(t_strategy.get("suggested_sell_price"))},
            {"项目": "建议高抛股数", "内容": f"{t_strategy.get('suggested_sell_shares', 0)}股"},
            {"项目": "建议卖出金额", "内容": format_amount(t_strategy.get("suggested_sell_amount"))},
            {"项目": "卖一手参考金额", "内容": format_amount(t_strategy.get("one_lot_sell_amount"))},
            {"项目": "止损线", "内容": format_price(t_strategy.get("stop_loss"))},
            {"项目": "止盈线", "内容": format_price(t_strategy.get("take_profit"))},
            {"项目": "VWAP", "内容": format_price(t_strategy.get("vwap"))},
            {"项目": "第一低位跌破概率", "内容": f"{low_model.get('break_first_probability')}%"},
            {"项目": "第二低位跌破概率", "内容": f"{low_model.get('break_second_probability')}%"},
            {"项目": "主力行为评分", "内容": f"{main_force.get('score')} / {main_force.get('label')}"},
            {"项目": "短线/中期/长期评分", "内容": f"{potential.get('short_score')} / {potential.get('mid_score')} / {potential.get('long_score')}"},
            {"项目": "是否禁止做T", "内容": "是" if t_strategy.get("forbid_t") else "否"},
            {"项目": "禁止原因", "内容": "；".join(t_strategy.get("forbid_reasons") or ["未触发"])},
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("今日操作记录")
max_count = int(watch.get("max_t_trades_per_day", 1))
st.write(f"今日已操作次数：{today_count} / {max_count}")
col_a, col_b = st.columns(2)
if col_a.button("记录一次做T操作", disabled=today_count >= max_count):
    increment_t_operation(code)
    st.success("已记录一次做T操作")
    st.rerun()
if today_count >= max_count:
    col_b.warning("已达到单日最大做T次数，请停止当日做T。")
else:
    col_b.info("仍需结合价格区间、评分和风险等级控制操作频率。")

st.subheader("今日已触发提醒")
alerts = [item for item in get_recent_alerts(30) if item["code"] == code]
if alerts:
    st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
else:
    st.info("暂无已保存提醒。")

render_risk_footer()
