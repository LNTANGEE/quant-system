from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.db import get_today_operation_count, get_watchlist, get_watch_stock, init_db
from models.indicators import add_intraday_vwap, enrich_indicators, latest_snapshot, support_resistance
from models.private_steward import build_steward_advice
from utils.analysis import analyze_stock
from utils.helpers import format_amount, format_pct, format_price, format_zone, normalize_code, render_mobile_style, render_risk_footer


init_db()
render_mobile_style()

st.title("个股详情")
st.caption("多因子概率量化框架：实时行情、分时/日K、低位区间、主力行为、做T建议、基本面与多周期评分。")

watchlist = get_watchlist()
options = [f"{item['code']} {item.get('name', '')}" for item in watchlist]
col1, col2 = st.columns([2, 1])
selected = col1.selectbox("从自选股选择", options, index=0 if options else None, placeholder="暂无自选股")
manual_code = col2.text_input("或输入代码", placeholder="例如 300308")
code = normalize_code(manual_code or (selected.split()[0] if selected else ""))

if not code:
    st.info("请选择或输入股票代码。")
    render_risk_footer()
    st.stop()

watch = get_watch_stock(code) or {}
with st.spinner("正在加载个股行情与模型..."):
    result = analyze_stock(code, watch=watch, include_minute=True)

if result.get("error"):
    st.error(result["error"])
    render_risk_footer()
    st.stop()

quote = result["quote"]
daily = result["daily"]
minute = result["minute"]
low_model = result["low_model"]
high_model = result.get("high_model", {})
next_hour = result.get("next_hour_high", {})
t_strategy = result["t_strategy"]
main_force = result["main_force"]
risk = result["risk"]
sector = result["sector"]
fundamentals = result["fundamentals"]
potential = result["potential"]
steward = build_steward_advice(result, watch, get_today_operation_count(code))

st.subheader(f"{code} {quote.get('name', '')}")
metric_cols = st.columns(6)
metric_cols[0].metric("当前价", format_price(quote.get("price")), format_pct(quote.get("pct_chg")))
metric_cols[1].metric("今日开盘", format_price(quote.get("open")))
metric_cols[2].metric("最高/最低", f"{format_price(quote.get('high'))} / {format_price(quote.get('low'))}")
metric_cols[3].metric("成交额", format_amount(quote.get("amount")))
metric_cols[4].metric("量比/换手", f"{format_price(quote.get('volume_ratio'))} / {format_pct(quote.get('turnover'))}")
metric_cols[5].metric("风险等级", f"{risk.get('risk_level')}({risk.get('risk_score')})")

forecast_cols = st.columns(5)
forecast_cols[0].metric("今日预估最低价区间", format_zone(low_model.get("estimated_low_zone")))
forecast_cols[1].metric("最低区触达概率", format_pct(low_model.get("estimated_low_reach_probability")))
forecast_cols[2].metric("今日预估最高价区间", format_zone(high_model.get("estimated_high_zone")))
forecast_cols[3].metric("最高触达概率", format_pct(high_model.get("estimated_high_reach_probability")))
forecast_cols[4].metric("做T执行计划", t_strategy.get("current_action", "观察"))
st.caption(low_model.get("estimated_low_label", ""))
st.caption(high_model.get("estimated_high_label", ""))
st.info(t_strategy.get("observe_plan", ""))

if next_hour.get("available"):
    hour_horizon = int(next_hour.get("horizon_minutes") or 60)
    st.subheader(f"未来{hour_horizon}个交易分钟最高价预测")
    hour_cols = st.columns(4)
    hour_cols[0].metric("预估最高价", format_price(next_hour.get("predicted_high")))
    hour_cols[1].metric("预估最高区间", format_zone(next_hour.get("predicted_high_zone")))
    hour_cols[2].metric(
        "突破今日高点概率",
        format_pct(next_hour.get("break_day_high_probability")),
    )
    hour_cols[3].metric("预测置信度", format_pct(next_hour.get("confidence")))
    st.caption(
        f"较现价预期上行 {format_pct(next_hour.get('expected_upside_pct'))} · "
        f"窗口 {next_hour.get('window_start', '--')} 至 {next_hour.get('window_end', '--')} · "
        f"样本/有效样本 {next_hour.get('sample_size', 0)}/"
        f"{next_hour.get('effective_sample_size', 0)} · "
        f"方法 {next_hour.get('method', '--')}"
    )
    if next_hour.get("method") == "atr_realized_volatility_fallback":
        st.warning(next_hour.get("label", "当前使用低样本回退模型，结果仅作观察。"))
    else:
        st.info(next_hour.get("label", ""))
else:
    st.warning(
        "未来一小时最高价预测暂不可用："
        f"{next_hour.get('label') or next_hour.get('explain') or '分钟行情不足'}"
    )

st.subheader("私人管家")
steward_cols = st.columns(4)
steward_cols[0].metric("管家信号", steward["signal"])
steward_cols[1].metric("做T判断", steward["signal_level"])
steward_cols[2].metric("低吸数量", f"{steward['buy_shares']}股 / {format_amount(steward['buy_amount'])}")
steward_cols[3].metric("高抛数量", f"{steward['sell_shares']}股 / {format_amount(steward['sell_amount'])}")
position_cols = st.columns(4)
position_cols[0].metric("持仓数量", f"{steward['holding_shares']}股")
position_cols[1].metric("成本价", format_price(steward["cost_price"], 3))
position_cols[2].metric("持仓市值", format_amount(steward["position_value"]))
position_cols[3].metric("相对成本", f"{steward['position_profit_pct']:.2f}%")
st.success(steward["current_advice"])
st.info(steward["low_judgement"])
st.info(steward.get("high_judgement", ""))
for alert in steward["realtime_alerts"]:
    st.warning(alert) if "风控" in alert or "弱势" in alert else st.info(alert)
entry_col, exit_col = st.columns(2)
entry_col.success(steward["entry_timing"])
exit_col.warning(steward["exit_timing"])
st.caption(steward["risk_guard"])

tabs = st.tabs(["分时图", "日K图", "技术与高低位", "主力与做T", "基本面与板块", "潜力评分"])

with tabs[0]:
    if minute.empty:
        st.warning("分钟K数据暂不可用，请稍后重试或检查AKShare接口。")
    else:
        minute_plot = add_intraday_vwap(minute)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=minute_plot["datetime"],
                y=minute_plot["close"],
                mode="lines",
                name="分时价格",
                line=dict(color="#2563eb", width=2),
            )
        )
        if "vwap" in minute_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=minute_plot["datetime"],
                    y=minute_plot["vwap"],
                    mode="lines",
                    name="VWAP",
                    line=dict(color="#f97316", width=1.5),
                )
            )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    if daily.empty:
        st.warning("日K数据暂不可用。")
    else:
        enriched = enrich_indicators(daily).tail(160)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=enriched["date"],
                open=enriched["open"],
                high=enriched["high"],
                low=enriched["low"],
                close=enriched["close"],
                name="日K",
            )
        )
        for ma, color in [("ma5", "#2563eb"), ("ma10", "#16a34a"), ("ma20", "#f97316")]:
            fig.add_trace(go.Scatter(x=enriched["date"], y=enriched[ma], name=ma.upper(), line=dict(width=1.2, color=color)))
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    snap = latest_snapshot(daily)
    sr = support_resistance(daily)
    low_stat = low_model.get("statistical_model", {})
    high_stat = high_model.get("statistical_model", {})
    cols = st.columns(3)
    cols[0].metric("今日预估最低价区间", format_zone(low_model.get("estimated_low_zone")))
    cols[1].metric("第一低位区", format_zone(low_model.get("first_low_zone")))
    cols[2].metric("极限低位区", format_zone(low_model.get("extreme_low_zone")))
    high_cols = st.columns(3)
    high_cols[0].metric("今日预估最高价区间", format_zone(high_model.get("estimated_high_zone")))
    high_cols[1].metric("正常反弹高位区", format_zone(high_model.get("first_high_zone")))
    high_cols[2].metric("极限高位观察区", format_zone(high_model.get("extreme_high_zone")))
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "基准预估高点", "值": format_price(high_model.get("base_case_high"))},
                {"项目": "强势预估高点", "值": format_price(high_model.get("strong_case_high"))},
                {"项目": "冲高回落观察高点", "值": format_price(high_model.get("spike_case_high"))},
                {"项目": "统计校准高位区", "值": format_zone(high_model.get("statistical_high_zone"))},
                {"项目": "高位统计样本/有效样本", "值": f"{high_stat.get('sample_size', 0)} / {high_stat.get('effective_sample_size', 0)}"},
                {"项目": "盘中进度/统计置信度", "值": f"{format_pct(high_stat.get('day_progress', 0) * 100)} / {format_pct(high_stat.get('confidence'))}"},
                {"项目": "预估最高区间触达概率", "值": format_pct(high_model.get("estimated_high_reach_probability"))},
                {"项目": "高位模型置信度", "值": format_pct(high_model.get("estimated_high_confidence"))},
                {"项目": "当前价距离第一高位区", "值": format_pct(high_model.get("distance_to_first_high_pct"))},
                {"项目": "当前价距离第二高位区", "值": format_pct(high_model.get("distance_to_second_high_pct"))},
                {"项目": "今日再创新高概率", "值": format_pct(high_model.get("new_high_probability"))},
                {"项目": "突破第一高位区概率", "值": format_pct(high_model.get("break_first_high_probability"))},
                {"项目": "突破第二高位区概率", "值": format_pct(high_model.get("break_second_high_probability"))},
                {"项目": "明日惯性高开概率", "值": format_pct(high_model.get("tomorrow_gap_up_probability"))},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "基准预估低点", "值": format_price(low_model.get("base_case_low"))},
                {"项目": "弱势预估低点", "值": format_price(low_model.get("weak_case_low"))},
                {"项目": "恐慌预估低点", "值": format_price(low_model.get("panic_case_low"))},
                {"项目": "统计校准低位区", "值": format_zone(low_model.get("statistical_low_zone"))},
                {"项目": "低位统计样本/有效样本", "值": f"{low_stat.get('sample_size', 0)} / {low_stat.get('effective_sample_size', 0)}"},
                {"项目": "盘中进度/统计置信度", "值": f"{format_pct(low_stat.get('day_progress', 0) * 100)} / {format_pct(low_stat.get('confidence'))}"},
                {"项目": "预估最低区间触达概率", "值": format_pct(low_model.get("estimated_low_reach_probability"))},
                {"项目": "模型置信度", "值": format_pct(low_model.get("estimated_low_confidence"))},
                {"项目": "当前价距离第一低位区", "值": format_pct(low_model.get("distance_to_first_pct"))},
                {"项目": "当前价距离第二低位区", "值": format_pct(low_model.get("distance_to_second_pct"))},
                {"项目": "今日再创新低概率", "值": format_pct(low_model.get("new_low_probability"))},
                {"项目": "跌破第一低位区概率", "值": format_pct(low_model.get("break_first_probability"))},
                {"项目": "跌破第二低位区概率", "值": format_pct(low_model.get("break_second_probability"))},
                {"项目": "明日惯性低开概率", "值": format_pct(low_model.get("tomorrow_gap_down_probability"))},
                {"项目": "支撑位", "值": format_price(sr.get("support"))},
                {"项目": "压力位", "值": format_price(sr.get("resistance"))},
                {"项目": "MA5/MA10/MA20", "值": f"{format_price(snap.get('ma5'))} / {format_price(snap.get('ma10'))} / {format_price(snap.get('ma20'))}"},
                {"项目": "ATR14 / RSI14", "值": f"{format_price(snap.get('atr14'))} / {format_price(snap.get('rsi14'))}"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[3]:
    cols = st.columns(4)
    cols[0].metric("主力行为评分", f"{main_force.get('score')} / {main_force.get('label')}")
    cols[1].metric("承接评分", main_force.get("absorption_score"))
    cols[2].metric("出货风险评分", main_force.get("distribution_risk"))
    cols[3].metric("尾盘风险评分", main_force.get("tail_risk"))
    st.write(main_force.get("explain"))
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "今日低吸区", "值": format_zone(t_strategy.get("low_buy_zone"))},
                {"项目": "今日高抛区", "值": format_zone(t_strategy.get("high_sell_zone"))},
                {"项目": "正T建议", "值": t_strategy.get("positive_t")},
                {"项目": "反T建议", "值": t_strategy.get("reverse_t")},
                {"项目": "建议做T比例", "值": f"{t_strategy.get('suggested_ratio', 0) * 100:.0f}%"},
                {"项目": "建议低吸价格", "值": format_price(t_strategy.get("suggested_buy_price"))},
                {"项目": "建议低吸股数", "值": f"{t_strategy.get('suggested_buy_shares', 0)}股"},
                {"项目": "建议买入金额", "值": format_amount(t_strategy.get("suggested_buy_amount"))},
                {"项目": "买一手所需金额", "值": format_amount(t_strategy.get("one_lot_buy_amount"))},
                {"项目": "非做T观察买入", "值": f"{t_strategy.get('observe_buy_shares', 0)}股 / {format_amount(t_strategy.get('observe_buy_amount'))}"},
                {"项目": "建议高抛价格", "值": format_price(t_strategy.get("suggested_sell_price"))},
                {"项目": "建议高抛股数", "值": f"{t_strategy.get('suggested_sell_shares', 0)}股"},
                {"项目": "建议卖出金额", "值": format_amount(t_strategy.get("suggested_sell_amount"))},
                {"项目": "卖一手参考金额", "值": format_amount(t_strategy.get("one_lot_sell_amount"))},
                {"项目": "执行摘要", "值": t_strategy.get("execution_plan")},
                {"项目": "止损线", "值": format_price(t_strategy.get("stop_loss"))},
                {"项目": "止盈线", "值": format_price(t_strategy.get("take_profit"))},
                {"项目": "做T评分", "值": f"{t_strategy.get('t_score')} / {t_strategy.get('t_score_label')}"},
                {"项目": "是否禁止做T", "值": "是" if t_strategy.get("forbid_t") else "否"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[4]:
    st.write(f"所属行业：{sector.get('industry', '未知')}；板块强弱：{sector.get('sector_strength', '数据不足')}；相对板块强弱：{format_pct(sector.get('relative_strength', 0))}")
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "营业收入", "值": fundamentals.get("revenue", "--")},
                {"项目": "营收同比", "值": format_pct(fundamentals.get("revenue_yoy"))},
                {"项目": "净利润", "值": fundamentals.get("net_profit", "--")},
                {"项目": "净利润同比", "值": format_pct(fundamentals.get("net_profit_yoy"))},
                {"项目": "毛利率/净利率", "值": f"{format_pct(fundamentals.get('gross_margin'))} / {format_pct(fundamentals.get('net_margin'))}"},
                {"项目": "ROE/ROA", "值": f"{format_pct(fundamentals.get('roe'))} / {format_pct(fundamentals.get('roa'))}"},
                {"项目": "负债率", "值": format_pct(fundamentals.get("debt_ratio"))},
                {"项目": "PE/PB/PS/PEG", "值": f"{format_price(fundamentals.get('pe'))} / {format_price(fundamentals.get('pb'))} / -- / --"},
                {"项目": "估值历史分位", "值": "预留接口"},
                {"项目": "机构/基金持仓变化", "值": "预留接口"},
                {"项目": "数据说明", "值": fundamentals.get("data_quality", "")},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[5]:
    cols = st.columns(4)
    cols[0].metric("短线评分", f"{potential.get('short_score')} / {potential.get('short_label')}")
    cols[1].metric("中期评分", f"{potential.get('mid_score')} / {potential.get('mid_label')}")
    cols[2].metric("长期评分", f"{potential.get('long_score')} / {potential.get('long_label')}")
    cols[3].metric("综合评分", f"{potential.get('comprehensive_score')} / {potential.get('stock_type')}")
    st.info(potential.get("operation_suggestion"))
    factor_cols = st.columns(3)
    factor_cols[0].dataframe(pd.DataFrame(potential.get("short_factors", {}).items(), columns=["短线因子", "得分"]), use_container_width=True, hide_index=True)
    factor_cols[1].dataframe(pd.DataFrame(potential.get("mid_factors", {}).items(), columns=["中期因子", "得分"]), use_container_width=True, hide_index=True)
    factor_cols[2].dataframe(pd.DataFrame(potential.get("long_factors", {}).items(), columns=["长期因子", "得分"]), use_container_width=True, hide_index=True)

render_risk_footer()
