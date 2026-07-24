from __future__ import annotations

import pandas as pd
import streamlit as st

from data.stock_data import get_hist_daily, get_minute_kline
from database.db import get_stock_pool, get_watchlist, init_db
from models.backtest import (
    run_low_zone_backtest,
    run_next_hour_high_backtest,
    run_potential_backtest,
    run_t_strategy_backtest,
    summarize_next_hour_high_backtest,
    summarize_returns,
)
from utils.helpers import normalize_code, parse_code_lines, render_mobile_style, render_risk_footer


init_db()
render_mobile_style()

st.title("回测验证")
st.caption("使用历史K线验证低位区间、做T策略和短线潜力评分。第一版为简化回测，不含真实滑点和手续费。")

watchlist = get_watchlist()
pool = get_stock_pool(include_watchlist=True)
default_codes = [item["code"] for item in watchlist[:5]] or [item["code"] for item in pool[:5]]

with st.sidebar:
    st.header("回测参数")
    selected_text = st.text_area("股票代码列表", value="\n".join(default_codes), height=140)
    lookback = st.slider("回测交易日数量", 30, 260, 120, step=10)
    top_n = st.slider("每日潜力股Top N", 1, 10, 10)
    adjust = st.selectbox("复权方式", ["qfq", "", "hfq"], format_func=lambda x: {"qfq": "前复权", "": "不复权", "hfq": "后复权"}[x])

codes = parse_code_lines(selected_text)
if not codes:
    st.info("请输入至少一个股票代码。")
    render_risk_footer()
    st.stop()

histories = {}
with st.spinner("正在获取历史K线..."):
    for code in codes:
        hist = get_hist_daily(code, adjust=adjust)
        if not hist.empty:
            histories[normalize_code(code)] = hist

if not histories:
    st.warning("未获取到可回测的历史K线。")
    render_risk_footer()
    st.stop()

tabs = st.tabs(["低位区间命中率", "做T策略胜率", "潜力股筛选回测", "未来一小时高点回测"])

with tabs[0]:
    code = st.selectbox("选择股票", list(histories.keys()), key="low_code")
    low_bt = run_low_zone_backtest(histories[code], lookback=lookback)
    if low_bt.empty:
        st.warning("样本不足，无法回测。")
    else:
        hit_first = low_bt["hit_first"].mean() * 100
        hit_second = low_bt["hit_second"].mean() * 100
        hit_extreme = low_bt["hit_extreme"].mean() * 100
        cols = st.columns(4)
        cols[0].metric("样本数", len(low_bt))
        cols[1].metric("第一低位区触达率", f"{hit_first:.2f}%")
        cols[2].metric("第二低位区触达率", f"{hit_second:.2f}%")
        cols[3].metric("极限低位区触达率", f"{hit_extreme:.2f}%")
        st.dataframe(low_bt.tail(80), use_container_width=True, hide_index=True)

with tabs[1]:
    code = st.selectbox("选择股票", list(histories.keys()), key="t_code")
    t_bt = run_t_strategy_backtest(histories[code], lookback=lookback)
    if t_bt.empty:
        st.warning("未触发足够的低吸样本。")
    else:
        summary = summarize_returns(t_bt["return_pct"].tolist())
        cols = st.columns(5)
        cols[0].metric("样本数", summary["样本数"])
        cols[1].metric("胜率", f"{summary['胜率']}%")
        cols[2].metric("平均收益", f"{summary['平均收益']}%")
        cols[3].metric("最大回撤", f"{summary['最大回撤']}%")
        cols[4].metric("连续失败次数", summary["连续失败次数"])
        st.dataframe(t_bt.tail(100), use_container_width=True, hide_index=True)

with tabs[2]:
    result = run_potential_backtest(histories, lookback=lookback, top_n=top_n)
    picks = result.get("picks", pd.DataFrame())
    if picks.empty:
        st.warning("样本不足，无法执行潜力股回测。")
    else:
        summary = result["summary"]
        st.subheader("收益统计")
        rows = []
        for horizon, values in summary.items():
            row = {"周期": horizon}
            row.update(values)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("评分阈值表现差异")
        st.dataframe(result["threshold_summary"], use_container_width=True, hide_index=True)
        st.subheader("每日短线评分Top记录")
        st.dataframe(picks.tail(200), use_container_width=True, hide_index=True)
        st.caption("市场环境与板块分组在第一版以中性假设处理，后续可接指数与行业历史序列后扩展。")

with tabs[3]:
    st.caption(
        "按历史分钟线逐时点向前滚动：预测时只使用锚点之前的数据，"
        "真实标签为其后60个交易分钟的最高价，午间休市不计时。"
    )
    code = st.selectbox("选择股票", list(histories.keys()), key="hour_high_code")
    max_anchors = st.slider("最大回测锚点数", 20, 120, 60, step=10)
    if st.button("运行未来一小时高点回测", type="primary"):
        now = pd.Timestamp.now(tz="Asia/Shanghai")
        calendar_days = min(max(int(lookback * 1.6), 45), 365)
        with st.spinner("正在获取5分钟历史数据并执行无未来数据泄漏回测..."):
            minute_history = get_minute_kline(
                code,
                period="5",
                start_date=(now - pd.Timedelta(days=calendar_days)).strftime("%Y-%m-%d 09:30:00"),
                end_date=now.strftime("%Y-%m-%d 15:00:00"),
                ttl=1800,
                use_realtime_fallback=False,
            )
            hour_bt = run_next_hour_high_backtest(
                minute_history,
                histories[code],
                horizon_minutes=60,
                max_anchors=max_anchors,
            )
        if hour_bt.empty:
            st.warning("历史5分钟数据或完整60分钟预测窗口不足，暂时无法回测。")
        else:
            summary = summarize_next_hour_high_backtest(hour_bt)
            cols = st.columns(6)
            cols[0].metric("样本数", summary["样本数"])
            cols[1].metric("区间覆盖率", f"{summary['区间覆盖率']:.2f}%")
            cols[2].metric("区间下沿触达率", f"{summary['区间下沿触达率']:.2f}%")
            cols[3].metric("中位绝对误差", f"{summary['中位绝对误差']:.3f}%")
            cols[4].metric("平均预测偏差", f"{summary['平均预测偏差']:.3f}%")
            cols[5].metric("突破概率Brier分数", f"{summary['突破概率Brier分数']:.4f}")
            st.dataframe(
                hour_bt.rename(
                    columns={
                        "anchor_time": "预测时点",
                        "window_end": "窗口结束",
                        "actual_high": "实际最高价",
                        "predicted_high": "预估最高价",
                        "zone_low": "区间下沿",
                        "zone_high": "区间上沿",
                        "interval_hit": "区间命中",
                        "lower_bound_reached": "触达下沿",
                        "error_pct": "预测偏差%",
                        "absolute_error_pct": "绝对误差%",
                        "actual_break_day_high": "实际突破日高",
                        "predicted_break_probability": "预测突破概率%",
                        "brier_score": "Brier分数",
                        "confidence": "置信度",
                        "method": "方法",
                        "sample_size": "样本数",
                        "effective_sample_size": "有效样本",
                    }
                ).tail(120),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "区间覆盖率衡量真实最高价是否落入预测区间；"
                "Brier分数越低表示“突破今日高点概率”校准越好。"
            )

render_risk_footer()
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.stock_data import get_hist_daily
from database.db import get_stock_pool, get_watchlist, init_db
from models.backtest import (
    run_low_zone_backtest,
    run_potential_backtest,
    run_t_strategy_backtest,
    summarize_returns,
)
from utils.helpers import normalize_code, parse_code_lines, render_mobile_style, render_risk_footer


init_db()
render_mobile_style()

st.title("回测验证")
st.caption("使用历史K线验证低位区间、做T策略和短线潜力评分。第一版为简化回测，不含真实滑点和手续费。")

watchlist = get_watchlist()
pool = get_stock_pool(include_watchlist=True)
default_codes = [item["code"] for item in watchlist[:5]] or [item["code"] for item in pool[:5]]

with st.sidebar:
    st.header("回测参数")
    selected_text = st.text_area("股票代码列表", value="\n".join(default_codes), height=140)
    lookback = st.slider("回测交易日数量", 30, 260, 120, step=10)
    top_n = st.slider("每日潜力股Top N", 1, 10, 10)
    adjust = st.selectbox("复权方式", ["qfq", "", "hfq"], format_func=lambda x: {"qfq": "前复权", "": "不复权", "hfq": "后复权"}[x])

codes = parse_code_lines(selected_text)
if not codes:
    st.info("请输入至少一个股票代码。")
    render_risk_footer()
    st.stop()

histories = {}
with st.spinner("正在获取历史K线..."):
    for code in codes:
        hist = get_hist_daily(code, adjust=adjust)
        if not hist.empty:
            histories[normalize_code(code)] = hist

if not histories:
    st.warning("未获取到可回测的历史K线。")
    render_risk_footer()
    st.stop()

tabs = st.tabs(["低位区间命中率", "做T策略胜率", "潜力股筛选回测"])

with tabs[0]:
    code = st.selectbox("选择股票", list(histories.keys()), key="low_code")
    low_bt = run_low_zone_backtest(histories[code], lookback=lookback)
    if low_bt.empty:
        st.warning("样本不足，无法回测。")
    else:
        hit_first = low_bt["hit_first"].mean() * 100
        hit_second = low_bt["hit_second"].mean() * 100
        hit_extreme = low_bt["hit_extreme"].mean() * 100
        cols = st.columns(4)
        cols[0].metric("样本数", len(low_bt))
        cols[1].metric("第一低位区触达率", f"{hit_first:.2f}%")
        cols[2].metric("第二低位区触达率", f"{hit_second:.2f}%")
        cols[3].metric("极限低位区触达率", f"{hit_extreme:.2f}%")
        st.dataframe(low_bt.tail(80), use_container_width=True, hide_index=True)

with tabs[1]:
    code = st.selectbox("选择股票", list(histories.keys()), key="t_code")
    t_bt = run_t_strategy_backtest(histories[code], lookback=lookback)
    if t_bt.empty:
        st.warning("未触发足够的低吸样本。")
    else:
        summary = summarize_returns(t_bt["return_pct"].tolist())
        cols = st.columns(5)
        cols[0].metric("样本数", summary["样本数"])
        cols[1].metric("胜率", f"{summary['胜率']}%")
        cols[2].metric("平均收益", f"{summary['平均收益']}%")
        cols[3].metric("最大回撤", f"{summary['最大回撤']}%")
        cols[4].metric("连续失败次数", summary["连续失败次数"])
        st.dataframe(t_bt.tail(100), use_container_width=True, hide_index=True)

with tabs[2]:
    result = run_potential_backtest(histories, lookback=lookback, top_n=top_n)
    picks = result.get("picks", pd.DataFrame())
    if picks.empty:
        st.warning("样本不足，无法执行潜力股回测。")
    else:
        summary = result["summary"]
        st.subheader("收益统计")
        rows = []
        for horizon, values in summary.items():
            row = {"周期": horizon}
            row.update(values)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("评分阈值表现差异")
        st.dataframe(result["threshold_summary"], use_container_width=True, hide_index=True)
        st.subheader("每日短线评分Top记录")
        st.dataframe(picks.tail(200), use_container_width=True, hide_index=True)
        st.caption("市场环境与板块分组在第一版以中性假设处理，后续可接指数与行业历史序列后扩展。")

render_risk_footer()
