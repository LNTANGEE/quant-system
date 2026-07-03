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
