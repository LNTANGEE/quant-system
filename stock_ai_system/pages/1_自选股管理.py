from __future__ import annotations

import pandas as pd
import streamlit as st

from data.stock_data import get_stock_name
from database.db import (
    add_watch_stock,
    delete_watch_stock,
    get_watchlist,
    init_db,
    update_watch_stock,
)
from utils.helpers import render_mobile_style, render_risk_footer


init_db()
render_mobile_style()

st.title("自选股管理")
st.caption("维护持仓、成本、手动现价、可卖数量、单次金额、做T次数和提醒阈值。手动现价用于行情接口不可用时兜底。")

with st.form("add_watch_stock"):
    cols = st.columns([1, 1, 1])
    code = cols[0].text_input("股票代码", placeholder="例如 300308")
    name = cols[1].text_input("股票名称，可选")
    auto_name = cols[2].checkbox("自动查询名称", value=True)
    pos_cols = st.columns(5)
    shares = pos_cols[0].number_input("持仓数量", min_value=0, step=100)
    cost_price = pos_cols[1].number_input("成本价", min_value=0.0, step=0.01, format="%.3f")
    manual_price = pos_cols[2].number_input("手动现价", min_value=0.0, step=0.01, format="%.3f")
    sellable_quantity = pos_cols[3].number_input("可卖数量", min_value=0, step=100)
    max_trade_amount = pos_cols[4].number_input("最大单次金额", min_value=0.0, step=1000.0, value=5000.0)
    rule_cols = st.columns(2)
    max_t_trades = rule_cols[0].number_input("单日最大做T次数", min_value=0, max_value=20, value=1, step=1)
    alert_threshold = rule_cols[1].number_input("提醒阈值(%)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
    submitted = st.form_submit_button("添加到自选股")
    if submitted:
        final_name = name
        if auto_name and code and not name:
            with st.spinner("正在查询股票名称..."):
                final_name = get_stock_name(code)
        if add_watch_stock(code, final_name):
            update_watch_stock(
                code,
                shares=int(shares),
                cost_price=float(cost_price),
                manual_price=float(manual_price),
                sellable_quantity=int(sellable_quantity),
                max_trade_amount=float(max_trade_amount),
                max_t_trades_per_day=int(max_t_trades),
                alert_threshold=float(alert_threshold),
            )
            st.success("已添加或更新自选股")
            st.rerun()
        else:
            st.warning("请输入有效的6位股票代码")

watchlist = get_watchlist()
if not watchlist:
    st.info("暂无自选股。")
    render_risk_footer()
    st.stop()

df = pd.DataFrame(watchlist)
editable = df[
    [
        "code",
        "name",
        "shares",
        "cost_price",
        "manual_price",
        "sellable_quantity",
        "max_trade_amount",
        "max_t_trades_per_day",
        "alert_threshold",
    ]
].rename(
    columns={
        "code": "代码",
        "name": "名称",
        "shares": "持仓数量",
        "cost_price": "成本价",
        "manual_price": "手动现价(可选)",
        "sellable_quantity": "可卖数量",
        "max_trade_amount": "最大单次操作金额",
        "max_t_trades_per_day": "单日最大做T次数",
        "alert_threshold": "提醒阈值(%)",
    }
)

edited = st.data_editor(
    editable,
    use_container_width=True,
    hide_index=True,
    column_config={
        "代码": st.column_config.TextColumn(disabled=True),
        "持仓数量": st.column_config.NumberColumn(min_value=0, step=100),
        "成本价": st.column_config.NumberColumn(min_value=0.0, format="%.3f"),
        "手动现价(可选)": st.column_config.NumberColumn(min_value=0.0, format="%.3f", help="公开行情不可用时，用这个价格兜底计算。"),
        "可卖数量": st.column_config.NumberColumn(min_value=0, step=100),
        "最大单次操作金额": st.column_config.NumberColumn(min_value=0.0, step=1000.0),
        "单日最大做T次数": st.column_config.NumberColumn(min_value=0, max_value=20, step=1),
        "提醒阈值(%)": st.column_config.NumberColumn(min_value=0.1, max_value=50.0, step=0.1),
    },
)

col_save, col_delete = st.columns([1, 2])
if col_save.button("保存修改", type="primary"):
    for _, row in edited.iterrows():
        update_watch_stock(
            row["代码"],
            name=row["名称"],
            shares=int(row["持仓数量"]),
            cost_price=float(row["成本价"]),
            manual_price=float(row["手动现价(可选)"]),
            sellable_quantity=int(row["可卖数量"]),
            max_trade_amount=float(row["最大单次操作金额"]),
            max_t_trades_per_day=int(row["单日最大做T次数"]),
            alert_threshold=float(row["提醒阈值(%)"]),
        )
    st.success("自选股设置已保存")
    st.rerun()

delete_codes = col_delete.multiselect("选择要删除的股票", edited["代码"].tolist())
if delete_codes and st.button("删除选中股票"):
    for item in delete_codes:
        delete_watch_stock(item)
    st.success("已删除选中股票")
    st.rerun()

render_risk_footer()
