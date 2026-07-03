from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from data.stock_data import get_quote, get_stock_name
from database.db import add_sold_position, get_sold_positions, init_db, update_sold_position_status
from utils.helpers import format_amount, format_pct, format_price, normalize_code, render_mobile_style, render_risk_footer, safe_div


init_db()
render_mobile_style()

st.title("卖出后接回提醒")
st.caption("记录卖出价格后，按回调幅度给出第一笔、第二笔、第三笔接回观察提醒；若不跌反涨，提示不追高。")

with st.form("sell_record"):
    cols = st.columns(5)
    code = cols[0].text_input("股票代码", placeholder="300308")
    name = cols[1].text_input("股票名称，可选")
    sell_price = cols[2].number_input("卖出价格", min_value=0.0, step=0.01, format="%.3f")
    sell_amount = cols[3].number_input("卖出金额", min_value=0.0, step=1000.0)
    sell_date = cols[4].date_input("卖出日期", value=date.today())
    note = st.text_input("备注", placeholder="例如 高抛减仓")
    submitted = st.form_submit_button("保存卖出记录")
    if submitted:
        code_norm = normalize_code(code)
        final_name = name or get_stock_name(code_norm)
        if code_norm and sell_price > 0:
            add_sold_position(code_norm, final_name, sell_price, sell_amount, sell_date.isoformat(), note)
            st.success("卖出记录已保存")
            st.rerun()
        else:
            st.warning("请输入有效代码和卖出价格")

records = get_sold_positions(active_only=True)
if not records:
    st.info("暂无活跃卖出记录。")
    render_risk_footer()
    st.stop()

rows = []
for record in records:
    quote = get_quote(record["code"])
    price = quote.get("price", 0)
    pullback = safe_div(price - record["sell_price"], record["sell_price"]) * 100
    if pullback <= -12:
        reminder = "回调12%-15%区间附近，第三笔接回观察提醒，仍需确认风险等级。"
    elif pullback <= -8:
        reminder = "回调8%-10%区间附近，第二笔接回观察提醒。"
    elif pullback <= -5:
        reminder = "回调约5%，第一笔接回观察提醒。"
    elif pullback > 0:
        reminder = "卖出后不跌反涨，提示不要追高，等待新的低吸区。"
    else:
        reminder = "尚未触发接回区间，继续观察。"
    rows.append(
        {
            "ID": record["id"],
            "股票代码": record["code"],
            "股票名称": record["name"] or quote.get("name", ""),
            "卖出价格": format_price(record["sell_price"]),
            "卖出金额": format_amount(record["sell_amount"]),
            "卖出日期": record["sell_date"],
            "当前价格": format_price(price),
            "相对卖出价回调幅度": format_pct(pullback),
            "接回提醒": reminder,
            "备注": record.get("note", ""),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

close_id = st.selectbox("标记为已完成/不再提醒", [""] + [str(row["ID"]) for row in rows])
if close_id and st.button("更新状态"):
    update_sold_position_status(int(close_id), "closed")
    st.success("已更新状态")
    st.rerun()

render_risk_footer()
