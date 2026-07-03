from __future__ import annotations

import streamlit as st

from config import AKSHARE_TIMEOUT_SECONDS, TUSHARE_TOKEN
from database.db import get_setting, init_db, set_setting
from utils.helpers import render_mobile_style, render_risk_footer


init_db()
render_mobile_style()

st.title("系统设置")
st.caption("数据源、Token、实时行情、Level-2、大单资金和企业微信提醒的预留配置。")

st.subheader("数据源")
st.write("第一版数据源：AKShare")
st.write(f"AKShare超时设置：{AKSHARE_TIMEOUT_SECONDS}秒")

st.subheader("Tushare Pro Token")
stored_token = get_setting("tushare_token", TUSHARE_TOKEN)
token = st.text_input("Token", value=stored_token or "", type="password", help="第一版仅保存配置，后续可接入Tushare Pro数据。")
if st.button("保存Token"):
    set_setting("tushare_token", token)
    st.success("Token已保存到本地SQLite")

st.subheader("后续扩展接口")
paid_realtime = st.text_input("付费实时行情接口地址", value=get_setting("paid_realtime_url", ""))
level2_url = st.text_input("Level-2接口地址", value=get_setting("level2_url", ""))
fund_flow_url = st.text_input("大单资金接口地址", value=get_setting("fund_flow_url", ""))
wechat_webhook = st.text_input("企业微信Webhook", value=get_setting("wechat_webhook", ""), type="password")

if st.button("保存扩展接口配置"):
    set_setting("paid_realtime_url", paid_realtime)
    set_setting("level2_url", level2_url)
    set_setting("fund_flow_url", fund_flow_url)
    set_setting("wechat_webhook", wechat_webhook)
    st.success("扩展配置已保存")

st.info(
    "第一版主力资金为代理评分：使用成交量、量比、换手率、VWAP、分时结构和相对板块强弱模拟资金强弱。"
)

render_risk_footer()
