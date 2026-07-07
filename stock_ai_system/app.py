from __future__ import annotations

import pandas as pd
import streamlit as st

from config import APP_NAME, PAGE_ICON, TUSHARE_TOKEN
from data.sector_data import get_market_status, get_strong_weak_sectors
from data.stock_data import get_realtime_quotes
from database.db import add_watch_stock, get_today_operation_count, get_watchlist, init_db
from models.private_steward import build_steward_advice
from utils.analysis import analyze_many
from utils.helpers import format_amount, format_pct, format_price, format_zone, render_mobile_style, render_risk_footer


st.set_page_config(page_title=APP_NAME, page_icon=PAGE_ICON, layout="wide")
render_mobile_style()


def _sector_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["板块", "涨跌幅", "成交额", "上涨家数", "下跌家数", "领涨股"])
    return pd.DataFrame(
        {
            "板块": df.get("sector", ""),
            "涨跌幅": df.get("pct_chg", pd.Series(dtype=float)).map(format_pct),
            "成交额": df.get("amount", df.get("total_mv", pd.Series(dtype=float))).map(format_amount),
            "上涨家数": df.get("up_count", ""),
            "下跌家数": df.get("down_count", ""),
            "领涨股": df.get("leader", ""),
        }
    )


def _overview_rows(results: list[dict], watchlist: list[dict]) -> pd.DataFrame:
    watch_by_code = {item.get("code", ""): item for item in watchlist}
    rows = []
    for item in results:
        watch = watch_by_code.get(item.get("code", ""), {})
        if item.get("error"):
            rows.append(
                {
                    "代码": item["code"],
                    "名称": item.get("quote", {}).get("name", ""),
                    "当前价": "--",
                    "成本价": format_price(watch.get("cost_price"), 3),
                    "相对成本": "--",
                    "涨跌幅": "--",
                    "主力评分": "--",
                    "短线": "--",
                    "中期": "--",
                    "长期": "--",
                    "今日预估最低价": "--",
                    "今日预估最高价": "--",
                    "最高触达概率": "--",
                    "今日低吸区": "--",
                    "今日高抛区": "--",
                    "管家信号": "数据不足",
                    "可买": "--",
                    "可卖": "--",
                    "何时进": "--",
                    "何时出": "--",
                    "当前建议操作": "等待行情恢复",
                    "做T建议": item["error"],
                    "风险": "数据不足",
                    "潜力模型建议": "等待数据恢复",
                }
            )
            continue
        quote = item["quote"]
        potential = item["potential"]
        low_model = item["low_model"]
        high_model = item.get("high_model", {})
        t_strategy = item["t_strategy"]
        risk = item["risk"]
        main_force = item["main_force"]
        steward = build_steward_advice(item, watch, get_today_operation_count(item["code"]))
        rows.append(
            {
                "代码": item["code"],
                "名称": quote.get("name", ""),
                "当前价": format_price(quote.get("price")),
                "成本价": format_price(steward.get("cost_price"), 3),
                "相对成本": f"{steward.get('position_profit_pct', 0):.2f}%",
                "涨跌幅": format_pct(quote.get("pct_chg")),
                "主力评分": f"{main_force.get('score', 0):.0f} / {main_force.get('label', '')}",
                "短线": f"{potential.get('short_score', 0):.0f}",
                "中期": f"{potential.get('mid_score', 0):.0f}",
                "长期": f"{potential.get('long_score', 0):.0f}",
                "综合": f"{potential.get('comprehensive_score', 0):.0f}",
                "今日预估最低价": format_zone(low_model.get("estimated_low_zone")),
                "今日预估最高价": format_zone(high_model.get("estimated_high_zone")),
                "最高触达概率": format_pct(high_model.get("estimated_high_reach_probability")),
                "今日低吸区": format_zone(low_model.get("first_low_zone")),
                "今日高抛区": format_zone(t_strategy.get("high_sell_zone")),
                "管家信号": f"{steward.get('signal')} / {steward.get('signal_level')}",
                "可买": f"{steward.get('buy_shares', 0)}股 / {format_amount(steward.get('buy_amount'))}",
                "可卖": f"{steward.get('sell_shares', 0)}股 / {format_amount(steward.get('sell_amount'))}",
                "何时进": steward.get("entry_timing", ""),
                "何时出": steward.get("exit_timing", ""),
                "当前建议操作": steward.get("current_advice", ""),
                "做T建议": t_strategy.get("current_action", ""),
                "风险": f"{risk.get('risk_level', '')}({risk.get('risk_score', 0):.0f})",
                "潜力模型建议": potential.get("operation_suggestion", ""),
            }
        )
    return pd.DataFrame(rows)


init_db()

st.title(APP_NAME)
st.caption("第一版MVP：公开行情 + 技术指标 + 概率区间 + 风险分级 + 本地自选股")

with st.sidebar:
    st.header("系统状态")
    st.write("数据源：AKShare")
    st.write("Tushare Pro Token：" + ("已配置" if TUSHARE_TOKEN else "未配置，已预留接口"))
    auto_refresh = st.checkbox("打开首页时自动刷新行情", value=False)
    if st.button("刷新首页行情"):
        st.session_state["home_refresh_market"] = True
    st.divider()
    with st.form("quick_add"):
        st.subheader("快速添加自选股")
        code = st.text_input("股票代码", placeholder="例如 300308")
        name = st.text_input("名称，可选")
        submitted = st.form_submit_button("添加")
        if submitted:
            if add_watch_stock(code, name):
                st.success("已加入自选股")
                st.rerun()
            else:
                st.warning("请输入有效的6位股票代码")

watchlist = get_watchlist()
load_market = auto_refresh or st.session_state.get("home_refresh_market", False)
if load_market:
    with st.spinner("正在刷新公开行情数据..."):
        spot_df = get_realtime_quotes()
        market = get_market_status(spot_df)
        strong, weak = get_strong_weak_sectors()
else:
    spot_df = pd.DataFrame()
    market = {
        "status": "待点击刷新",
        "up_count": 0,
        "down_count": 0,
        "median_pct": 0,
        "amount": 0,
    }
    strong, weak = pd.DataFrame(), pd.DataFrame()
    st.info("首页已使用安全启动模式打开，避免公开行情接口卡住页面。点击左侧“刷新首页行情”后，会加载AKShare/备用行情、强弱板块和自选股评分。")

metric_cols = st.columns(4)
is_sample_market = spot_df.attrs.get("source") == "eastmoney_spot_fallback" if not spot_df.empty else False
metric_cols[0].metric("今日大盘状态", market.get("status", "数据不足"))
metric_cols[1].metric("上涨/下跌家数", f"{market.get('up_count', 0)} / {market.get('down_count', 0)}")
metric_cols[2].metric("市场中位涨跌幅", format_pct(market.get("median_pct", 0)))
metric_cols[3].metric("行情样本成交额" if is_sample_market else "全市场成交额", format_amount(market.get("amount", 0)))
if is_sample_market:
    st.caption("当前AKShare全市场接口不可用，首页市场状态按东方财富直连备用源可获取样本估算；板块强弱使用东方财富板块直连备用源。")

left, right = st.columns(2)
with left:
    st.subheader("今日强势板块")
    st.dataframe(_sector_table(strong), use_container_width=True, hide_index=True)
with right:
    st.subheader("今日弱势板块")
    st.dataframe(_sector_table(weak), use_container_width=True, hide_index=True)

st.subheader("自选股智能盯盘总览")
if not watchlist:
    st.info("还没有自选股。请在侧边栏或“自选股管理页”添加股票代码。")
else:
    if load_market:
        with st.spinner("正在计算自选股评分、低吸区和做T区间..."):
            results = analyze_many(watchlist, include_minute=False)
        overview = _overview_rows(results, watchlist)
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.caption("距离低吸区、跌破概率、接回提醒等更细内容可在对应页面查看。")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "代码": item.get("code", ""),
                        "名称": item.get("name", ""),
                        "持仓数量": item.get("shares", 0),
                        "成本价": item.get("cost_price", 0),
                        "可卖数量": item.get("sellable_quantity", 0),
                        "状态": "待刷新行情",
                    }
                    for item in watchlist
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

render_risk_footer()
