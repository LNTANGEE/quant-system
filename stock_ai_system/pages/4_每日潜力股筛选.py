from __future__ import annotations

import pandas as pd
import streamlit as st

from database.db import add_alert, get_stock_pool, get_watchlist, init_db, remove_pool_stock, upsert_pool_stock
from utils.analysis import analyze_many
from utils.helpers import format_pct, format_price, format_zone, parse_code_lines, render_mobile_style, render_risk_footer


init_db()
render_mobile_style()

st.title("每日潜力股筛选")
st.caption("第一版扫描范围：自选股 + 手动导入股票池。默认先过滤高风险股票。")

with st.expander("手动导入股票池", expanded=False):
    text = st.text_area("输入股票代码，支持空格、逗号、换行分隔", placeholder="300308\n000001\n600519")
    source = st.text_input("来源标签", value="manual")
    col_import, col_clear = st.columns(2)
    if col_import.button("导入股票池", type="primary"):
        codes = parse_code_lines(text)
        for code in codes:
            upsert_pool_stock(code, source=source or "manual")
        st.success(f"已导入 {len(codes)} 只股票")
        st.rerun()

pool = get_stock_pool(include_watchlist=True)
if not pool:
    st.info("暂无股票池。请先添加自选股或手动导入股票。")
    render_risk_footer()
    st.stop()

with st.sidebar:
    st.header("筛选设置")
    scan_upper = max(1, min(len(pool), 100))
    if scan_upper == 1:
        max_scan = 1
        st.caption("当前股票池只有1只股票，本次扫描数量固定为1。")
    else:
        max_scan = st.slider("本次最多扫描数量", 1, scan_upper, min(scan_upper, 30))
    include_minute = st.checkbox("包含分钟K评分", value=False, help="更接近实时，但会增加AKShare请求时间。")
    st.write(f"股票池数量：{len(pool)}")
    manual_codes = [item["code"] for item in pool if item.get("source") != "watchlist"]
    remove_code = st.selectbox("删除手动池股票", [""] + manual_codes)
    if remove_code and st.button("删除该股票"):
        remove_pool_stock(remove_code)
        st.success("已删除")
        st.rerun()

with st.spinner("正在计算潜力评分和风险过滤..."):
    results = analyze_many(pool, include_minute=include_minute, limit=max_scan)

records = []
errors = []
for item in results:
    if item.get("error"):
        errors.append({"代码": item["code"], "错误": item["error"]})
        continue
    records.append(item["potential"])

if errors:
    st.warning("部分股票数据获取失败，已跳过。")
    st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

if not records:
    st.info("本次没有可展示的股票。")
    render_risk_footer()
    st.stop()

df = pd.DataFrame(records)
df["high_risk"] = df["risk_level"].isin(["高", "极高"]) | (df["risk_score"].fillna(0) >= 55)
safe_df = df[~df["high_risk"]].copy()
high_risk_df = df[df["high_risk"]].copy()


def short_table(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame()
    ordered = source.sort_values("short_score", ascending=False).copy()
    return pd.DataFrame(
        {
            "股票代码": ordered["code"],
            "股票名称": ordered["name"],
            "所属板块": ordered["sector"],
            "当前价": ordered["price"].map(format_price),
            "今日涨跌幅": ordered["pct_chg"].map(format_pct),
            "短线评分": ordered["short_score"],
            "今日低吸区": ordered["low_buy_zone"].map(format_zone),
            "今日高抛区": ordered["high_sell_zone"].map(format_zone),
            "风险等级": ordered["risk_level"],
            "操作建议": ordered["operation_suggestion"],
        }
    )


def mid_table(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame()
    ordered = source.sort_values("mid_score", ascending=False).copy()
    rows = []
    for _, row in ordered.iterrows():
        fundamental = row.get("fundamental", {}) or {}
        rows.append(
            {
                "股票代码": row["code"],
                "股票名称": row["name"],
                "所属行业": row["industry"],
                "中期评分": row["mid_score"],
                "业绩增长": f"营收{format_pct(fundamental.get('revenue_yoy'))} / 净利{format_pct(fundamental.get('net_profit_yoy'))}",
                "估值分位": "预留接口",
                "技术趋势": row["mid_label"],
                "建议观察价": format_price(row.get("suggest_observe_price")),
                "建议低吸区": format_zone(row.get("suggest_low_zone")),
            }
        )
    return pd.DataFrame(rows)


def long_table(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame()
    ordered = source.sort_values("long_score", ascending=False).copy()
    rows = []
    for _, row in ordered.iterrows():
        factors = row.get("long_factors", {}) or {}
        rows.append(
            {
                "股票代码": row["code"],
                "股票名称": row["name"],
                "长期评分": row["long_score"],
                "行业地位": factors.get("公司行业地位代理"),
                "成长性": factors.get("成长性"),
                "盈利质量": factors.get("盈利质量"),
                "估值安全边际": factors.get("估值安全边际"),
                "长期风险": row["risk_level"],
                "是否适合长期跟踪": row["long_label"],
            }
        )
    return pd.DataFrame(rows)


def alert_messages(source: pd.DataFrame) -> list[dict]:
    alerts = []
    for _, row in source.iterrows():
        messages = []
        if row["short_score"] >= 75:
            messages.append("短线评分超过75")
        if row["mid_score"] >= 75:
            messages.append("中期评分超过75")
        if row["long_score"] >= 80:
            messages.append("长期评分超过80")
        price = row.get("price")
        zone = row.get("low_buy_zone") or {}
        if zone.get("low", 0) <= price <= zone.get("high", 0):
            messages.append("股票进入低吸区")
        factors = row.get("short_factors", {}) or {}
        if row.get("sector") and factors.get("板块强度", 0) >= 16:
            messages.append("板块强度靠前")
        if factors.get("个股相对强度", 0) >= 12:
            messages.append("个股强于板块")
        if row["risk_score"] >= 55:
            messages.append("风险评分突然升高或处于高位")
        if "高位" in "；".join(row.get("risk_flags", [])):
            messages.append("出现高位放量滞涨风险")
        if messages:
            alerts.append(
                {
                    "股票": f"{row['code']} {row['name']}",
                    "短线评分": row["short_score"],
                    "中期评分": row["mid_score"],
                    "长期评分": row["long_score"],
                    "当前价": format_price(row["price"]),
                    "今日低吸区": format_zone(row["low_buy_zone"]),
                    "提醒原因": "；".join(messages),
                    "系统判断": row["operation_suggestion"],
                    "code": row["code"],
                    "trigger_price": row["price"],
                }
            )
    return alerts


tabs = st.tabs(["短线潜力榜", "中期潜力榜", "长期潜力榜", "低吸观察", "高风险剔除", "潜力股提醒"])

with tabs[0]:
    st.dataframe(short_table(safe_df.head(len(safe_df))), use_container_width=True, hide_index=True)

with tabs[1]:
    st.dataframe(mid_table(safe_df), use_container_width=True, hide_index=True)

with tabs[2]:
    st.dataframe(long_table(safe_df), use_container_width=True, hide_index=True)

with tabs[3]:
    observe = safe_df[
        (safe_df["stock_type"].isin(["低吸观察型", "中期趋势型", "长期成长型"]))
        | ((safe_df["short_score"] >= 55) & (safe_df["short_score"] < 75))
    ].sort_values("comprehensive_score", ascending=False)
    st.dataframe(short_table(observe), use_container_width=True, hide_index=True)

with tabs[4]:
    if high_risk_df.empty:
        st.success("本次扫描未发现需要剔除的高风险股票。")
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "股票代码": high_risk_df["code"],
                    "股票名称": high_risk_df["name"],
                    "风险等级": high_risk_df["risk_level"],
                    "风险评分": high_risk_df["risk_score"],
                    "风险原因": high_risk_df["risk_flags"].map(lambda x: "；".join(x or [])),
                    "处理": "剔除短线榜，仅观察",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tabs[5]:
    alerts = alert_messages(df)
    if not alerts:
        st.info("当前未触发潜力股提醒条件。")
    else:
        alert_df = pd.DataFrame(alerts).drop(columns=["code", "trigger_price"])
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        if st.button("保存本次提醒到本地数据库"):
            for alert in alerts:
                add_alert(
                    alert["code"],
                    "potential",
                    f"{alert['提醒原因']}。{alert['系统判断']}",
                    alert["trigger_price"],
                )
            st.success("提醒已保存")

render_risk_footer()
