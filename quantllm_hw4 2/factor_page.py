"""网页平台的「因子评价」页（由 app.py 调用）。"""

from __future__ import annotations

import hashlib
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from bt import Costs
from bt.factor_eval import BUILTIN, CUSTOM_TEMPLATE, EvalConfig, compile_custom, deep_dive, evaluate, lookahead_check

BLUE, ORANGE, GREEN, RED, GREY, INK = "#2a78d6", "#eb6834", "#2f9e6e", "#c8402f", "#9a9893", "#1a1a19"
ACCENT = {0: BLUE, 1: ORANGE, 2: GREEN}
QUESTIONS = [
    ("有没有预测力", "IC / RankIC：每天算「今天的因子排名」和「之后 h 天收益排名」的相关系数，再取平均。大于 0 说明分数高的股票之后涨得多。"),
    ("稳不稳", "ICIR、t 值、IC 为正的比例、分年 IC：IC 的平均除以波动；只在某一年有效的因子不可靠。"),
    ("是不是单调", "分组收益：按因子分组，看各组之后的收益是否从低到高排好。IC 为正但不单调，说明效应只在某一头。"),
    ("能不能赚钱", "多空与做多回测：A 股基本不能做空，真正要看「只买最高组」在回测系统里扣费后能否跑赢股票池等权。"),
    ("贵不贵", "换手与衰减：排名变得越快，调仓越频繁，费用越高；IC 随持有期怎么变，决定多久调一次仓。"),
    ("是不是重复", "因子相关性：两个因子高度相关，等于同一个信号算了两遍。"),
]
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
              hovermode="x unified", legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), font=dict(size=12))
GRID = "rgba(128,128,128,0.15)"


def _short(res, k):
    return res.summary.loc[k, "name"] if k in res.summary.index else k


def _color(keys, k, focus):
    if k == "POOL":
        return INK
    if k == focus:
        return BLUE
    others = [x for x in keys if x != focus]
    i = others.index(k) if k in others else 0
    palette = [ORANGE, GREEN, "#8a6fd1", "#c79a2b", "#3fa7b5", "#d1689b", GREY]
    return palette[i % len(palette)]


def render(data, lo, hi):
    with st.sidebar:
        st.markdown('<p class="step">② 选择因子</p>', unsafe_allow_html=True)
        mode = st.radio("评价方式", ["单因子深度评价", "多因子对照"], horizontal=True, key="fe_mode",
                        help="单因子：只看一个因子，额外给出分样本、多空拆解、成本盈亏平衡、参数敏感性。多因子：并排比较。")
        if mode == "单因子深度评价":
            src = st.radio("因子来源", ["内建", "自定义"], horizontal=True, key="fe_src")
            if src == "内建":
                one = st.selectbox("因子", list(BUILTIN), index=0, format_func=lambda k: f"{k} · {BUILTIN[k].name}")
                chosen, use_custom = [one], False
            else:
                chosen, use_custom = [], True
        else:
            chosen = st.multiselect("内建因子", list(BUILTIN), default=list(BUILTIN),
                                    format_func=lambda k: f"{k} · {BUILTIN[k].name}")
            use_custom = st.toggle("加入自定义因子", value=False, help="在右侧编辑器里写 factor(d) 函数。")
        custom_name = st.text_input("自定义因子名称", value="POS20") if use_custom else None

        st.markdown('<p class="step">③ 样本与口径</p>', unsafe_allow_html=True)
        start = st.date_input("开始日期", value=lo, min_value=lo, max_value=hi, key="fe_start")
        end = st.date_input("结束日期", value=hi, min_value=lo, max_value=hi, key="fe_end")
        exclude_st = st.toggle("剔除 ST 股票", value=True, key="fe_st")
        horizon = st.select_slider("持有期（交易日）", options=[1, 2, 3, 5, 10, 20], value=5,
                                   help="IC、分组收益都用这个持有期的远期收益：次日开盘买入，持有 h 日后开盘卖出。")
        groups = st.select_slider("分组数", options=[5, 10, 20], value=10)
        rebalance = st.select_slider("分组与回测的调仓间隔（交易日）", options=[1, 5, 10, 20], value=5)

        st.markdown('<p class="step">④ 做多回测</p>', unsafe_allow_html=True)
        do_bt = st.toggle("用回测系统检验「只买最高组」", value=True, help="每个因子多跑一次完整回测，全样本约 8 秒 / 个。")
        top_pct = st.select_slider("买入比例", options=[5, 10, 20, 30], value=10, format_func=lambda v: f"最高 {v}%")
        capital = st.number_input("初始资金（元）", value=1e8, step=1e7, format="%.0f", key="fe_cap")
        with st.expander("费用"):
            comm = st.number_input("佣金（万分之，双边）", value=1.0, step=0.5, key="fe_comm") / 1e4
            stamp = st.number_input("印花税（万分之，卖出）", value=5.0, step=0.5, key="fe_stamp") / 1e4
            slip = st.number_input("滑点（bp，单边）", value=5.0, step=1.0, key="fe_slip") / 1e4
            part = st.number_input("单只成交额上限（前一日成交额的 %，0 为不限）", value=10.0, step=1.0, key="fe_part")

    st.title("因子评价")
    st.markdown('<p class="flow">第 t 日收盘 <b>算因子</b> → 第 t+1 日开盘 <b>买得到才计入</b> → 持有 h 日 <b>算收益</b>'
                '　·　同一个股票池、同一套成交规则，与策略回测共用引擎</p>', unsafe_allow_html=True)

    with st.expander("因子评价在回答什么（第一次用请先看）", expanded=False):
        cols = st.columns(3)
        for i, (q, a) in enumerate(QUESTIONS):
            with cols[i % 3]:
                st.markdown(f"**{q}**  \n{a}")
        st.caption("所有因子都约定为「值越大，预期收益越高」。表中的 t 值为 Newey-West 调整后的 t 值；单调性为各组平均超额与组号的秩相关（1 为完全单调）。")

    code = None
    if use_custom:
        st.subheader("自定义因子")
        if "fe_code" not in st.session_state:
            st.session_state.fe_code = CUSTOM_TEMPLATE
        st.text_area("factor(d)", key="fe_code", height=300, label_visibility="collapsed")
        code = st.session_state.fe_code

    run = st.button("▶ 运行因子评价", type="primary", disabled=data is None or (not chosen and not use_custom))
    if data is None:
        st.caption("先在左侧载入数据。")
        return

    if run:
        factors = {k: BUILTIN[k] for k in chosen}
        if use_custom:
            try:
                fn = compile_custom(code)
                with st.spinner("检查自定义因子是否用到未来数据…"):
                    ok, diff = lookahead_check(data, fn)
                if not ok:
                    st.error(f"自定义因子用到了未来数据：把数据截断后重算，同一天的因子值变了（最大差 {diff:.4g}）。"
                             "第 t 行只能用第 t 日及以前的数据。")
                    return
                name = (custom_name or "CUSTOM").strip() or "CUSTOM"
                factors[name] = fn(data)
            except Exception:
                st.error("自定义因子代码有错误")
                st.code(traceback.format_exc(limit=3), language="text")
                return
        cfg = EvalConfig(start=str(start), end=str(end), horizon=horizon, groups=groups, rebalance=rebalance,
                         exclude_st=exclude_st, backtest=do_bt, top_frac=top_pct / 100, capital=float(capital),
                         costs=Costs(commission=comm, min_commission=0.0, transfer=0.0, stamp_old=stamp,
                                     stamp_new=stamp, slippage=slip),
                         max_participation=part / 100 if part > 0 else None)
        key = hashlib.md5(repr((id(data), sorted(factors), repr(cfg), code if use_custom else None)).encode()).hexdigest()
        cache = st.session_state.setdefault("fe_cache", {})
        if key not in cache:
            bar = st.progress(0.0, text="准备中…")
            try:
                prog = lambda p, m: bar.progress(min(p, 1.0) * (0.5 if len(factors) == 1 else 1.0), text=m)
                ev = evaluate(data, factors, cfg, progress=prog)
                dd = None
                if len(factors) == 1:                       # 只评价一个因子时，加做单因子深度评价
                    (k1, f1), = factors.items()
                    dd = deep_dive(data, k1, f1, cfg, progress=lambda p, m: bar.progress(0.5 + 0.5 * min(p, 1.0), text="深度评价：" + m))
                cache[key] = (ev, dd)
            except Exception as e:
                bar.empty()
                st.error(f"评价失败：{e}")
                st.code(traceback.format_exc(limit=4), language="text")
                return
            bar.empty()
        st.session_state.fe_result, st.session_state.fe_deep = cache[key]

    res = st.session_state.get("fe_result")
    if res is not None:
        show(res, st.session_state.get("fe_deep"))


def _fmt(v, kind):
    if v is None or not np.isfinite(v):
        return "—"
    return {"pct": f"{v:+.1%}", "ic": f"{v:.3f}", "bp": f"{v:+.1f}bp", "x": f"{v:.2f}"}[kind]


def show(res, dd=None):
    S = res.summary
    keys = list(S.index)
    st.divider()
    info = res.info
    st.caption(f"IC 评价区间 {info['start']} 至 {info['end']}（{info['days']:,} 个交易日）· 股票池平均 {info['pool_size']:,.0f} 只"
               f"{'（剔除 ST）' if info['exclude_st'] else ''} · 持有期 {info['horizon']} 日 · {info['groups']} 组 · 每 {info['rebalance']} 日调仓"
               + (f" · 做多回测至 {info['bt_end']}" if res.nav is not None else ""))

    focus = st.selectbox("重点看哪个因子", keys, index=0, format_func=lambda k: f"{k} · {_short(res, k)}")
    r = S.loc[focus]
    sig = abs(r.t_nw) >= 2
    top_word = "跑赢" if r.top_ex_bp > 0 else "跑输"
    text = (f"**{r['name']}** 的 RankIC 为 **{_fmt(r.rankic_mean, 'ic')}**（t = {r.t_nw:.1f}），"
            f"{'统计上显著' if sig else '统计上不显著'}；分组单调性 **{_fmt(r.monotonic, 'x')}**，"
            f"最高组每 {info['horizon']} 日相对股票池 {top_word} **{abs(r.top_ex_bp):.1f}bp**，最低组 {_fmt(r.bottom_ex_bp, 'bp')}。")
    if "bt_excess_ann" in S.columns and np.isfinite(r.get("bt_excess_ann", np.nan)):
        text += f" 按真实规则只买最高组，扣费后年化超额 **{_fmt(r.bt_excess_ann, 'pct')}**。"
    text += f" 结论：**{r.verdict}**。"
    st.markdown(text)

    # ---------------- 总表
    st.subheader("评价总表")
    cols = ["name", "verdict", "rankic_mean", "icir_ann", "t_nw", "ic_pos_ratio", "monotonic", "top_ex_bp", "bottom_ex_bp",
            "ls_ann", "top_turnover"] + [c for c in ["bt_excess_ann", "bt_ir", "bt_ann", "bt_mdd", "bt_turnover"] if c in S.columns]
    tbl = S[cols].copy()
    for c in ["ic_pos_ratio", "top_turnover", "ls_ann", "bt_excess_ann", "bt_ann", "bt_mdd"]:
        if c in tbl:
            tbl[c] = tbl[c] * 100
    NC = st.column_config.NumberColumn
    st.dataframe(tbl, width="stretch", column_config={
        "_index": st.column_config.TextColumn("代码"), "name": "因子", "verdict": "结论",
        "rankic_mean": NC("RankIC", format="%.3f"), "icir_ann": NC("ICIR（年化）", format="%.2f"),
        "t_nw": NC("t 值", format="%.1f"), "ic_pos_ratio": NC("IC>0 比例", format="%.0f%%"),
        "monotonic": NC("单调性", format="%.2f"), "top_ex_bp": NC("最高组超额 bp", format="%+.1f"),
        "bottom_ex_bp": NC("最低组超额 bp", format="%+.1f"), "ls_ann": NC("多空年化（不可做空）", format="%+.1f%%"),
        "top_turnover": NC("最高组每期换出", format="%.0f%%"), "bt_excess_ann": NC("做多年化超额", format="%+.1f%%"),
        "bt_ir": NC("做多信息比率", format="%.2f"), "bt_ann": NC("做多年化", format="%+.1f%%"),
        "bt_mdd": NC("做多最大回撤", format="%.1f%%"), "bt_turnover": NC("做多年换手（倍）", format="%.1f"),
    })
    with st.expander("因子定义"):
        for k in keys:
            st.markdown(f"- **{k} · {S.loc[k, 'name']}**：{S.loc[k, 'definition']}")

    # ---------------- 重点因子
    st.subheader(f"深入看：{_short(res, focus)}")
    c1, c2 = st.columns(2)
    with c1:
        g = res.deciles[focus]
        fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values,
                               marker_color=[RED if v < 0 else BLUE for v in g.values],
                               hovertemplate="第 %{x} 组<br>%{y:+.1f}bp<extra></extra>"))
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=400,
                          title=dict(text=f"分组超额（bp / {res.info['horizon']} 日，组号越大因子值越高）", font=dict(size=14)))
        fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="#bdbcb7")
        st.plotly_chart(fig, width="stretch")
    with c2:
        if res.nav is not None:
            fig = go.Figure()
            for k in ["POOL"] + [x for x in keys if x != focus] + [focus]:
                if k not in res.nav:
                    continue
                fig.add_scatter(x=res.nav.index, y=res.nav[k], name="股票池等权" if k == "POOL" else _short(res, k),
                                line=dict(color=_color(keys, k, focus), width=2.4 if k in (focus, "POOL") else 1.2),
                                opacity=1 if k in (focus, "POOL") else 0.45)
            fig.add_hline(y=1, line=dict(color="#bdbcb7", width=1))
            fig.update_layout(**LAYOUT, height=400, title=dict(text="做多最高组的净值（扣费）", font=dict(size=14)))
            fig.update_yaxes(gridcolor=GRID)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("未开启做多回测。")
    c3, c4 = st.columns(2)
    with c3:
        y = res.yearly[focus]
        fig = go.Figure(go.Bar(x=[str(i) for i in y.index], y=y.values, marker_color=[RED if v < 0 else BLUE for v in y.values],
                               hovertemplate="%{x} 年<br>RankIC %{y:.3f}<extra></extra>"))
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=400, title=dict(text="分年 RankIC", font=dict(size=14)))
        fig.update_yaxes(gridcolor=GRID, zerolinecolor="#bdbcb7")
        st.plotly_chart(fig, width="stretch")
    with c4:
        cum = res.ic.fillna(0).cumsum()
        fig = go.Figure()
        for k in [x for x in keys if x != focus] + [focus]:
            fig.add_scatter(x=cum.index, y=cum[k], name=_short(res, k), line=dict(color=_color(keys, k, focus), width=2.4 if k == focus else 1.2),
                            opacity=1 if k == focus else 0.5)
        fig.add_hline(y=0, line=dict(color="#bdbcb7", width=1))
        fig.update_layout(**LAYOUT, height=400, title=dict(text="累计 RankIC（斜率稳定向上 = 一直有效）", font=dict(size=14)))
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")

    if dd is not None and dd.key in keys:
        show_deep(res, dd)
        _downloads(res)
        return

    # ---------------- 全部因子
    st.subheader("全部因子对照")
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=[f"{k} · {_short(res, k)}" for k in keys],
                        vertical_spacing=0.18 if nrow > 1 else 0.1)
    for i, k in enumerate(keys):
        g = res.deciles[k]
        fig.add_bar(x=[str(j) for j in g.index], y=g.values, row=i // ncol + 1, col=i % ncol + 1, showlegend=False,
                    marker_color=[RED if v < 0 else (BLUE if k == focus else GREY) for v in g.values],
                    hovertemplate=f"{k} · 第 %{{x}} 组<br>%{{y:+.1f}}bp<extra></extra>")
    fig.update_layout(**{**LAYOUT, "hovermode": "closest", "margin": dict(l=10, r=10, t=80, b=10)}, height=260 * nrow + 40,
                      title=dict(text=f"分组超额（同一纵轴，bp / {res.info['horizon']} 日）", font=dict(size=14)))
    fig.update_yaxes(gridcolor=GRID)
    fig.update_annotations(font_size=12)
    st.plotly_chart(fig, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        fig = go.Figure()
        for k in [x for x in keys if x != focus] + [focus]:
            fig.add_scatter(x=[f"{h} 日" for h in res.decay.index], y=res.decay[k], name=_short(res, k), mode="lines+markers",
                            line=dict(color=_color(keys, k, focus), width=2.4 if k == focus else 1.4))
        fig.add_hline(y=0, line=dict(color="#bdbcb7", width=1))
        fig.update_layout(**LAYOUT, height=420, title=dict(text="IC 衰减：不同持有期的平均 RankIC", font=dict(size=14)))
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")
    with c6:
        lab = [_short(res, k) for k in keys]
        z = res.corr.loc[keys, keys].to_numpy(dtype=float)
        fig = go.Figure(go.Heatmap(z=z, x=lab, y=lab, zmin=-1, zmax=1, colorscale=[[0, ORANGE], [0.5, "#f4f3ef"], [1, BLUE]],
                                   text=np.round(z, 2), texttemplate="%{text:.2f}", hovertemplate="%{y} × %{x}<br>%{z:.2f}<extra></extra>",
                                   showscale=False))
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=420, title=dict(text="因子相关性（平均横截面秩相关）", font=dict(size=14)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    _downloads(res)


def _downloads(res):
    out = res.summary.copy()
    d1, d2, d3 = st.columns(3)
    d1.download_button("下载评价总表 CSV", out.to_csv().encode("utf-8-sig"), "factor_summary.csv")
    d2.download_button("下载每日 RankIC CSV", res.ic.to_csv().encode("utf-8-sig"), "rankic_daily.csv")
    d3.download_button("下载分组收益 CSV", res.deciles.to_csv().encode("utf-8-sig"), "factor_groups.csv")


def show_deep(res, dd):
    info = res.info
    st.subheader(f"单因子深度评价：{dd.name}")
    st.caption("只选一个因子时才会出现。回答三个追问：效应在哪里成立、钱从哪一头来、成本扣完还剩多少。")

    # 分样本
    c1, c2 = st.columns([1.1, 1])
    with c1:
        sub = dd.subsets.copy()
        sub["label"] = sub["group"] + " · " + sub["subset"]
        sub = sub.iloc[::-1]
        colors = [GREY if g == "全样本" else (BLUE if abs(t) >= 2 else "#c9c7c1") for g, t in zip(sub["group"], sub["t"])]
        fig = go.Figure(go.Bar(y=sub["label"], x=sub["rankic"], orientation="h", marker_color=colors,
                               text=[f"t={t:.1f}" for t in sub["t"]], textposition="outside", cliponaxis=False,
                               customdata=sub["days"], hovertemplate="%{y}<br>RankIC %{x:.3f}<br>%{customdata} 天<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#bdbcb7", width=1))
        fig.update_layout(**{**LAYOUT, "hovermode": "closest", "margin": dict(l=10, r=50, t=40, b=10)}, height=440,
                          title=dict(text="分样本 RankIC（蓝色：|t| ≥ 2，显著）", font=dict(size=14)))
        fig.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=dd.legs.index, y=dd.legs["最低组"], name="最低组（做空端）", line=dict(color=RED, width=2))
        fig.add_scatter(x=dd.legs.index, y=dd.legs["最高组"], name="最高组（做多端）", line=dict(color=BLUE, width=2.4))
        fig.add_hline(y=1, line=dict(color="#bdbcb7", width=1))
        fig.update_layout(**LAYOUT, height=440, title=dict(text="多空拆解：两头各自相对股票池的累计超额（零费用）", font=dict(size=14)))
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")

    # 稳定性与成本
    c3, c4 = st.columns([1.4, 1])
    with c3:
        ric = dd.rolling_ic
        fig = go.Figure(go.Scatter(x=ric.index, y=ric.values, line=dict(color=BLUE, width=2), fill="tozeroy",
                                   fillcolor="rgba(42,120,214,0.10)", name="滚动 120 日 RankIC"))
        fig.add_hline(y=0, line=dict(color="#bdbcb7", width=1))
        fig.update_layout(**{**LAYOUT, "showlegend": False}, height=320, title=dict(text="滚动 120 日平均 RankIC（跌破 0 = 那段时间失效）", font=dict(size=14)))
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")
    with c4:
        be = dd.breakeven
        st.markdown("**成本盈亏平衡（只买最高组）**")
        m1, m2 = st.columns(2)
        m1.metric(f"最高组每 {info['horizon']} 日超额（零费用）", f"{be['top_ex_bp']:+.1f}bp")
        m2.metric("每次调仓换掉的持仓", f"{be['turnover']:.0%}")
        m3, m4 = st.columns(2)
        m3.metric("盈亏平衡往返成本", "无" if not np.isfinite(be["breakeven_rt_bp"]) else f"{be['breakeven_rt_bp']:.1f}bp")
        m4.metric("当前设定的往返成本", f"{be['cost_rt_bp']:.1f}bp")
        if not np.isfinite(be["breakeven_rt_bp"]):
            st.error("最高组扣费前就跑输股票池：成本再低也赚不到钱。")
        elif be["breakeven_rt_bp"] > be["cost_rt_bp"]:
            st.success(f"超额足以覆盖成本：往返成本低于 {be['breakeven_rt_bp']:.1f}bp 时仍有超额。")
        else:
            st.warning(f"超额不够付成本：往返成本要降到 {be['breakeven_rt_bp']:.1f}bp 以下才赚钱。")
        st.caption("往返成本 = 买入 + 卖出的佣金、印花税、过户费与滑点之和。每次调仓的成本 = 换掉的比例 × 往返成本。")

    # 参数扫描
    if dd.sweep is not None:
        sw = dd.sweep
        st.markdown("**参数敏感性：回看天数换成多少，结论还成立吗**")
        c5, c6 = st.columns([1, 1.2])
        with c5:
            fig = go.Figure(go.Bar(x=[f"{n} 日" for n in sw.index], y=sw["rankic"],
                                   marker_color=[BLUE if n == 5 else GREY for n in sw.index],
                                   text=[f"t={t:.1f}" for t in sw["t"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="回看 %{x}<br>RankIC %{y:.3f}<extra></extra>"))
            fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=320, title=dict(text="不同回看天数的 RankIC（蓝色为 5DR）", font=dict(size=14)))
            fig.update_yaxes(gridcolor=GRID)
            st.plotly_chart(fig, width="stretch")
        with c6:
            t = sw.copy()
            t.index = [f"{n} 日" for n in t.index]
            t["top_turnover"] = t["top_turnover"] * 100
            NC = st.column_config.NumberColumn
            st.dataframe(t, width="stretch", column_config={
                "_index": st.column_config.TextColumn("回看"), "rankic": NC("RankIC", format="%.3f"), "t": NC("t 值", format="%.1f"),
                "monotonic": NC("单调性", format="%.2f"), "top_ex_bp": NC("最高组超额 bp", format="%+.1f"),
                "bottom_ex_bp": NC("最低组超额 bp", format="%+.1f"), "top_turnover": NC("最高组每期换出", format="%.0f%%")})
            st.caption("这是敏感性检查，不是调参：在同一份数据上挑最好的回看天数会高估效果。")
