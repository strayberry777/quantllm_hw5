"""A 股策略回测台（Streamlit）。

    streamlit run app.py

左侧选数据、区间、资金、股票池和交易规则；中间写策略代码；点「运行回测」看结果。
撮合、费用、涨跌停等全部由 bt/ 里经过测试的引擎完成，页面只负责输入和展示。
"""

from __future__ import annotations

import hashlib
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bt import Costs, MarketData, summarize
from bt.api import RunConfig, compile_strategy, run_strategy
from bt import strategies as S
from bt.engine import Backtester, frictionless_nav
from bt.metrics import by_year
from bt.templates import TEMPLATES

ROOT = Path(__file__).resolve().parent
FIELDS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")
BLUE, GREY, ORANGE, RED = "#2a78d6", "#8a8984", "#eb6834", "#c8402f"

PEERS = ["603986.SH", "300223.SZ", "301308.SZ", "688525.SH", "688008.SH", "688766.SH", "688110.SH", "688123.SH",
         "688416.SH", "001309.SZ", "000021.SZ", "600667.SH", "300042.SZ", "300475.SZ", "688981.SH", "688347.SH",
         "002371.SZ", "688012.SH", "600584.SH", "002156.SZ", "002185.SZ", "603501.SH", "688041.SH", "688256.SH",
         "688072.SH", "688126.SH", "688249.SH", "688082.SH", "688120.SH", "300782.SZ", "300661.SZ", "002049.SZ",
         "688019.SH", "688361.SH"]
POOLS = {
    "全部股票": None,
    "长鑫科技 688825.SH": ["688825.SH"],
    "存储 / 半导体同类股（34 只）": PEERS,
    "长鑫 + 同类股（35 只）": ["688825.SH"] + PEERS,
    "自定义": "custom",
}

st.set_page_config(page_title="A 股回测与因子评价", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
h1 {font-size: 2.0rem !important; letter-spacing: -0.02em;}
.flow {color: #6b6a66; font-size: 0.95rem; margin-top: -0.6rem; margin-bottom: 1.2rem;}
.flow b {color: #2a78d6; font-weight: 600;}
.step {font-size: 0.78rem; letter-spacing: 0.08em; color: #8a8984; text-transform: uppercase; margin-bottom: -0.4rem;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
textarea {font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace !important; font-size: 13px !important;}
</style>
""", unsafe_allow_html=True)


# ====================================================================== 数据
@st.cache_resource(show_spinner="读取行情数据…")
def load_dir(path: str) -> MarketData:
    return MarketData.load(path)


@st.cache_resource(show_spinner="读取上传的数据…")
def load_uploaded(key: str, _files: tuple) -> MarketData:
    tmp = Path(tempfile.mkdtemp(prefix="bt_upload_"))
    for f in _files:
        (tmp / f.name).write_bytes(f.getvalue())
    return MarketData.load(tmp)


def check_dir(path: str) -> list[str]:
    p = Path(path).expanduser()
    return [f for f in FIELDS if not (p / f"{f}.feather").exists()]


with st.sidebar:
    st.header("回测设置")

    st.markdown('<p class="step">① 数据</p>', unsafe_allow_html=True)
    src = st.radio("数据来源", ["本机文件夹", "上传文件"], horizontal=True, label_visibility="collapsed")
    data = None
    if src == "本机文件夹":
        default_dir = str(ROOT / "raw") if (ROOT / "raw" / "close.feather").exists() else ""
        path = st.text_input("Feather 数据文件夹", value=default_dir, placeholder="例如 /Users/你/Downloads/feathers")
        if path:
            missing = check_dir(path)
            if missing:
                st.error(f"文件夹里缺少：{', '.join(m + '.feather' for m in missing)}")
            else:
                data = load_dir(str(Path(path).expanduser()))
    else:
        files = st.file_uploader("选择 7 个 .feather 文件", type="feather", accept_multiple_files=True)
        names = {f.name for f in files or []}
        missing = [f for f in FIELDS if f"{f}.feather" not in names]
        if files and missing:
            st.error(f"还缺：{', '.join(m + '.feather' for m in missing)}")
        elif files:
            key = hashlib.md5("".join(sorted(f"{f.name}{f.size}" for f in files)).encode()).hexdigest()
            data = load_uploaded(key, tuple(f for f in files if f.name[:-8] in FIELDS))

    if data is not None:
        st.caption(f"{data.dates[0].date()} 至 {data.dates[-1].date()} · {len(data.dates):,} 个交易日 · {len(data.symbols):,} 只股票")
        lo, hi = data.dates[0].date(), data.dates[-1].date()
    else:
        lo, hi = pd.Timestamp("2020-01-02").date(), pd.Timestamp("2026-10-16").date()

    st.markdown('<p class="step">功能</p>', unsafe_allow_html=True)
    page = st.radio("功能", ["策略回测", "因子评价"], horizontal=True, label_visibility="collapsed", key="page")

if page == "因子评价":
    from factor_page import render as render_factor_page
    render_factor_page(data, lo, hi)
    st.stop()

with st.sidebar:
    st.markdown('<p class="step">② 回测口径</p>', unsafe_allow_html=True)
    mode = st.radio("口径", ["课程作业规格", "自定义"], horizontal=True, label_visibility="collapsed",
                    help="课程作业规格：非 ST 全市场、10 亿元、佣金万一 + 印花税万五、2025 年首个交易日建仓、2026 年 9 月最后一个交易日清仓。")
    SPEC = mode == "课程作业规格"
    if SPEC:
        pool_name, symbols = "全部股票", None
        start, end = pd.Timestamp("2025-01-02").date(), pd.Timestamp("2026-09-30").date()
        capital = 1e9
        strict, exclude_st, part, liquidate, open_on_start = True, True, 0.10, True, True
        costs = Costs(commission=0.0001, min_commission=0.0, transfer=0.0, stamp_old=0.0005, stamp_new=0.0005,
                      slippage=0.0)
        st.markdown("""
| 项目 | 设置 |
|---|---|
| 股票池 | 全部股票，剔除 ST |
| 初始资金 | 10 亿元 |
| 佣金 | 万一（双边，无最低） |
| 印花税 | 万五（卖出） |
| 建仓 | 2025-01-02 开盘 |
| 清仓 | 2026-09-30 收盘 |
| 容量 | 单只 ≤ 前一日成交额 10% |
""")
        st.caption("想改参数请切换到「自定义」。")
    else:
        st.markdown('<p class="step">③ 股票池</p>', unsafe_allow_html=True)
        pool_name = st.selectbox("股票池", list(POOLS), index=0, label_visibility="collapsed")
        symbols = POOLS[pool_name]
        if symbols == "custom":
            raw = st.text_area("代码（逗号或换行分隔）", placeholder="688825.SH, 603986.SH, 300223.SZ", height=90)
            symbols = [s.strip().upper() for s in raw.replace("\n", ",").replace("，", ",").split(",") if s.strip()] or None
        exclude_st = st.toggle("剔除 ST 股票", value=True,
                               help="数据没有 ST 标记，按 ±5% 涨跌幅行为推断，见 docs/ST_INFERENCE.md。")

        st.markdown('<p class="step">④ 区间与资金</p>', unsafe_allow_html=True)
        default_start = pd.Timestamp("2025-01-02").date()
        if data is not None and symbols and symbols != "custom" and all(x in data.symbols for x in symbols):
            firsts = [data.dates[int(np.argmax(data.traded[:, data.col(x)]))].date() for x in symbols]
            default_start = max(default_start, min(firsts))      # 例如长鑫：从上市日开始
        start = st.date_input("开始日期", value=min(max(lo, default_start), hi), min_value=lo, max_value=hi,
                              key=f"start_{pool_name}")
        open_on_start = st.checkbox("开始日期当天开盘建仓", value=True,
                                    help="勾选：用前一交易日收盘的信号，在开始日期当天开盘成交。不勾选：开始日期收盘才出第一个信号。")
        end = st.date_input("结束日期", value=min(pd.Timestamp("2026-09-30").date(), hi), min_value=lo, max_value=hi)
        liquidate = st.checkbox("结束日期收盘清仓", value=True)
        capital = st.number_input("初始资金（元）", min_value=10_000.0, value=1_000_000_000.0, step=1_000_000.0,
                                  format="%.0f")

        st.markdown('<p class="step">⑤ 交易规则</p>', unsafe_allow_html=True)
        strict = st.toggle("A 股真实规则", value=True,
                           help="开：整手、开盘涨停买不进 / 跌停卖不出、停牌顺延。关：理想化（分数股、无涨跌停），用来看信号本身。")
        part_pct = st.number_input("单只成交额上限（前一日成交额的 %，0 为不限）", value=10.0, step=1.0, min_value=0.0)
        part = part_pct / 100 if part_pct > 0 else None
        with st.expander("费用"):
            comm = st.number_input("佣金（万分之，双边）", value=1.0, step=0.5) / 1e4
            min_comm = st.number_input("单笔最低佣金（元）", value=0.0, step=1.0)
            stamp_old = st.number_input("印花税 2023-08-28 前（万分之，卖出）", value=10.0, step=0.5) / 1e4
            stamp_new = st.number_input("印花税 2023-08-28 起（万分之，卖出）", value=5.0, step=0.5) / 1e4
            transfer = st.number_input("过户费（十万分之）", value=0.0, step=0.5) / 1e5
            slip = st.number_input("滑点（bp，单边）", value=0.0, step=1.0) / 1e4
            zero = st.checkbox("零费用（看扣费前表现）")
        costs = Costs.zero() if zero else Costs(comm, min_comm, transfer, stamp_old, stamp_new, slip)

    bench_mode = st.selectbox("基准", ["同口径股票池等权（含费用）", "股票池全部等权（零费用）", "指定股票买入持有", "不显示"],
                              help="同口径：与模板相同的筛选（上市满 60 日、剔除流动性最差 20%、剔除收盘一字板"
                                   "及 ST），每 5 日等权再平衡，按同样的费用、整手、成交额上限和清仓规则成交。")
    bench_sym = st.text_input("基准代码", value="688825.SH") if bench_mode == "指定股票买入持有" else None


# ====================================================================== 页面头
st.title("A 股策略回测台")
st.markdown('<p class="flow">第 t 日收盘 <b>算信号</b> → 第 t+1 日开盘 <b>撮合成交</b> → 收盘 <b>估值</b>'
            '　·　费用、整手、T+1、涨跌停、停牌、除权由引擎统一处理</p>', unsafe_allow_html=True)

left, right = st.columns([1.55, 1], gap="large")

with left:
    st.subheader("01 · 策略代码")
    if "code" not in st.session_state:
        st.session_state.tpl = list(TEMPLATES)[2]
        st.session_state.code = TEMPLATES[st.session_state.tpl]

    def _load_template():
        st.session_state.code = TEMPLATES[st.session_state.tpl]

    st.selectbox("从模板开始", list(TEMPLATES), key="tpl", on_change=_load_template)
    st.text_area("Python", key="code", height=520, label_visibility="collapsed")
    run = st.button("▶ 运行回测", type="primary", width="stretch", disabled=data is None)
    if data is None:
        st.caption("先在左侧载入数据。")

with right:
    st.subheader("02 · 接口与口径")
    st.markdown("""
**写一个继承 `Strategy` 的类，实现 `on_bar(self, ctx)`**，每个交易日收盘调用一次。

| 返回 | 含义（下一交易日开盘执行） |
|---|---|
| `{"688825.SH": 0.5, ...}` | 目标权重，合计 ≤ 1，余下为现金 |
| `None` | 不调仓 |
| `{}` | 清仓 |

没写到的已持仓股票，目标视为 0。
""")
    with st.expander("ctx 能用什么"):
        st.markdown("""
- `ctx.history(field, n)`：最近 n 天（含今天）的数据框，行是日期、列是股票。价格默认**后复权**，可直接算收益；`field` 可选 `open/high/low/close/amt`
- `ctx.current(field)`：今天的数据（不复权）
- `ctx.universe`：股票池里今天有成交的代码
- `ctx.listed_days`：上市以来的交易日数
- `ctx.limit_status`：今天收盘 1 涨停 / −1 跌停 / 0
- `ctx.positions` / `ctx.weights` / `ctx.cash` / `ctx.nav`：当前账户
- `ctx.date`、`ctx.bar_index`（从回测起点数第几天）
- `ctx.equal_weight(代码列表)`：等权字典

`ctx` 只给到**今天收盘为止**的数据，策略写不出未来函数。
""")
    st.info("""
**成交规则**　开盘价 ± 滑点成交，先卖后买，现金不足按比例缩量。
主板 / 创业板 100 股一手；科创板至少 200 股。
开盘封涨停买不进、封跌停卖不出；停牌不能交易。没成交的目标顺延到下一个交易日。
分红送股按复权因子调整持股数。
""")
    st.warning("数据没有 ST 标记和公司行为明细：ST 股的 5% 涨跌停用「一字板」规则近似；复权按 cumadj 计算。")


# ====================================================================== 运行
def benchmark(d: MarketData, res_index, i0: int, i1: int, pool: list[str] | None, ex_st: bool,
              run_cfg: dict) -> pd.Series | None:
    if bench_mode == "不显示":
        return None
    W = np.full(d.close.shape, np.nan)
    mask = np.ones(len(d.symbols), bool) if not pool else np.isin(d.symbols, pool)
    if bench_mode == "指定股票买入持有":
        if bench_sym not in d.symbols:
            st.warning(f"基准代码 {bench_sym} 不在数据里")
            return None
        j = d.col(bench_sym)
        first = max(i0, int(np.argmax(d.traded[:, j])))
        W[first] = 0.0
        W[first, j] = 1.0
        nav = frictionless_nav(d, W, start=i0, end=i1)
        return pd.Series(nav, index=res_index, name=f"{bench_sym} 买入持有（零费用）")
    if bench_mode == "同口径股票池等权（含费用）":
        uni = bench_universe(id(d), ex_st) if mask.sum() > 50 else (d.traded & (~d.st if ex_st else True))
        for t in range(i0, i1 - 1, 5):
            ok = uni[t] & mask
            if ok.any():
                W[t] = np.where(ok, 1.0 / ok.sum(), 0.0)
        bt_ = Backtester(d, run_cfg["costs"], capital=run_cfg["capital"], lots=run_cfg["lots"], limits=run_cfg["lots"],
                         band=0.0002, max_participation=run_cfg["part"], liquidate_at_end=run_cfg["liquidate"])
        r = bt_.run(W, start=i0, end=i1)
        return pd.Series((r.nav / run_cfg["capital"]).to_numpy(), index=res_index,
                         name="同口径股票池等权（含费用" + ("，剔除 ST）" if ex_st else "）"))
    for t in range(i0, i1, 5):
        ok = mask & d.traded[t] & (d.traded[t + 1] if t + 1 < len(d.dates) else True)
        if ex_st:
            ok &= ~d.st[t]
        if ok.any():
            W[t] = np.where(ok, 1.0 / ok.sum(), 0.0)
    nav = frictionless_nav(d, W, start=i0, end=i1)
    return pd.Series(nav, index=res_index, name="股票池全部等权（每 5 日再平衡，零费用" + ("，剔除 ST）" if ex_st else "）"))


@st.cache_resource(show_spinner=False)
def bench_universe(_key: int, ex_st: bool) -> np.ndarray:
    return S.universe(data, exclude_st=ex_st)


if run:
    try:
        cls = compile_strategy(st.session_state.code)
        strat = cls()
    except Exception:
        st.error("策略代码有错误")
        st.code(traceback.format_exc(limit=3), language="text")
        st.stop()
    if start >= end:
        st.error("开始日期要早于结束日期")
        st.stop()
    bar = st.progress(0.0, text="回测中…")
    cfg = RunConfig(start=str(start), end=str(end), capital=float(capital), symbols=symbols, costs=costs,
                    lots=strict, limits=strict, retry=True, exclude_st=exclude_st, max_participation=part,
                    liquidate_at_end=liquidate, first_trade_on_start=open_on_start)
    try:
        res, i0, i1 = run_strategy(data, strat, cfg,
                                   progress=lambda p, dt: bar.progress(min(p, 1.0), text=f"回测中… {dt.date()}"))
    except Exception:
        bar.empty()
        st.error("运行时出错（多半是策略代码里的问题）")
        st.code(traceback.format_exc(limit=4), language="text")
        st.stop()
    bar.empty()
    run_cfg = {"costs": costs, "capital": float(capital), "lots": strict, "part": part, "liquidate": liquidate}
    st.session_state.result = (res, i0, i1, symbols, cls.__name__, float(capital), exclude_st, run_cfg)


# ====================================================================== 结果
if "result" in st.session_state and data is not None:
    res, i0, i1, pool, sname, cap, ex_st, run_cfg = st.session_state.result
    nav = res.nav / cap
    with st.spinner("计算基准…"):
        bench = benchmark(data, nav.index, i0, i1, pool, ex_st, run_cfg)
    m = summarize(res.nav, bench=bench, turnover=res.daily.turnover, fees=res.daily.fees)

    st.divider()
    st.subheader(f"03 · 结果　{sname}")
    st.caption(f"{nav.index[0].date()} 至 {nav.index[-1].date()} · {m['days']} 个交易日 · "
               f"{'A 股真实规则' if res.meta['lots'] else '理想化规则'} · 初始资金 {cap:,.0f} 元")

    if res.meta["lots"] and res.daily.exposure.iloc[1:].mean() < 0.6 and res.daily.positions.max() > 20:
        st.warning(f"平均仓位只有 {res.daily.exposure.iloc[1:].mean():.0%}：资金分到每只股票太少，整手约束下很多股票连 1 手都买不了，"
                   "最低 5 元佣金也会被放大。持有几百只股票的策略建议把初始资金设为 1 亿元，或关闭「A 股真实规则」。")
    k = st.columns(7)
    k[0].metric("总收益", f"{m['total_return']:+.2%}")
    k[1].metric("年化收益", f"{m['ann_return']:+.2%}")
    k[2].metric("夏普", f"{m['sharpe']:.2f}" if np.isfinite(m["sharpe"]) else "—")
    k[3].metric("最大回撤", f"{m['max_drawdown']:.2%}")
    k[4].metric("相对基准", f"{m['excess_total']:+.2%}" if bench is not None else "—")
    k[5].metric("年化换手", f"{m['turnover_ann']:.1f} 倍")
    fee = m["fees_total"]
    k[6].metric("费用合计", f"{fee / 1e4:,.0f} 万元" if fee >= 1e6 else f"{fee:,.0f} 元")
    if bench is not None:
        st.caption(f"基准：{bench.name}，同期总收益 {m['bench_total']:+.2%}。"
                   f"「相对基准」= (1 + 策略收益) / (1 + 基准收益) − 1。费用合计占初始资金 {m['fees_total'] / cap:.2%}。")

    fig = go.Figure()
    if bench is not None:
        fig.add_scatter(x=bench.index, y=bench, name=bench.name, line=dict(color=GREY, width=1.6))
    fig.add_scatter(x=nav.index, y=nav, name="策略（扣费后）", line=dict(color=BLUE, width=2.2))
    fig.add_hline(y=1, line=dict(color="#bdbcb7", width=1))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title="净值",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, width="stretch")

    dd = nav / nav.cummax() - 1
    c1, c2 = st.columns([1.6, 1])
    with c1:
        f2 = go.Figure()
        f2.add_scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=RED, width=1), name="回撤",
                       fillcolor="rgba(200,64,47,0.18)")
        f2.add_scatter(x=res.daily.index, y=res.daily.exposure, name="仓位", line=dict(color=BLUE, width=1.2),
                       yaxis="y2")
        f2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                         title=dict(text="回撤与仓位", font=dict(size=14)),
                         yaxis=dict(tickformat=".0%", gridcolor="rgba(128,128,128,0.15)"),
                         yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.05], showgrid=False),
                         legend=dict(orientation="h", y=1.15, x=0.6),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(f2, width="stretch")
    with c2:
        st.markdown("**分年表现**")
        yr = by_year(res.nav, bench * cap if bench is not None else None)
        yr = yr.rename(columns={"strategy": "策略", "bench": "基准", "excess": "超额"}).rename_axis("年份")
        st.dataframe(yr.style.format("{:+.2%}"), width="stretch")

    t1, t2, t3, t4 = st.tabs([f"成交记录（{len(res.trades)}）", f"未成交（{len(res.rejects)}）", "期末持仓", "每日明细"])
    with t1:
        tr = res.trades.copy()
        tr["date"] = tr.date.dt.date
        tr["side"] = tr.side.map({"BUY": "买入", "SELL": "卖出"})
        tr.columns = ["日期", "代码", "方向", "股数", "成交价", "金额", "费用"]
        NC = st.column_config.NumberColumn
        st.dataframe(tr, width="stretch", height=320, hide_index=True,
                     column_config={"股数": NC(format="%,.0f"), "成交价": NC(format="%.3f"),
                                    "金额": NC(format="%,.0f"), "费用": NC(format="%,.2f")})
        st.download_button("下载成交记录 CSV", res.trades.to_csv(index=False).encode("utf-8-sig"), "trades.csv")
    with t2:
        if res.rejects.empty:
            st.caption("没有因涨跌停或停牌而未成交的订单。")
        else:
            rj = res.rejects.copy()
            rj["date"] = rj.date.dt.date
            rj.columns = ["日期", "代码", "原因"]
            st.dataframe(rj, width="stretch", height=320, hide_index=True)
    with t3:
        last = res.positions.iloc[-1]
        last = last[last > 0]
        if last.empty:
            st.caption("期末空仓。")
        else:
            px = pd.Series(data.mark[i1 - 1], index=data.symbols)[last.index]
            pos = pd.DataFrame({"股数": last, "收盘价": px, "市值": last * px})
            pos["权重"] = pos["市值"] / res.nav.iloc[-1]
            st.dataframe(pos.sort_values("市值", ascending=False).style.format(
                {"股数": "{:,.0f}", "收盘价": "{:.2f}", "市值": "{:,.0f}", "权重": "{:.2%}"}), width="stretch")
    with t4:
        daily = res.daily.copy()
        daily.index = daily.index.date
        daily.columns = ["净值（元）", "现金", "仓位", "当日换手", "当日费用", "持股数量"]
        NC = st.column_config.NumberColumn
        st.dataframe(daily, width="stretch", height=320,
                     column_config={"净值（元）": NC(format="%,.0f"), "现金": NC(format="%,.0f"),
                                    "仓位": NC(format="percent"), "当日换手": NC(format="percent"),
                                    "当日费用": NC(format="%,.2f")})
        st.download_button("下载每日净值 CSV", res.daily.to_csv().encode("utf-8-sig"), "daily.csv")
