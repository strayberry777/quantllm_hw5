"""把 report/candidates.csv 渲染成图片（docs/img/loop_candidates.png）。"""
import asyncio
import html

import pandas as pd
from playwright.async_api import async_playwright

from common import HERE, ROOT

df = pd.read_csv(HERE / "report" / "candidates.csv")
pct = lambda v: "" if pd.isna(v) else f"{v:+.1%}"
num = lambda v, f="{:.2f}": "" if pd.isna(v) else f.format(v)
rows = []
for _, r in df.iterrows():
    dec = r["决策"]
    cls = "up" if "晋升" in dec else ("base" if "基准" in dec else "rej")
    rows.append(f"""<tr class="{cls}"><td class="id">{r['实验']}<br><span>第 {r['轮次']} 轮</span></td>
<td class="name">{html.escape(r['因子'])}<br><span>{html.escape(str(r['假设']))}</span></td>
<td><b class="pill {cls}">{html.escape(dec)}</b></td>
<td>{num(r.get('验证 RankIC'), '{:.4f}')}</td><td>{num(r.get('验证 IC t'), '{:.1f}')}</td><td>{num(r.get('训练 IC t'), '{:.1f}')}</td>
<td>{num(r.get('夏普'))}</td><td>{pct(r.get('最大回撤'))}</td><td>{pct(r.get('年化超额'))}</td><td>{num(r.get('年换手（倍）'), '{:.0f}')}</td></tr>""")
page = f"""<html><head><meta charset="utf-8"><style>
body{{font-family:'Noto Sans CJK SC',sans-serif;background:#fcfcfb;color:#1a1a19;margin:0;padding:24px;width:1500px}}
h1{{font-size:20px;margin:0 0 4px}} p{{margin:0 0 14px;color:#6b6a66;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}}
th{{text-align:right;color:#8a8984;font-weight:500;padding:8px;border-bottom:1px solid #e4e3df;background:#f4f3ef}}
td{{text-align:right;padding:7px 8px;border-bottom:1px solid #eeede9;vertical-align:top}}
th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}
td span{{color:#9a9893;font-size:11px}} td.name{{max-width:520px}} td.id{{white-space:nowrap}}
.pill{{display:inline-block;padding:1px 8px;border-radius:4px;font-weight:500;font-size:11.5px;white-space:nowrap}}
.pill.up{{background:#e3f1e8;color:#2f7a52}} .pill.base{{background:#eef0e6;color:#6b7a3a}} .pill.rej{{background:#f3f1ec;color:#8a8984}}
tr.up td{{background:#f4faf6}}
</style></head><body>
<h1>全部候选 · 成功与失败同样保留</h1>
<p>{len(df)} 个实验按时间顺序排列；决策规则固定：验证期 IC t ≥ 当时的 Bonferroni 门槛且为正、训练期 t ≥ 2 同向、与池内因子相关 &lt; 0.7。夏普、回撤、超额为验证期「做多最高 10%」扣费后结果。</p>
<table><tr><th>实验 / 轮次</th><th>因子 / 假设</th><th>决策</th><th>验证 RankIC</th><th>验证 IC t</th><th>训练 IC t</th><th>夏普</th><th>最大回撤</th><th>年化超额</th><th>年换手</th></tr>
{''.join(rows)}</table></body></html>"""
(HERE / "report" / "candidates.html").write_text(page)


async def shot():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1548, "height": 800})
        await pg.goto((HERE / "report" / "candidates.html").as_uri())
        await pg.screenshot(path=str(ROOT / "docs" / "img" / "loop_candidates.png"), full_page=True)
        await b.close()

asyncio.run(shot())
print("ok")
