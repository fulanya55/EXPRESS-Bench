"""Render a complete evaluation report without browser JavaScript."""

from collections import Counter
from html import escape
from pathlib import Path

from evaluation import _parse_mark


def render_report(data, report_path: Path):
    def h(value):
        return escape(str(value), quote=True)

    def fmt(value, digits=2):
        return "—" if value is None else f"{value:,.{digits}f}"

    def money(value):
        return f"${value:,.4f}"

    def bar(value, color="blue"):
        width = min(100.0, max(0.0, float(value or 0)))
        return (
            f'<span class="bar-track"><span class="bar-fill {color}" '
            f'style="width:{width:.2f}%"></span></span>'
        )

    status = data["status"]
    partial = data["partial"]
    api = data["api_total"]
    analysis = data["analysis"]
    rows = data["rows"]
    completed, target = status["completed"], status["target"]
    progress_pct = completed / target * 100 if target else 0.0
    state_label = {"running": "测评运行中", "stopped": "测评暂停", "complete": "测评完成"}[status["state"]]

    types = sorted(data["by_type"], key=lambda item: item["c"] or -1, reverse=True)
    type_rows = "".join(
        f'<tr><th scope="row">{h(item["type"])}</th><td class="num">{item["count"]}</td>'
        f'<td class="num">{item["scored_count"]}</td>'
        f'<td class="measure">{bar(item["c"])}<span>{fmt(item["c"])}%</span></td>'
        f'<td class="measure">{bar(item["epath"], "green")}<span>{fmt(item["epath"])}%</span></td>'
        f'<td class="num">{item["tokens"]:,}</td><td class="num">{money(item["cost"])}</td></tr>'
        for item in types
    )
    rank_rows = "".join(
        f'<tr><th scope="row">GPU {item["rank"]}</th>'
        f'<td class="measure">{bar(item["completed"] / item["total"] * 100 if item["total"] else 0, "orange")}'
        f'<span>{item["completed"]}/{item["total"]}</span></td>'
        f'<td class="num">{fmt(item["completed"] / item["total"] * 100 if item["total"] else 0, 1)}%</td>'
        f'<td class="num">{money(item["cost_usd"])}</td></tr>'
        for item in data["rank_progress"]
    )

    accuracy_counts = Counter()
    for row in rows:
        components = row["eac"].split(",", 1)
        if len(components) == 2:
            try:
                accuracy_counts[str(int(_parse_mark(components[1])))] += 1
                continue
            except ValueError:
                pass
        accuracy_counts["格式异常"] += 1
    accuracy_rows = "".join(
        f'<tr><th scope="row">{h(label)}</th><td class="measure">'
        f'{bar(count / completed * 100 if completed else 0)}<span>{fmt(count / completed * 100 if completed else 0, 1)}%</span></td>'
        f'<td class="num">{count}</td></tr>'
        for label in ["5", "4", "3", "2", "1", "0", "格式异常"]
        if (count := accuracy_counts.get(label, 0))
    )
    detail_rows = ""
    for row in rows:
        c_text = "—" if row["c"] is None else f'{fmt(row["c"])}%'
        epath_text = "—" if row["epath"] is None else f'{fmt(row["epath"])}%'
        detail_rows += (
            f'<tr><td class="num">{row["index"]}</td><td>{h(row["type"])}</td>'
            f'<td class="question">{h(row["question"])}</td><td>{h(row["eac"])}</td>'
            f'<td class="num">{c_text}</td><td class="num">{epath_text}</td>'
            f'<td class="num">{row["tokens"]:,}</td><td class="num">{money(row["cost"])}</td></tr>'
        )

    top = types[0] if types else None
    bottom = types[-1] if types else None
    gap = partial["c"] - partial["epath"] if partial["c"] is not None and partial["epath"] is not None else None
    analysis_lines = [
        f'已保存 <strong>{completed:,}</strong> 条，覆盖 <strong>{fmt(progress_pct, 1)}%</strong>；其中 <strong>{partial["scored_count"]:,}</strong> 条有有效评分。',
        f'C_avg 为 <strong>{fmt(partial["c"])}%</strong>，E_path 为 <strong>{fmt(partial["epath"])}%</strong>，路径项平均降低约 <strong>{fmt(gap)} 个百分点</strong>。',
        f'当前最高类型为 <strong>{h(top["type"])}</strong>（{fmt(top["c"])}%），最低为 <strong>{h(bottom["type"])}</strong>（{fmt(bottom["c"])}%）。' if top and bottom else "",
        f'已保存单题平均成本 <strong>{money(analysis["average_question_cost"])}</strong>；全部已发出请求累计 <strong>{money(api["cost"])}</strong>。',
    ]
    if data["invalid_scores"]:
        ids = ", ".join(str(item["index"]) for item in data["invalid_scores"])
        analysis_lines.append(f'评分格式异常的题目编号：<strong>{h(ids)}</strong>；保留单题结果，不计入平均评分。')
    analysis_html = "".join(f"<li>{line}</li>" for line in analysis_lines if line)

    # A compact SVG cost curve; all other charts use static HTML bars.
    cumulative = 0.0
    max_index = max((row["index"] for row in rows), default=1)
    points = []
    for row in rows:
        cumulative += row["cost"]
        x = 58 + 620 * row["index"] / max(max_index, 1)
        y = 245 - 200 * cumulative / max(partial["cost"], 0.0001)
        points.append(f"{x:.1f},{y:.1f}")
    cost_svg = (
        '<svg viewBox="0 0 730 285" role="img" aria-label="已保存问题的累计 API 费用">'
        '<line class="axis" x1="58" y1="245" x2="678" y2="245"/>'
        '<line class="axis" x1="58" y1="45" x2="58" y2="245"/>'
        f'<polyline class="trend" points="{h(" ".join(points))}"/>'
        '<text class="tick" x="58" y="268">0</text>'
        f'<text class="tick" x="678" y="268" text-anchor="end">题目编号 {max_index}</text>'
        f'<text class="tick" x="58" y="36">{money(partial["cost"])}</text></svg>'
    )

    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store"><title>EXPRESS-Bench 当前评测报告</title>
<style>
:root{{color-scheme:light dark;--bg:light-dark(#f7f8fa,#111827);--fg:light-dark(#172033,#f3f4f6);--muted:light-dark(#536174,#b4becc);--surface:light-dark(#fff,#1b2433);--border:light-dark(#d9dee7,#344155);--blue:light-dark(#2563eb,#76a7ff);--green:light-dark(#15803d,#61d394);--orange:light-dark(#c2410c,#ffad70);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);line-height:1.5}}main{{max-width:1180px;margin:auto;padding:26px 18px 48px}}h1{{font-size:28px;font-weight:500;margin:0 0 4px}}h2{{font-size:18px;font-weight:500;margin:0 0 14px}}.muted{{color:var(--muted);font-size:13px}}.status{{margin:18px 0;padding:12px 16px;border-left:4px solid var(--orange);background:color-mix(in srgb,var(--orange) 10%,transparent)}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}.stat,.panel{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:15px}}.stat .label{{color:var(--muted);font-size:13px}}.stat .value{{font-size:25px;font-variant-numeric:tabular-nums;margin-top:3px}}.section{{margin-top:22px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.progress-track,.bar-track{{display:inline-block;background:color-mix(in srgb,var(--border) 65%,transparent);overflow:hidden;border-radius:9px;vertical-align:middle}}.progress-track{{display:block;width:100%;height:14px;margin:12px 0}}.bar-track{{height:10px;width:65%}}.bar-fill{{display:block;height:100%}}.bar-fill.blue,.progress-fill{{background:var(--blue)}}.bar-fill.green{{background:var(--green)}}.bar-fill.orange{{background:var(--orange)}}.progress-fill{{height:100%}}.meta{{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--muted)}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 8px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}}th{{font-weight:500}}thead th{{background:var(--surface);position:sticky;top:0}}.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.measure{{min-width:170px}}.measure>span:last-child{{display:inline-block;width:33%;text-align:right;white-space:nowrap}}.table-wrap{{overflow:auto;max-height:640px}}.question{{min-width:250px}}ol{{margin:0;padding-left:22px}}li{{margin:6px 0}}.axis{{stroke:var(--border);stroke-width:1}}.trend{{fill:none;stroke:var(--blue);stroke-width:2}}.tick{{fill:var(--muted);font-size:12px}}svg{{display:block;width:100%;height:auto}}@media(max-width:760px){{.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid{{grid-template-columns:1fr}}main{{padding:18px 10px}}.stat .value{{font-size:22px}}}}
</style></head><body><main>
<h1>EXPRESS-Bench Fine-EQA 测评报告</h1>
<div class="muted">快照时间：{h(status["generated_at"])} · 输出目录：experiment_result_full8_v4</div>
<div class="status"><strong>{state_label}</strong>：已保存 {completed:,} / {target:,} 条。{'' if status["state"] == "complete" else '以下指标只代表当前样本，仍会变化。'}</div>
<div class="stats">
<div class="stat"><div class="label">C_avg · {partial["scored_count"]:,} 条有效评分</div><div class="value">{fmt(partial["c"])}%</div></div>
<div class="stat"><div class="label">C*_avg</div><div class="value">{fmt(partial["cstar"])}%</div></div>
<div class="stat"><div class="label">E_path · {partial["path_count"]:,} 条有效路径</div><div class="value">{fmt(partial["epath"])}%</div></div>
<div class="stat"><div class="label">平均目标距离 · {partial["goal_count"]:,} 条有效距离</div><div class="value">{fmt(partial["goal"])} m</div></div>
</div>
<section class="section panel"><h2>进度与 API 消耗</h2><strong>{completed:,} / {target:,}（{fmt(progress_pct,1)}%）</strong><div class="progress-track" role="progressbar" aria-label="测评进度" aria-valuemin="0" aria-valuemax="{target}" aria-valuenow="{completed}"><div class="progress-fill" style="width:{progress_pct:.2f}%"></div></div><div class="meta"><span>已发出请求：{api["calls"]:,} 次</span><span>Tokens：{api["total_tokens"]:,}</span><span>累计费用：{money(api["cost"])}</span><span>已保存结果费用：{money(partial["cost"])}</span></div></section>
<section class="section panel"><h2>当前阶段分析</h2><ol>{analysis_html}</ol></section>
<div class="grid section"><section class="panel"><h2>问题类型表现</h2><div class="table-wrap"><table><thead><tr><th>类型</th><th class="num">数量</th><th class="num">有效</th><th>C_avg</th><th>E_path</th><th class="num">Tokens</th><th class="num">费用</th></tr></thead><tbody>{type_rows}</tbody></table></div><p class="muted">蓝色：C_avg；绿色：E_path。</p></section>
<section class="panel"><h2>8 卡分片进度</h2><div class="table-wrap"><table><thead><tr><th>卡</th><th>完成</th><th class="num">进度</th><th class="num">API 费用</th></tr></thead><tbody>{rank_rows}</tbody></table></div></section></div>
<div class="grid section"><section class="panel"><h2>评分准确度分布</h2><div class="table-wrap"><table><thead><tr><th>得分</th><th>占比</th><th class="num">数量</th></tr></thead><tbody>{accuracy_rows}</tbody></table></div></section>
<section class="panel"><h2>单题累计费用</h2>{cost_svg}<p class="muted">曲线仅统计已保存的 {completed:,} 条结果；上方累计费用包含全部已发出请求。</p></section></div>
<section class="section panel"><h2>已保存单题结果（{completed:,} 条）</h2><div class="table-wrap"><table><thead><tr><th class="num">编号</th><th>类型</th><th>问题</th><th>EAC</th><th class="num">C</th><th class="num">E_path</th><th class="num">Tokens</th><th class="num">费用</th></tr></thead><tbody>{detail_rows}</tbody></table></div></section>
</main></body></html>'''
    report_path.write_text(html, encoding="utf-8")
