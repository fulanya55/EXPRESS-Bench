#!/usr/bin/env python3
"""Refresh the standalone HTML report from per-question checkpoints.

The evaluator writes one ``<question>/result.pkl`` at a time.  This report
therefore intentionally works while torchrun is still running and treats a
malformed or half-written checkpoint as a skipped row for this snapshot.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiment_result_full8_v4"
DATASET_PATH = ROOT / "data" / "express-bench.json"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.html"
sys.path.insert(0, str(ROOT))

from evaluation import _parse_mark


def finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_result(result):
    """Return compact JSON-safe row data and metric components."""
    eac = str(result.get("EAC", "")).replace("Your mark:", "").strip()
    grade = accuracy = None
    score_error = None
    pieces = eac.split(",", 1)
    if len(pieces) == 2:
        try:
            grade = _parse_mark(pieces[0])
            accuracy = _parse_mark(pieces[1])
        except ValueError as exc:
            score_error = str(exc)
    else:
        score_error = "EAC does not contain two comma-separated components"

    c = grade * accuracy if grade is not None and accuracy is not None else None
    c_percent = 100.0 * max(0.0, min(c, 5.0)) / 5.0 if c is not None else None
    cstar_percent = (
        100.0 * max(0.0, min(accuracy, 5.0)) / 5.0
        if accuracy is not None
        else None
    )
    path_len = finite(result.get("path_len"))
    geodesic = finite(result.get("geodesic_distance"))
    epath = None
    if c_percent is not None and path_len is not None and geodesic is not None:
        denominator = max(path_len, geodesic)
        if denominator > 0:
            epath = c_percent * geodesic / denominator
    goal_dis = finite(result.get("goal_dis"))
    usage = result.get("api_usage") or {}
    row = {
        "index": int(result["question_ind"]),
        "type": str(result.get("type", "unknown")),
        "question": str(result.get("question", "")),
        "answer": str(result.get("answer", "")),
        "eac": eac,
        "gen_answer": str(result.get("gen_answer", "")),
        "c": c_percent,
        "cstar": cstar_percent,
        "epath": epath,
        "path_len": path_len,
        "goal_dis": goal_dis,
        "cnt_step": int(result.get("cnt_step", 0) or 0),
        "tokens": int(usage.get("total_tokens", 0) or 0),
        "cost": float(usage.get("cost_usd", 0.0) or 0.0),
    }
    return row, score_error


def mean(values):
    return float(np.mean(values)) if values else None


def metric_summary(rows):
    scored = [row for row in rows if row["c"] is not None]
    path_rows = [row for row in rows if row["epath"] is not None]
    goal_rows = [row for row in rows if row["goal_dis"] is not None]
    return {
        "count": len(rows),
        "scored_count": len(scored),
        "path_count": len(path_rows),
        "goal_count": len(goal_rows),
        "c": mean([row["c"] for row in scored]),
        "cstar": mean([row["cstar"] for row in scored]),
        "epath": mean([row["epath"] for row in path_rows]),
        "goal": mean([row["goal_dis"] for row in goal_rows]),
        "tokens": sum(row["tokens"] for row in rows),
        "cost": sum(row["cost"] for row in rows),
    }


def read_rank_progress():
    states = []
    for path in sorted(OUTPUT_DIR.glob("progress.rank*.json")):
        try:
            states.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    rank_progress = []
    for path, state in zip(sorted(OUTPUT_DIR.glob("progress.rank*.json")), states):
        match = re.search(r"rank(\d+)", path.name)
        rank_progress.append({
            "rank": int(match.group(1)) if match else len(rank_progress),
            "completed": int(state.get("completed", 0)),
            "total": int(state.get("total", 0)),
            "calls": int(state.get("calls", 0)),
            "total_tokens": int(state.get("total_tokens", 0)),
            "cost_known": bool(state.get("cost_known", False)),
            "cost_usd": float(state.get("cost_usd", 0.0) or 0.0),
        })
    return rank_progress


def read_jsonl_usage():
    """Aggregate append-only request records while the run is active."""
    usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_known": True,
    }
    found = False
    for path in sorted(OUTPUT_DIR.glob("api_usage.rank*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            found = True
            usage["calls"] += 1
            usage["prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
            usage["total_tokens"] += int(record.get("total_tokens", 0) or 0)
            if record.get("cost_usd") is None:
                usage["cost_known"] = False
            else:
                usage["cost_usd"] += float(record.get("cost_usd", 0.0) or 0.0)
    return usage if found else None


def process_active():
    return subprocess.run(
        ["tmux", "has-session", "-t", "express-eval"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def load_rows():
    rows = []
    invalid = []
    for path in sorted(OUTPUT_DIR.glob("*/result.pkl")):
        try:
            with path.open("rb") as handle:
                result = pickle.load(handle)
            row, score_error = parse_result(result)
        except Exception as exc:  # a checkpoint may be mid-write during refresh
            continue
        rows.append(row)
        if score_error:
            invalid.append({"index": row["index"], "error": score_error, "eac": row["eac"]})
    return sorted(rows, key=lambda row: row["index"]), invalid


def make_data():
    rows, invalid = load_rows()
    by_type_rows = defaultdict(list)
    for row in rows:
        by_type_rows[row["type"]].append(row)
    by_type = []
    for type_name in sorted(by_type_rows):
        summary = metric_summary(by_type_rows[type_name])
        summary["type"] = type_name
        by_type.append(summary)

    rank_progress = read_rank_progress()
    progress_completed = sum(item["completed"] for item in rank_progress)
    progress_target = sum(item["total"] for item in rank_progress)
    target = len(json.loads(DATASET_PATH.read_text(encoding="utf-8")))
    active = process_active()

    api_path = OUTPUT_DIR / "api_usage.json"
    if api_path.is_file():
        try:
            api = json.loads(api_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            api = {}
    else:
        api = {}
    jsonl_api = read_jsonl_usage()
    if not api and jsonl_api:
        api = jsonl_api
    if not api:
        api = {
            "calls": sum(item["calls"] for item in rank_progress),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": sum(item["total_tokens"] for item in rank_progress),
            "cost_usd": sum(item["cost_usd"] for item in rank_progress),
            "cost_known": bool(rank_progress) and all(item["cost_known"] for item in rank_progress),
        }
    api_total = {
        "calls": int(api.get("calls", 0)),
        "prompt_tokens": int(api.get("prompt_tokens", 0)),
        "completion_tokens": int(api.get("completion_tokens", 0)),
        "total_tokens": int(api.get("total_tokens", 0)),
        "cost": float(api.get("cost_usd", api.get("cost", 0.0)) or 0.0),
        "cost_known": bool(api.get("cost_known", True)),
        "source": "api_usage.json" if api_path.is_file() else ("api_usage.rank*.jsonl" if jsonl_api else "progress.rank*.json"),
    }
    summary = metric_summary(rows)
    status = {
        "completed": len(rows),
        "target": target,
        "progress_file_completed": progress_completed,
        "progress_file_target": progress_target or target,
        "max_index": max((row["index"] for row in rows), default=None),
        "active_process": active,
        "state": "complete" if len(rows) >= target else ("running" if active else "stopped"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    eac_counts = Counter(row["eac"] for row in rows)
    return {
        "status": status,
        "partial": summary,
        "invalid_scores": invalid,
        "analysis": {
            "average_steps": mean([row["cnt_step"] for row in rows]),
            "saved_tokens": summary["tokens"],
            "saved_cost": summary["cost"],
            "average_question_cost": summary["cost"] / len(rows) if rows else 0.0,
            "average_question_tokens": summary["tokens"] / len(rows) if rows else 0.0,
        },
        "by_type": by_type,
        "eac_counts": dict(eac_counts),
        "rank_progress": rank_progress,
        "api_total": api_total,
        "rows": rows,
    }


def refresh_html(data):
    html = REPORT_PATH.read_text(encoding="utf-8")
    html = html.replace(
        '<div class="status"><strong>部分运行</strong>：当前没有评测进程，尚未生成最终合并的 <code>results.pkl</code>。已读取 <b id="done"></b> 条单题结果；指标只代表这一阶段样本，不能作为 2044 条完整基准结果。</div>',
        '<div class="status"><strong id="status-title"></strong>：<span id="status-text"></span></div>',
    )
    html = html.replace(
        '曲线只统计已保存的 100 条单题结果；API 总用量还包含中断时未完成的请求。',
        '曲线统计 <b id="saved-count-caption"></b> 条已保存结果；API 总用量包含已发出请求。',
    )
    html = html.replace(
        '<section class="section panel"><h2>问题类型汇总</h2>',
        '<section class="section panel"><h2>当前阶段分析</h2><ul id="analysis-list" class="analysis-list"></ul></section>\n  <section class="section panel"><h2>问题类型汇总</h2>',
    )
    html = html.replace(
        '<th>类型</th><th class="num">数量</th><th class="num">C_avg</th>',
        '<th>类型</th><th class="num">数量</th><th class="num">有效评分</th><th class="num">C_avg</th>',
    )
    html = html.replace(
        '<h2>EAC 分布</h2>',
        '<h2>EAC 分布（Top 12）</h2>',
    )
    html = html.replace(
        '</style>',
        '.analysis-list{margin:0;padding-left:20px}.analysis-list li{margin:6px 0}.analysis-list strong{font-variant-numeric:tabular-nums}\n</style>',
        1,
    )
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html, count = re.subn(
        r"const DATA = .*?;\nconst fmt",
        "const DATA = " + payload + ";\nconst fmt",
        html,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Could not locate DATA block in the existing report")

    html = html.replace(
        "el('done').textContent = DATA.status.completed + ' / ' + DATA.status.target;",
        "el('done').textContent = DATA.status.completed + ' / ' + DATA.status.target;\n"
        "el('status-title').textContent = DATA.status.state === 'complete' ? '评测完成' : (DATA.status.state === 'running' ? '评测运行中' : '评测暂停');\n"
        "el('status-text').textContent = '已读取 ' + DATA.status.completed + ' / ' + DATA.status.target + ' 条结果；' + (DATA.status.state === 'complete' ? '已生成完整结果。' : '当前指标只代表已保存样本。');\n"
        "el('saved-count-caption').textContent = DATA.status.completed;",
    )
    html = html.replace(
        "el('progress-bar').parentElement.setAttribute('aria-valuenow',DATA.status.completed);",
        "el('progress-bar').parentElement.setAttribute('aria-valuenow',DATA.status.completed);\n"
        "el('progress-bar').parentElement.setAttribute('aria-valuemax',DATA.status.target);",
    )
    html = html.replace(
        "function drawEac(){const svg=el('eac-chart');svg.innerHTML='';const items=Object.entries(DATA.eac_counts).sort((a,b)=>b[1]-a[1]);const left=180,right=680,top=28,row=28,max=Math.max(...items.map(x=>x[1]));items.forEach(([label,n],i)=>{const y=top+i*row;svgText(svg,left-10,y+14,label,'tick','end');rect(svg,left,y,(right-left)*n/max,13,'bar-c');svgText(svg,left+(right-left)*n/max+6,y+11,String(n),'tick','start')})}",
        "function drawEac(){const svg=el('eac-chart');svg.innerHTML='';const all=Object.entries(DATA.eac_counts).sort((a,b)=>b[1]-a[1]);const topItems=all.slice(0,12);const other=all.slice(12).reduce((s,x)=>s+x[1],0);if(other)topItems.push(['其他',other]);const left=180,right=680,top=28,row=23,max=Math.max(...topItems.map(x=>x[1]));topItems.forEach(([label,n],i)=>{const y=top+i*row;svgText(svg,left-10,y+14,label,'tick','end');rect(svg,left,y,(right-left)*n/max,13,'bar-c');svgText(svg,left+(right-left)*n/max+6,y+11,String(n),'tick','start')})}",
    )
    html = html.replace(
        "tr.innerHTML=`<td>${d.type}</td><td class=\"num\">${d.count}</td><td class=\"num\">${fmt(d.c)}%</td>",
        "tr.innerHTML=`<td>${d.type}</td><td class=\"num\">${d.count}</td><td class=\"num\">${d.scored_count}</td><td class=\"num\">${d.c==null?'—':fmt(d.c)+'%'}</td>",
    )
    html = html.replace(
        "el('type-filter').addEventListener('change',()=>{drawType();fillTables()});drawType();drawRank();drawEac();drawCost();fillTables();",
        "function fillAnalysis(){const list=el('analysis-list');const sorted=DATA.by_type.filter(x=>x.c!=null).sort((a,b)=>b.c-a.c);const strongest=sorted[0];const weakest=sorted[sorted.length-1];const gap=DATA.partial.c-DATA.partial.epath;const items=[`已保存 <strong>${DATA.status.completed.toLocaleString()}</strong> 条，覆盖 <strong>${fmt(DATA.status.completed/DATA.status.target*100,1)}%</strong>；有效评分 ${DATA.partial.scored_count.toLocaleString()} 条。`,`整体 C_avg 为 <strong>${fmt(DATA.partial.c)}%</strong>，E_path 为 <strong>${fmt(DATA.partial.epath)}%</strong>，路径执行使平均分降低约 <strong>${fmt(gap)} 个百分点</strong>。`,strongest?`类型表现最高：<strong>${strongest.type}</strong>（C_avg ${fmt(strongest.c)}%）；最低：<strong>${weakest.type}</strong>（${fmt(weakest.c)}%）。`:'' ,`已保存结果累计 API 成本 <strong>${money(DATA.analysis.saved_cost)}</strong>；全量已发出请求累计 <strong>${money(DATA.api_total.cost)}</strong>。`];list.innerHTML=items.filter(Boolean).map(x=>`<li>${x}</li>`).join('')}el('type-filter').addEventListener('change',()=>{drawType();fillTables()});drawType();drawRank();drawEac();drawCost();fillTables();fillAnalysis();",
    )
    REPORT_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    data = make_data()
    refresh_html(data)
    print(json.dumps({
        "report": str(REPORT_PATH),
        "state": data["status"]["state"],
        "completed": data["status"]["completed"],
        "target": data["status"]["target"],
        "partial": data["partial"],
        "api_total": data["api_total"],
        "invalid_scores": data["invalid_scores"],
    }, ensure_ascii=False, indent=2))
