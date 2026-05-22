import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

RAW_DIR = Path(r"E:\DoCode\1 VN-Legal-AI\01_Report_CD1_DistillationLaw\data_review\raw")
REPORT_DIR = Path(r"E:\DoCode\1 VN-Legal-AI\01_Report_CD1_DistillationLaw\data_review\report")
REPORT_PATH = REPORT_DIR / "raw_analysis.html"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def summarize_per_sample(task_data: Dict[str, Any]) -> Dict[str, Any]:
    per_sample = task_data.get("per_sample", []) or []
    if not isinstance(per_sample, list):
        return {}

    total = len(per_sample)
    if total == 0:
        return {
            "n": 0,
        }

    if task_data.get("task") in {"mc", "nli"}:
        unknown_count = 0
        incorrect_samples: List[Tuple[str, Optional[str], Optional[str]]] = []
        for row in per_sample:
            pred = row.get("pred")
            if pred is None or pred == "unknown":
                unknown_count += 1
            if row.get("correct") is False:
                incorrect_samples.append(
                    (
                        str(row.get("id", "")),
                        str(row.get("gold", "")),
                        str(pred) if pred is not None else None,
                    )
                )
        return {
            "n": total,
            "unknown_rate": unknown_count / total,
            "incorrect_samples": incorrect_samples[:10],
        }

    if task_data.get("task") == "syllogism":
        rouge_l = [row.get("rougeL") for row in per_sample if row.get("rougeL") is not None]
        if not rouge_l:
            return {"n": total}
        rouge_l_sorted = sorted(rouge_l)
        low_threshold = rouge_l_sorted[max(0, int(0.1 * len(rouge_l_sorted)) - 1)]
        high_threshold = rouge_l_sorted[min(len(rouge_l_sorted) - 1, int(0.9 * len(rouge_l_sorted)))]
        low_samples = [row for row in per_sample if row.get("rougeL", 1.0) <= low_threshold][:10]
        high_samples = [row for row in per_sample if row.get("rougeL", 0.0) >= high_threshold][:10]
        return {
            "n": total,
            "rougeL_mean": mean(rouge_l),
            "rougeL_p10": low_threshold,
            "rougeL_p90": high_threshold,
            "low_samples": low_samples,
            "high_samples": high_samples,
        }

    return {"n": total}


def load_model_folder(folder: Path) -> Dict[str, Any]:
    summary_path = folder / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    model_id = folder.name

    task_files = {
        "mc": folder / "mc_results.json",
        "nli": folder / "nli_results.json",
        "syllogism": folder / "syllogism_results.json",
    }
    task_data: Dict[str, Any] = {}
    for task, path in task_files.items():
        if path.exists():
            data = read_json(path)
            task_data[task] = data

    per_sample_summary = {
        task: summarize_per_sample(data) for task, data in task_data.items()
    }

    return {
        "model_id": model_id,
        "summary": summary,
        "tasks": task_data,
        "per_sample": per_sample_summary,
    }


def load_eval_files(raw_dir: Path) -> List[Dict[str, Any]]:
    eval_files = sorted(raw_dir.glob("eval_cpt_*.json"))
    entries = []
    for path in eval_files:
        data = read_json(path)
        adapter = data.get("adapter")
        model_id = None
        if isinstance(adapter, str) and "outputs/" in adapter:
            model_id = adapter.split("outputs/")[-1].split("/")[0]
        entries.append(
            {
                "file": path.name,
                "model_id": model_id or "unknown",
                "eval_loss": data.get("eval_loss"),
                "perplexity": data.get("perplexity"),
                "total_tokens": data.get("total_tokens"),
                "model": data.get("model"),
                "adapter": adapter,
                "config": data.get("config"),
                "timestamp": data.get("timestamp"),
            }
        )
    return entries


def build_table(headers: List[str], rows: List[List[str]]) -> str:
    thead = "".join(f"<th>{html_escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return "".join([
        "<table>",
        f"<thead><tr>{thead}</tr></thead>",
        "<tbody>",
        "".join(body_rows),
        "</tbody></table>",
    ])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_folders = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    models = [load_model_folder(folder) for folder in sorted(model_folders)]
    eval_entries = load_eval_files(RAW_DIR)

    summary_rows = []
    for model in models:
        summary = model["summary"]
        summary_rows.append([
            html_escape(model["model_id"]),
            html_escape(str(summary.get("adapter", "-"))),
            pct(summary.get("nli_accuracy")),
            pct(summary.get("mc_accuracy")),
            fmt_float(summary.get("syllo_rouge1")),
            fmt_float(summary.get("syllo_rougeL")),
        ])

    task_rows = []
    for model in models:
        for task_name, task_data in model["tasks"].items():
            overall = task_data.get("overall", {})
            by_thinking = task_data.get("by_thinking", {})
            thinking_true = by_thinking.get("thinking_true", {})
            thinking_false = by_thinking.get("thinking_false", {})

            if task_name in {"mc", "nli"}:
                task_rows.append([
                    html_escape(model["model_id"]),
                    task_name,
                    pct(overall.get("accuracy")),
                    pct(overall.get("f1")) if task_name == "nli" else "-",
                    str(overall.get("n", "-")),
                    pct(thinking_true.get("accuracy")),
                    pct(thinking_false.get("accuracy")),
                ])
            else:
                task_rows.append([
                    html_escape(model["model_id"]),
                    task_name,
                    fmt_float(overall.get("rouge1")),
                    fmt_float(overall.get("rouge2")),
                    fmt_float(overall.get("rougeL")),
                    str(overall.get("n", "-")),
                    fmt_float(thinking_true.get("rougeL")),
                    fmt_float(thinking_false.get("rougeL")),
                ])

    eval_rows = []
    for entry in eval_entries:
        config = entry.get("config")
        if isinstance(config, list):
            config_text = ", ".join(str(item) for item in config)
        else:
            config_text = str(config) if config is not None else "-"
        eval_rows.append([
            html_escape(entry["model_id"]),
            html_escape(entry["file"]),
            fmt_float(entry.get("eval_loss")),
            fmt_float(entry.get("perplexity")),
            str(entry.get("total_tokens", "-")),
            html_escape(str(entry.get("model", "-"))),
            html_escape(config_text),
            html_escape(str(entry.get("timestamp", "-"))),
        ])

    diagnostic_blocks = []
    for model in models:
        per_sample = model["per_sample"]
        blocks = []
        for task_name, stats in per_sample.items():
            if task_name in {"mc", "nli"}:
                unknown_rate = stats.get("unknown_rate")
                incorrect = stats.get("incorrect_samples", [])
                rows = []
                for sample_id, gold, pred in incorrect:
                    rows.append([html_escape(sample_id), html_escape(gold), html_escape(str(pred))])
                blocks.append(
                    "".join([
                        f"<h4>{html_escape(task_name.upper())}</h4>",
                        f"<p>Unknown rate: {pct(unknown_rate)} | Samples: {stats.get('n', '-') }</p>",
                        build_table(["id", "gold", "pred"], rows) if rows else "<p>No incorrect samples captured.</p>",
                    ])
                )
            elif task_name == "syllogism":
                low_samples = stats.get("low_samples", [])
                high_samples = stats.get("high_samples", [])
                low_rows = [[html_escape(row.get("id", "")), fmt_float(row.get("rougeL"))] for row in low_samples]
                high_rows = [[html_escape(row.get("id", "")), fmt_float(row.get("rougeL"))] for row in high_samples]
                blocks.append(
                    "".join([
                        "<h4>SYLLOGISM</h4>",
                        f"<p>RougeL mean: {fmt_float(stats.get('rougeL_mean'))}, p10: {fmt_float(stats.get('rougeL_p10'))}, p90: {fmt_float(stats.get('rougeL_p90'))}</p>",
                        "<p>Low RougeL samples</p>",
                        build_table(["id", "rougeL"], low_rows) if low_rows else "<p>No low samples.</p>",
                        "<p>High RougeL samples</p>",
                        build_table(["id", "rougeL"], high_rows) if high_rows else "<p>No high samples.</p>",
                    ])
                )
        diagnostic_blocks.append(
            "".join([
                f"<section class='card'><h3>{html_escape(model['model_id'])}</h3>",
                "".join(blocks) if blocks else "<p>No per-sample details.</p>",
                "</section>",
            ])
        )

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Raw Evaluation Analysis</title>
<style>
    :root {
        --bg: #0f172a;
        --panel: #111827;
        --panel-2: #0b1120;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --accent: #f97316;
        --accent-2: #38bdf8;
    }
    body {
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #1e293b, #0b1120 55%, #020617);
        color: var(--text);
    }
    header {
        padding: 32px 24px 16px;
        text-align: center;
    }
    header h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: 0.02em;
    }
    header p {
        margin-top: 8px;
        color: var(--muted);
    }
    main {
        max-width: 1100px;
        margin: 0 auto;
        padding: 0 24px 48px;
        display: flex;
        flex-direction: column;
        gap: 24px;
    }
    .card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(2, 6, 23, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.6);
    }
    h2 {
        margin-top: 0;
        font-size: 1.4rem;
        color: var(--accent);
    }
    h3 {
        margin-top: 0;
        color: var(--accent-2);
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        font-size: 0.92rem;
    }
    th, td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        text-align: left;
    }
    th {
        color: var(--muted);
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    tbody tr:hover {
        background: rgba(148, 163, 184, 0.08);
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
    }
</style>
</head>
<body>
<header>
    <h1>Raw Evaluation Analysis</h1>
    <p>Auto-generated from data_review/raw</p>
</header>
<main>
    <section class="card">
        <h2>Summary metrics (by model folder)</h2>
        {{summary_table}}
    </section>
    <section class="card">
        <h2>Task-level metrics</h2>
        {{task_table}}
    </section>
    <section class="card">
        <h2>CPT eval loss & perplexity</h2>
        {{eval_table}}
    </section>
    <section class="card">
        <h2>Diagnostics & sample-level signals</h2>
        <div class="grid">
            {{diagnostics}}
        </div>
    </section>
</main>
</body>
</html>
"""

    summary_table = build_table(
        ["model", "adapter", "nli_acc", "mc_acc", "syllo_r1", "syllo_rL"],
        summary_rows,
    )
    task_table = build_table(
        [
            "model",
            "task",
            "metric_1",
            "metric_2",
            "n",
            "thinking",
            "direct",
        ],
        task_rows,
    )
    eval_table = build_table(
        ["model", "file", "eval_loss", "ppl", "tokens", "model_name", "config", "timestamp"],
        eval_rows,
    )

    html = (
        html.replace("{{summary_table}}", summary_table)
        .replace("{{task_table}}", task_table)
        .replace("{{eval_table}}", eval_table)
        .replace("{{diagnostics}}", "".join(diagnostic_blocks) or "<p>No diagnostics available.</p>")
    )

    REPORT_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
