"""
Deep analysis for qwen3-0b6-cpt-sft model.
Generates a detailed HTML report covering:
  1. Error analysis per task (MC, NLI, Syllogism)
  2. Thinking vs Direct inference comparison
  3. CPT → SFT pipeline evaluation
"""
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

RAW_DIR = Path(r"E:\DoCode\1 VN-Legal-AI\01_Report_CD1_DistillationLaw\data_review\raw")
MODEL_DIR = RAW_DIR / "qwen3-0b6-cpt-sft"
CPT_EVAL = RAW_DIR / "eval_cpt_qwen3-0b6-lora_20260521_141451.json"
REPORT_PATH = Path(r"E:\DoCode\1 VN-Legal-AI\01_Report_CD1_DistillationLaw\data_review\report\deep_analysis_qwen3_0b6.html")

ALL_MODEL_DIRS = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def f4(v):
    return f"{v:.4f}"


# ── MC analysis ──────────────────────────────────────────────
def analyze_mc(data):
    samples = data["per_sample"]
    total = len(samples)
    null_preds = [s for s in samples if s["pred"] is None]
    non_null = [s for s in samples if s["pred"] is not None]
    correct = [s for s in samples if s["correct"]]

    gold_dist = Counter(s["gold"] for s in samples)
    pred_dist = Counter(s["pred"] for s in non_null)
    correct_by_gold = Counter(s["gold"] for s in correct)

    thinking_samples = [s for s in samples if s["thinking"]]
    direct_samples = [s for s in samples if not s["thinking"]]

    thinking_null = sum(1 for s in thinking_samples if s["pred"] is None)
    direct_null = sum(1 for s in direct_samples if s["pred"] is None)

    wrong_non_null = [s for s in non_null if not s["correct"]]
    wrong_pred_dist = Counter(s["pred"] for s in wrong_non_null)
    wrong_gold_dist = Counter(s["gold"] for s in wrong_non_null)

    confusion = {}
    for s in non_null:
        key = (s["gold"], s["pred"])
        confusion[key] = confusion.get(key, 0) + 1

    return {
        "total": total,
        "null_count": len(null_preds),
        "null_rate": len(null_preds) / total,
        "non_null_count": len(non_null),
        "correct_count": len(correct),
        "accuracy": len(correct) / total,
        "accuracy_non_null": len(correct) / len(non_null) if non_null else 0,
        "gold_dist": dict(sorted(gold_dist.items())),
        "pred_dist": dict(sorted(pred_dist.items())),
        "correct_by_gold": dict(sorted(correct_by_gold.items())),
        "thinking_n": len(thinking_samples),
        "thinking_null": thinking_null,
        "thinking_null_rate": thinking_null / len(thinking_samples) if thinking_samples else 0,
        "thinking_acc": sum(1 for s in thinking_samples if s["correct"]) / len(thinking_samples) if thinking_samples else 0,
        "direct_n": len(direct_samples),
        "direct_null": direct_null,
        "direct_null_rate": direct_null / len(direct_samples) if direct_samples else 0,
        "direct_acc": sum(1 for s in direct_samples if s["correct"]) / len(direct_samples) if direct_samples else 0,
        "wrong_pred_dist": dict(sorted(wrong_pred_dist.items())),
        "wrong_gold_dist": dict(sorted(wrong_gold_dist.items())),
        "confusion": confusion,
    }


# ── NLI analysis ─────────────────────────────────────────────
def analyze_nli(data):
    samples = data["per_sample"]
    total = len(samples)

    gold_dist = Counter(s["gold"] for s in samples)
    pred_dist = Counter(s["pred"] for s in samples)

    tp = sum(1 for s in samples if s["gold"] == "relevant" and s["pred"] == "relevant")
    fp = sum(1 for s in samples if s["gold"] == "not_relevant" and s["pred"] == "relevant")
    fn = sum(1 for s in samples if s["gold"] == "relevant" and s["pred"] != "relevant")
    tn = sum(1 for s in samples if s["gold"] == "not_relevant" and s["pred"] != "relevant")

    precision_rel = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_rel = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_rel = 2 * precision_rel * recall_rel / (precision_rel + recall_rel) if (precision_rel + recall_rel) > 0 else 0

    precision_nr = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_nr = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_nr = 2 * precision_nr * recall_nr / (precision_nr + recall_nr) if (precision_nr + recall_nr) > 0 else 0

    thinking = [s for s in samples if s["thinking"]]
    direct = [s for s in samples if not s["thinking"]]

    def nli_sub(subset):
        n = len(subset)
        gold_r = sum(1 for s in subset if s["gold"] == "relevant")
        gold_nr = n - gold_r
        pred_r = sum(1 for s in subset if s["pred"] == "relevant")
        pred_nr = sum(1 for s in subset if s["pred"] == "not_relevant")
        pred_unk = sum(1 for s in subset if s["pred"] not in ("relevant", "not_relevant"))
        acc = sum(1 for s in subset if s["correct"]) / n if n else 0
        sub_tp = sum(1 for s in subset if s["gold"] == "relevant" and s["pred"] == "relevant")
        sub_fp = sum(1 for s in subset if s["gold"] == "not_relevant" and s["pred"] == "relevant")
        sub_fn = sum(1 for s in subset if s["gold"] == "relevant" and s["pred"] != "relevant")
        sub_tn = sum(1 for s in subset if s["gold"] == "not_relevant" and s["pred"] != "relevant")
        return {
            "n": n, "gold_relevant": gold_r, "gold_not_relevant": gold_nr,
            "pred_relevant": pred_r, "pred_not_relevant": pred_nr, "pred_unknown": pred_unk,
            "accuracy": acc, "tp": sub_tp, "fp": sub_fp, "fn": sub_fn, "tn": sub_tn,
        }

    return {
        "total": total,
        "gold_dist": dict(sorted(gold_dist.items())),
        "pred_dist": dict(sorted(pred_dist.items())),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_relevant": precision_rel,
        "recall_relevant": recall_rel,
        "f1_relevant": f1_rel,
        "precision_not_relevant": precision_nr,
        "recall_not_relevant": recall_nr,
        "f1_not_relevant": f1_nr,
        "macro_f1": (f1_rel + f1_nr) / 2,
        "thinking": nli_sub(thinking),
        "direct": nli_sub(direct),
    }


# ── Syllogism analysis ───────────────────────────────────────
def analyze_syllogism(data):
    samples = data["per_sample"]
    total = len(samples)
    scores = [s["rougeL"] for s in samples]
    scores_sorted = sorted(scores)

    buckets = {"0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.4": 0,
               "0.4-0.5": 0, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8+": 0}
    for v in scores:
        if v < 0.1: buckets["0.0-0.1"] += 1
        elif v < 0.2: buckets["0.1-0.2"] += 1
        elif v < 0.3: buckets["0.2-0.3"] += 1
        elif v < 0.3: buckets["0.2-0.3"] += 1
        elif v < 0.4: buckets["0.3-0.4"] += 1
        elif v < 0.5: buckets["0.4-0.5"] += 1
        elif v < 0.6: buckets["0.5-0.6"] += 1
        elif v < 0.7: buckets["0.6-0.7"] += 1
        elif v < 0.8: buckets["0.7-0.8"] += 1
        else: buckets["0.8+"] += 1

    bottom_10 = sorted(samples, key=lambda s: s["rougeL"])[:10]
    top_10 = sorted(samples, key=lambda s: -s["rougeL"])[:10]

    thinking = [s for s in samples if s["thinking"]]
    direct = [s for s in samples if not s["thinking"]]

    def syllo_sub(subset):
        n = len(subset)
        sc = [s["rougeL"] for s in subset]
        return {
            "n": n,
            "mean": mean(sc) if sc else 0,
            "median": median(sc) if sc else 0,
            "stdev": stdev(sc) if len(sc) > 1 else 0,
            "min": min(sc) if sc else 0,
            "max": max(sc) if sc else 0,
        }

    return {
        "total": total,
        "mean": mean(scores),
        "median": median(scores),
        "stdev": stdev(scores),
        "min": min(scores),
        "max": max(scores),
        "p10": scores_sorted[int(0.1 * len(scores_sorted))],
        "p25": scores_sorted[int(0.25 * len(scores_sorted))],
        "p75": scores_sorted[int(0.75 * len(scores_sorted))],
        "p90": scores_sorted[int(0.9 * len(scores_sorted))],
        "buckets": buckets,
        "bottom_10": bottom_10,
        "top_10": top_10,
        "thinking": syllo_sub(thinking),
        "direct": syllo_sub(direct),
    }


# ── Cross-model comparison ───────────────────────────────────
def load_all_summaries():
    results = []
    for d in ALL_MODEL_DIRS:
        sp = d / "summary.json"
        if sp.exists():
            s = load(sp)
            s["model_id"] = d.name
            results.append(s)
    return results


# ── HTML generation ───────────────────────────────────────────
def bar_cell(value, max_val=1.0, color="#f97316"):
    w = min(value / max_val * 100, 100)
    return (
        f'<div style="position:relative;height:22px;background:rgba(148,163,184,0.1);border-radius:4px">'
        f'<div style="width:{w:.1f}%;height:100%;background:{color};border-radius:4px;opacity:0.7"></div>'
        f'<span style="position:absolute;left:6px;top:1px;font-size:0.82rem">{value:.3f}</span>'
        f'</div>'
    )


def build_html(mc, nli, syllo, cpt_eval, all_summaries):
    # ── MC section ──
    mc_gold_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td>{mc['correct_by_gold'].get(k, 0)}</td>"
        f"<td>{pct(mc['correct_by_gold'].get(k, 0) / v) if v else '-'}</td></tr>"
        for k, v in mc["gold_dist"].items()
    )
    mc_pred_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in mc["pred_dist"].items()
    )
    mc_confusion_rows = ""
    for gold_label in ["A", "B", "C", "D"]:
        cells = ""
        for pred_label in ["A", "B", "C", "D"]:
            val = mc["confusion"].get((gold_label, pred_label), 0)
            bg = "rgba(34,197,94,0.3)" if gold_label == pred_label and val > 0 else (
                "rgba(239,68,68,0.2)" if val > 0 else ""
            )
            cells += f'<td style="background:{bg};text-align:center">{val}</td>'
        mc_confusion_rows += f"<tr><td><strong>{gold_label}</strong></td>{cells}</tr>"

    # ── NLI section ──
    nli_cm = f"""
    <table>
    <thead><tr><th></th><th>Pred: relevant</th><th>Pred: not_relevant</th><th>Pred: unknown</th></tr></thead>
    <tbody>
    <tr><td><strong>Gold: relevant</strong></td>
        <td style="background:rgba(34,197,94,0.3);text-align:center">{nli['tp']}</td>
        <td style="text-align:center">{nli['fn']}</td>
        <td style="text-align:center">0</td></tr>
    <tr><td><strong>Gold: not_relevant</strong></td>
        <td style="background:rgba(239,68,68,0.2);text-align:center">{nli['fp']}</td>
        <td style="background:rgba(34,197,94,0.3);text-align:center">{nli['tn']}</td>
        <td style="text-align:center">{nli['total'] - nli['tp'] - nli['fp'] - nli['fn'] - nli['tn']}</td></tr>
    </tbody></table>
    """

    # ── Syllogism histogram (CSS bars) ──
    syllo_hist = ""
    max_bucket = max(syllo["buckets"].values()) if syllo["buckets"] else 1
    for rng, cnt in syllo["buckets"].items():
        w = cnt / max_bucket * 100 if max_bucket else 0
        syllo_hist += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<span style="width:60px;font-size:0.82rem;text-align:right;color:#94a3b8">{rng}</span>'
            f'<div style="flex:1;height:20px;background:rgba(148,163,184,0.1);border-radius:4px">'
            f'<div style="width:{w:.1f}%;height:100%;background:#38bdf8;border-radius:4px;opacity:0.7"></div>'
            f'</div>'
            f'<span style="width:24px;font-size:0.82rem">{cnt}</span>'
            f'</div>'
        )

    syllo_bottom = "".join(
        f"<tr><td>{s['id']}</td><td>{'T' if s['thinking'] else 'D'}</td>"
        f"<td>{bar_cell(s['rougeL'], color='#ef4444')}</td></tr>"
        for s in syllo["bottom_10"]
    )
    syllo_top = "".join(
        f"<tr><td>{s['id']}</td><td>{'T' if s['thinking'] else 'D'}</td>"
        f"<td>{bar_cell(s['rougeL'], color='#22c55e')}</td></tr>"
        for s in syllo["top_10"]
    )

    # ── Cross-model comparison ──
    cross_rows = ""
    for s in all_summaries:
        is_current = s["model_id"] == "qwen3-0b6-cpt-sft"
        highlight = "background:rgba(249,115,22,0.15);" if is_current else ""
        marker = " *" if is_current else ""
        cross_rows += (
            f'<tr style="{highlight}">'
            f'<td><strong>{s["model_id"]}{marker}</strong></td>'
            f'<td>{pct(s.get("nli_accuracy", 0))}</td>'
            f'<td>{pct(s.get("mc_accuracy", 0))}</td>'
            f'<td>{f4(s.get("syllo_rougeL", 0))}</td>'
            f'<td>{f4(s.get("syllo_rouge1", 0))}</td>'
            f'</tr>'
        )

    # ── Thinking vs Direct summary table ──
    td_nli = nli["thinking"]
    dd_nli = nli["direct"]
    td_syllo = syllo["thinking"]
    dd_syllo = syllo["direct"]

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Deep Analysis — qwen3-0b6-cpt-sft</title>
<style>
:root {{
    --bg:#0f172a; --panel:#111827; --text:#e2e8f0; --muted:#94a3b8;
    --accent:#f97316; --accent2:#38bdf8; --green:#22c55e; --red:#ef4444;
}}
* {{ box-sizing: border-box; }}
body {{
    margin:0; font-family:"IBM Plex Sans","Segoe UI",sans-serif;
    background:radial-gradient(circle at top,#1e293b,#0b1120 55%,#020617);
    color:var(--text); line-height:1.6;
}}
header {{ padding:40px 24px 12px; text-align:center; }}
header h1 {{ margin:0; font-size:2.2rem; letter-spacing:0.02em; }}
header p {{ margin-top:6px; color:var(--muted); font-size:0.95rem; }}
main {{ max-width:1100px; margin:0 auto; padding:0 24px 60px; display:flex; flex-direction:column; gap:28px; }}
.card {{
    background:linear-gradient(135deg,rgba(15,23,42,0.92),rgba(2,6,23,0.92));
    border:1px solid rgba(148,163,184,0.18); border-radius:16px;
    padding:24px; box-shadow:0 10px 30px rgba(15,23,42,0.5);
}}
h2 {{ margin:0 0 16px; font-size:1.35rem; color:var(--accent); }}
h3 {{ margin:20px 0 10px; color:var(--accent2); font-size:1.1rem; }}
h4 {{ margin:16px 0 8px; color:var(--muted); font-size:0.95rem; text-transform:uppercase; letter-spacing:0.06em; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:0.9rem; }}
th,td {{ padding:9px 12px; border-bottom:1px solid rgba(148,163,184,0.15); text-align:left; }}
th {{ color:var(--muted); font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.07em; }}
tbody tr:hover {{ background:rgba(148,163,184,0.06); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; }}
.metric-box {{
    background:rgba(148,163,184,0.06); border-radius:12px; padding:16px 20px; text-align:center;
}}
.metric-box .value {{ font-size:2rem; font-weight:700; }}
.metric-box .label {{ font-size:0.82rem; color:var(--muted); margin-top:4px; }}
.tag {{ display:inline-block; padding:2px 10px; border-radius:6px; font-size:0.82rem; font-weight:600; }}
.tag-red {{ background:rgba(239,68,68,0.2); color:#fca5a5; }}
.tag-yellow {{ background:rgba(234,179,8,0.2); color:#fde047; }}
.tag-green {{ background:rgba(34,197,94,0.2); color:#86efac; }}
.tag-blue {{ background:rgba(56,189,248,0.2); color:#7dd3fc; }}
.insight {{
    background:rgba(249,115,22,0.08); border-left:3px solid var(--accent);
    padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0; font-size:0.92rem;
}}
.insight strong {{ color:var(--accent); }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media (max-width:700px) {{ .two-col,.grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
    <h1>Deep Analysis: qwen3-0b6-cpt-sft</h1>
    <p>Qwen3-0.6B Base → CPT (LoRA) → SFT | ViLawQA Benchmark</p>
</header>
<main>

<!-- ═══ OVERVIEW ═══ -->
<section class="card">
    <h2>1. Overview</h2>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr))">
        <div class="metric-box">
            <div class="value" style="color:var(--red)">{pct(mc['accuracy'])}</div>
            <div class="label">MC Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--accent)">{pct(nli['tp'] + nli['tn'], d=0)}/{nli['total']}</div>
            <div class="label">NLI Accuracy ({pct((nli['tp']+nli['tn'])/nli['total'])})</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--accent2)">0.{str(round(syllo['mean']*1000)).zfill(3)}</div>
            <div class="label">Syllogism ROUGE-L</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--muted)">{f4(cpt_eval['perplexity'])}</div>
            <div class="label">CPT Perplexity</div>
        </div>
    </div>
    <div class="insight">
        <strong>Tóm tắt:</strong> Model 0.6B sau CPT+SFT cho thấy khả năng language modeling tốt (PPL=2.19) nhưng
        downstream task performance rất hạn chế — đặc biệt MC (10%) và NLI bị bias nghiêm trọng.
        Đây là pattern điển hình của model nhỏ: học được surface-level patterns nhưng thiếu reasoning depth.
    </div>
</section>

<!-- ═══ MC ERROR ANALYSIS ═══ -->
<section class="card">
    <h2>2. Multiple Choice — Error Analysis</h2>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
        <div class="metric-box">
            <div class="value" style="color:var(--red)">{mc['null_count']}/{mc['total']}</div>
            <div class="label">Null predictions ({pct(mc['null_rate'])})</div>
        </div>
        <div class="metric-box">
            <div class="value">{mc['non_null_count']}</div>
            <div class="label">Có prediction</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--green)">{mc['correct_count']}</div>
            <div class="label">Đúng</div>
        </div>
        <div class="metric-box">
            <div class="value">{pct(mc['accuracy_non_null'])}</div>
            <div class="label">Acc (chỉ non-null)</div>
        </div>
    </div>

    <div class="insight">
        <strong>Vấn đề #1 — Format Compliance:</strong> 68% câu trả lời là null, nghĩa là model sinh ra text
        nhưng không tuân thủ format yêu cầu (chỉ trả lời A/B/C/D). Đây là vấn đề SFT instruction-following,
        không phải knowledge. Model 0.6B thiếu capacity để vừa reasoning vừa format output.
    </div>

    <h3>2.1 Phân bố Gold vs Predicted labels</h3>
    <div class="two-col">
        <div>
            <h4>Gold label distribution</h4>
            <table>
            <thead><tr><th>Label</th><th>Count</th><th>Correct</th><th>Acc per label</th></tr></thead>
            <tbody>{mc_gold_rows}</tbody>
            </table>
        </div>
        <div>
            <h4>Predicted distribution (non-null only)</h4>
            <table>
            <thead><tr><th>Pred</th><th>Count</th></tr></thead>
            <tbody>{mc_pred_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="insight">
        <strong>Vấn đề #2 — Position Bias (C-bias):</strong> Trong {mc['non_null_count']} predictions non-null,
        model dự đoán "{max(mc['pred_dist'], key=mc['pred_dist'].get)}" chiếm {mc['pred_dist'].get(max(mc['pred_dist'], key=mc['pred_dist'].get), 0)}/{mc['non_null_count']} lần
        ({pct(mc['pred_dist'].get(max(mc['pred_dist'], key=mc['pred_dist'].get), 0)/mc['non_null_count'])}).
        Model chưa bao giờ đúng khi dự đoán B hoặc D — chỉ đúng với A ({mc['correct_by_gold'].get('A', 0)} lần)
        và C ({mc['correct_by_gold'].get('C', 0)} lần).
    </div>

    <h3>2.2 Confusion Matrix (non-null predictions)</h3>
    <table>
    <thead><tr><th>Gold ↓ / Pred →</th><th>A</th><th>B</th><th>C</th><th>D</th></tr></thead>
    <tbody>{mc_confusion_rows}</tbody>
    </table>
</section>

<!-- ═══ NLI ERROR ANALYSIS ═══ -->
<section class="card">
    <h2>3. NLI — Error Analysis</h2>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
        <div class="metric-box">
            <div class="value" style="color:var(--accent)">{pct((nli['tp']+nli['tn'])/nli['total'])}</div>
            <div class="label">Overall Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--red)">{pct(nli['macro_f1'])}</div>
            <div class="label">Macro F1</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--green)">{pct(nli['recall_relevant'])}</div>
            <div class="label">Recall (relevant)</div>
        </div>
        <div class="metric-box">
            <div class="value" style="color:var(--red)">{pct(nli['recall_not_relevant'])}</div>
            <div class="label">Recall (not_relevant)</div>
        </div>
    </div>

    <h3>3.1 Confusion Matrix</h3>
    {nli_cm}

    <div class="insight">
        <strong>Vấn đề chính — Majority-class Collapse:</strong> Model predict "relevant" cho {nli['pred_dist'].get('relevant', 0)}/{nli['total']} mẫu
        ({pct(nli['pred_dist'].get('relevant', 0)/nli['total'])}).
        Chỉ {nli['tn']} mẫu được predict đúng là "not_relevant" (recall = {pct(nli['recall_not_relevant'])}).
        Đây là classifier suy biến — accuracy cao giả tạo vì dataset có {nli['gold_dist'].get('relevant', 0)}/{nli['total']}
        mẫu relevant ({pct(nli['gold_dist'].get('relevant', 0)/nli['total'])}).
    </div>

    <h3>3.2 Per-class Metrics</h3>
    <table>
    <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>
    <tr><td>relevant</td><td>{pct(nli['precision_relevant'])}</td><td>{pct(nli['recall_relevant'])}</td>
        <td>{pct(nli['f1_relevant'])}</td><td>{nli['gold_dist'].get('relevant', 0)}</td></tr>
    <tr><td>not_relevant</td><td>{pct(nli['precision_not_relevant'])}</td><td>{pct(nli['recall_not_relevant'])}</td>
        <td>{pct(nli['f1_not_relevant'])}</td><td>{nli['gold_dist'].get('not_relevant', 0)}</td></tr>
    <tr style="border-top:2px solid rgba(148,163,184,0.3)"><td><strong>Macro avg</strong></td>
        <td>{pct((nli['precision_relevant']+nli['precision_not_relevant'])/2)}</td>
        <td>{pct((nli['recall_relevant']+nli['recall_not_relevant'])/2)}</td>
        <td>{pct(nli['macro_f1'])}</td><td>{nli['total']}</td></tr>
    </tbody></table>
</section>

<!-- ═══ SYLLOGISM ANALYSIS ═══ -->
<section class="card">
    <h2>4. Syllogism — ROUGE-L Distribution</h2>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr))">
        <div class="metric-box">
            <div class="value" style="color:var(--accent2)">{f4(syllo['mean'])}</div>
            <div class="label">Mean ROUGE-L</div>
        </div>
        <div class="metric-box">
            <div class="value">{f4(syllo['median'])}</div>
            <div class="label">Median</div>
        </div>
        <div class="metric-box">
            <div class="value">{f4(syllo['stdev'])}</div>
            <div class="label">Std Dev</div>
        </div>
        <div class="metric-box">
            <div class="value">{f4(syllo['p10'])} — {f4(syllo['p90'])}</div>
            <div class="label">P10 — P90</div>
        </div>
    </div>

    <h3>4.1 ROUGE-L Histogram</h3>
    <div style="max-width:500px">
    {syllo_hist}
    </div>

    <div class="insight">
        <strong>Phân tích:</strong> Phân bố ROUGE-L có variance cao (std={f4(syllo['stdev'])}), cho thấy model
        rất không ổn định. Có {syllo['buckets'].get('0.0-0.1', 0)} mẫu gần như fail hoàn toàn (&lt;0.1) nhưng cũng có
        {syllo['buckets'].get('0.7-0.8', 0) + syllo['buckets'].get('0.8+', 0)} mẫu xuất sắc (&gt;0.7).
        Model có thể sinh text pháp luật đúng khi câu hỏi gần với training data, nhưng hallucinate khi gặp patterns mới.
    </div>

    <h3>4.2 Bottom 10 (worst)</h3>
    <table>
    <thead><tr><th>ID</th><th>Mode</th><th>ROUGE-L</th></tr></thead>
    <tbody>{syllo_bottom}</tbody>
    </table>

    <h3>4.3 Top 10 (best)</h3>
    <table>
    <thead><tr><th>ID</th><th>Mode</th><th>ROUGE-L</th></tr></thead>
    <tbody>{syllo_top}</tbody>
    </table>
</section>

<!-- ═══ THINKING VS DIRECT ═══ -->
<section class="card">
    <h2>5. Thinking vs Direct — So sánh</h2>

    <table>
    <thead><tr><th>Task</th><th>Metric</th><th>Thinking</th><th>Direct</th><th>Delta</th><th>Verdict</th></tr></thead>
    <tbody>
    <tr><td>MC</td><td>Accuracy</td><td>{pct(mc['thinking_acc'])}</td><td>{pct(mc['direct_acc'])}</td>
        <td style="color:{'var(--green)' if mc['direct_acc']>mc['thinking_acc'] else 'var(--red)'}">
        {'+' if mc['direct_acc']-mc['thinking_acc']>=0 else ''}{pct(mc['direct_acc']-mc['thinking_acc'])}</td>
        <td><span class="tag tag-green">Direct tốt hơn</span></td></tr>
    <tr><td>MC</td><td>Null rate</td><td>{pct(mc['thinking_null_rate'])}</td><td>{pct(mc['direct_null_rate'])}</td>
        <td>{pct(mc['direct_null_rate']-mc['thinking_null_rate'])}</td>
        <td><span class="tag tag-yellow">Tương đương</span></td></tr>
    <tr><td>NLI</td><td>Accuracy</td><td>{pct(td_nli['accuracy'])}</td><td>{pct(dd_nli['accuracy'])}</td>
        <td>{'+' if td_nli['accuracy']>dd_nli['accuracy'] else ''}{pct(td_nli['accuracy']-dd_nli['accuracy'])}</td>
        <td><span class="tag tag-blue">Thinking nhỉnh hơn</span></td></tr>
    <tr><td>NLI</td><td>Bias ratio</td>
        <td>{td_nli['pred_relevant']}/{td_nli['n']} ({pct(td_nli['pred_relevant']/td_nli['n'])})</td>
        <td>{dd_nli['pred_relevant']}/{dd_nli['n']} ({pct(dd_nli['pred_relevant']/dd_nli['n'])})</td>
        <td>-</td>
        <td><span class="tag tag-red">Cả hai bias</span></td></tr>
    <tr><td>Syllogism</td><td>ROUGE-L mean</td><td>{f4(td_syllo['mean'])}</td><td>{f4(dd_syllo['mean'])}</td>
        <td>{'+' if dd_syllo['mean']>td_syllo['mean'] else ''}{f4(dd_syllo['mean']-td_syllo['mean'])}</td>
        <td><span class="tag tag-yellow">Tương đương</span></td></tr>
    <tr><td>Syllogism</td><td>ROUGE-L stdev</td><td>{f4(td_syllo['stdev'])}</td><td>{f4(dd_syllo['stdev'])}</td>
        <td>-</td>
        <td><span class="tag tag-yellow">Cả hai biến động</span></td></tr>
    </tbody></table>

    <div class="insight">
        <strong>Kết luận Thinking vs Direct:</strong>
        Với model 0.6B, thinking mode <strong>không giúp ích</strong> — thậm chí có hại ở MC (null rate cao hơn,
        model "nghĩ" dài nhưng không extract được answer). Ở NLI, thinking tăng nhẹ accuracy nhưng cả hai mode đều
        bị majority-class collapse. Syllogism gần như không khác biệt. Model quá nhỏ để hưởng lợi từ chain-of-thought —
        thinking tokens chiếm capacity mà không cải thiện reasoning.
    </div>
</section>

<!-- ═══ CPT → SFT PIPELINE ═══ -->
<section class="card">
    <h2>6. Đánh giá Pipeline CPT → SFT</h2>

    <h3>6.1 CPT Quality</h3>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">
        <div class="metric-box">
            <div class="value" style="color:var(--green)">{cpt_eval['perplexity']}</div>
            <div class="label">Perplexity</div>
        </div>
        <div class="metric-box">
            <div class="value">{cpt_eval['eval_loss']}</div>
            <div class="label">Eval Loss</div>
        </div>
        <div class="metric-box">
            <div class="value">{cpt_eval['total_tokens']:,}</div>
            <div class="label">Total Tokens</div>
        </div>
    </div>

    <div class="insight">
        <strong>CPT đánh giá:</strong> Perplexity 2.19 là tốt — model đã học được language patterns của corpus pháp luật VN.
        Tuy nhiên, PPL thấp trên CPT eval set không đảm bảo downstream performance. Khoảng cách lớn giữa
        language modeling capability (PPL=2.19) và task performance (MC=10%, NLI collapse) cho thấy <strong>bottleneck
        nằm ở SFT, không phải CPT</strong>. Model cần:
        (1) Nhiều SFT data đa dạng hơn, đặc biệt cho MC format compliance;
        (2) Balanced NLI training data;
        (3) Có thể cần model lớn hơn cho reasoning tasks.
    </div>

    <h3>6.2 So sánh với các model khác</h3>
    <table>
    <thead><tr><th>Model</th><th>NLI Acc</th><th>MC Acc</th><th>Syllo ROUGE-L</th><th>Syllo ROUGE-1</th></tr></thead>
    <tbody>{cross_rows}</tbody>
    </table>

    <div class="insight">
        <strong>So sánh:</strong> qwen3-0b6-cpt-sft là model yếu nhất trong 4 variants. Đáng chú ý,
        <strong>qwen3-0b6-cpt-qlora-sft</strong> (cùng base model, khác training method) đạt MC=45% vs 10% —
        cho thấy vấn đề có thể nằm ở hyperparameters hoặc SFT data configuration, không chỉ model size.
        qwen3-1b7-cpt-sft (model lớn hơn 3x) cho kết quả tốt nhất ở syllogism (0.5952 ROUGE-L),
        xác nhận model size quan trọng cho generative reasoning.
    </div>
</section>

<!-- ═══ RECOMMENDATIONS ═══ -->
<section class="card">
    <h2>7. Recommendations</h2>
    <table>
    <thead><tr><th>#</th><th>Vấn đề</th><th>Mức độ</th><th>Giải pháp đề xuất</th></tr></thead>
    <tbody>
    <tr><td>1</td><td>MC null rate 68%</td><td><span class="tag tag-red">Critical</span></td>
        <td>Thêm SFT examples với strict format enforcement. Cân nhắc constrained decoding (force output A/B/C/D).
            So sánh với qwen3-0b6-cpt-qlora-sft để tìm config khác biệt.</td></tr>
    <tr><td>2</td><td>NLI majority-class collapse</td><td><span class="tag tag-red">Critical</span></td>
        <td>Cân bằng training data (upsample "not_relevant" hoặc downsample "relevant").
            Thêm hard negatives vào SFT data. Dùng focal loss hoặc class weights.</td></tr>
    <tr><td>3</td><td>MC C-position bias</td><td><span class="tag tag-yellow">Medium</span></td>
        <td>Augment SFT data bằng answer shuffling (đổi vị trí đáp án đúng random).
            Đảm bảo gold label distribution đều trong training set.</td></tr>
    <tr><td>4</td><td>Syllogism variance cao</td><td><span class="tag tag-yellow">Medium</span></td>
        <td>Phân tích mẫu ROUGE~0 để hiểu failure mode (hallucination? wrong topic? empty output?).
            Cần đọc actual output text để diagnostic thêm.</td></tr>
    <tr><td>5</td><td>Thinking mode không hiệu quả</td><td><span class="tag tag-blue">Info</span></td>
        <td>Với 0.6B params, disable thinking mode để tiết kiệm inference tokens.
            Thinking chỉ nên dùng cho model ≥1.7B.</td></tr>
    </tbody></table>
</section>

</main>
</body>
</html>"""
    return html


def main():
    mc_data = load(MODEL_DIR / "mc_results.json")
    nli_data = load(MODEL_DIR / "nli_results.json")
    syllo_data = load(MODEL_DIR / "syllogism_results.json")
    cpt_eval = load(CPT_EVAL)

    mc = analyze_mc(mc_data)
    nli = analyze_nli(nli_data)
    syllo = analyze_syllogism(syllo_data)
    all_summaries = load_all_summaries()

    html = build_html(mc, nli, syllo, cpt_eval, all_summaries)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
