"""
Full cross-model analysis for VN-Legal-AI distillation experiment.
Covers 4 SFT models + 5 CPT checkpoints.

Usage:
    python data_review/src/full_analysis.py

Output:
    data_review/report/full_analysis.html
"""
import json
import html as html_mod
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "report" / "full_analysis.html"

# ── helpers ──────────────────────────────────────────────────
def load(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))

def pct(v: float, d: int = 1) -> str:
    return f"{v*100:.{d}f}%"

def f4(v: float) -> str:
    return f"{v:.4f}"

def esc(s: str) -> str:
    return html_mod.escape(str(s))


# ── per-model analysis ──────────────────────────────────────
def analyze_mc(data: dict) -> dict:
    samples = data["per_sample"]
    total = len(samples)
    null_preds = [s for s in samples if s["pred"] is None]
    non_null = [s for s in samples if s["pred"] is not None]
    correct = [s for s in samples if s["correct"]]

    gold_dist = Counter(s["gold"] for s in samples)
    pred_dist = Counter(s["pred"] for s in non_null)
    correct_by_gold = Counter(s["gold"] for s in correct)

    thinking = [s for s in samples if s["thinking"]]
    direct = [s for s in samples if not s["thinking"]]

    confusion: dict[tuple, int] = {}
    for s in non_null:
        k = (s["gold"], s["pred"])
        confusion[k] = confusion.get(k, 0) + 1

    return {
        "total": total,
        "null_count": len(null_preds),
        "null_rate": len(null_preds) / total,
        "non_null": len(non_null),
        "correct": len(correct),
        "acc": len(correct) / total,
        "acc_non_null": len(correct) / len(non_null) if non_null else 0,
        "gold_dist": dict(sorted(gold_dist.items())),
        "pred_dist": dict(sorted(pred_dist.items())),
        "correct_by_gold": dict(sorted(correct_by_gold.items())),
        "confusion": confusion,
        "think_n": len(thinking),
        "think_null": sum(1 for s in thinking if s["pred"] is None),
        "think_acc": sum(1 for s in thinking if s["correct"]) / len(thinking) if thinking else 0,
        "direct_n": len(direct),
        "direct_null": sum(1 for s in direct if s["pred"] is None),
        "direct_acc": sum(1 for s in direct if s["correct"]) / len(direct) if direct else 0,
    }


def analyze_nli(data: dict) -> dict:
    samples = data["per_sample"]
    total = len(samples)
    gold_dist = Counter(s["gold"] for s in samples)
    pred_dist = Counter(s["pred"] for s in samples)

    tp = sum(1 for s in samples if s["gold"] == "relevant" and s["pred"] == "relevant")
    fp = sum(1 for s in samples if s["gold"] == "not_relevant" and s["pred"] == "relevant")
    fn = sum(1 for s in samples if s["gold"] == "relevant" and s["pred"] != "relevant")
    tn = sum(1 for s in samples if s["gold"] == "not_relevant" and s["pred"] != "relevant")

    prec_r = tp / (tp + fp) if (tp + fp) else 0
    rec_r = tp / (tp + fn) if (tp + fn) else 0
    f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) else 0
    prec_nr = tn / (tn + fn) if (tn + fn) else 0
    rec_nr = tn / (tn + fp) if (tn + fp) else 0
    f1_nr = 2 * prec_nr * rec_nr / (prec_nr + rec_nr) if (prec_nr + rec_nr) else 0

    thinking = [s for s in samples if s["thinking"]]
    direct = [s for s in samples if not s["thinking"]]

    return {
        "total": total,
        "gold_dist": dict(sorted(gold_dist.items())),
        "pred_dist": dict(sorted(pred_dist.items())),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": (tp + tn) / total,
        "prec_r": prec_r, "rec_r": rec_r, "f1_r": f1_r,
        "prec_nr": prec_nr, "rec_nr": rec_nr, "f1_nr": f1_nr,
        "macro_f1": (f1_r + f1_nr) / 2,
        "bias_ratio": pred_dist.get("relevant", 0) / total,
        "think_n": len(thinking),
        "think_acc": sum(1 for s in thinking if s["correct"]) / len(thinking) if thinking else 0,
        "think_bias": sum(1 for s in thinking if s["pred"] == "relevant") / len(thinking) if thinking else 0,
        "direct_n": len(direct),
        "direct_acc": sum(1 for s in direct if s["correct"]) / len(direct) if direct else 0,
        "direct_bias": sum(1 for s in direct if s["pred"] == "relevant") / len(direct) if direct else 0,
    }


def analyze_syllo(data: dict) -> dict:
    samples = data["per_sample"]
    scores = sorted(s["rougeL"] for s in samples)
    n = len(scores)

    buckets = {}
    labels = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5","0.5-0.6","0.6-0.7","0.7-0.8","0.8+"]
    for lb in labels:
        buckets[lb] = 0
    for v in scores:
        if v < 0.1: buckets["0.0-0.1"] += 1
        elif v < 0.2: buckets["0.1-0.2"] += 1
        elif v < 0.3: buckets["0.2-0.3"] += 1
        elif v < 0.4: buckets["0.3-0.4"] += 1
        elif v < 0.5: buckets["0.4-0.5"] += 1
        elif v < 0.6: buckets["0.5-0.6"] += 1
        elif v < 0.7: buckets["0.6-0.7"] += 1
        elif v < 0.8: buckets["0.7-0.8"] += 1
        else: buckets["0.8+"] += 1

    thinking = [s for s in samples if s["thinking"]]
    direct = [s for s in samples if not s["thinking"]]

    bottom5 = sorted(samples, key=lambda s: s["rougeL"])[:5]
    top5 = sorted(samples, key=lambda s: -s["rougeL"])[:5]

    return {
        "total": n,
        "mean": mean(scores),
        "median": median(scores),
        "std": stdev(scores) if n > 1 else 0,
        "min": scores[0], "max": scores[-1],
        "p10": scores[int(0.1 * n)],
        "p90": scores[int(0.9 * n)],
        "buckets": buckets,
        "bottom5": bottom5,
        "top5": top5,
        "think_n": len(thinking),
        "think_mean": mean(s["rougeL"] for s in thinking) if thinking else 0,
        "direct_n": len(direct),
        "direct_mean": mean(s["rougeL"] for s in direct) if direct else 0,
    }


# ── load everything ──────────────────────────────────────────
def load_model(folder: Path) -> dict:
    mid = folder.name
    summary = load(folder / "summary.json") if (folder / "summary.json").exists() else {}
    mc_raw = load(folder / "mc_results.json") if (folder / "mc_results.json").exists() else None
    nli_raw = load(folder / "nli_results.json") if (folder / "nli_results.json").exists() else None
    syllo_raw = load(folder / "syllogism_results.json") if (folder / "syllogism_results.json").exists() else None
    return {
        "id": mid,
        "summary": summary,
        "mc": analyze_mc(mc_raw) if mc_raw else None,
        "nli": analyze_nli(nli_raw) if nli_raw else None,
        "syllo": analyze_syllo(syllo_raw) if syllo_raw else None,
    }


def load_cpt_evals() -> list[dict]:
    entries = []
    for p in sorted(RAW.glob("eval_cpt_*.json")):
        d = load(p)
        adapter = d.get("adapter", "")
        mid = adapter.split("outputs/")[-1].split("/")[0] if "outputs/" in adapter else p.stem
        entries.append({
            "file": p.name, "model_id": mid,
            "base_model": d.get("model", "?"),
            "loss": d.get("eval_loss"), "ppl": d.get("perplexity"),
            "tokens": d.get("total_tokens"),
            "config": d.get("config"), "timestamp": d.get("timestamp"),
        })
    return entries


# ── cross-model error overlap ───────────────────────────────
def compute_overlap(models: list[dict], task: str) -> dict:
    """Which sample IDs are wrong across all / most models."""
    if task in ("mc", "nli"):
        per_model_wrong: dict[str, set] = {}
        all_ids: set = set()
        for m in models:
            td = m.get(task)
            if not td:
                continue
            # we need raw per_sample — reload
            folder = RAW / m["id"]
            raw = load(folder / f"{task}_results.json")
            wrong = {s["id"] for s in raw["per_sample"] if not s["correct"]}
            per_model_wrong[m["id"]] = wrong
            all_ids |= {s["id"] for s in raw["per_sample"]}

        hard_for_all = set.intersection(*per_model_wrong.values()) if per_model_wrong else set()
        easy_for_all = all_ids - set.union(*per_model_wrong.values()) if per_model_wrong else set()

        return {
            "hard_count": len(hard_for_all),
            "easy_count": len(easy_for_all),
            "total": len(all_ids),
            "hard_ids": sorted(hard_for_all)[:15],
            "easy_ids": sorted(easy_for_all)[:15],
        }
    return {}


# ── HTML builder ─────────────────────────────────────────────
CSS = """
:root{--bg:#0f172a;--text:#e2e8f0;--muted:#94a3b8;--accent:#f97316;--accent2:#38bdf8;--green:#22c55e;--red:#ef4444;--yellow:#eab308}
*{box-sizing:border-box}
body{margin:0;font-family:"IBM Plex Sans","Segoe UI",sans-serif;background:radial-gradient(circle at top,#1e293b,#0b1120 55%,#020617);color:var(--text);line-height:1.6}
header{padding:40px 24px 12px;text-align:center}
header h1{margin:0;font-size:2.2rem;letter-spacing:.02em}
header .sub{margin-top:6px;color:var(--muted);font-size:.95rem}
main{max-width:1200px;margin:0 auto;padding:0 24px 60px;display:flex;flex-direction:column;gap:28px}
.card{background:linear-gradient(135deg,rgba(15,23,42,.92),rgba(2,6,23,.92));border:1px solid rgba(148,163,184,.18);border-radius:16px;padding:24px;box-shadow:0 10px 30px rgba(15,23,42,.5)}
h2{margin:0 0 16px;font-size:1.35rem;color:var(--accent)}
h3{margin:20px 0 10px;color:var(--accent2);font-size:1.1rem}
h4{margin:16px 0 8px;color:var(--muted);font-size:.95rem;text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:.9rem}
th,td{padding:9px 12px;border-bottom:1px solid rgba(148,163,184,.15);text-align:left}
th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.07em}
tbody tr:hover{background:rgba(148,163,184,.06)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}
.mb{background:rgba(148,163,184,.06);border-radius:12px;padding:16px 20px;text-align:center}
.mb .v{font-size:2rem;font-weight:700}
.mb .l{font-size:.82rem;color:var(--muted);margin-top:4px}
.tag{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.82rem;font-weight:600}
.tr{background:rgba(239,68,68,.2);color:#fca5a5}
.ty{background:rgba(234,179,8,.2);color:#fde047}
.tg{background:rgba(34,197,94,.2);color:#86efac}
.tb{background:rgba(56,189,248,.2);color:#7dd3fc}
.insight{background:rgba(249,115,22,.08);border-left:3px solid var(--accent);padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0;font-size:.92rem}
.insight strong{color:var(--accent)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.hl{background:rgba(249,115,22,.12)}
.best{background:rgba(34,197,94,.12)}
.worst{background:rgba(239,68,68,.1)}
.hist-row{display:flex;align-items:center;gap:8px;margin:3px 0}
.hist-label{width:60px;font-size:.82rem;text-align:right;color:#94a3b8}
.hist-bar-wrap{flex:1;height:18px;background:rgba(148,163,184,.1);border-radius:4px;overflow:hidden}
.hist-bar{height:100%;border-radius:4px;opacity:.75}
.hist-count{width:24px;font-size:.82rem}
@media(max-width:700px){.two-col,.grid{grid-template-columns:1fr}}
"""

BUCKET_COLORS = {
    "0.0-0.1": "#ef4444", "0.1-0.2": "#f97316", "0.2-0.3": "#eab308",
    "0.3-0.4": "#38bdf8", "0.4-0.5": "#38bdf8", "0.5-0.6": "#22c55e",
    "0.6-0.7": "#22c55e", "0.7-0.8": "#10b981", "0.8+": "#10b981",
}


def cell_color(val: float, best: float, worst: float, higher_better: bool = True) -> str:
    if val == best:
        return "background:rgba(34,197,94,.18)"
    if val == worst:
        return "background:rgba(239,68,68,.12)"
    return ""


def build_html(models: list[dict], cpts: list[dict],
               mc_overlap: dict, nli_overlap: dict) -> str:
    parts: list[str] = []

    # ── helper: metric box
    def mb(value: str, label: str, color: str = "var(--text)") -> str:
        return f'<div class="mb"><div class="v" style="color:{color}">{value}</div><div class="l">{label}</div></div>'

    # ── 1. OVERVIEW DASHBOARD ──
    parts.append('<section class="card"><h2>1. Overview Dashboard</h2>')
    parts.append('<table><thead><tr><th>Model</th><th>Size</th><th>MC Acc</th><th>MC Null%</th>'
                 '<th>NLI Acc</th><th>NLI Macro-F1</th><th>NLI Bias%</th>'
                 '<th>Syllo ROUGE-L</th><th>Syllo Std</th></tr></thead><tbody>')

    model_labels = {
        "qwen3-0b6-cpt-sft": ("0.6B LoRA", "var(--red)"),
        "qwen3-0b6-cpt-qlora-sft": ("0.6B QLoRA", "var(--green)"),
        "gwen3-1.7b-pretrain-vlsp-sft": ("1.7B VLSP-PT", "var(--accent2)"),
        "qwen3-1b7-cpt-sft": ("1.7B CPT-LoRA", "var(--accent)"),
    }

    mc_accs = [m["mc"]["acc"] for m in models if m["mc"]]
    nli_accs = [m["nli"]["acc"] for m in models if m["nli"]]
    syllo_means = [m["syllo"]["mean"] for m in models if m["syllo"]]
    best_mc, worst_mc = max(mc_accs), min(mc_accs)
    best_nli, worst_nli = max(nli_accs), min(nli_accs)
    best_syllo, worst_syllo = max(syllo_means), min(syllo_means)

    for m in models:
        mid = m["id"]
        lbl, _ = model_labels.get(mid, (mid, "var(--text)"))
        mc, nli, sy = m["mc"], m["nli"], m["syllo"]
        row_cls = ""
        if mc and mc["acc"] == best_mc:
            row_cls = ' class="best"'
        elif mc and mc["acc"] == worst_mc:
            row_cls = ' class="worst"'

        parts.append(f'<tr{row_cls}><td><strong>{esc(mid)}</strong></td><td>{lbl}</td>')
        if mc:
            parts.append(f'<td style="{cell_color(mc["acc"],best_mc,worst_mc)}">{pct(mc["acc"])}</td>')
            parts.append(f'<td>{pct(mc["null_rate"])}</td>')
        else:
            parts.append('<td>-</td><td>-</td>')
        if nli:
            parts.append(f'<td style="{cell_color(nli["acc"],best_nli,worst_nli)}">{pct(nli["acc"])}</td>')
            parts.append(f'<td>{pct(nli["macro_f1"])}</td>')
            parts.append(f'<td>{pct(nli["bias_ratio"])}</td>')
        else:
            parts.append('<td>-</td><td>-</td><td>-</td>')
        if sy:
            parts.append(f'<td style="{cell_color(sy["mean"],best_syllo,worst_syllo)}">{f4(sy["mean"])}</td>')
            parts.append(f'<td>{f4(sy["std"])}</td>')
        else:
            parts.append('<td>-</td><td>-</td>')
        parts.append('</tr>')

    parts.append('</tbody></table>')
    parts.append('<div class="insight"><strong>Ranking:</strong> '
                 'qwen3-0b6-cpt-qlora-sft (0.6B) dẫn đầu MC (45%) và NLI (71.6%). '
                 'qwen3-1b7-cpt-sft (1.7B) dẫn đầu Syllogism (0.595). '
                 'qwen3-0b6-cpt-sft (0.6B LoRA) yếu nhất ở tất cả tasks.</div>')
    parts.append('</section>')

    # ── 2. CPT PERPLEXITY ──
    parts.append('<section class="card"><h2>2. CPT Perplexity &amp; Scaling</h2>')
    parts.append('<table><thead><tr><th>Adapter</th><th>Base Model</th><th>Method</th>'
                 '<th>Eval Loss</th><th>Perplexity</th><th>Tokens</th></tr></thead><tbody>')
    for c in sorted(cpts, key=lambda x: x["ppl"]):
        method = "QLoRA" if "qlora" in c["model_id"] else "LoRA"
        parts.append(f'<tr><td>{esc(c["model_id"])}</td><td>{esc(c["base_model"])}</td>'
                     f'<td>{method}</td><td>{f4(c["loss"])}</td><td>{f4(c["ppl"])}</td>'
                     f'<td>{c["tokens"]:,}</td></tr>')
    parts.append('</tbody></table>')
    parts.append('<div class="insight"><strong>Scaling law:</strong> '
                 'PPL giảm rõ theo model size: 4B (1.83-1.86) &lt; 1.7B (1.96) &lt; 0.6B (2.19-2.24). '
                 'LoRA vs QLoRA gần như tương đương ở CPT (0.6B: 2.19 vs 2.24; 4B: 1.86 vs 1.83). '
                 'Tất cả dùng cùng corpus (745K tokens, SeaLLM config).</div>')
    parts.append('</section>')

    # ── 3. MC DEEP DIVE ──
    parts.append('<section class="card"><h2>3. Multiple Choice — Cross-model Deep Dive</h2>')

    # 3.1 Null rate + prediction distribution per model
    parts.append('<h3>3.1 Format Compliance (Null Rate)</h3>')
    parts.append('<table><thead><tr><th>Model</th><th>Null Rate</th><th>Non-null</th>'
                 '<th>Acc (overall)</th><th>Acc (non-null only)</th>'
                 '<th>Dominant Pred</th></tr></thead><tbody>')
    for m in models:
        mc = m["mc"]
        if not mc:
            continue
        dom = max(mc["pred_dist"], key=mc["pred_dist"].get) if mc["pred_dist"] else "-"
        dom_pct = mc["pred_dist"].get(dom, 0) / mc["non_null"] if mc["non_null"] else 0
        parts.append(f'<tr><td>{esc(m["id"])}</td><td>{pct(mc["null_rate"])}</td>'
                     f'<td>{mc["non_null"]}</td><td>{pct(mc["acc"])}</td>'
                     f'<td>{pct(mc["acc_non_null"])}</td>'
                     f'<td>{dom} ({pct(dom_pct)})</td></tr>')
    parts.append('</tbody></table>')

    parts.append('<div class="insight"><strong>Key finding:</strong> '
                 'qwen3-0b6-cpt-sft có null rate 68% — tệ nhất. '
                 'qwen3-0b6-cpt-qlora-sft chỉ 19% null — cùng base model nhưng SFT method khác biệt. '
                 'Điều này chứng minh format compliance là vấn đề SFT, không phải model capacity.</div>')

    # 3.2 Confusion matrices side by side
    parts.append('<h3>3.2 Confusion Matrices (non-null predictions)</h3>')
    parts.append('<div class="two-col">')
    for m in models:
        mc = m["mc"]
        if not mc:
            continue
        parts.append(f'<div><h4>{esc(m["id"])}</h4><table style="font-size:.85rem">')
        parts.append('<thead><tr><th>G\\P</th><th>A</th><th>B</th><th>C</th><th>D</th></tr></thead><tbody>')
        for g in "ABCD":
            parts.append(f'<tr><td><strong>{g}</strong></td>')
            for p in "ABCD":
                v = mc["confusion"].get((g, p), 0)
                bg = "rgba(34,197,94,.3)" if g == p and v > 0 else ("rgba(239,68,68,.15)" if v > 0 else "")
                parts.append(f'<td style="background:{bg};text-align:center">{v}</td>')
            parts.append('</tr>')
        parts.append('</tbody></table></div>')
    parts.append('</div>')

    # 3.3 Per-gold-label accuracy
    parts.append('<h3>3.3 Accuracy by Gold Label</h3>')
    parts.append('<table><thead><tr><th>Model</th><th>A</th><th>B</th><th>C</th><th>D</th></tr></thead><tbody>')
    for m in models:
        mc = m["mc"]
        if not mc:
            continue
        parts.append(f'<tr><td>{esc(m["id"])}</td>')
        for g in "ABCD":
            total_g = mc["gold_dist"].get(g, 0)
            correct_g = mc["correct_by_gold"].get(g, 0)
            a = correct_g / total_g if total_g else 0
            color = "var(--green)" if a > 0.4 else ("var(--red)" if a < 0.1 else "var(--text)")
            parts.append(f'<td style="color:{color}">{correct_g}/{total_g} ({pct(a)})</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')

    # 3.4 hard samples
    parts.append('<h3>3.4 Hard MC Samples (wrong for ALL models)</h3>')
    parts.append(f'<p>{mc_overlap["hard_count"]}/{mc_overlap["total"]} samples sai ở tất cả 4 models.</p>')
    if mc_overlap["hard_ids"]:
        parts.append('<p style="font-size:.85rem;color:var(--muted)">IDs: ' +
                     ", ".join(mc_overlap["hard_ids"][:15]) + '</p>')

    parts.append('</section>')

    # ── 4. NLI DEEP DIVE ──
    parts.append('<section class="card"><h2>4. NLI — Cross-model Deep Dive</h2>')

    parts.append('<h3>4.1 Confusion Matrices</h3>')
    parts.append('<table><thead><tr><th>Model</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th>'
                 '<th>Acc</th><th>Prec(rel)</th><th>Rec(rel)</th><th>Rec(nr)</th><th>Macro-F1</th>'
                 '<th>Bias%</th></tr></thead><tbody>')
    for m in models:
        nli = m["nli"]
        if not nli:
            continue
        parts.append(f'<tr><td>{esc(m["id"])}</td>'
                     f'<td>{nli["tp"]}</td><td>{nli["fp"]}</td><td>{nli["fn"]}</td><td>{nli["tn"]}</td>'
                     f'<td>{pct(nli["acc"])}</td><td>{pct(nli["prec_r"])}</td>'
                     f'<td>{pct(nli["rec_r"])}</td><td>{pct(nli["rec_nr"])}</td>'
                     f'<td>{pct(nli["macro_f1"])}</td>'
                     f'<td>{pct(nli["bias_ratio"])}</td></tr>')
    parts.append('</tbody></table>')

    parts.append('<div class="insight"><strong>Bias pattern:</strong> '
                 'Tất cả models đều bias toward "relevant" nhưng mức độ khác nhau. '
                 'qwen3-0b6-cpt-sft bias nặng nhất (97.9% predict relevant, recall nr = 4.8%). '
                 'qwen3-0b6-cpt-qlora-sft tốt hơn nhiều (bias ~87%, macro-F1 gấp đôi).</div>')

    # NLI thinking vs direct
    parts.append('<h3>4.2 Thinking vs Direct — NLI</h3>')
    parts.append('<table><thead><tr><th>Model</th><th>Think Acc</th><th>Direct Acc</th><th>Delta</th>'
                 '<th>Think Bias%</th><th>Direct Bias%</th></tr></thead><tbody>')
    for m in models:
        nli = m["nli"]
        if not nli:
            continue
        delta = nli["direct_acc"] - nli["think_acc"]
        color = "var(--green)" if delta > 0.02 else ("var(--red)" if delta < -0.02 else "var(--muted)")
        parts.append(f'<tr><td>{esc(m["id"])}</td>'
                     f'<td>{pct(nli["think_acc"])}</td><td>{pct(nli["direct_acc"])}</td>'
                     f'<td style="color:{color}">{"+" if delta >= 0 else ""}{pct(delta)}</td>'
                     f'<td>{pct(nli["think_bias"])}</td><td>{pct(nli["direct_bias"])}</td></tr>')
    parts.append('</tbody></table>')
    parts.append('<div class="insight"><strong>Pattern:</strong> '
                 'Direct mode consistently có accuracy cao hơn hoặc tương đương thinking ở NLI cho tất cả models. '
                 'gwen3-1.7b-pretrain-vlsp-sft: direct 77.1% vs thinking 65.1% (delta +12%). '
                 'Thinking ở NLI binary classification task không hiệu quả — model nhỏ "overthink" rồi default to majority.</div>')

    # 4.3 hard NLI
    parts.append('<h3>4.3 Hard NLI Samples</h3>')
    parts.append(f'<p>{nli_overlap["hard_count"]}/{nli_overlap["total"]} samples sai ở tất cả 4 models.</p>')

    parts.append('</section>')

    # ── 5. SYLLOGISM DEEP DIVE ──
    parts.append('<section class="card"><h2>5. Syllogism — Cross-model Deep Dive</h2>')

    parts.append('<h3>5.1 Distribution Statistics</h3>')
    parts.append('<table><thead><tr><th>Model</th><th>Mean</th><th>Median</th><th>Std</th>'
                 '<th>P10</th><th>P90</th><th>Min</th><th>Max</th></tr></thead><tbody>')
    for m in models:
        sy = m["syllo"]
        if not sy:
            continue
        parts.append(f'<tr><td>{esc(m["id"])}</td>'
                     f'<td>{f4(sy["mean"])}</td><td>{f4(sy["median"])}</td><td>{f4(sy["std"])}</td>'
                     f'<td>{f4(sy["p10"])}</td><td>{f4(sy["p90"])}</td>'
                     f'<td>{f4(sy["min"])}</td><td>{f4(sy["max"])}</td></tr>')
    parts.append('</tbody></table>')

    # 5.2 Histograms side by side
    parts.append('<h3>5.2 ROUGE-L Histograms</h3>')
    parts.append('<div class="two-col">')
    for m in models:
        sy = m["syllo"]
        if not sy:
            continue
        max_b = max(sy["buckets"].values()) or 1
        parts.append(f'<div><h4>{esc(m["id"])}</h4>')
        for rng, cnt in sy["buckets"].items():
            w = cnt / max_b * 100
            c = BUCKET_COLORS.get(rng, "#38bdf8")
            parts.append(f'<div class="hist-row"><span class="hist-label">{rng}</span>'
                         f'<div class="hist-bar-wrap"><div class="hist-bar" style="width:{w:.0f}%;background:{c}"></div></div>'
                         f'<span class="hist-count">{cnt}</span></div>')
        parts.append('</div>')
    parts.append('</div>')

    # 5.3 Thinking vs direct syllogism
    parts.append('<h3>5.3 Thinking vs Direct — Syllogism</h3>')
    parts.append('<table><thead><tr><th>Model</th><th>Think Mean</th><th>Direct Mean</th><th>Delta</th></tr></thead><tbody>')
    for m in models:
        sy = m["syllo"]
        if not sy:
            continue
        delta = sy["direct_mean"] - sy["think_mean"]
        parts.append(f'<tr><td>{esc(m["id"])}</td>'
                     f'<td>{f4(sy["think_mean"])} (n={sy["think_n"]})</td>'
                     f'<td>{f4(sy["direct_mean"])} (n={sy["direct_n"]})</td>'
                     f'<td>{"+" if delta >= 0 else ""}{f4(delta)}</td></tr>')
    parts.append('</tbody></table>')

    parts.append('</section>')

    # ── 6. THINKING VS DIRECT GLOBAL ──
    parts.append('<section class="card"><h2>6. Thinking vs Direct — Global Summary</h2>')
    parts.append('<table><thead><tr><th>Model</th><th>MC Think</th><th>MC Direct</th>'
                 '<th>NLI Think</th><th>NLI Direct</th><th>Syllo Think</th><th>Syllo Direct</th>'
                 '<th>Verdict</th></tr></thead><tbody>')
    for m in models:
        mc, nli, sy = m["mc"], m["nli"], m["syllo"]
        if not (mc and nli and sy):
            continue
        mc_win = "D" if mc["direct_acc"] > mc["think_acc"] else ("T" if mc["think_acc"] > mc["direct_acc"] else "=")
        nli_win = "D" if nli["direct_acc"] > nli["think_acc"] else ("T" if nli["think_acc"] > nli["direct_acc"] else "=")
        sy_win = "D" if sy["direct_mean"] > sy["think_mean"] else ("T" if sy["think_mean"] > sy["direct_mean"] else "=")
        verdict = f"MC:{mc_win} NLI:{nli_win} Sy:{sy_win}"
        direct_wins = sum(1 for x in [mc_win, nli_win, sy_win] if x == "D")
        tag_cls = "tg" if direct_wins >= 2 else ("tr" if direct_wins == 0 else "ty")

        parts.append(f'<tr><td>{esc(m["id"])}</td>')
        parts.append(f'<td>{pct(mc["think_acc"])}</td><td>{pct(mc["direct_acc"])}</td>')
        parts.append(f'<td>{pct(nli["think_acc"])}</td><td>{pct(nli["direct_acc"])}</td>')
        parts.append(f'<td>{f4(sy["think_mean"])}</td><td>{f4(sy["direct_mean"])}</td>')
        parts.append(f'<td><span class="tag {tag_cls}">{verdict}</span></td></tr>')
    parts.append('</tbody></table>')
    parts.append('<div class="insight"><strong>Kết luận:</strong> '
                 'Direct mode tốt hơn hoặc tương đương thinking ở hầu hết cases. '
                 'NLI: Direct LUÔN thắng (4/4 models). MC: Direct thắng 3/4 (trừ qlora-sft). '
                 'Syllogism: không rõ winner. '
                 'Với model ≤1.7B, thinking tokens là overhead không cần thiết cho classification tasks.</div>')
    parts.append('</section>')

    # ── 7. KEY: LoRA vs QLoRA (cùng 0.6B) ──
    parts.append('<section class="card"><h2>7. LoRA vs QLoRA — Cùng Qwen3-0.6B, cùng CPT</h2>')
    lora = next((m for m in models if m["id"] == "qwen3-0b6-cpt-sft"), None)
    qlora = next((m for m in models if m["id"] == "qwen3-0b6-cpt-qlora-sft"), None)
    if lora and qlora:
        parts.append('<table><thead><tr><th>Metric</th><th>LoRA SFT</th><th>QLoRA SFT</th><th>Delta</th><th>Impact</th></tr></thead><tbody>')
        comparisons = [
            ("MC Accuracy", lora["mc"]["acc"], qlora["mc"]["acc"], True),
            ("MC Null Rate", lora["mc"]["null_rate"], qlora["mc"]["null_rate"], False),
            ("NLI Accuracy", lora["nli"]["acc"], qlora["nli"]["acc"], True),
            ("NLI Macro-F1", lora["nli"]["macro_f1"], qlora["nli"]["macro_f1"], True),
            ("NLI Bias%", lora["nli"]["bias_ratio"], qlora["nli"]["bias_ratio"], False),
            ("Syllo ROUGE-L", lora["syllo"]["mean"], qlora["syllo"]["mean"], True),
            ("Syllo Std", lora["syllo"]["std"], qlora["syllo"]["std"], False),
        ]
        for label, v_l, v_q, higher_better in comparisons:
            delta = v_q - v_l
            if "Rate" in label or "Bias" in label or "Std" in label:
                # lower is better
                improved = delta < 0
            else:
                improved = delta > 0
            tag = '<span class="tag tg">QLoRA wins</span>' if improved else '<span class="tag tr">LoRA wins</span>'
            parts.append(f'<tr><td>{label}</td><td>{pct(v_l)}</td><td>{pct(v_q)}</td>'
                         f'<td>{"+" if delta >= 0 else ""}{pct(delta)}</td><td>{tag}</td></tr>')
        parts.append('</tbody></table>')

        parts.append('<div class="insight"><strong>Root cause analysis:</strong> '
                     'QLoRA SFT vượt trội toàn diện trên cùng base + CPT. Điều này loại trừ model size '
                     'và CPT quality là bottleneck. Nguyên nhân có thể là:'
                     '<ul style="margin:8px 0 0;padding-left:20px">'
                     '<li><strong>LoRA rank/alpha khác nhau</strong> — QLoRA có thể dùng rank cao hơn vì quantized base tiết kiệm VRAM</li>'
                     '<li><strong>Learning rate</strong> — LoRA full-precision có thể cần lr khác</li>'
                     '<li><strong>Regularization effect</strong> — 4-bit quantization trong QLoRA có thể đóng vai trò regularizer, giảm overfitting SFT data</li>'
                     '<li><strong>Training epochs/steps</strong> — có thể LoRA SFT bị overfit hoặc underfit</li>'
                     '</ul>'
                     '→ Cần diff configs/model/qwen3_0b6_lora.yaml vs qwen3_0b6_qlora.yaml để xác nhận.</div>')
    parts.append('</section>')

    # ── 8. SCALING ANALYSIS ──
    parts.append('<section class="card"><h2>8. Scaling Analysis — 0.6B vs 1.7B</h2>')
    m06 = next((m for m in models if m["id"] == "qwen3-0b6-cpt-qlora-sft"), None)  # best 0.6B
    m17 = next((m for m in models if m["id"] == "qwen3-1b7-cpt-sft"), None)
    m17v = next((m for m in models if m["id"] == "gwen3-1.7b-pretrain-vlsp-sft"), None)
    if m06 and m17:
        parts.append('<table><thead><tr><th>Metric</th><th>0.6B QLoRA (best)</th><th>1.7B CPT-LoRA</th>'
                     '<th>1.7B VLSP-PT</th><th>Winner</th></tr></thead><tbody>')
        rows_data = [
            ("MC Acc", m06["mc"]["acc"], m17["mc"]["acc"], m17v["mc"]["acc"] if m17v else None),
            ("NLI Acc", m06["nli"]["acc"], m17["nli"]["acc"], m17v["nli"]["acc"] if m17v else None),
            ("NLI Macro-F1", m06["nli"]["macro_f1"], m17["nli"]["macro_f1"], m17v["nli"]["macro_f1"] if m17v else None),
            ("Syllo ROUGE-L", m06["syllo"]["mean"], m17["syllo"]["mean"], m17v["syllo"]["mean"] if m17v else None),
        ]
        for label, v06, v17, v17v in rows_data:
            vals = [("0.6B QLoRA", v06), ("1.7B CPT", v17)]
            if v17v is not None:
                vals.append(("1.7B VLSP", v17v))
            winner = max(vals, key=lambda x: x[1])
            parts.append(f'<tr><td>{label}</td><td>{pct(v06)}</td><td>{pct(v17)}</td>')
            parts.append(f'<td>{pct(v17v) if v17v is not None else "-"}</td>')
            parts.append(f'<td><span class="tag tb">{winner[0]}</span></td></tr>')
        parts.append('</tbody></table>')

        parts.append('<div class="insight"><strong>Scaling insights:</strong>'
                     '<ul style="margin:8px 0 0;padding-left:20px">'
                     '<li><strong>MC:</strong> 0.6B QLoRA (45%) &gt; 1.7B CPT (32%) &gt; 1.7B VLSP (16%). '
                     'Model nhỏ hơn nhưng SFT tốt hơn THẮNG model lớn hơn — SFT quality &gt; model size cho classification.</li>'
                     '<li><strong>NLI:</strong> 0.6B QLoRA (71.6%) dẫn đầu — cùng pattern.</li>'
                     '<li><strong>Syllogism:</strong> 1.7B CPT (0.595) vượt xa 0.6B QLoRA (0.483). '
                     'Generative reasoning CẦN model lớn hơn — size matters cho generation.</li>'
                     '</ul></div>')
    parts.append('</section>')

    # ── 9. RECOMMENDATIONS ──
    parts.append('<section class="card"><h2>9. Actionable Recommendations</h2>')
    parts.append('<table><thead><tr><th>#</th><th>Action</th><th>Priority</th><th>Expected Impact</th><th>Details</th></tr></thead><tbody>')
    recs = [
        ("1", "Diff LoRA vs QLoRA SFT configs", "tr", "Urgent",
         "So sánh lora_rank, lr, epochs, data mix giữa 2 configs. "
         "Đây là experiment có ROI cao nhất — cùng compute budget nhưng MC 10%→45%."),
        ("2", "Apply QLoRA SFT recipe cho 1.7B model", "tr", "High",
         "1.7B + QLoRA SFT recipe có thể đạt MC &gt;50% VÀ Syllogism &gt;0.6 — best of both worlds."),
        ("3", "Disable thinking mode cho classification", "ty", "Medium",
         "Direct mode tốt hơn NLI ở 4/4 models, MC ở 3/4. Tiết kiệm ~40% tokens."),
        ("4", "Answer shuffling cho MC SFT data", "ty", "Medium",
         "Giảm position bias (C-bias 75% ở 0.6B LoRA). Random swap vị trí đáp án đúng."),
        ("5", "Upsample not_relevant trong NLI SFT", "ty", "Medium",
         "Tất cả models bias toward relevant. Cần balanced training data."),
        ("6", "Investigate ROUGE=0 syllogism failures", "tb", "Research",
         "2 mẫu ROUGE=0.0 ở 0.6B LoRA. Cần đọc actual output để diagnose (empty? hallucination? wrong language?)."),
        ("7", "Train 4B model qua SFT", "tb", "Future",
         "CPT đã có (PPL=1.83). Nếu SFT quality tốt, 4B có thể dominate tất cả tasks."),
    ]
    for num, action, tag_cls, priority, detail in recs:
        parts.append(f'<tr><td>{num}</td><td><strong>{action}</strong></td>'
                     f'<td><span class="tag {tag_cls}">{priority}</span></td>'
                     f'<td>{detail}</td><td></td></tr>')
    parts.append('</tbody></table></section>')

    # ── assemble ──
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Full Cross-Model Analysis — VN Legal AI</title>
<style>{CSS}</style>
</head>
<body>
<header>
<h1>Full Cross-Model Analysis</h1>
<p class="sub">VN-Legal-AI Distillation Experiment — 4 SFT Models + 5 CPT Checkpoints</p>
<p class="sub" style="font-size:.85rem">Generated: 2026-05-22 | Benchmark: ViLawQA (MC/NLI/Syllogism)</p>
</header>
<main>
{"".join(parts)}
</main>
</body>
</html>"""


# ── main ─────────────────────────────────────────────────────
def main():
    model_dirs = sorted(p for p in RAW.iterdir() if p.is_dir())
    models = [load_model(d) for d in model_dirs]
    cpts = load_cpt_evals()

    mc_overlap = compute_overlap(models, "mc")
    nli_overlap = compute_overlap(models, "nli")

    html = build_html(models, cpts, mc_overlap, nli_overlap)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Report → {OUT}")
    print(f"Models: {[m['id'] for m in models]}")
    print(f"CPT evals: {len(cpts)}")
    print(f"MC hard samples (wrong for all): {mc_overlap['hard_count']}/{mc_overlap['total']}")
    print(f"NLI hard samples: {nli_overlap['hard_count']}/{nli_overlap['total']}")


if __name__ == "__main__":
    main()
