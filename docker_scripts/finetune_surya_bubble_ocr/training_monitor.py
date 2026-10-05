"""Local Rich telemetry, append-only JSONL and a self-contained HTML/SVG report."""

import html
import json
import math
import os
import re
import sys
import time
import uuid
import csv
import subprocess
import base64
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if re.search(r"(^|_)(token|secret|password|api_key|credentials)$", str(key), re.I)
            else json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--"
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def svg_chart(history, series):
    colors = ("#8bb9d5", "#d6b584", "#9cbd9d")
    datasets = [[(r["step"], r[key]) for r in history if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])] for key in series]
    points = [p for dataset in datasets for p in dataset]
    if not points:
        return '<p class="empty">En attente de mesures</p>'
    xmin, xmax = min(x for x, y in points), max(x for x, y in points)
    ymin, ymax = min(y for x, y in points), max(y for x, y in points)
    xr, yr = max(xmax - xmin, 1), ymax - ymin or max(abs(ymax) * 0.1, 0.01)
    content = ['<svg viewBox="0 0 540 210" role="img" aria-label="Courbes par etape">']
    for i in range(4):
        y = 18 + i * 52
        content.append(f'<line x1="62" x2="524" y1="{y}" y2="{y}" stroke="#303740"/><text x="56" y="{y+4}" text-anchor="end">{ymax-i*yr/3:.3g}</text>')
    for dataset, color in zip(datasets, colors):
        coords = [(62 + (x-xmin)/xr*462, 18 + (ymax-y)/yr*156) for x, y in dataset]
        encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        content.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2"/>')
        if len(coords) == 1:
            x, y = coords[0]
            content.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    content.append(f'<text x="62" y="202">{xmin}</text><text x="524" y="202" text-anchor="end">{xmax} steps</text></svg>')
    content.append('<div class="legend">' + " ".join(f'<span style="color:{color}">{html.escape(key)}</span>' for key, color in zip(series, colors)) + '</div>')
    return "".join(content)


def write_dashboard(path, summary, history):
    s = json_safe(summary)
    def fmt(key, scale=1):
        value = s.get(key)
        return f"{value * scale:.4g}" if isinstance(value, (int, float)) else "--"
    refresh = '<meta http-equiv="refresh" content="15">' if s.get("status") in {"training", "validating", "exporting"} else ""
    stats = "".join(f'<div class="stat"><small>{name}</small><strong>{value}</strong></div>' for name, value in (
        ("Loss", fmt("loss")), ("CER validation", fmt("eval_cer", 100) + " %"),
        ("Bulles exactes", fmt("eval_exact_match", 100) + " %"), ("Pic VRAM allouée", fmt("gpu_peak_gib") + " GiB")))
    charts = "".join('<section><h2>' + title + '</h2>' + svg_chart(history, series) + '</section>' for title, series in (
        ("Perte", ("loss", "eval_loss")), ("Erreurs de caractères et de mots", ("eval_cer", "eval_wer")),
        ("Bulles exactes", ("eval_exact_match",)), ("Sorties vides / limite de tokens", ("eval_blank_rate", "eval_token_limit_rate")),
        ("Mémoire GPU / GiB", ("gpu_allocated_gib", "gpu_reserved_gib", "gpu_peak_gib")), ("Learning rate · premier groupe", ("learning_rate",)),
        ("Norme des gradients", ("grad_norm",)), ("Débit entraînement / bulles par seconde", ("train_bubbles_per_second",)),
        ("Temps par étape / secondes", ("step_seconds",)), ("GPU / utilisation %", ("gpu_utilization_percent",)),
        ("GPU / puissance W", ("gpu_power_watts",)), ("GPU / température °C", ("gpu_temperature_c",))))
    title = html.escape(str(s.get("run_name", "Surya BBox")))
    status = html.escape(str(s.get("status", "running")))
    checkpoint = html.escape(str(s.get("best_model_checkpoint") or "--"))
    document = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>{title}</title>
<style>:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#101318;color:#ecf0f4}}*{{box-sizing:border-box}}body{{max-width:1280px;margin:auto;padding:36px 28px}}header{{border-bottom:1px solid #303740;padding-bottom:24px}}h1{{font-size:32px;letter-spacing:-1px;margin:8px 0}}h2{{font-size:15px;font-weight:500;color:#bcc9d5}}p,small{{color:#8e9eac;line-height:1.6}}.eyebrow{{font-size:12px;letter-spacing:3px;color:#8aa7c0}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#303740;border:1px solid #303740;margin:24px 0}}.stat{{background:#161b22;padding:20px}}strong{{display:block;font-size:26px;margin-top:8px;font-variant-numeric:tabular-nums}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}section{{background:#161b22;border:1px solid #303740;padding:22px;min-width:0}}svg{{width:100%;height:auto}}svg text{{font:11px monospace;fill:#8e9eac}}.legend{{font-size:12px;display:flex;gap:18px;flex-wrap:wrap}}.empty{{min-height:180px;display:grid;place-content:center}}progress{{width:100%;height:5px;accent-color:#8bb9d5}}footer{{margin-top:28px;overflow-wrap:anywhere}}@media(max-width:700px){{body{{padding:22px 16px}}.stats{{grid-template-columns:1fr 1fr}}.charts{{grid-template-columns:1fr}}}}</style></head><body>
<header><div class="eyebrow">PONEGLYPH / TRAINING</div><h1>{title}</h1><p>{status} / etape {s.get('step',0)} sur {s.get('max_steps',0)}</p><progress max="{max(1,s.get('max_steps',1))}" value="{s.get('step',0)}"></progress></header>
<div class="stats">{stats}</div><div class="charts">{charts}</div><footer><p>Debit moyen : {fmt('optimizer_steps_per_second')} step/s | Temps : {s.get('elapsed_display','--')} | ETA : {s.get('eta_display','--')}<br>Meilleur checkpoint : {checkpoint}</p><small>Rapport local autonome. Courbes de cette session ; metrics.jsonl conserve toutes les sessions. Actualisation toutes les 15 secondes pendant le run. ETA indicative, validation et sauvegarde incluses.</small></footer></body></html>'''
    path = Path(path)
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self, output_dir, run_name="Surya OCR"):
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.enabled = True
        self.summary = {"run_name": run_name, "session_id": uuid.uuid4().hex, "status": "training"}
        self.history = []
        self.started = time.perf_counter()
        self.initial_step = 0
        self.last_render = 0.0
        self.last_dashboard = 0.0
        self.live = None
        self.io_warning = False
        self.last_gpu_poll = 0.0
        self.hardware = {}
        self.step_started = None
        self.step_times = []

    def _gpu_metrics(self):
        try:
            import torch
            if torch.cuda.is_available():
                if time.perf_counter() - self.last_gpu_poll > 15:
                    self.last_gpu_poll = time.perf_counter()
                    try:
                        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,temperature.gpu",
                                                 "--format=csv,noheader,nounits", "-i", "0"],
                                                capture_output=True, text=True, timeout=2, check=True)
                        values = [float(x.strip()) for x in result.stdout.strip().split(",")]
                        self.hardware = dict(zip(("gpu_utilization_percent", "gpu_power_watts", "gpu_temperature_c"), values))
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                return {**self.hardware, "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        except (ImportError, RuntimeError):
            pass
        return {}

    def _snapshot(self, state):
        elapsed = time.perf_counter() - self.started
        rate = max(0, state.global_step - self.initial_step) / max(elapsed, 1e-9)
        self.summary.update(step=state.global_step, max_steps=state.max_steps, epoch=state.epoch,
                            elapsed_seconds=elapsed, elapsed_display=duration(elapsed), optimizer_steps_per_second=rate,
                            eta_display=duration((state.max_steps-state.global_step)/rate if rate else None),
                            best_model_checkpoint=state.best_model_checkpoint, best_metric=state.best_metric)

    def _renderable(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
        from rich.table import Table
        s = self.summary
        progress = Progress(TextColumn("[bold]Step {task.completed:.0f}/{task.total:.0f}"), BarColumn(), TaskProgressColumn(), expand=True)
        progress.add_task("train", total=max(1, s.get("max_steps", 1)), completed=s.get("step", 0))
        grid = Table.grid(expand=True, padding=(0, 2))
        for _ in range(4):
            grid.add_column()
        def f(key, digits=4):
            value = s.get(key)
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "--"
        grid.add_row("Loss", f("loss"), "Eval loss", f("eval_loss"))
        grid.add_row("CER / WER", f("eval_cer") + " / " + f("eval_wer"), "Exact match", f("eval_exact_match"))
        grid.add_row("Learning rate", f"{s['learning_rate']:.2e}" if s.get("learning_rate") is not None else "--", "Grad norm", f("grad_norm", 3))
        grid.add_row("VRAM / peak GiB", f("gpu_allocated_gib", 2) + " / " + f("gpu_peak_gib", 2), "Step/s", f("optimizer_steps_per_second", 3))
        grid.add_row("Elapsed", s.get("elapsed_display", "--"), "ETA", s.get("eta_display", "--"))
        return Panel(Group(progress, grid), title=self.run_name, subtitle=str(s["status"]), border_style="blue", padding=(1, 2))

    def _persist(self, event, values=None):
        if not self.enabled:
            return
        record = {"event": event, "session_id": self.summary["session_id"], "time": datetime.now(timezone.utc).isoformat(),
                  "step": self.summary.get("step", 0), **(values or {})}
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
            atomic_json(self.output_dir / "training_summary.json", self.summary)
            now = time.perf_counter()
            if event != "metrics" or now - self.last_dashboard >= 15:
                write_dashboard(self.output_dir / "training_dashboard.html", self.summary, self.history)
                self.last_dashboard = now
        except OSError as exc:
            if not self.io_warning:
                print(f"WARNING: Cannot persist monitoring files: {exc}", flush=True)
                self.io_warning = True

    def on_train_begin(self, args, state, control, **kwargs):
        self.enabled = state.is_world_process_zero
        if not self.enabled:
            return
        self.started = time.perf_counter()
        self.initial_step = state.global_step
        self._snapshot(state)
        if sys.stdout.isatty() and os.getenv("SURYA_RICH", "1").lower() not in {"0", "false", "off"}:
            try:
                from rich.live import Live
                self.live = Live(self._renderable(), auto_refresh=False)
                self.live.start()
            except ImportError:
                pass
        self._persist("start", self.summary)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started is not None:
            self.step_times.append(time.perf_counter() - self.step_started)
        if self.enabled and self.live is not None and time.perf_counter() - self.last_render >= 1:
            self._snapshot(state)
            self.live.update(self._renderable(), refresh=True)
            self.last_render = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or not logs:
            return
        self._snapshot(state)
        values = {**logs, **self._gpu_metrics()}
        if self.step_times:
            seconds = sum(self.step_times) / len(self.step_times)
            values.update(step_seconds=seconds,
                          train_bubbles_per_second=args.per_device_train_batch_size * args.gradient_accumulation_steps / seconds)
            self.step_times.clear()
        self.summary.update(json_safe(values))
        self.history.append(json_safe({"step": state.global_step, **values}))
        self._persist("metrics", values)
        if self.live is not None:
            self.live.update(self._renderable(), refresh=True)
        else:
            keys = ("loss", "eval_loss", "eval_cer", "eval_wer", "eval_exact_match", "learning_rate", "grad_norm", "train_bubbles_per_second", "gpu_allocated_gib")
            details = " | ".join(f"{key}={values[key]:.5g}" for key in keys if isinstance(values.get(key), (int, float)))
            print(f"[Surya] step {state.global_step}/{state.max_steps} | {details} | ETA {self.summary['eta_display']}", flush=True)
        for key in ("loss", "eval_loss", "grad_norm"):
            if isinstance(values.get(key), (int, float)) and not math.isfinite(values[key]):
                raise FloatingPointError(f"Non-finite {key} at step {state.global_step}.")

    def on_save(self, args, state, control, **kwargs):
        if self.enabled:
            self._snapshot(state)
            self._persist("checkpoint", {"path": str(self.output_dir / f"checkpoint-{state.global_step}")})

    def on_train_end(self, args, state, control, **kwargs):
        self.close("trained", state)

    def close(self, status, state=None):
        if not self.enabled:
            return
        if state is not None:
            self._snapshot(state)
        self.summary["status"] = status
        self._persist(status, self.summary)
        if self.live is not None:
            self.live.update(self._renderable(), refresh=True)
            self.live.stop()
            self.live = None


def write_final_report(output_dir, benchmark, summary):
    """Export durable PNG/SVG figures, CSV and a portable HTML report for this run."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Levenshtein import editops

    output_dir = Path(output_dir)
    summary = {**summary, "status": "complete"}
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        summary["configuration"] = json.loads(config_path.read_text(encoding="utf-8"))
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    history = []
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event") == "metrics":
                history.append(row)
    keys = sorted({k for row in history for k in row})
    if keys:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    samples = benchmark["samples"]
    metrics = benchmark["metrics"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "#f7f6f2",
                         "axes.facecolor": "#f7f6f2", "axes.titleweight": "bold"})
    colors = ("#067b83", "#d07b35", "#7764ab")
    figures = []

    def save(fig, name, title):
        fig.suptitle(title, fontsize=19, fontweight="bold", x=.05, ha="left")
        fig.tight_layout(rect=(0, 0, 1, .96), pad=2.0)
        fig.savefig(plot_dir / f"{name}.png", dpi=180)
        fig.savefig(plot_dir / f"{name}.svg")
        plt.close(fig)
        figures.append((name, title))

    def curves(ax, series, title):
        for i, (key, label, scale) in enumerate(series):
            points = [(r["step"], r[key] * scale) for r in history if isinstance(r.get(key), (int, float))]
            if points:
                ax.plot(*zip(*points), label=label, color=colors[i % 3], linewidth=1.7,
                        marker="o" if len(points) < 15 else None, markersize=4)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("Étape d'optimisation")
        if ax.lines:
            ax.legend(frameon=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, series, title in zip(axes.flat, (
        (("loss", "Entraînement", 1), ("eval_loss", "Validation", 1)),
        (("eval_cer", "CER", 100), ("eval_wer", "WER", 100)),
        (("eval_exact_match", "Exact match", 100),),
        (("learning_rate", "LR premier groupe (vision)", 1),),
        (("grad_norm", "Norme L2", 1),),
        (("eval_token_limit_rate", "Limite de tokens", 100), ("eval_blank_rate", "Sortie vide", 100))),
        ("Loss par bulle", "Erreurs de reconnaissance / %", "Bulles parfaitement lues / %",
         "Taux d'apprentissage", "Gradients avant clipping", "Échecs de génération / %")):
        curves(ax, series, title)
    save(fig, "learning", "SURYA OCR · Apprentissage")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, series, title in zip(axes.flat, (
        (("train_bubbles_per_second", "Bulles/s (batch nominal)", 1),),
        (("step_seconds", "Secondes/étape", 1),),
        (("gpu_allocated_gib", "Allouée après étape", 1), ("gpu_reserved_gib", "Réservée", 1), ("gpu_peak_gib", "Pic alloué", 1)),
        (("gpu_utilization_percent", "Utilisation", 1),),
        (("gpu_power_watts", "Puissance", 1),),
        (("gpu_temperature_c", "Température", 1),)),
        ("Débit des étapes GPU", "Durée des étapes (hors validation)", "Mémoire GPU / GiB",
         "Occupation GPU / % · échantillonnée", "Puissance GPU / W", "Température GPU / °C")):
        curves(ax, series, title)
    save(fig, "performance", "RTX 5090 · Débit et ressources")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    buckets = metrics["error_distribution_by_length"]
    labels = list(buckets)
    axes[0, 0].bar(labels, [buckets[k]["count"] for k in labels], color=colors[0])
    axes[0, 0].set_title("Test · effectifs par longueur de référence", loc="left")
    axes[0, 1].bar(labels, [100*buckets[k]["cer"] for k in labels], color=colors[1])
    axes[0, 1].set_title("Test · CER par longueur / %", loc="left")
    slices = benchmark["slice_metrics"]
    names = {"short_len_le_5": "1–5 caractères", "very_short_len_le_2": "1–2 caractères",
             "has_accents": "Accents", "has_punctuation": "Ponctuation",
             "onomatopoeia_like": "Onomatopées (heuristique)", "reference_multiline": "Multiligne"}
    axes[1, 0].barh([f"{names.get(k,k)} (n={v['sample_count']})" for k,v in slices.items()],
                    [100*v["cer"] for v in slices.values()], color=colors[0])
    axes[1, 0].set_title("Test · CER par catégorie / %", loc="left")
    distances = Counter(min(s["levenshtein"], 5) for s in samples)
    axes[1, 1].bar(["0", "1", "2", "3", "4", "5+"], [distances[i] for i in range(6)], color=colors[2])
    axes[1, 1].set_title("Test · distance d'édition par bulle", loc="left")
    save(fig, "quality", "SURYA OCR · Qualité sur le test réservé")

    operations, confusions = Counter(), Counter()
    for s in samples:
        for op, i, j in editops(s["reference"], s["prediction"]):
            operations[op] += 1
            before = s["reference"][i] if op != "insert" else "∅"
            after = s["prediction"][j] if op != "delete" else "∅"
            confusions[f"{before!r} → {after!r}"] += 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].bar(["Substitutions", "Suppressions", "Insertions"],
                [operations[k] for k in ("replace", "delete", "insert")], color=colors)
    axes[0].set_title("Opérations au niveau caractère", loc="left")
    pairs = confusions.most_common(15)[::-1]
    if pairs:
        axes[1].barh([p[0] for p in pairs], [p[1] for p in pairs], color=colors[1])
    axes[1].set_title("Confusions les plus fréquentes", loc="left")
    save(fig, "errors", "SURYA OCR · Analyse des erreurs")
    atomic_json(output_dir / "error_analysis.json", {"operations": dict(operations), "confusions": confusions.most_common(100)})

    sample_keys = ("id", "page_id", "reference", "prediction", "cer", "levenshtein", "generated_tokens", "hit_token_limit", "amortized_seconds")
    with (output_dir / "predictions_test.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    cards = "".join(f'<div class="card"><small>{name}</small><strong>{value}</strong></div>' for name, value in (
        ("CER", f"{metrics['cer']*100:.3f} %"), ("WER", f"{metrics['wer']*100:.3f} %"),
        ("Bulles exactes", f"{metrics['exact_match']*100:.2f} %"), ("Bulles de test", str(len(samples))),
        ("Sorties vides", f"{metrics['blank_rate']*100:.2f} %"), ("Limite de tokens", str(metrics['token_limit_count']))))
    plots = "".join('<section><h2>' + title + '</h2><img alt="' + title + '" src="data:image/svg+xml;base64,' +
                    base64.b64encode((plot_dir / f"{name}.svg").read_bytes()).decode() + '"></section>' for name, title in figures)
    worst = sorted(samples, key=lambda s: (s["cer"], s["levenshtein"]), reverse=True)[:30]
    rows = "".join(f"<tr><td>{html.escape(str(s['id']))}</td><td>{html.escape(s['reference'])}</td>"
                   f"<td>{html.escape(s['prediction'])}</td><td>{s['cer']*100:.1f} %</td></tr>" for s in worst)
    text = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Surya OCR · Rapport d'entraînement</title>
<style>*{{box-sizing:border-box}}body{{margin:auto;max-width:1400px;background:#101820;color:#ecf1f3;font:16px system-ui;padding:42px 30px}}h1{{font-size:clamp(32px,5vw,58px);margin:16px 0}}h2{{font-size:20px}}p,small{{color:#a5b8c3;line-height:1.6}}.eyebrow{{letter-spacing:4px;color:#52bdc1;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:30px 0}}.card{{background:#1a2732;padding:24px;border-top:2px solid #52bdc1}}strong{{display:block;font-size:32px;margin-top:10px}}section{{margin:32px 0}}img{{width:100%;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{text-align:left;padding:14px 10px;border-bottom:1px solid #33424e;vertical-align:top;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1a2732;padding:20px}}@media(max-width:700px){{body{{padding:22px 16px}}.cards{{grid-template-columns:1fr 1fr}}strong{{font-size:26px}}}}</style></head><body>
<header><div class="eyebrow">PONEGLYPH / SURYA OCR / RTX 5090</div><h1>Lire chaque bulle.</h1><p>Fine-tuning sur crops · Test réservé par page · Meilleur checkpoint choisi sur le CER de validation</p></header>
<div class="cards">{cards}</div><p>Les courbes d'apprentissage portent sur la validation. Les métriques ci-dessus et les analyses d'erreurs portent uniquement sur le test réservé. Les erreurs portent sur les prédictions après le post-traitement OCR existant : normalisation des espaces et retrait des guillemets extérieurs appariés. Les catégories peuvent se recouper. Le débit d'entraînement est calculé par étape, hors validation et sauvegarde ; le dernier batch peut être incomplet.</p>{plots}
<section><h2>30 erreurs à examiner</h2><table><thead><tr><th>Bulle</th><th>Référence</th><th>Prédiction</th><th>CER</th></tr></thead><tbody>{rows}</tbody></table></section>
<details><summary>Toutes les métriques de test</summary><pre>{html.escape(json.dumps(json_safe(metrics),ensure_ascii=False,indent=2))}</pre></details>
<details><summary>Configuration et bilan d'exécution</summary><pre>{html.escape(json.dumps(json_safe(summary),ensure_ascii=False,indent=2))}</pre></details>
<footer><p>Rapport autonome · Graphiques disponibles en PNG et SVG dans plots/ · Mesures brutes dans metrics.jsonl et metrics.csv · Prédictions dans predictions_test.csv</p></footer></body></html>'''
    (output_dir / "report.html").write_text(text, encoding="utf-8")
    if (output_dir / "final_merged").is_dir():
        model_card = f'''---
base_model: datalab-to/surya-ocr-2
library_name: transformers
pipeline_tag: image-text-to-text
language:
- fr
tags:
- ocr
- manga
- surya
---

# Surya OCR — Poneglyph, crops de bulles

Fine-tuning local sur RTX 5090. Entrée : un crop RGB de bulle ; sortie : sa transcription française.
Prompt : `{benchmark.get('prompt', '')}`

Meilleur checkpoint sélectionné uniquement sur le CER de validation :
`{Path(str(summary.get('best_model_checkpoint', ''))).name}`.

## Test réservé

| Métrique | Valeur |
| --- | ---: |
| Bulles | {len(samples)} |
| CER | {metrics['cer']*100:.3f} % |
| WER | {metrics['wer']*100:.3f} % |
| Exact match | {metrics['exact_match']*100:.2f} % |
| Distance de Levenshtein moyenne | {metrics['avg_levenshtein']:.4f} |
| Sorties vides | {metrics['blank_rate']*100:.2f} % |
| Générations à la limite de tokens | {metrics['token_limit_count']} |

Split strict par page. Le fichier `benchmark_test.json` contient toutes les
prédictions, les sous-catégories et les intervalles de confiance par bootstrap
de pages. La normalisation des espaces et le post-traitement sont décrits dans
ce fichier. Pas de comparaison avec un autre modèle.

## Chargement

Poids BF16 au format Transformers standard (`transformers==5.14.1`).
Charger `AutoProcessor` et `AutoModelForImageTextToText` depuis ce dossier.
Les patches Liger d'entraînement ne sont pas nécessaires au rechargement.
`generation_config.json` contient les identifiants EOS/PAD et le budget de génération.

La recette, les versions des packages et les empreintes des données sont dans
le dossier parent : `run_config.json`, `package_versions.json`, `training_summary.json`.
Le rapport autonome est `../report.html` et les figures PNG/SVG sont dans `../plots/`.
'''
        (output_dir / "final_merged" / "README.md").write_text(model_card, encoding="utf-8")
    return ["report.html", "metrics.csv", "predictions_test.csv", "error_analysis.json", *[f"plots/{name}.png" for name, _ in figures]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild a Surya report from completed run artifacts; no model inference.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.output_dir / "training_summary.json").read_text(encoding="utf-8"))
    benchmark = json.loads((args.output_dir / "final_merged" / "benchmark_test.json").read_text(encoding="utf-8"))
    artifacts = write_final_report(args.output_dir, benchmark, summary)
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
