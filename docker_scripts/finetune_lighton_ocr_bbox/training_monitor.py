"""Local Rich telemetry, append-only JSONL and a self-contained HTML/SVG report."""

import html
import json
import math
import os
import re
import sys
import time
import uuid
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
        ("F1 @ IoU 0.5", fmt("eval_f1@0_5", 100) + " %"), ("VRAM allouee", fmt("gpu_allocated_gib") + " GiB")))
    charts = "".join('<section><h2>' + title + '</h2>' + svg_chart(history, series) + '</section>' for title, series in (
        ("Perte", ("loss", "eval_loss")), ("Qualite des generations", ("eval_cer", "eval_f1@0_5", "eval_mean_iou")),
        ("Memoire GPU / GiB", ("gpu_allocated_gib", "gpu_reserved_gib")), ("Learning rate", ("learning_rate",))))
    title = html.escape(str(s.get("run_name", "LightOn BBox")))
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
    def __init__(self, output_dir, run_name="LightOn BBox"):
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

    def _gpu_metrics(self):
        try:
            import torch
            if torch.cuda.is_available():
                return {"gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
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
        grid.add_row("CER", f("eval_cer"), "F1 / IoU", f("eval_f1@0_5") + " / " + f("eval_mean_iou"))
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
        if sys.stdout.isatty() and os.getenv("LIGHTON_BBOX_RICH", "1").lower() not in {"0", "false", "off"}:
            try:
                from rich.live import Live
                self.live = Live(self._renderable(), auto_refresh=False)
                self.live.start()
            except ImportError:
                pass
        self._persist("start", self.summary)

    def on_step_end(self, args, state, control, **kwargs):
        if self.enabled and self.live is not None and time.perf_counter() - self.last_render >= 1:
            self._snapshot(state)
            self.live.update(self._renderable(), refresh=True)
            self.last_render = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or not logs:
            return
        self._snapshot(state)
        values = {**logs, **self._gpu_metrics()}
        self.summary.update(json_safe(values))
        self.history.append(json_safe({"step": state.global_step, **values}))
        self._persist("metrics", values)
        if self.live is not None:
            self.live.update(self._renderable(), refresh=True)
        else:
            keys = ("loss", "eval_loss", "eval_cer", "eval_mean_iou", "eval_f1@0_5", "learning_rate", "grad_norm", "gpu_allocated_gib")
            details = " | ".join(f"{key}={values[key]:.5g}" for key in keys if isinstance(values.get(key), (int, float)))
            print(f"[LightOn] step {state.global_step}/{state.max_steps} | {details} | ETA {self.summary['eta_display']}", flush=True)
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
