"""Bubble CER dashboard and local Trackio; no remote telemetry or webhook."""
import html
import json
import math
import os
import time
from pathlib import Path

from training_monitor import TrainingMonitorCallback, atomic_json, svg_chart


def write_dashboard(path, summary, history):
    phase = summary.get("status", "preparing")
    titles = {"preparing": "Préparation", "calibrating": "Calibration 5090", "baseline": "Benchmark initial de validation",
              "training": "Entraînement", "validating": "Validation complète", "exporting": "Export safetensors",
              "testing": "Test final", "complete": "Terminé", "failed": "Erreur", "smoke_complete": "Test technique terminé",
              "validation_complete": "Validation terminée · choix final du modèle"}
    def fmt(key, scale=1, suffix=""):
        value = summary.get(key)
        return f"{value * scale:.4g}{suffix}" if isinstance(value, (int, float)) else "—"
    final = phase == "complete"
    prefix = "test_" if final else "eval_"
    cards = (("CER test" if final else "CER validation", fmt(prefix + "cer", 100, " %")),
             ("WER test" if final else "WER validation", fmt(prefix + "wer", 100, " %")),
             ("Bulles exactes", fmt(prefix + "exact_match", 100, " %")), ("Loss", fmt("loss")),
             ("Pic VRAM", fmt("gpu_peak_gib", suffix=" GiB")), ("Débit", fmt("train_bubbles_per_second", suffix=" bulles/s")))
    historic = summary.get("training_curve_scoring_version") == "legacy-surya-quote-strip"
    plot_note = " · courbes historiques" if historic else ""
    plots = (("CER · caractères / WER · mots" + plot_note, ("eval_cer", "eval_wer")), ("Bulles exactes" + plot_note, ("eval_exact_match",)),
             ("Perte d'entraînement et validation", ("loss", "eval_loss")), ("Sorties vides / tronquées", ("eval_blank_rate", "eval_token_limit_rate")),
             ("Learning rate · premier groupe vision", ("learning_rate",)), ("Norme des gradients", ("grad_norm",)),
             ("VRAM · GiB", ("gpu_allocated_gib", "gpu_reserved_gib", "gpu_peak_gib")), ("Débit · bulles/s", ("train_bubbles_per_second",)),
             ("GPU · utilisation %", ("gpu_utilization_percent",)), ("GPU · watts", ("gpu_power_watts",)),
             ("GPU · température °C", ("gpu_temperature_c",)), ("Temps par update · secondes", ("step_seconds",)))
    chart_html = "".join(f"<section><h2>{title}</h2>{svg_chart(history, series)}</section>" for title, series in plots)
    card_html = "".join(f"<div><small>{title}</small><strong>{value}</strong></div>" for title, value in cards)
    step, maximum = summary.get("step", 0), summary.get("max_steps", 0)
    generation = phase in {"baseline", "validating", "testing"} and summary.get("generation_total", 0) > 0
    progress_step = summary.get("generation_processed", 0) if generation else step
    progress_max = summary.get("generation_total", 0) if generation else maximum
    generation_note = (f" · {progress_step} / {progress_max} bulles · reste environ {summary.get('generation_remaining_seconds', 0)/60:.1f} min"
                       if generation else "")
    refresh = '' if phase in {"complete", "failed", "smoke_complete", "calibrated", "validation_complete"} else '<meta http-equiv="refresh" content="15">'
    links = "".join(f'<a href="{p.name}">{p.stem}</a>' for p in path.parent.iterdir()
                    if p.name in {"run_config.json", "dataset_audit.json", "baseline_validation.json", "report.html", "model_verification.json", "training_summary.json", "metrics.csv", "experiment_report.md", "experiment_curves.png", "comparison_surya.json", "hf_publication.json", "visual_data_audit.html", "visual_data_audit.json"})
    error = f"<p class='error'>{html.escape(str(summary['error']))}</p>" if summary.get("error") else ""
    if summary.get("model_url"):
        links = f'<a href="{html.escape(summary["model_url"],quote=True)}">Modèle Hugging Face</a>' + links
    document = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>Poneglyph · LightOnOCR-3 · Bulles</title>
<style>:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151b;color:#edf2f7}}*{{box-sizing:border-box}}body{{max-width:1380px;margin:auto;padding:30px}}header{{border-bottom:1px solid #334250;padding:20px 0}}h1{{font-size:34px;letter-spacing:-1px;margin:12px 0}}.tag{{color:#a6bdd8;font:12px monospace;letter-spacing:2px}}p,small{{color:#a0b0c0;line-height:1.6}}progress{{width:100%;height:8px;accent-color:#a6bdd8}}.stats{{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:#334250;margin:24px 0;border:1px solid #334250}}.stats>div{{padding:18px;background:#17212a}}strong{{display:block;font-size:26px;margin:10px 0;font-variant-numeric:tabular-nums}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}section{{border:1px solid #334250;background:#17212a;padding:22px;min-width:0}}h2{{font-size:15px;color:#c5d5e5;font-weight:500}}svg{{width:100%;height:auto}}svg text{{font:11px monospace;fill:#98acbd}}.legend{{display:flex;gap:14px;font-size:12px;flex-wrap:wrap}}.empty{{min-height:175px;display:grid;place-content:center}}footer{{margin:26px 0;overflow-wrap:anywhere}}a{{color:#b9d5f0;margin-right:18px;text-decoration:none}}.error{{color:#ff9589}}@media(max-width:900px){{.stats{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:640px){{body{{padding:18px}}.charts{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr 1fr}}}}</style></head><body>
<header><div class="tag">PONEGLYPH / RTX 5090 / TRANSCRIPTION DE BULLES</div><h1>LightOnOCR-3 · 0.8B · Bulles</h1><p>{titles.get(phase, html.escape(phase))}{generation_note} · update {step} / {maximum or '—'} · époque {fmt('epoch')}</p>{error}<progress value="{progress_step}" max="{max(1,progress_max)}"></progress><p>Temps : {summary.get('elapsed_display','—')} · ETA mesurée : {summary.get('eta_display','—')} · Prompt vide · thinking désactivé</p></header>
<div class="stats">{card_html}</div><p>Meilleur CER de validation : {fmt('best_metric',100,' %')} · Baseline originale : {fmt('baseline_cer',100,' %')} · 1 648 bulles de validation, 184 pages. Comparaison finale Surya : mêmes 1 612 crops, 185 pages.</p><div class="charts">{chart_html}</div>
<footer><p>Meilleur checkpoint : {html.escape(str(summary.get('best_model_checkpoint') or 'En attente'))}</p>{links}<p>Sélection exclusivement sur le CER de validation. Test réservé à la comparaison finale. Poids maîtres FP32, autocast BF16, AdamW fused, FLA/SM120 et Liger avec backward. Métriques persistées en JSONL, CSV, TensorBoard et Trackio local.</p></footer></body></html>'''
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)


class BubbleMonitor(TrainingMonitorCallback):
    def __init__(self, output_dir, config):
        super().__init__(output_dir, run_name="LightOnOCR-3 0.8B / bulles / RTX 5090")
        self.trackio = None
        self.summary.update(status="preparing")
        self.summary["training_curve_scoring_version"] = config.get("training_curve_scoring_version", config.get("scoring_version", "text-v2-preserve-visible-quotes"))
        baseline_path = Path(output_dir) / "baseline_validation.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
            self.summary.update({"baseline_" + k: v for k, v in baseline.items() if isinstance(v, (int, float))})
        if config.get("resume"):
            path = Path(output_dir) / "metrics.jsonl"
            if path.exists():
                self.history = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                                if json.loads(line).get("event") == "metrics"]
        self._persist("preparing")
        if os.getenv("LIGHTON3_BUBBLE_TRACKIO", "1") == "1":
            import trackio
            os.environ.pop("TRACKIO_WEBHOOK_URL", None)
            trackio.init(project="poneglyph-lighton3-bubble", name=Path(output_dir).name,
                         config=config, space_id=None, resume="allow", auto_log_gpu=False)
            self.trackio = trackio

    def phase(self, status, **values):
        self.summary.update(status=status, **values)
        self._persist(status, values)

    def _persist(self, event, values=None):
        previous = self.last_dashboard
        super()._persist(event, values)
        if self.last_dashboard != previous:
            write_dashboard(self.output_dir / "training_dashboard.html", self.summary, self.history)

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            super().on_log(args, state, control, logs=logs, **kwargs)
        except FloatingPointError as exc:
            if self.trackio:
                self.trackio.alert(title="Entraînement interrompu", text=str(exc), level=self.trackio.AlertLevel.ERROR)
            raise
        if self.trackio and logs:
            values = {"optimizer_step": state.global_step, **logs, **self.hardware}
            self.trackio.log({k: v for k, v in values.items() if isinstance(v, (int, float)) and math.isfinite(v)})

    def on_train_begin(self, args, state, control, **kwargs):
        self.summary["status"] = "training"
        return super().on_train_begin(args, state, control, **kwargs)

    def close(self, status, state=None):
        super().close(status, state)
        if self.trackio and status in {"complete", "failed", "smoke_complete", "validation_complete"}:
            self.trackio.finish()
            self.trackio = None
