"""Reuse the existing Poneglyph dashboard, with local Trackio and GPU telemetry."""

import math
import html
import json
import os
import sys
import time
from pathlib import Path

for support_root in (Path(__file__).resolve().parent.parent, Path("/app")):
    if (support_root / "finetune_lighton_ocr_bbox/training_monitor.py").is_file():
        sys.path.insert(0, str(support_root))
        break
from finetune_lighton_ocr_bbox.training_monitor import TrainingMonitorCallback, atomic_json, svg_chart


def write_dashboard(path, summary, history):
    phase = summary.get("status", "preparing")
    labels = {"preparing": "Préparation", "loading_model": "Chargement du modèle", "calibrating": "Calibration GPU",
              "baseline": "Évaluation initiale", "training": "Entraînement", "validating": "Validation",
              "exporting": "Export du meilleur modèle", "testing": "Benchmark test complet",
              "complete": "Terminé", "failed": "Erreur", "smoke_complete": "Test technique terminé"}
    step, maximum = summary.get("step", 0), summary.get("max_steps", 0)
    percent = 100 * step / maximum if maximum else 0
    def number(key, multiplier=1, suffix=""):
        value = summary.get(key)
        return f"{value * multiplier:.3g}{suffix}" if isinstance(value, (int, float)) else "—"
    final_test = phase == "complete" and isinstance(summary.get("test_cer"), (int, float))
    quality_prefix = "test_" if final_test else "eval_"
    quality_label = f"CER test · {summary.get('test_num_samples', '—')} pages" if final_test else "CER validation"
    stats = (("Avancement", f"{percent:.1f} %"), (quality_label, number(quality_prefix + "cer", 100, " %")),
             ("WER test" if final_test else "WER validation", number(quality_prefix + "wer", 100, " %")),
             ("Texte exact · bulles", number(quality_prefix + "exact_bubble_text_rate", 100, " %")),
             ("Texte exact · pages / ordre", number(quality_prefix + "page_text_exact", 100, " %")),
             ("Pic VRAM · dernière session" if final_test else "VRAM allouée", number("gpu_peak_gib" if final_test else "gpu_allocated_gib", suffix=" GiB")),
             ("F1 / IoU ≥ 0.5", number(quality_prefix + "f1@0_5", 100, " %")), ("IoU moyen", number(quality_prefix + "mean_iou")),
             ("Pages tronquées", f"{summary.get('test_truncated_pages', 0)} / {summary.get('test_num_samples', '—')}") if final_test else
             ("Activité GPU", number("gpu_utilization_percent", suffix=" %")))
    cards = "".join(f"<div><small>{label}</small><strong>{value}</strong></div>" for label, value in stats)
    baseline = {"step": 0, **{"eval_" + key.removeprefix("baseline_"): value for key, value in summary.items() if key.startswith("baseline_")}}
    chart_history = [baseline, *history] if "eval_cer" in baseline else history
    charts = "".join(f"<section><h2>{label}</h2>{svg_chart(chart_history, keys)}</section>" for label, keys in (
        ("Loss · train / validation", ("loss", "eval_loss")),
        ("Transcription · CER / ordre de lecture", ("eval_cer", "eval_reading_order_cer")),
        ("Mots · WER / ordre de lecture", ("eval_wer", "eval_reading_order_wer")),
        ("Texte exact · bulles / pages", ("eval_exact_bubble_text_rate", "eval_page_text_exact")),
        ("Localisation · F1@0.5 / F1@0.75 / IoU", ("eval_f1@0_5", "eval_f1@0_75", "eval_mean_iou")),
        ("VRAM · Go", ("gpu_allocated_gib", "gpu_reserved_gib")),
        ("Learning rate", ("learning_rate",)),
        ("GPU · activité %", ("gpu_utilization_percent",))))
    reports = sorted(path.parent.glob("validation_step_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    reports += [p for p in (path.parent / "baseline_validation.json", path.parent / "test_benchmark.json", path.parent / "ocr_metrics_extended.json", path.parent / "refinement_comparison.json", path.parent / "ocr_refinement_comparison.svg", path.parent / "results.json", path.parent / "training_curves.svg", path.parent / "model_verification.json", path.parent / "export_precision_comparison.json", path.parent / "run_config.json", path.parent / "hardware_profile.json") if p.exists()]
    links = "".join(f'<a href="{html.escape(p.name)}">{html.escape(p.stem)}</a>' for p in reports)
    overlays = sorted((path.parent / "test_benchmark_overlays").glob("*.jpg"))[:8] if final_test else sorted(path.parent.glob("validation_step_*_overlays/*.jpg"))[-8:]
    if not overlays:
        overlays = sorted(path.parent.glob("baseline_validation_overlays/*.jpg"))[:8]
    previews = "".join(f'<a href="{p.relative_to(path.parent).as_posix()}"><img loading="lazy" src="{p.relative_to(path.parent).as_posix()}" alt="Page {html.escape(p.stem)} : annotations vertes et prédictions rouges"><span>{html.escape(p.stem)}</span></a>' for p in overlays)
    refresh = '<meta http-equiv="refresh" content="15">' if phase not in {"complete", "failed", "smoke_complete", "calibrated"} else ""
    error = f'<p class="error">{html.escape(str(summary["error"]))}</p>' if summary.get("error") else ""
    test_class = "error" if summary.get("test_truncated_pages", 0) else ""
    test_note = (f"<p class='{test_class}'>Test complet : {summary.get('test_truncated_pages', 0)} sortie(s) tronquée(s) / "
                 f"{summary.get('test_num_samples', '—')} pages, incluses dans les scores. "
                 "Voir les réponses et intervalles de confiance dans test_benchmark.</p>") if final_test else ""
    if final_test and summary.get("init_model"):
        test_note += "<p>Le même test avait déjà été évalué avant cette consolidation. Le checkpoint a été sélectionné uniquement sur le CER de validation.</p>"
    checkpoint = html.escape(str(summary.get("best_model_checkpoint") or "En attente de validation"))
    selection_label = "Meilleure CER validation" if summary.get("selection_metric") == "cer" else "Score validation"
    best_display = number("best_metric", 100, " %") if summary.get("selection_metric") == "cer" else number("best_metric")
    document = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>Poneglyph · LightOnOCR-3 0.8B</title>
<style>:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151b;color:#edf2f7}}*{{box-sizing:border-box}}body{{max-width:1380px;margin:auto;padding:32px}}header{{padding:24px 0;border-bottom:1px solid #334250}}h1{{font-size:34px;letter-spacing:-1px;margin:14px 0}}.tag{{color:#89b9d5;font:12px monospace;letter-spacing:2px}}p,small{{color:#98acbd;line-height:1.6}}.meta{{display:flex;gap:22px;flex-wrap:wrap}}progress{{width:100%;height:7px;accent-color:#8bb9d5}}.stats{{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:#334250;margin:24px 0;border:1px solid #334250}}.stats>div{{padding:18px;background:#17212a}}strong{{display:block;font-size:26px;margin:10px 0;font-variant-numeric:tabular-nums}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}section{{border:1px solid #334250;background:#17212a;padding:22px;min-width:0}}h2{{font-size:15px;color:#c1d4e2;font-weight:500}}svg{{width:100%;height:auto}}svg text{{font:11px monospace;fill:#98acbd}}.legend{{display:flex;gap:14px;font-size:12px;flex-wrap:wrap}}.empty{{min-height:175px;display:grid;place-content:center}}.links{{display:flex;gap:14px;flex-wrap:wrap}}a{{color:#a2cbe3;text-decoration:none}}footer{{margin:28px 0;overflow-wrap:anywhere}}.previews{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}}.previews img{{width:100%;height:280px;object-fit:contain;background:#0b0f13}}.previews span{{display:block;padding:6px;font-size:12px}}.error{{color:#ff9589}}@media(max-width:1000px){{.stats{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:680px){{body{{padding:18px}}h1{{font-size:26px}}.charts{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr 1fr}}.previews{{grid-template-columns:1fr 1fr}}}}</style></head><body>
<header><div class="tag">PONEGLYPH / RTX 5090 / FULL PAGE GROUNDING</div><h1>LightOnOCR-3 · 0.8B</h1><p>{labels.get(phase, html.escape(phase))} · étape {step} / {maximum or '—'} · époque {number('epoch')}</p>{error}<progress value="{step}" max="{max(1,maximum)}"></progress>
<div class="meta"><p>Temps session : {summary.get('elapsed_display','—')} · ETA : {summary.get('eta_display','—')}</p><p>GPU · dernière mesure : {number('gpu_temperature_c',suffix=' °C')} · {number('gpu_power_w',suffix=' W')}</p><p>Prompt : <code>grounding</code> · thinking désactivé</p></div></header>
<div class="stats">{cards}</div>{test_note}<div class="charts">{charts}</div><footer><p>Meilleur checkpoint : {checkpoint}</p><p>{selection_label} : {best_display} · Débit : {number('optimizer_steps_per_second',suffix=' step/s')}</p><div class="links">{links}</div><h2>Contrôle visuel · annotations vertes / prédictions rouges</h2><div class="previews">{previews or '<p>Disponible après la première génération.</p>'}</div><small>Validation sur pages réservées. Le test final ne sélectionne pas le modèle. Actualisation 15 s pendant l'entraînement ; ETA mesurée, indicative. Métriques complètes : metrics.jsonl, TensorBoard et Trackio local.</small></footer></body></html>'''
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)


class Monitor(TrainingMonitorCallback):
    def __init__(self, output_dir, config):
        previous_summary = Path(output_dir) / "training_summary.json"
        previous_summary = json.loads(previous_summary.read_text(encoding="utf-8")) if previous_summary.exists() else {}
        super().__init__(output_dir, run_name="LightOnOCR-3 0.8B / grounding / RTX 5090")
        self.summary.update({key: value for key, value in previous_summary.items() if key.startswith("baseline_")})
        history_path = Path(output_dir) / "metrics.jsonl"
        if config.get("resume_step") is not None and history_path.exists():
            self.history = [record for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
                            for record in [json.loads(line)]
                            if record.get("event") == "metrics" and record.get("step", 0) <= config["resume_step"]]
        self.config = config
        self.trackio = None
        self.nvml = None
        self.last_telemetry = 0
        self.telemetry = {}
        self.summary.update(status="preparing", phase="preparing", **{k: config[k] for k in ("model_id", "revision", "train_mode")})
        self.summary.update({key: config[key] for key in ("selection_metric", "init_model", "bbox_loss_weight", "eos_loss_weight") if key in config})
        self._persist("preparing", self.summary)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            pass
        if os.getenv("LIGHTON3_TRACKIO", "1") == "1":
            import trackio
            # No Space, bucket or webhook: metrics stay on this machine.
            os.environ.pop("TRACKIO_WEBHOOK_URL", None)
            trackio.init(project="poneglyph-lighton3-bbox", name=Path(output_dir).name, config=config, space_id=None, resume="allow", auto_log_gpu=False)
            self.trackio = trackio

    def phase(self, status, **values):
        self.summary.update(status=status, phase=status, **values)
        self._persist(status, self.summary)

    def _persist(self, event, values=None):
        previous = self.last_dashboard
        super()._persist(event, values)
        if self.enabled and self.last_dashboard != previous:
            write_dashboard(self.output_dir / "training_dashboard.html", self.summary, self.history)

    def _gpu_metrics(self):
        metrics = super()._gpu_metrics()
        if self.nvml and time.perf_counter() - self.last_telemetry >= 5:
            api, handle = self.nvml
            try:
                utilization = api.nvmlDeviceGetUtilizationRates(handle)
                self.telemetry = {"gpu_utilization_percent": utilization.gpu,
                                  "gpu_temperature_c": api.nvmlDeviceGetTemperature(handle, api.NVML_TEMPERATURE_GPU),
                                  "gpu_power_w": api.nvmlDeviceGetPowerUsage(handle) / 1000,
                                  "gpu_device_used_gib": api.nvmlDeviceGetMemoryInfo(handle).used / 2**30}
                self.last_telemetry = time.perf_counter()
            except Exception:
                pass
        return {**metrics, **self.telemetry}

    def on_train_begin(self, args, state, control, **kwargs):
        self.summary.update(status="training", phase="training")
        super().on_train_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        # Detached runs also get a heartbeat and live progress between log events.
        if time.perf_counter() - self.last_dashboard >= 10:
            self._snapshot(state)
            self._persist("heartbeat", {})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.trackio:
            values = {k: v for k, v in {**logs, **self._gpu_metrics()}.items() if isinstance(v, (float, int)) and math.isfinite(v)}
            self.trackio.log({**values, "optimizer_step": state.global_step})
            for key in ("loss", "eval_loss", "grad_norm"):
                if isinstance(logs.get(key), (float, int)) and not math.isfinite(logs[key]):
                    self.trackio.alert(title=f"Non-finite {key}", text=f"Step {state.global_step}", level=self.trackio.AlertLevel.ERROR)
        if logs and any(key.startswith("eval_") for key in logs):
            self.summary.update(status="training", phase="training")
        super().on_log(args, state, control, logs, **kwargs)

    def close(self, status, state=None):
        self.summary["phase"] = status
        super().close(status, state)
        if status in {"complete", "failed", "smoke_complete", "calibrated"} and self.trackio:
            self.trackio.finish()
            self.trackio = None
