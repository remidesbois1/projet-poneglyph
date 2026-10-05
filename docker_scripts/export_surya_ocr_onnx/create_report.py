"""Summarize actual browser DOM-exported measurements, with standalone plots."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j-1] + 1, previous + (char != other))
    return row[-1]


def main(directory):
    results = json.loads((directory / "browser_validation.json").read_text(encoding="utf-8"))
    stress = json.loads((directory / "browser_stress.json").read_text(encoding="utf-8"))
    examples = {x["id"]: x for x in json.loads((directory / "fixtures/examples.json").read_text(encoding="utf-8"))}
    manifest = json.loads((directory / "browser_manifest.json").read_text())
    total = np.array([x["timings"]["totalMs"] for x in results])
    vision = np.array([x["timings"]["visionMs"] for x in results])
    prefill = np.array([x["timings"]["prefillMs"] for x in results])
    tokens = np.array([x["tokens"] for x in results])
    metrics = {"samples": len(results), "provider": "WebGPU only", "gpu": results[0]["gpu"],
               "latency_ms": {"min": int(total.min()), "p50": float(np.median(total)), "p95": float(np.percentile(total, 95)), "max": int(total.max()), "mean": float(total.mean())},
               "decode_tokens_per_second": float(tokens.sum() / ((total - prefill).sum() / 1000)),
               "complete_generations": sum(not x["tokenLimit"] for x in results),
               "native_conformity": sum(x["text"] == x["reference"] for x in results),
               "blank_outputs": sum(not x["text"] for x in results),
               "validation_cer": sum(distance(examples[x["id"]]["reference"], x["text"]) for x in results) / sum(len(examples[x["id"]]["reference"]) for x in results),
               "validation_exact": sum(x["text"] == examples[x["id"]]["reference"] for x in results) / len(results),
               "export_bytes": sum(x["bytes"] for x in manifest["files"].values()),
               "source_sha256": manifest["source_sha256"], "stress": stress,
               "scope": "24 fixed length-stratified validation crops; loaded model; not an independent full test-set evaluation"}
    (directory / "browser_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#101828", "axes.facecolor": "#101828", "axes.edgecolor": "#42526b", "text.color": "#e8eef8", "axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8", "grid.color": "#334155"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    ax = axes[0, 0]
    ax.scatter(tokens, total / 1000, c="#61d9c2", s=36)
    ax.set(xlabel="Tokens générés", ylabel="Durée totale (s)", title="Durée selon la longueur de transcription")
    ax.grid(alpha=.35)
    ax = axes[0, 1]
    x = np.arange(len(results))
    ax.bar(x, vision, color="#61d9c2", label="Image")
    ax.bar(x, prefill - vision, bottom=vision, color="#818cf8", label="Prompt")
    ax.bar(x, total - prefill, bottom=prefill, color="#fbbf24", label="Génération")
    ax.set(xlabel="Bulle de validation", ylabel="Durée (ms)", title="Répartition du temps sur GPU")
    ax.legend(frameon=False)
    ax = axes[1, 0]
    ax.plot(np.sort(total) / 1000, np.arange(1, len(total) + 1) / len(total) * 100, drawstyle="steps-post", color="#818cf8", linewidth=2)
    ax.set(xlabel="Durée totale (s)", ylabel="Bulles terminées (%)", title="Distribution des latences")
    ax.grid(alpha=.35)
    ax = axes[1, 1]
    ax.scatter(tokens, tokens / ((total - prefill) / 1000), c="#fbbf24", s=36)
    ax.set(xlabel="Tokens générés", ylabel="Tokens / seconde", title="Débit de génération, hors préremplissage")
    ax.grid(alpha=.35)
    fig.savefig(directory / "performance.svg")
    fig.savefig(directory / "performance.png", dpi=160)
    plt.close(fig)
    svg = (directory / "performance.svg").read_text()
    svg = svg[svg.index("<svg"):]
    cards = [("Générations complètes", f'{metrics["complete_generations"]}/{len(results)}'),
             ("Latence médiane", f'{np.median(total)/1000:.2f} s'),
             ("Débit de génération", f'{metrics["decode_tokens_per_second"]:.1f} tok/s'),
             ("Latence p95", f'{np.percentile(total,95)/1000:.2f} s')]
    card_html = ''.join(f'<article><small>{label}</small><strong>{value}</strong></article>' for label, value in cards)
    rows = ''.join(f'<tr><td>{html.escape(r["id"])}</td><td>{r["tokens"]}</td><td>{r["timings"]["visionMs"]}</td><td>{r["timings"]["prefillMs"]}</td><td>{r["timings"]["totalMs"]}</td><td>{"Oui" if not r["tokenLimit"] else "Limite"}</td></tr>' for r in results + stress)
    page = f'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Surya · Validation WebGPU</title>
<style>body{{background:#101828;color:#e8eef8;font:16px system-ui;margin:0}}main{{max-width:1120px;margin:auto;padding:48px 28px}}h1{{font-size:44px;margin:12px 0}}p{{color:#aebdd0;line-height:1.6}}.eyebrow{{color:#61d9c2;letter-spacing:3px;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:32px 0}}article{{border:1px solid #334155;border-radius:12px;padding:22px}}small{{color:#aebdd0}}strong{{display:block;font-size:30px;margin-top:12px}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;border-bottom:1px solid #334155;text-align:left}}a{{color:#61d9c2}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:32px}}}}</style>
<main><div class="eyebrow">PONEGLYPH · RTX 5090 · 5 OCTOBRE 2026</div><h1>Surya tourne sur WebGPU.</h1>
<p>24 bulles de validation, puis une image de 864 × 1 184 px et un appel de contrôle après remise à zéro des états. Mesures relevées dans le navigateur, modèle déjà chargé, téléchargement exclu. Exécution GPU imposée, sans repli CPU.</p>
<div class="cards">{card_html}</div>{svg}
<p>Sur ce petit échantillon de validation : CER {metrics["validation_cer"]*100:.3f} %, transcription exacte {metrics["validation_exact"]*100:.2f} %. Ces résultats servent au contrôle du portage ; ils ne remplacent pas l'évaluation indépendante sur le test complet. Les deux cas de résistance sont exclus des agrégats ci-dessus.</p>
<p>Export : {metrics["export_bytes"]/1e9:.3f} Go. Poids et matmuls FP16, états récurrents FP32, préremplissage par blocs de 32, décodeur spécialisé, caches conservés sur GPU.</p>
<table><thead><tr><th>Bulle</th><th>Tokens</th><th>Image (ms)</th><th>Premier token (ms)</th><th>Total (ms)</th><th>Terminée</th></tr></thead><tbody>{rows}</tbody></table>
<p><a href="browser_metrics.json">Métriques JSON</a> · <a href="performance.png">Figure PNG</a> · <a href="https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/tree/main/onnx">Export Hugging Face</a> · <a href="http://127.0.0.1:3000/sandbox/surya">Essayer Surya</a></p>
<p>Empreinte du modèle source : <code>{manifest["source_sha256"]}</code></p></main></html>'''
    (directory / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k != "stress"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory)
