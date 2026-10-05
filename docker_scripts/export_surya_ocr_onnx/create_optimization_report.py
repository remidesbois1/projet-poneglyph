"""Reproducible metrics and plots from real browser exports, including the full test set."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from create_report import distance


def quality(rows):
    chars = words = char_errors = word_errors = exact = 0
    errors = []
    for row in rows:
        ref, pred = row["groundTruth"], row["text"]
        ce, we = distance(ref, pred), distance(ref.split(), pred.split())
        chars += len(ref)
        words += len(ref.split())
        char_errors += ce
        word_errors += we
        exact += ref == pred
        if ce:
            errors.append({"id": row["id"], "reference": ref, "prediction": pred, "edits": ce})
    return {"samples": len(rows), "characters": chars, "words": words, "character_errors": char_errors,
            "word_errors": word_errors, "cer": char_errors / max(1, chars), "wer": word_errors / max(1, words),
            "exact": exact, "exact_rate": exact / max(1, len(rows)), "blank": sum(not r["text"] for r in rows),
            "truncated": sum(r["tokenLimit"] for r in rows), "native_conformity": sum(r["text"] == r["reference"] for r in rows)}, errors


def distribution(values):
    return {key: float(value) for key, value in zip(["min", "p50", "p90", "p95", "p99", "max"], np.percentile(values, [0, 50, 90, 95, 99, 100]))}


def main(directory):
    production = json.loads((directory / "production-3passes.json").read_text(encoding="utf-8"))
    test = json.loads((directory / "full-test1612.json").read_text(encoding="utf-8"))
    sweep = json.loads((directory / "sweep-8settings.json").read_text(encoding="utf-8"))
    measured = [r for r in production if not r["id"].startswith("stress-")]
    stress = [r for r in production if r["id"].startswith("stress-")]
    q, errors = quality(test)
    total = np.array([r["timings"]["totalMs"] for r in measured])
    first = np.array([r["timings"]["prefillMs"] for r in measured])
    tokens = np.array([r["tokens"] for r in measured])
    decode = total - first
    metrics = {"hardware": "RTX 5090 / Chromium / WebGPU", "model_loaded": True, "download_excluded": True,
               "production_runs": len(production), "measured_runs": len(measured), "unique_crops": len({r["id"] for r in measured}),
               "latency_ms": distribution(total), "first_token_ms": distribution(first),
               "decode_tokens_per_second": float(tokens.sum() * 1000 / decode.sum()),
               "native_conformity": sum(r["text"] == r["reference"] for r in production),
               "test": q, "stress": stress, "configuration": {"gemv": "subgroups", "token_batch": 8,
                   "weights": "FP16", "accumulation_and_recurrent_state": "FP32", "max_new_tokens": 256},
               "decoder_buffers": measured[0].get("diagnostics"),
               "method": "24 fixed validation crops and two stress cases, three production worker passes after warm-up; separate complete 1612-crop test evaluation."}
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    # Explicit allowlist: no crops, IDs, references or individual predictions in the public summary.
    public = {key: metrics[key] for key in ["hardware", "model_loaded", "download_excluded", "production_runs",
        "measured_runs", "unique_crops", "latency_ms", "first_token_ms", "decode_tokens_per_second",
        "native_conformity", "test", "configuration", "method"]}
    public["decode_rate_definition"] = "Sum of output text tokens / sum of time after first token; includes EOS detection and unused steps in a GPU submission."
    public["custom_decoder_memory_bytes"] = {"weights": metrics["decoder_buffers"]["weightBytes"],
        "example_sequence": metrics["decoder_buffers"]["sequenceBytes"], "scope": "Custom decoder buffers only, not total browser or ONNX Runtime VRAM."}
    release_dir = directory.parent / "onnx-decode"
    if (release_dir / "hf_publication.json").exists():
        release = json.loads((release_dir / "hf_publication.json").read_text())
        public["artifact_revision"] = release["revision"]
    if (release_dir / "browser_manifest.json").exists():
        manifest = json.loads((release_dir / "browser_manifest.json").read_text())
        public["source_sha256"] = manifest["source_sha256"]
        public["artifacts"] = manifest["files"]
    (directory / "webgpu_metrics.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "test-errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / "measurements.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["corpus", "id", "pass", "tokens", "total_ms", "first_token_ms", "native_equal", "token_limit"])
        for corpus, rows in [("production", production), ("test", test)]:
            for r in rows:
                writer.writerow([corpus, r["id"], r["repeat"], r["tokens"], r["timings"]["totalMs"], r["timings"]["prefillMs"], r["text"] == r["reference"], r["tokenLimit"]])
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#0c1525", "axes.facecolor": "#0c1525",
        "axes.edgecolor": "#42526b", "text.color": "#eef3fb", "axes.labelcolor": "#cbd5e1", "xtick.color": "#a2b3ca",
        "ytick.color": "#a2b3ca", "grid.color": "#334155", "font.size": 10})
    figures = []
    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=170)
        fig.savefig(directory / f"{name}.svg")
        plt.close(fig)
        svg = (directory / f"{name}.svg").read_text()
        figures.append(svg[svg.index("<svg"):])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    ax.plot(np.sort(total), np.arange(1, len(total) + 1) / len(total) * 100, color="#5fe0c4", linewidth=2, drawstyle="steps-post")
    ax.set(xlabel="Durée par bulle (ms)", ylabel="Appels terminés (%)", title="Distribution des latences")
    ax = axes[0, 1]
    ax.scatter(tokens, total, color="#94a3ff", alpha=.65, s=25)
    ax.set(xlabel="Tokens produits", ylabel="Durée totale (ms)", title="Latence selon la longueur")
    ax = axes[1, 0]
    ids = list(dict.fromkeys(r["id"] for r in measured))
    f = [np.median([r["timings"]["prefillMs"] for r in measured if r["id"] == key]) for key in ids]
    d = [np.median([r["timings"]["totalMs"] - r["timings"]["prefillMs"] for r in measured if r["id"] == key]) for key in ids]
    ax.bar(range(len(ids)), f, color="#94a3ff", label="Jusqu’au premier token")
    ax.bar(range(len(ids)), d, bottom=f, color="#5fe0c4", label="Génération")
    ax.set(xlabel="Bulle, par longueur croissante", ylabel="Durée médiane (ms)", title="Répartition du temps observé")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    ax.scatter(tokens, tokens * 1000 / np.maximum(decode, 1), color="#ffd580", alpha=.6, s=25)
    ax.set(xlabel="Tokens produits", ylabel="Tokens / seconde", title="Débit de génération")
    for ax in axes.flat:
        ax.grid(alpha=.25, axis="y")
    save(fig, "latencies")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    ax = axes[0]
    all_total = np.array([r["timings"]["totalMs"] for r in test])
    ax.hist(all_total, bins=35, color="#5fe0c4", alpha=.9)
    ax.set(xlabel="Durée (ms)", ylabel="Bulles", title=f"Endurance · {len(test):,} appels".replace(",", " "))
    ax = axes[1]
    bins = [0, 10, 30, 60, 120, 10000]
    subsets = [[r for r in test if lo < len(r["groundTruth"]) <= hi] for lo, hi in zip(bins, bins[1:])]
    values = [quality(rows)[0]["cer"] * 100 for rows in subsets]
    ax.bar(["1–10", "11–30", "31–60", "61–120", ">120"], values, color="#94a3ff")
    ax.set(xlabel="Longueur de la référence (caractères)", ylabel="CER (%)", title="Erreurs selon la longueur")
    ax = axes[2]
    runs = sorted(set(r["repeat"] for r in measured))
    ax.bar([str(i + 1) for i in runs], [np.median([r["timings"]["totalMs"] for r in measured if r["repeat"] == i]) for i in runs], color="#ffd580")
    ax.set(xlabel="Passe de production", ylabel="Latence médiane (ms)", title="Stabilité des mesures")
    for ax in axes:
        ax.grid(alpha=.25, axis="y")
    save(fig, "quality-and-stability")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ax = axes[0]
    for mode, color in [("shared", "#94a3ff"), ("subgroup", "#5fe0c4")]:
        values = [np.median([r["tokens"] * 1000 / (r["timings"]["totalMs"] - r["timings"]["prefillMs"]) for r in sweep if r["options"]["gemv"] == mode and r["options"]["tokenBatch"] == n]) for n in [1, 2, 4, 8]]
        ax.plot([1, 2, 4, 8], values, "o-", label="Sous-groupes" if mode == "subgroup" else "Mémoire partagée", color=color)
    ax.set(xlabel="Tokens par soumission GPU", ylabel="Tokens / seconde", title="Choix du réglage · 3 répétitions")
    ax.legend(frameon=False, fontsize=8)
    profile_path = directory / "final-profile.json"
    ax = axes[1]
    if profile_path.exists():
        prof = json.loads(profile_path.read_text())[0]["profile"]
        entries = sorted(prof.items(), key=lambda pair: pair[1]["gpuMs"], reverse=True)[:8][::-1]
        ax.barh([name.replace(".weight", "") for name, _ in entries], [v["gpuMs"] for _, v in entries], color="#5fe0c4")
        ax.set(xlabel="Temps GPU cumulé (ms)", title="Profil des kernels · passe instrumentée")
    else:
        ax.axis("off")
    save(fig, "tuning-and-profile")
    cards = [("Latence médiane", f'{np.median(total):.0f} ms'), ("Débit de génération", f'{metrics["decode_tokens_per_second"]:.0f} tok/s'),
             ("Premier token médian", f'{np.median(first):.0f} ms'), ("CER · test complet", f'{q["cer"] * 100:.3f} %')]
    cards_html = ''.join(f'<article><small>{name}</small><strong>{value}</strong></article>' for name, value in cards)
    mismatch_count = q["samples"] - q["native_conformity"]
    rows_html = ''.join(f'<tr><td>{html.escape(r["id"])}</td><td>{r["tokens"]}</td><td>{r["timings"]["prefillMs"]}</td><td>{r["timings"]["totalMs"]}</td><td>{"Oui" if r["text"] == r["reference"] else "Écart"}</td></tr>' for r in production)
    page = f'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Surya · WebGPU optimisé</title>
<style>body{{background:#0c1525;color:#eef3fb;font:16px system-ui;margin:0}}main{{max-width:1160px;margin:auto;padding:44px 28px}}h1{{font-size:46px;margin:14px 0}}h2{{font-size:25px;margin:32px 0 12px}}p{{color:#b5c4d9;line-height:1.65}}.eyebrow{{color:#5fe0c4;font-size:12px;letter-spacing:3px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}}article{{border:1px solid #35415b;border-radius:14px;padding:20px}}small{{color:#b5c4d9}}strong{{display:block;font-size:30px;margin-top:10px}}svg{{width:100%;height:auto}}.note{{border-left:3px solid #ffd580;padding:12px 20px;background:#142036}}a{{color:#5fe0c4}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #35415b}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:34px}}}}</style>
<main><div class="eyebrow">PONEGLYPH · RTX 5090 · WEBGPU</div><h1>Surya, accéléré sur WebGPU.</h1>
<p>Un décodeur GPU spécialisé, des caches réutilisés, un prompt ONNX allégé et des soumissions de 8 tokens. Les poids restent en FP16 ; les accumulations du décodeur et les états récurrents restent en FP32.</p>
<div class="cards">{cards_html}</div>
<p>{len(production)} appels de production sur 26 images, après échauffement, et {len(test)} bulles du test complet. Téléchargement et chargement initial exclus. Les deux images de résistance sont exclues des agrégats de latence de production. Débit : total des tokens texte divisé par le temps après le premier token, détection d’EOS comprise.</p>
{figures[0]}
<h2>Qualité et endurance</h2><p>Test complet : CER <b>{q["cer"]*100:.3f} %</b>, WER <b>{q["wer"]*100:.3f} %</b>, transcriptions exactes <b>{q["exact_rate"]*100:.2f} %</b>. {q["blank"]} sortie vide, {q["truncated"]} limite atteinte. {q["character_errors"]} éditions de caractères sur {q["characters"]} caractères de référence.</p>
<p class="note">Conformité au benchmark natif enregistré : {q["native_conformity"]}/{len(test)} transcriptions identiques, {mismatch_count} différentes. Le moteur optimisé n’est donc pas identique bit à bit au chemin natif. Les scores ci-dessus sont mesurés sur ses propres sorties. Le chemin ONNX d’origine reste disponible avec <code>mode: 'onnx'</code>.</p>
{figures[1]}
<h2>Profilage et choix des réglages</h2><p>Huit configurations, trois répétitions chacune sur la même bulle longue. Le profil des kernels utilise des requêtes de timestamps GPU dans une passe séparée ; son instrumentation est exclue des mesures de production.</p>{figures[2]}
<h2>Reproduire et inspecter</h2><p><a href="metrics.json">Métriques JSON</a> · <a href="measurements.csv">Mesures CSV</a> · <a href="latencies.png">Figure PNG</a> · <a href="full-test1612.json">Prédictions du test, locales</a> · <a href="http://127.0.0.1:3003/sandbox/surya">Essayer Surya</a></p>
<details><summary>Les {len(production)} appels de contrôle en production</summary><table><thead><tr><th>Bulle</th><th>Tokens</th><th>Premier token (ms)</th><th>Total (ms)</th><th>Conforme</th></tr></thead><tbody>{rows_html}</tbody></table></details>
<p>Les crops et références détaillées restent sur ce poste. Les fichiers publiés sur Hugging Face contiennent les poids, graphes, index vérifiés et documentation.</p></main></html>'''
    (directory / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key not in ["stress"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory)
