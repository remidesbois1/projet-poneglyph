"""Combine both training phases and the final paired test into durable figures."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def history(directory):
    return [json.loads(line) for line in (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
            if json.loads(line).get('event') == 'metrics']


def report(phase1, phase2, selected, comparison, average=None):
    first, second = history(phase1), history(phase2)
    validations = [[r for r in values if 'eval_cer' in r] for values in (first, second)]
    old, new = comparison['surya'], comparison['lighton3']
    verification = read(selected / 'model_verification.json')
    configs = [read(p / 'run_config.json') for p in (phase1, phase2)]
    candidates = [{'run': p.name, 'native_validation_cer': read(p/'model_verification.json')['fresh_native_bf16_validation_cer']}
                  for p in [phase1, phase2] + ([average] if average else [])]
    if selected.name != min(candidates,key=lambda c:c['native_validation_cer'])['run']:
        raise ValueError('Release does not have the lowest native validation CER among evaluated candidates')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.5), constrained_layout=True)
    colors = ['#377c9b', '#d08b40']
    offset = 0
    for phase, rows in enumerate(validations):
        x = np.arange(1, len(rows)+1) + offset
        label = 'Fine-tuning' if phase == 0 else 'Consolidation'
        for col, key, title in [(0,'eval_cer','CER validation (%)'), (1,'eval_wer','WER validation (%)'), (2,'eval_exact_match','Bulles exactes validation (%)')]:
            axes[0,col].plot(x,[r[key]*100 for r in rows],marker='o',color=colors[phase],label=label)
            axes[0,col].set_title(title.replace('validation', 'validation · historique'))
            axes[0,col].set_xlabel('Évaluation complète · 1 648 crops')
        offset += len(rows)
    axes[0,0].set_yscale('log')
    axes[0,1].set_yscale('log')
    axes[0,0].legend(frameon=False)
    if average:
        axes[0,0].scatter([offset+1], [read(average/'model_verification.json')['legacy_native_validation_cer']*100], color='#497d45',marker='*',s=160,label='Lissage natif 75/25 · historique')
        axes[0,0].legend(frameon=False)
    update_offset = 0
    for phase, rows in enumerate((first, second)):
        for col, key, title in [(0,'loss','Perte · EOS ×1 puis ×2'),(1,'train_bubbles_per_second','Débit entraînement (bulles/s)'),(2,'gpu_peak_gib','Pic VRAM allouée (GiB)')]:
            points = [(r['step']+update_offset,r[key]) for r in rows if key in r and isinstance(r[key],(int,float))]
            if points:
                axes[1,col].plot(*zip(*points),color=colors[phase],alpha=.8)
            axes[1,col].set_title(title)
            axes[1,col].set_xlabel('Updates réalisés · deux phases')
        update_offset += max(r['step'] for r in rows)
    axes[1,0].set_yscale('log')
    for col,key,title,scale in [(0,'cer','CER test (%)',100),(1,'wer','WER test (%)',100),(2,'exact_match','Bulles exactes test (%)',100)]:
        axis = axes[2,col]
        values = [old[key]*scale,new[key]*scale]
        axis.bar(['Surya','LightOnOCR-3'],values,color=['#8c98a4','#377c9b'],width=.55)
        axis.set_title(title)
        axis.set_ylim(0,max(values)*1.23)
        for position,value in enumerate(values):
            axis.text(position,value,f'{value:.3f} %',ha='center',va='bottom')
    fig.suptitle('Poneglyph · LightOnOCR-3 0.8B · Transcription de bulles\nSélection sur validation · Test identique : 1 612 crops / 185 pages',fontsize=17)
    fig.savefig(selected / 'experiment_curves.png',dpi=190)
    fig.savefig(selected / 'experiment_curves.svg')
    plt.close(fig)
    summary = {'schema':1,'selected_run':selected.name,'verification':verification,'comparison':comparison,'validation_candidates':candidates,
               'phases':[{'run':p.name,'config':c,'training_metrics':read(p/'train_results.json')} for p,c in zip((phase1,phase2),configs)]}
    (selected / 'experiment_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    ci = new['confidence_intervals']['cer_95']
    delta = comparison['new_minus_surya_cer_95']
    historic = comparison['historical_quote_stripping']
    text = f'''# LightOnOCR-3 0.8B — transcription de bulles

Modèle retenu : `{selected.name}`, checkpoint `{Path(verification['selected_checkpoint']).name}`.
Validation native BF16 : **{verification['fresh_native_bf16_validation_cer']*100:.4f} % de CER**.

| Même test figé | Surya | LightOnOCR-3 |
| --- | ---: | ---: |
| CER | {old['cer']*100:.4f} % | {new['cer']*100:.4f} % |
| WER | {old['wer']*100:.4f} % | {new['wer']*100:.4f} % |
| Exact match | {old['exact_match']*100:.2f} % | {new['exact_match']*100:.2f} % |
| Erreurs de caractères | {comparison['surya_character_errors']} | {comparison['lighton3_character_errors']} |

Test : 1 612 crops, 185 pages, {comparison['reference_characters']:,} caractères.
Références et IDs vérifiés identiques à Surya. CER à 95 % par bootstrap de pages :
{ci[0]*100:.4f}–{ci[1]*100:.4f} %. Différence appariée de CER (LightOn − Surya), IC 95 % :
{delta[0]*100:+.4f}–{delta[1]*100:+.4f} point. Le test avait été évalué pour Surya ;
il n'a servi à aucun choix de checkpoint, LR ou pondération dans cet entraînement.

Initialisation originale LightOnOCR-3, pas les poids bbox (splits incompatibles).
7496 / 1648 / 1612 crops répartis strictement par pages ; zéro copie de pixels entre splits.
Prompt vide, image seule, thinking désactivé. Safetensors natif BF16 ; les
{verification['all_native_tensors_verified']} tensors égalent le cast BF16 du checkpoint FP32 retenu.
Validation et test natifs vérifiés sans les patches Liger d'entraînement.

![Courbes et comparaison](experiment_curves.png)

Les courbes de perte des deux phases utilisent des pondérations EOS différentes.
Le CER inclut intégralement les sorties erronées, vides ou tronquées.
Le protocole corrigé préserve les guillemets visibles et normalise uniquement
les espaces. Scores test historiques (suppression des guillemets des prédictions) :
Surya {historic['surya']['cer']*100:.4f} %, LightOn {historic['lighton3']['cer']*100:.4f} %.
Les courbes de validation gardent ce protocole historique ; les scores de test
et la sélection native finale utilisent le protocole corrigé pour les deux modèles.
Audit visuel local : `visual_data_audit.html` (34 crops, deux erreurs d’annotation
certaines et plusieurs cas à confirmer ; aucun label ni split modifié).
Voir `report.html` pour les erreurs détaillées et `experiment_summary.json` pour la recette.
'''
    (selected / 'experiment_report.md').write_text(text,encoding='utf-8')
    model_dir = selected / 'final_merged'
    shutil.copyfile(selected / 'experiment_curves.png', model_dir / 'training_curves.png')
    card = model_dir / 'README.md'
    beginning, end = '<!-- experiment:start -->', '<!-- experiment:end -->'
    previous = card.read_text(encoding='utf-8')
    if beginning in previous:
        previous = previous.split(beginning)[0] + previous.split(end,1)[1]
    phase_note = ('The consolidation checkpoint was retained.' if selected.name == phase2.name
                  else 'The first-phase checkpoint was retained; consolidation did not improve native validation CER.')
    if verification.get('checkpoint_average'):
        weight = verification['checkpoint_average']['second_parent_weight']
        phase_note = f'The FP32 interpolation candidate ({1-weight:.0%} first-phase checkpoint 1175 / {weight:.0%} consolidation checkpoint 705) was retained.'
    candidate_table = '\n'.join(f"| {c['run']} | {c['native_validation_cer']*100:.4f}% |" for c in candidates)
    block = f'''\n{beginning}
## Reproducible comparison

| Identical frozen test | Surya bubble fine-tune | This LightOnOCR-3 model |
| --- | ---: | ---: |
| CER | {old['cer']*100:.4f}% | {new['cer']*100:.4f}% |
| WER | {old['wer']*100:.4f}% | {new['wer']*100:.4f}% |
| Exact match | {old['exact_match']*100:.2f}% | {new['exact_match']*100:.2f}% |
| Character edits / 52,064 reference characters | {comparison['surya_character_errors']} | {comparison['lighton3_character_errors']} |

IDs, pages, references, CER, WER and exact match were independently checked.
Paired page-cluster bootstrap, 5,000 resamples: LightOn minus Surya CER
95% interval **{delta[0]*100:+.4f} to {delta[1]*100:+.4f} percentage points**.
Surya comparison: [surya-bubble-ocr-poneglyph frozen benchmark](https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/blob/7d7b358c545cfe757329f780da6ed4100bb5909f/benchmark_test.json), dated 2026-10-05.

First phase: decoder LR 1.2e-5, six epochs, 60 warmup updates, cosine decay,
effective batch 32 (32 crops / update), mild photometric augmentation.
Second phase: initialized from first-phase checkpoint 1175, decoder LR 3e-6,
three epochs, ten warmup updates, no augmentation and EOS loss weight doubled.
Vision and merger LR multipliers are 0.25 and 0.5 throughout.
{phase_note} Selection compared native BF16 validation CER before opening test.
Native validation CER of this export: **{verification['fresh_native_bf16_validation_cer']*100:.4f}%**.

| Candidate evaluated before test | Native BF16 validation CER |
| --- | ---: |
{candidate_table}

Scoring was corrected before opening this run's test: visible quotes are kept,
and only whitespace is normalized for both models. Historical quote-stripping
test CER: Surya {historic['surya']['cer']*100:.4f}%, LightOn {historic['lighton3']['cer']*100:.4f}%.
The paired CER interval above includes zero; these data do not establish a clear
CER advantage over Surya. Validation curves show the historical scoring; native
candidate selection and test comparison use the corrected protocol.

## Annotation audit

34 local crops were visually inspected: 18 targeted validation cases, six
training repetition/onomatopoeia cases and ten random training crops (seed 42).
Two validation annotations were clearly incorrect (repetition count and an
onomatopoeia); other stylized cases remain uncertain. All 16 inspected training
crops were consistent, which does not certify the full training dataset.
Of 169 non-exact validation outputs under historical scoring, 55 differed only
in case (67 of 264 character edits). Annotation capitalization conventions are
mixed, especially for onomatopoeia. No labels or splits were changed for these
results. Review images and detailed private error examples remain local.

![Training, validation and frozen test comparison](training_curves.png)

The two training loss curves use different EOS weighting. Test was evaluated
once for the retained release, after all checkpoint and hyperparameter decisions.
These scores measure this local manga dataset; the paired confidence interval
describes uncertainty relative to the historical Surya benchmark.
{end}
'''
    card.write_text(previous.rstrip()+'\n'+block,encoding='utf-8')
    print(json.dumps({'selected':selected.name,'native_validation_cer':verification['fresh_native_bf16_validation_cer'],
                      'test_cer':new['cer'],'test_wer':new['wer'],'test_exact_match':new['exact_match']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1',type=Path,required=True)
    parser.add_argument('--phase2',type=Path,required=True)
    parser.add_argument('--selected',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--average',type=Path)
    args=parser.parse_args()
    report(args.phase1,args.phase2,args.selected,read(args.comparison),args.average)
