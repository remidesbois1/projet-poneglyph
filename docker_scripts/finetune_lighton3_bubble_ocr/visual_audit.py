"""Local-only record of 34 crops actually inspected during the training audit."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil


VAL = {
    '3804': ('annotation_erronee', 'Certain', 'Trois HA visibles (un sur la première ligne, deux sur la seconde). La référence en compte quatre ; le modèle en produit correctement trois.'),
    '10780': ('annotation_erronee', 'Certain', 'Le bruitage visible est GLURP, annoté Glups. La prédiction Glupo reste imparfaite ; cette cible ne représente pas le texte visible.'),
    '786': ('evaluation', 'Certain', 'Les guillemets sont visibles et correctement générés. Le protocole hérité les supprimait seulement de la prédiction, ajoutant deux erreurs artificielles. La casse Roi/Pirates reste une convention d’annotation.'),
    '11070': ('modele', 'Certain', 'DIGNES est lisible dans l’image. L’annotation est correcte ; le modèle omet ce mot.'),
    '206': ('modele', 'Certain', 'ZIZANIE et TONNEAUX sont bien visibles. Le modèle déforme ces deux mots.'),
    '5959': ('modele', 'Certain', 'FRÉROT est visible ; le modèle produit froid. Cible correcte.'),
    '10781': ('modele', 'Certain', 'Quatre GLOUB sont visibles. La référence est cohérente ; le modèle produit trois Gloup.'),
    '10392': ('modele', 'Certain', 'TEMPE et REDEVIENT sont lisibles. Le modèle les altère ; lambda/Lambda relève également de la casse.'),
    '7041': ('controle', 'Certain', 'ATCHOUM !! est visible. Le checkpoint retenu transcrit correctement Atchoum !! ; certaines consolidations perdaient le O.'),
    '1415': ('a_confirmer', 'Probable', 'Le petit lettrage semble YAAHHH plutôt que YAAAHH. Nombre de A/H à confirmer sur la page originale ; aucune correction automatique.'),
    '3240': ('a_confirmer', 'Probable', 'Le lettrage ressemble à BROLOM avec une ponctuation expressive. Les points de suspension de la référence semblent douteux ; lecture verticale à confirmer.'),
    '10803': ('a_confirmer', 'Probable', 'Deux segments BRO / BROOO sont visibles. La ponctuation terminale semble des points d’exclamation plutôt que des points de suspension ; à confirmer.'),
    '5401': ('ambigu', 'Ambigu', 'PLOM apparaît deux fois. La ponctuation est petite et stylisée ; la virgule et les points de suspension de la cible doivent être revus sur la page originale.'),
    '10785': ('ambigu', 'Ambigu', 'Lettrage incliné et très stylisé. Le premier caractère et le nombre de A/H sont difficiles à trancher ; ni cible ni prédiction ne sont certifiées.'),
    '10782': ('modele', 'Probable', 'Deux segments BRO / BROOO semblent visibles ; la sortie Bpooooo contracte le bruitage. Lecture verticale difficile.'),
    '6202': ('modele', 'Probable', 'Le bruitage stylisé paraît PSCHIII. La cible est plausible ; le modèle remplace les I par des H.'),
    '1417': ('casse', 'Certain', 'ARGH... est visible. Le modèle produit Argh... ; la cible attend les majuscules. Les autres bruitages du dataset sont souvent en casse de phrase.'),
    '3153': ('casse', 'Certain', 'GWIP... est visible et produit par le modèle. La référence Gwip... impose une autre mise en forme.'),
}
TRAIN = ['3665','7866','6185','4528','11306','10948','9193','7992','3990','7614','3289','4029','5721','1780','7163','2260']
COLORS = {'annotation_erronee':'#d27f78','evaluation':'#cfab62','modele':'#8eb7cd','controle':'#8db99a','a_confirmer':'#d0b480','ambigu':'#b3a7c9','casse':'#9cadbf','train_coherent':'#8db99a'}


def audit(dataset, benchmark, output, second=None):
    payload=json.loads(benchmark.read_text(encoding='utf-8'))
    samples={str(s['id']):s for s in payload['samples']}
    later={} if second is None else {str(s['id']):s for s in json.loads(second.read_text(encoding='utf-8'))['samples']}
    records=[]
    for split, ids in [('val',list(VAL)),('train',TRAIN)]:
        metadata={str(s['id']):s for s in map(json.loads,(dataset/split/'metadata.jsonl').read_text(encoding='utf-8').splitlines())}
        for identity in ids:
            meta=metadata[identity]
            kind,confidence,note=VAL[identity] if split=='val' else ('train_coherent','Lecture cohérente','Texte et répétitions cohérents avec l’image inspectée ; mise en forme française des majuscules.')
            image=output/'audit_images'/f'{split}-{identity}.png'
            image.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(dataset/split/meta['image_file'],image)
            record={'split':split,'id':identity,'page_id':meta['page_id'],'image':f'audit_images/{image.name}',
                'reference':meta['text'],'category':kind,'confidence':confidence,'visual_observation':note}
            if split=='val':
                record.update(raw_prediction=samples[identity]['raw_prediction'],historical_scored_prediction=samples[identity]['prediction'])
                if identity in later:
                    record['consolidation_prediction']=later[identity]['raw_prediction']
            if identity=='10780':
                record['proposed_reference']='Glurp'
            elif identity=='3804':
                record['proposed_reference']="Ha, ha, ha !! Alors l'élastique, tu veux toujours essayer de venir en aide à ta petite camarade ?!"
            records.append(record)
    errors=[s for s in samples.values() if s['reference']!=s['prediction']]
    case_only=[s for s in errors if s['reference'].casefold()==s['prediction'].casefold()]
    overview={'crops_visually_inspected':len(records),'validation_reviewed':len(VAL),'training_reviewed':len(TRAIN),
        'selection':'18 targeted validation cases; train: six targeted repeated/onomatopoeia texts and ten random crops (seed 42)',
        'validation_non_exact_historical':len(errors),'case_only_crops':len(case_only),
        'historical_character_errors':sum(s['levenshtein'] for s in errors),
        'case_only_character_errors':sum(s['levenshtein'] for s in case_only),
        'categories':dict(Counter(r['category'] for r in records)), 'dataset_modified':False,
        'scope':'Only train/validation images inspected; no test-image audit; small sample, not a global dataset error-rate estimate'}
    result={'schema':1,'overview':overview,'records':records}
    (output/'visual_data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    esc=lambda s:html.escape(str(s))
    cards=[]
    for r in records:
        prediction=(f'<dt>Modèle retenu · sortie brute</dt><dd>{esc(r["raw_prediction"])}</dd>'
                    if 'raw_prediction' in r else '')
        if 'consolidation_prediction' in r:
            prediction+=f'<dt>Consolidation · sortie brute</dt><dd>{esc(r["consolidation_prediction"])}</dd>'
        cards.append(f'''<article style="--accent:{COLORS[r['category']]}"><div class="label">{esc(r['split'])} · ID {r['id']} · page {r['page_id']} · {esc(r['category'])} · {esc(r['confidence'])}</div><div class="crop"><img src="{r['image']}" alt="Crop {r['id']}"></div><dl><dt>Annotation actuelle</dt><dd>{esc(r['reference'])}</dd>{prediction}</dl><p class="observation">{esc(r['visual_observation'])}</p></article>''')
    document='''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Poneglyph · Audit visuel du dataset</title><style>
    :root{color-scheme:dark;font:15px system-ui;background:#111820;color:#e6edf4}*{box-sizing:border-box}body{margin:0 auto;padding:28px;max-width:1460px}h1{font-size:32px;letter-spacing:-.8px;margin:12px 0}h2{font-size:18px}p{line-height:1.6;color:#bac8d4}.tag,.label,dt{font-size:12px;color:#9fb2c2}.tag{letter-spacing:2px}header{border-bottom:1px solid #354250;padding-bottom:20px}.stats{display:flex;gap:28px;flex-wrap:wrap;margin:22px 0}.stats strong{font-size:27px;display:block}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}article{border:1px solid #354250;border-top:3px solid var(--accent);padding:18px;background:#18232d}.label{color:var(--accent);margin-bottom:14px}.crop{height:300px;background:#eff1f3;display:flex;align-items:center;justify-content:center;overflow:hidden}.crop img{max-width:100%;max-height:100%;width:auto;height:auto;image-rendering:auto;min-height:min(200px,100%)}dl{margin:18px 0}dt{margin-top:14px}dd{margin:6px 0;font-size:15px;white-space:pre-wrap;overflow-wrap:anywhere}.observation{border-top:1px solid #354250;padding-top:12px;color:#d4dfe7}a{color:#a9cce6}footer{margin:25px 0}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.grid{grid-template-columns:1fr}body{padding:16px}}</style></head><body>
    <header><div class="tag">PONEGLYPH / LIGHTONOCR-3 / AUDIT LOCAL</div><h1>Ce qui reste à apprendre — et ce qui est mal annoté</h1><p>34 crops effectivement examinés : 18 cas ciblés de validation, 6 bruitages/répétitions d’entraînement et 10 exemples d’entraînement tirés avec seed 42. Les images sont les fichiers originaux, affichés plus grands. Ce petit échantillon ne mesure pas la qualité de tout le dataset.</p><div class="stats"><div><strong>2</strong>annotations certainement incorrectes</div><div><strong>55 / 169</strong>écarts de validation uniquement de casse</div><div><strong>16 / 16</strong>crops train examinés cohérents</div></div><p>Le modèle fait aussi de vraies omissions et confusions. Le scoring hérité supprimait des guillemets réellement visibles : les sorties brutes sont maintenant évaluées en préservant guillemets, casse, accents et ponctuation. Anciens scores conservés ; comparaison Surya recalculée avec le même protocole.</p><p>Les fichiers du dataset et leurs splits restent intacts. Les propositions ci-dessous servent à la revue ; elles ne modifient aucun label. Aucun crop de test n’a été utilisé dans cet audit.</p></header><h2>Images, annotations et observations</h2><div class="grid">'''+''.join(cards)+'''</div><footer><a href="training_dashboard.html">Suivi et résultat final</a> · <a href="visual_data_audit.json">Audit structuré</a></footer></body></html>'''
    (output/'visual_data_audit.html').write_text(document,encoding='utf-8')
    print(json.dumps(overview,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--second',type=Path)
    args=parser.parse_args()
    audit(args.dataset,args.benchmark,args.output,args.second)
