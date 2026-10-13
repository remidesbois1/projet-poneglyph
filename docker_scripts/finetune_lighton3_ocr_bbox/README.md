# LightOnOCR-3 0.8B — Poneglyph / RTX 5090

Entraînement sur **pages entières**, textes et bbox, du seul
`lightonai/LightOnOCR-3-0.8B`. Le modèle utilise Qwen3.5.

Le prompt natif est exactement **`grounding`**, avec `enable_thinking=False` :

```text
![text](826,58,980,106)
Nous sommes dans un petit village maritime.
```

Les bbox sont normalisées sur 0–1000. Les zones supervisées sont les bulles/textes
validés de Poneglyph, dans leur ordre annoté ; les panneaux et illustrations n'ont
pas d'annotations de description. Cette spécialisation ne conserve donc pas une
supervision complète des descriptions d'images du modèle généraliste.

La fiche du modèle recommande des images limitées à 2048 px côté long. L'export
utilise les images source, sans agrandir les sources plus petites. Il réutilise
les accès Supabase/R2 du pipeline Surya, et contrôle les fuites de pages et les
images identiques entre train/val/test. Il refuse les pages contenant encore des
propositions non validées ou des zones acceptées sans ordre ; les candidats
rejetés sont volontairement exclus. Les métadonnées et la révision modèle
`36d8636925327c671bd8e479c517a004f7566bd9` sont figées par empreintes.

## Profil de qualité

- Mode hybride du dernier Surya : décodeur complet, embeddings/head liés,
  merger et 4 derniers blocs visuels ; `--train-mode full` permet tous les poids.
- Poids maîtres, gradients et états Adam en **FP32**, calcul matriciel sous
  autocast **BF16** ; export final BF16.
- AdamW fusionné, LR décodeur `1.2e-5`, merger ×0.5, vision ×0.25, decay 0.05,
  clipping 1.0, warmup 5 %, cosine avec LR final à 10 %.
- 5 époques maximum, arrêt anticipé après 2 validations sans amélioration de
  0.001 ; meilleur checkpoint selon le score combinant CER, F1@0.5 et IoU.
- Batch effectif 8 pages, calibration physique 1/2/4/8 sur les pages les plus
  longues avec de vrais backward/états optimizer ; marge VRAM mesurée.
  Le deuxième forward conserve les gradients FP32 déjà alloués, comme les
  microbatches suivants d'une accumulation ; cela évite de sous-estimer le pic.
- FLA Gated DeltaNet, causal-conv1d SM120, SDPA, Liger RMSNorm/SwiGLU et
  projection+cross-entropy fusionnée limitée aux tokens assistant. La loss est
  normalisée sur tous les tokens de la fenêtre d'accumulation, y compris la fin.
- Pas de quantification ; checkpointing seulement si nécessaire ; cache images
  borné, workers 0 pour éviter les blocages fork/CUDA déjà rencontrés avec Surya.
- Augmentations photométriques légères, sans déplacement des bbox.

## Lancement

La base Docker locale `poneglyph/surya-onnx:20261005` contient les kernels de ton
entraînement précédent. Le Dockerfile ajoute Liger et le suivi. Pour une autre
machine, fournir une base compatible via `--build-arg TRAINING_BASE=...`.

```powershell
.\run_training.ps1 -Build -Export -Smoke
docker logs -f lighton3-bbox-smoke
# Après réussite du smoke test :
.\run_training.ps1
docker logs -f lighton3-bbox-training
```

Les containers sont détachés : fermer le terminal n'arrête pas le run. Les sorties
et caches sont montés dans ce dossier. Une reprise conserve `--output` et
`--resume auto` (par défaut). La révision, les données et les paramètres principaux
sont vérifiés avant reprise. Supprimer uniquement un container terminé avant de
réutiliser son nom ; les checkpoints restent dans le dossier monté.
`-Recalibrate` permet une nouvelle mesure du batch physique en conservant le batch
effectif, les états optimizer, le scheduler et les données. Les courbes repartent
du checkpoint chargé ; les logs bruts conservent aussi les tentatives abandonnées.

## Suivi et validation

`outputs/<run>/training_dashboard.html` : loss train/validation, CER, F1/IoU,
LR, VRAM, avancement et ETA. Les mesures GPU (activité, température, puissance)
sont enregistrées dans `metrics.jsonl` et `training_summary.json`.
TensorBoard est dans `runs/`. Trackio reste local dans `trackio/`.
`report.py <run>` produit également `training_curves.png` / `.svg` et `results.json`
pour conserver ou comparer les résultats. Le rapport est généré après le benchmark
final ; on peut aussi l'actualiser pendant le run. Un trait vertical indique une
reprise, et les étapes abandonnées restent uniquement dans les logs bruts.

```powershell
# Dans un deuxième terminal, ouvrir un suivi local actualisé :
.\serve_dashboard.ps1
# Puis http://127.0.0.1:8766/training_dashboard.html
```

Le budget de génération est de 1280 tokens ; le contrôle préalable vérifie que
toutes les réponses supervisées tiennent entièrement dans ce budget.

La loss de validation utilise toutes les pages val. Une sélection fixe de 48
pages sparse/denses est générée à chaque époque, sans réponse dans le prompt.
Une baseline avant adaptation utilise ces mêmes pages. Les bbox sont appariées
par assignation optimale ; zones manquées et zones supplémentaires pénalisent le
CER. L'ordre de lecture, les sorties malformées et les troncatures sont mesurés.

`validation_step_*.json` conserve les réponses par page. Les dossiers `*_overlays`
superposent annotations vertes et prédictions rouges sur les pages les moins
réussies. Le **test complet** ne sert jamais à choisir le checkpoint ; le benchmark
final produit les métriques et des intervalles bootstrap à 95 % sur 1000 tirages.

Le CER est sensible à la casse et à la ponctuation, comme les annotations de
Poneglyph. Le modèle de base produit fréquemment des descriptions de panneaux
ou une transcription en majuscules : la baseline mesure son adéquation au
contrat manga de ce dataset, pas sa qualité OCR générale sur d'autres documents.
Dans ce run, le LR natif du Trainer correspond au premier groupe optimizer
(vision, ×0.25) ; les LR décodeur/merger sont explicités dans `run_config.json`.

Le modèle final natif est dans `final_merged/`, prêt à être chargé avec
`AutoProcessor` et `Qwen3_5ForConditionalGeneration`. Aucun upload automatique.

## Utilisation du modèle exporté

`infer.py` charge les poids et le processor sauvegardés, sans patch d'entraînement.
Il produit la sortie native, les bbox normalisées et en pixels, et peut enregistrer
une superposition des prédictions. Depuis le dossier de ce pipeline :

```powershell
$pageMount = "${PWD}:/workspace"
docker run --rm --gpus all -v $pageMount poneglyph/lighton3-bbox:5090 infer.py `
  --model /workspace/outputs/rtx5090-20261011/final_merged `
  --image /workspace/dataset/test/images/page_1003.jpg `
  --output /workspace/outputs/rtx5090-20261011/inference.json `
  --overlay /workspace/outputs/rtx5090-20261011/inference.jpg
```

Remplacer `page_1003.jpg` par la page voulue. Les résultats de benchmark utilisent
la génération greedy native, sans correction des erreurs ni filtre de répétition.

Source du contrat : [fiche officielle](https://huggingface.co/lightonai/LightOnOCR-3-0.8B).

## Run terminé — RTX 5090, 11 octobre 2026

Le run `rtx5090-20261011` a terminé ses 5 époques (555 updates).
Le checkpoint retenu est celui de l'**époque 3, étape 333**, choisi uniquement
sur la validation. Les données figées comprennent 882 pages train, 189 val et
189 test ; 11 030 zones validées au total.

| Mesure | Validation fixe, 48 pages | Test complet, 189 pages |
|---|---:|---:|
| CER, casse/ponctuation incluses | 1,20 % | 4,40 % |
| F1 bbox, IoU ≥ 0,5 | 99,01 % | 98,01 % |
| IoU moyen, zones manquées pénalisées | 0,905 | 0,901 |
| Pages tronquées | 0 | 1 |

Le rescoring `rescore_saved.py --run <run> --repo <repo>` ajoute le WER et des
exact match textuels aux prédictions figées, sans nouvelle inférence. Sur le test :
WER 3,41 %, WER avec ordre 4,41 %, texte exact par bulle 89,25 % (pondération
par nombre de bulles), texte de page exact dans l'ordre 38,10 % (72/189), toutes
zones et textes exacts avec IoU ≥ 0,5 42,86 % (81/189). L'ancien `exact_match`
compare littéralement la sortie entière, coordonnées incluses : 0 %. L'ancien
`matched_text_exact` est une moyenne par page (89,29 %), distincte du taux par bulle.
Les mots sont des tokens séparés par des espaces ; casse/ponctuation conservées,
zones manquées/ajoutées comptées comme suppressions/insertions.

`ocr_metrics_extended.json` conserve ces définitions, leurs intervalles bootstrap
et les comparaisons avec les prédictions historiques. Il contient notamment
42 pages communes avec Surya et 28 avec LightOnOCR-2, avec annotations identiques
et contrôle des pixels RGB des images conservées. Les benchmarks originaux
restent inchangés. Les petits sous-ensembles communs ne constituent pas un nouveau
benchmark commun complet ; les historiques de train des anciens runs ne sont pas
recertifiés par ce rescoring.

Une page de test (`1233`) produit une répétition de chiffres jusqu'au budget de
1280 tokens. L'échec est inclus dans les métriques, et reproduit après rechargement
natif sur une page seule. L'intervalle bootstrap à 95 % du CER est donc large :
0,79–12,06 %. Les réponses complètes et intervalles des autres mesures sont dans
`outputs/rtx5090-20261011/test_benchmark.json`.

Le modèle conseillé de ce premier run est `outputs/rtx5090-20261011/final_merged/` : poids BF16
nativement rechargeables, environ 1,71 Go. Le contrôle comparant ces poids aux
poids FP32 sous autocast, sur les mêmes 48 pages de validation, retrouve le même
CER et F1, avec un écart d'IoU inférieur à 0,001. `final_fp32/` conserve aussi
l'export FP32 de référence. `model_verification.json` documente le rechargement
natif et l'empreinte des poids BF16.

Le profil final utilise 1 page physique × 8 accumulations, sans checkpointing,
avec un pic PyTorch mesuré de 23,25 GiB après recalibration. Le run a repris au
checkpoint 111 après un premier batch physique de 2 pages trop proche du plafond
VRAM ; le scheduler et les états optimizer ont été conservés. Les tentatives
abandonnées restent visibles dans les logs bruts et le manifeste d'exécution.

Les courbes finales sont dans `training_curves.png` / `.svg`, le rapport dans
`results.json`, et le dashboard local dans `training_dashboard.html`.

## Phase suivante : priorité au CER

`run_cer_refinement.ps1` démarre une nouvelle phase depuis les poids FP32 du
checkpoint 333, dans `outputs/rtx5090-20261011-cer-refinement/`. Le premier run
reste conservé. Le checkpoint 333 était déjà le meilleur en CER sur les 48 pages
de validation ; cette phase change donc aussi la loss et le learning rate.

- Texte : poids 1 ; marqueurs/coordonnées : 0,2 ; fin de réponse EOS : 2.
  La pondération dépend de la position des tokens, pas de leur vocabulaire :
  les chiffres présents dans le texte restent pleinement supervisés.
- Nouveau AdamW/scheduler, LR décodeur 3e-6, 3 époques supplémentaires maximum.
  Images nettes, sans augmentation pendant cette phase de consolidation.
- Génération sur les **189 pages de validation** à chaque époque, sélection
  exclusivement sur le CER minimum, arrêt après 2 validations sans amélioration
  d'au moins 0,0002 (0,02 point de pourcentage).
- Évaluation du modèle initial sur ces mêmes pages avant toute mise à jour.
  L'export conserve le modèle initial si aucun candidat n'abaisse son CER d'au
  moins 0,0002. Le WER, les exact match et les bbox restent suivis.
- Calibration réelle du batch 1 avec les gradients/états Adam FP32 résidents et
  la nouvelle loss fusionnée ; accumulation pour 8 pages effectives.

```powershell
.\run_cer_refinement.ps1 -Build -Smoke
docker logs -f lighton3-cer-refinement-smoke
# Après réussite du contrôle :
.\run_cer_refinement.ps1
.\serve_dashboard.ps1 -RunName rtx5090-20261011-cer-refinement -Port 8767
```

Les splits restent figés. Aucune page de test n'entre dans le train ou la
sélection. Le jeu de test a déjà été évalué et inspecté lors du premier run :
un benchmark final sur ces pages mesure une comparaison sur ce test réutilisé,
et ne constitue pas une nouvelle mesure sur un jeu jamais consulté.

## Résultat de la consolidation CER

La phase est terminée (333 étapes). Le **checkpoint 111**, première époque de
reprise, est retenu : CER sur les 189 pages de validation **1,166 % → 1,040 %**.
Les époques 2 et 3 atteignent respectivement 1,148 % et 1,208 % de CER ; leurs
poids ne remplacent donc pas le meilleur candidat, même si certaines autres
mesures progressent.

| Même test de 189 pages | Premier run | Consolidation retenue |
|---|---:|---:|
| CER | 4,404 % | **1,066 %** |
| WER | 3,408 % | **2,224 %** |
| Texte exact par bulle | 89,247 % | **90,219 %** |
| Texte de page exact, dans l'ordre | 38,095 % | **40,212 %** |
| Sorties tronquées | 1 | **0** |

Le gain principal vient de la page 1233 : 1 810 → 9 erreurs de caractères,
avec fin de génération normale. Sur les 188 autres pages, le CER est presque
stable, 1,034 % → 1,061 % (548 → 562 erreurs). Cette reprise corrige donc un
échec majeur sans démontrer une amélioration uniforme de toute la transcription.
Sur les pages strictement communes, le CER est de 1,838 % contre 1,972 % pour
Surya (42 pages), et de 0,476 % contre 1,248 % pour LightOnOCR-2 (28 pages).
Ces petits sous-ensembles ne permettent pas d'établir un classement général.

Le nouvel export conseillé est
`outputs/rtx5090-20261011-cer-refinement/final_merged/` : BF16, 1 706 030 528 octets,
SHA-256 `3e7f25dbbdd87a0f1896af27aad5ee07f66364a24da4151d14ef6d125a98d14b`.
Ses 473 tenseurs correspondent exactement au checkpoint 111 après conversion
BF16. Le rechargement natif en processus neuf fonctionne et retrouve le même
CER sur la page de contrôle que la génération de validation ; les coordonnées
peuvent différer légèrement. Les données et le premier run restent conservés.

Le dashboard de cette phase est disponible sur le port local 8767.
`refinement_comparison.json` contient les écarts appariés page par page et les
intervalles bootstrap (5 000 rééchantillonnages, seed 42). Les intervalles de
validation sont descriptifs, car ce même jeu sélectionne le checkpoint.
`ocr_refinement_comparison.png` / `.svg` montrent la progression du texte ;
`training_curves.png` / `.svg` incluent CER, WER, texte exact, bbox et GPU.
`model_verification.json` documente les contrôles de l'export et les sources de
post-traitement. Le snapshot du code utilisé pour entraîner reste dans
`source_snapshot/`, indépendamment des enrichissements ultérieurs du rapport.
