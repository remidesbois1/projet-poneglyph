# Réentraînement PPOCR / YOLO — 3 novembre 2026

Entraînements réalisés sur RunPod, RTX 4090 24 Go, PyTorch 2.8.0 + CUDA 12.8. Le nouveau PP-OCR est retenu. Les nouveaux YOLO sont disponibles comme candidats ; les poids du détecteur de bulles actif sont conservés, avec un seuil passé de 0,45 à **0,25** après sélection sur validation.

## Résultats comparés

Même test historique : **129 pages, 1 221 bulles annotées, 40 136 caractères**. Les pages ont déjà été examinées lors des audits précédents ; ce n'est pas un test aveugle sur de nouveaux tomes.

| Mesure | Ancienne production | Version retenue |
|---|---:|---:|
| CER sur crops de référence, après post-traitement | 1,069 % | **0,904 %** |
| Textes exactement corrects sur crops | 1 015 / 1 221 | **1 038 / 1 221** |
| Sorties vides sur crops | 3 | **0** |
| CER depuis les pages entières | 1,624 % | **1,014 %** |
| Bulles annotées manquées | 12 | **4** |
| Détections sans annotation correspondante | 6 | 9 |
| Pages dont tous les textes annotés sont corrects | 28 / 129 | **37 / 129** |

Les pages entières passent par les fonctions réelles des workers dans Skia/ONNX Runtime Node CPU. Appariement spatial glouton un à un à IoU ≥ 0,5 ; une bulle manquée reçoit un texte vide. Les détections sans référence sont comptées séparément et ne sont pas pénalisées dans le CER. Les annotations n'étant pas démontrées exhaustives, ces détections ne sont pas toutes des faux positifs établis. Les « pages correctes » ne mesurent ni ces régions supplémentaires ni l'ordre de lecture.

Sur validation, garder les anciens poids YOLO à 0,25 réduit les annotations manquées de 16 à 7 sans augmenter les deux détections sans référence. Les YOLO réentraînés 800 px, tête seule et 1024 px donnent respectivement 8/14, 7/5 et 7/8 (manquées/sans référence) au même seuil. Ils ne justifient donc pas de remplacer les poids actifs. Les courbes complètes, les configurations et les métriques natives sont publiées.

L'ordre des annotations spatialement appariées reste exact sur **114/129 pages** avant et après. Cette mesure exploratoire ignore les régions manquées/supplémentaires ; elle ne remplace pas un benchmark exhaustif d'ordre. Les poids de panneaux et d'ordre n'ont pas changé.

## Données, entraînement et vérifications

- 1 326 pages, 11 670 bulles ; splits conservés par page : 9 256 train, 1 193 validation, 1 221 test. Hashes de toutes les images vérifiés, aucune image de page identique entre splits.
- Le détecteur utilise 1 065/127/129 pages : cinq pages à annotations non validées sont écartées de train/validation. Deux boîtes train sont bornées à l'image, sans modification des annotations d'évaluation.
- Les images du reconnaisseur reproduisent exactement les tenseurs de production : détection de lignes à 0,10, padding 2 px, espacement 8 px, hauteur 48, largeur 960–3200, fond #808080, canaux BGR. Aucun deuxième redimensionnement Python.
- 28 paires **train** ont été vérifiées visuellement : une ligne visible avait disparu du crop reconstruit. Elles sont exclues du seul entraînement du reconnaisseur, soit **9 228 paires effectives**. Aucune exclusion d'évaluation. Les SFX et les 24 fallbacks sans ligne sont conservés.
- PP-OCR : cinq époques FP32, backbone entraîné, statistiques BN figées, batch 1 × accumulation 16, LR tête 6e-6 / backbone 6e-7, seed 42. **Époque 3** retenue par CER brut de validation ; aucun pas d'optimisation ignoré.
- Le post-traitement PP-OCR conserve les guillemets et les chiffres romains. Le nettoyage générique supplémentaire a été retiré après comparaison sur validation (440 au lieu de 485 erreurs). Les règles lexicales apprises sur train restent figées.
- Export natif/ONNX : 9 exemples réels identiques au décodage et cinq largeurs dynamiques vérifiées. Chrome WebGPU AMD RDNA3 : 100 crops de validation, 29 → 22 erreurs, 99/100 textes bruts identiques au natif. Détecteur actif : 20 pages réelles dans Chrome/WASM. Ces sous-ensembles ne constituent pas un CER navigateur exhaustif.

[results-20261103.json](results-20261103.json) contient les valeurs exactes, protocoles, hashes et limites. Les scans, transcriptions et prédictions individuelles restent dans `outputs/`, ignoré par Git. Les modèles publics sont dans [PP-OCR](https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec) et [ReaderNet](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet) ; le registre partagé de l'application fixe leurs commits exacts.

Versions publiées et vérifiées : PP-OCR [`a6dddec`](https://huggingface.co/Remidesbois/pp-ocrv6-one-piece-bubble-line-rec/tree/a6dddec384c76a1ac3c2125f92403e775a3929d7), ReaderNet [`0f3cc90`](https://huggingface.co/Remidesbois/Poneglyph-ReaderNet/tree/0f3cc90737d90d8c3810b7397c923a2fc7be24e9). Après publication, les sept fichiers actifs ont été téléchargés et contrôlés ; les sorties sur deux pages complètes et dix crops Chrome/WebGPU reproduisent exactement les sorties évaluées. Les manifestes publics répondent sans authentification avec les en-têtes CORS requis.

Les **44 tests frontend ciblés**, les **4 tests Python**, les vérifications de tenseurs/splits/post-traitement et le **build de production** passent. Sur le Node 26.11.1 local, lancer Vitest avec `NODE_OPTIONS=--no-experimental-webstorage` pour laisser JSDOM fournir `localStorage`. Le pod et le volume temporaire de cette session ont été supprimés puis leur absence vérifiée ; le compte indique **0 $/h** après nettoyage. Variation observée du solde pendant la session : environ **0,91 $**. Les preuves agrégées de publication sont dans [published-20261103.json](published-20261103.json).

## Reproduire

Installer les dépendances Node avec `npm ci --prefix docker_scripts/retrain_ppocr_pipeline`. Créer un venv et installer `requirements.txt` en conservant les roues CUDA du template ; `setup_runpod.sh` impose torch 2.8.0 / torchvision 0.23.0. La récupération des anciens YOLO nécessite une installation **isolée** de `requirements-yolo-legacy.txt` (8.4.48), différente des candidats récents (8.4.201).

1. Rafraîchir le snapshot privé avec l'audit existant, puis `node prepare.cjs --output outputs/nouveau-dataset`. Le script vérifie les statuts, conserve les splits historiques et refuse de reprendre un export avec un protocole différent. `prefetch.cjs` prépare le cache de pages. Les accès Supabase/R2 sont uniquement en lecture.
2. Vérifier `node test_runtime.cjs` et `python -m unittest discover -s docker_scripts/retrain_ppocr_pipeline -p test_training.py`. L'audit de paires signale des suspects ; l'exclusion exige une revue visuelle et le statut train.
3. `package_inputs.py` prépare une archive privée avec une liste explicite de fichiers. Transférer par SSH ; aucune clé Supabase, R2, RunPod ou HF n'entre dans l'archive. Les scripts `launch_*.sh` documentent les commandes effectivement utilisées. `remote.py` et le garde-coût sont volontairement liés à l'état attribué de cette session ; ne pas les réutiliser avec des IDs devinés.
4. Évaluer les détecteurs sur **validation** avec `evaluate_detection.cjs`, figer la sélection, puis utiliser `evaluate_pipeline.cjs` pour le test. `evaluate_recognizer.py`, `export_recognizer.py` et `verify_browser.cjs` isolent reconnaissance, export et navigateur. `rescore_pipeline.cjs` permet une comparaison explicite du seul post-traitement à partir de sorties brutes immuables.
5. `build_release.py` reconstruit la publication de cette session depuis les résultats vérifiés. `publish_release.py` est un aperçu par défaut ; `--publish` publie la liste de fichiers et vérifie les téléchargements. Le commit parent et tous les hashes doivent correspondre. Ne jamais utiliser un envoi récursif de `outputs/` sur HF.

Après récupération et vérification des artefacts, supprimer les ressources temporaires de cette session. Les checkpoints durables et leurs exports sont conservés localement et sur HF.

## Prochaine amélioration utile

Le problème confirmé restant est la **perte de lignes avant reconnaissance**. Il faut annoter les lignes complètes, y compris celles que le détecteur omet, puis réentraîner ce détecteur. En parallèle, créer un jeu humainement étiqueté « classique / SFX » et un test sur de nouveaux tomes avec annotations exhaustives. Le minuscule sous-ensemble SFX inspecté ne permet pas d'annoncer une précision globale par catégorie, ni de valider un classifieur. Le résultat actuel s'améliore sensiblement mais ne garantit pas une transcription parfaite.
