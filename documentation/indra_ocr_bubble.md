# Indra OCR pour les bulles

Ce document décrit l'évaluation d'[Indra OCR](https://huggingface.co/datalab-to/indra-ocr) (Datalab, tag `v1.1.0`) sur les bulles de One Piece, le fine-tuning de ses deux réseaux, puis leur intégration comme OCR par défaut du navigateur. Le modèle publié est [`Remidesbois/indra-ocr-poneglyph-bubble`](https://huggingface.co/Remidesbois/indra-ocr-poneglyph-bubble) (révision `fea60b5`). Le code de l'évaluation et de l'entraînement est dans [`docker_scripts/eval_indra_ocr`](../docker_scripts/eval_indra_ocr/README.md).

Indra lit les pages avec deux petits réseaux convolutifs en ONNX : `det.onnx` (90 k paramètres, 390 Ko) détecte les lignes, `rec.onnx` (1,25 M paramètres, 5 Mo) les lit avec un décodage CTC. Ses autres modèles (orientation, mise en page, ordre de lecture, autres écritures, modèles de langue) ne servent pas pour une bulle déjà isolée.

## Protocole

- Mêmes données que le PP-OCRv6 publié le 3 novembre : crops des bulles annotées, splits par page de `dataset-v5`, soit 9 256 bulles en train, 1 193 en validation et 1 221 en test.
- Pas de vérité terrain par ligne : les cibles sont le texte de la bulle entière.
- Le pipeline actuel (A) passe par les fonctions réelles du worker de production (`ocr.worker.js`, ONNX Runtime Node).
- Le même post-traitement est appliqué à toutes les configurations : règles apprises sur les textes de train (lexique, espaces de ponctuation française, casse des débuts de phrase), puis la normalisation de l'audit (NFKC, apostrophes droites, points de suspension, espaces). Casse et ponctuation comptent.
- Tous les réglages (échelle du détecteur, mode de lecture, variante et époque des modèles) sont choisis sur la validation. Le test n'est évalué qu'une fois par configuration figée. Les pages de test ont déjà été inspectées lors d'audits précédents : ce n'est pas un test aveugle.

Les trois pipelines comparés :

| | Détection des lignes | Reconnaissance |
|---|---|---|
| A | YOLO26n (production) | PP-OCRv6 fine-tuné, lignes mises bout à bout |
| B | YOLO26n (production) | Indra `rec.onnx` |
| C | Indra `det.onnx` | Indra `rec.onnx` |

## Indra sans fine-tuning

Le lettrage est en capitales : Indra lit `J'EN AI ASSEZ…` là où l'annotation porte `J'en ai assez…`, d'où un CER sensible à la casse supérieur à 70 %. Le tableau ci-dessous mesure donc deux choses : le CER après passage en minuscules et remise en casse par les règles de production, et le CER insensible à la casse (« contenu »). Résultats sur la validation, pour choisir les réglages :

| Variante (validation) | CER remise en casse | CER contenu |
|---|---:|---:|
| A : production (CER final) | 1,16 % | 0,91 % |
| B, lignes lues séparément | 8,65 % | 6,59 % |
| B, lignes côte à côte en un seul passage | **7,64 %** | **5,38 %** |
| B, modèle de langue n-gram d'Indra | 9,25 % | 7,38 % |
| C, détecteur à 0,75 × la taille du crop, côte à côte | 7,81 % | 5,62 % |
| `indra_ocr` complet appliqué au crop | 61,98 % | 61,67 % |

- Le pipeline complet d'Indra n'est pas adapté à une bulle isolée : il ne produit aucun texte sur 488 bulles.
- Le détecteur d'Indra donne de bons résultats dès qu'on l'applique au crop à 0,75 × sa taille. À la taille d'une page (côté long 1 024 px), le texte est trop grand pour lui.
- Lire toutes les lignes en un seul passage, côte à côte, est légèrement meilleur que lignes séparées. C'est aussi la seule manière d'entraîner avec des textes de bulle sans annotation par ligne.

Sur le test, le zero-shot reste loin de la production : 7,35 % (B) et 8,02 % (C), contre 0,90 %.

## Fine-tuning

Aucun code d'entraînement n'est publié pour Indra. Les graphes ONNX publiés sont exécutés tels quels en PyTorch par un interpréteur (`onnx_torch.py`) : chaque poids devient un paramètre entraînable, et les poids appris sont réécrits dans une copie du graphe d'origine. Les modèles livrés gardent donc exactement les entrées, les sorties et les opérateurs d'Indra. La parité a été vérifiée : sorties identiques à ONNX Runtime, et graphe réexporté identique octet pour octet avant entraînement.

**Reconnaissance.** L'entrée d'entraînement reprend celle de l'inférence : les fenêtres de lignes d'Indra (`crop_line`, fenêtre ×3, hauteur 96) d'une bulle mises côte à côte. La cible est la transcription complète de la bulle. Les boîtes viennent de YOLO (60 %) ou du détecteur Indra (40 %), avec jitter des boîtes et augmentations photométriques. AdamW, lr 3e-4, 40 époques, EMA. Trois variantes ont été entraînées en parallèle en 5,3 minutes sur une RTX PRO 6000 (RunPod). Lire en capitales puis remettre la casse a donné de moins bons résultats qu'apprendre directement la casse des annotations.

**Détection.** Le détecteur a été entraîné avec les boîtes YOLO de production comme pseudo-annotations, via l'encodeur de cibles d'Indra (`encode_targets`). Les bulles sans ligne YOLO et les 28 oublis de lignes repérés par revue visuelle sont écartés. Entraînement en 3,6 minutes. Sur la validation, l'accord avec YOLO (F1 à IoU 0,5) passe de 96,5 % à 99,3 %, et le nombre de bulles sans ligne détectée de 77 à 0.

## Résultats sur le test

| Pipeline (test, 1 221 bulles) | CER | Exact | CER textes courts |
|---|---:|---:|---:|
| A : YOLO + PP-OCRv6 (production) | 0,902 % | 85,01 % | 4,37 % |
| B : YOLO + Indra rec fine-tuné | 0,912 % | 87,80 % | 6,53 % |
| C : Indra det d'origine + rec fine-tuné | 1,084 % | 85,83 % | 10,95 % |
| **C : Indra det + rec fine-tunés** | **0,690 %** | **88,53 %** | 6,36 % |
| C, implémentation navigateur (`indraOcr.js`) | 0,693 % | 88,53 % | 6,42 % |

- **C fine-tuné contre la production :** 277 erreurs de caractères contre 362. L'écart de CER est de −0,21 point, avec un intervalle de confiance à 95 % de [−0,41 ; −0,005] (bootstrap apparié sur les 129 pages). Le gain est significatif, mais de peu.
- **B vaut A en précision :** +0,01 point, intervalle [−0,09 ; +0,14].
- **Le fine-tuning du détecteur fait la différence.** Sans lui, le détecteur d'Indra ne trouve aucune ligne dans 43 bulles de test, surtout des onomatopées et du lettrage stylisé, et C reste derrière A.
- **Point faible : les textes courts** (12 caractères ou moins : cris, ponctuation seule, onomatopées). Ils restent moins bien lus que par PP-OCRv6.

## Vitesse et taille

Latence CPU médiane par bulle, mesurée sur les 300 premières bulles du test. Harnais Python, ONNX Runtime 1.31, sessions chaudes, une bulle à la fois, machine au repos :

| Pipeline | Détection | Reconnaissance | Total, 4 threads | Total, 1 thread |
|---|---:|---:|---:|---:|
| A : YOLO + PP-OCRv6 | 69,6 ms | 85,5 ms | 157,7 ms | 387,8 ms |
| B : YOLO + Indra rec | 69,6 ms | 3,7 ms | 73,8 ms | 141,4 ms |
| C : Indra det + rec | 1,2 ms | 3,2 ms | **4,4 ms** | **4,5 ms** |

Les mêmes 14 bulles d'une page du sandbox ont aussi été lues dans Chromium headless, via l'application réelle : Indra en WASM prend 12 ms par bulle (médiane, 47 ms pour la toute première), PP-OCRv6 en WebGPU 146 ms.

Côté téléchargement, Indra pèse 5,8 Mo : 0,39 Mo pour le détecteur, 5,04 Mo pour la reconnaissance et 0,38 Mo pour le manifeste. Le pipeline YOLO + PP-OCRv6 pèse 86,6 Mo.

## Intégration

- **Pipeline navigateur :** `frontend/src/lib/indraOcr.js` réimplémente en JavaScript pur l'ensemble du traitement : niveaux de gris, `page_to_input`, redimensionnements OpenCV `INTER_AREA` et `INTER_LINEAR`, décodage du détecteur, ordre des lignes, `crop_line`, assemblage et CTC. Sur le test, elle reproduit la référence Python à l'identique pour 1 219 textes sur 1 221 (1 193 sur 1 193 en validation).
- **Tests :** `indraOcr.test.js` compare cette implémentation à des sorties OpenCV et indra_ocr réelles, générées par `make_test_fixtures.py`.
- **Worker :** `frontend/src/workers/indra.worker.js` télécharge le manifeste et les deux ONNX depuis la révision fixée dans `packages/shared/src/model-registry.json` (`indra-bubble`), vérifie les tailles et les SHA-256, puis exécute le pipeline en WASM. Ces réseaux sont petits et leur taille d'entrée change à chaque bulle : le WebGPU n'apporterait rien et recompilerait ses shaders à chaque forme.
- **Choix du modèle :** `indraBubble` est le modèle OCR de bulles par défaut (`ocrModels.js`, `WorkerContext.jsx`). PP-OCRv6 reste sélectionnable.
- **Recherche par image :** elle charge et utilise directement le worker Indra, quel que soit le modèle choisi dans l'éditeur.
- **Post-traitement partagé :** les règles de texte sont dans `applyOcrTextRules` (`ocr-utils.js`) et servent aux deux recognizers. Le score de PP-OCRv6 est inchangé après ce déplacement (validation : 1,161 % avant et après).

## Limites et licence

- Les pseudo-annotations du détecteur viennent de YOLO, qui oublie parfois des lignes. Le détecteur fine-tuné peut avoir hérité de ces oublis.
- Le test a déjà été inspecté lors d'audits précédents. Un test aveugle sur de nouveaux tomes reste à faire.
- Les poids dérivent d'Indra OCR et restent soumis à sa licence OpenRAIL-M modifiée, incluse dans le dépôt du modèle : usage libre pour la recherche, l'usage personnel et les organisations sous le seuil de financement ou de chiffre d'affaires indiqué, et aucun usage en concurrence avec l'API de Datalab. Voir [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
