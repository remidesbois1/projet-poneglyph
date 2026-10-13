# LightOnOCR-3 0.8B — transcription de bulles

Fine-tuning natif sur RTX 5090, prompt vide/image seule, thinking désactivé.
Poids BF16 safetensors ; conversion et intégration ONNX ultérieures.

Les crops et les splits sont ceux du benchmark Surya du 5 octobre : 7 496 / 1 648 /
1 612 bulles, réparties sur 860 / 184 / 185 pages. L'audit vérifie IDs, pages,
existence des images, empreinte des métadonnées et des pixels, puis exclut du
train d'éventuelles copies exactes de crops de validation/test. Le test reste figé.

Initialisation : `lightonai/LightOnOCR-3-0.8B`, révision
`36d8636925327c671bd8e479c517a004f7566bd9`. Les poids bbox ne conviennent pas à ce
benchmark : leur train contient 85 pages de validation et 77 pages de test.

Recette reprise de Surya : modèle de langage complet + merger + 4 derniers blocs
vision, poids maîtres/états AdamW FP32, autocast BF16, AdamW fused, FLA/causal-conv1d
SM120, SDPA, Liger avec backward, projection uniquement des tokens supervisés.
Perte assistant/EOS normalisée par bulle et par fenêtre d'accumulation, légère
pondération des textes courts et onomatopées, grouping par longueur, augmentations
modérées. Pas de quantification des poids ou de l'optimiseur pendant le training.

LR décodeur `1.2e-5`, merger `6e-6`, vision `3e-6`, warmup 60 updates puis cosine
jusqu'à 10 % du LR. Six époques maximum, patience deux évaluations. Chaque validation
génère **toutes les 1 648 bulles**. Sélection exclusivement sur le CER ; WER, exact
match, sorties vides/tronquées, loss, gradients et métriques GPU suivis également.
Les hyperparamètres sont ajustés sur la validation, jamais sur les erreurs de test.

Une consolidation reprend exclusivement un checkpoint de ce même run bulle et
les mêmes empreintes de splits, avec un nouvel optimiseur, un LR plus faible,
`-NoAugment` et `-EosWeight 2`. La pondération EOS renormalise chaque bulle pour
préserver sa contribution globale. `-DeferTest` garde le test fermé : export et
validation du modèle natif d'abord, `-Mode finalize` seulement après le choix
final sur validation. Les validations affichent aussi le nombre de crops traités
et une ETA propre à cette phase.

## Exécution locale

Construire avec `run_training.ps1 -Build -Mode calibrate`, puis calibrer batch 16
et 32 ; les mesures incluent trois vrais updates et les buffers/états persistants.
Lancer l'audit CPU avec `audit_dataset.py --dataset ... --output .../dataset_audit.json`.
Faire un test Trainer avec `run_training.ps1 -RunName ...-smoke -SmokeSteps 2` (copier
l'audit dans ce dossier), puis benchmark initial `-Mode baseline` et entraînement
`-Mode train` avec batch/accumulation retenus. Les variables `LIGHTON3_BUBBLE_*`
permettent un nouveau run ; ne pas relancer un output existant sans reprise explicite.

`serve_dashboard.ps1` sert le suivi sur
<http://127.0.0.1:8768/training_dashboard.html>. JSONL, CSV, graphiques PNG/SVG,
rapport HTML autonome, TensorBoard et Trackio local sont conservés dans `outputs/`.
Aucun Space, webhook ou transmission de crops n'est activé.

Le meilleur modèle est sauvegardé en BF16 ; chaque tensor est comparé au cast BF16
du checkpoint FP32 choisi sur validation. Validation puis test se font avec un
rechargement Transformers standard, sans les patches d'entraînement. Le test inclut
des IC à 95 % par bootstrap de pages (5 000 réplications) et des scores bruts.

`publish_model.py` n'accepte que le dépôt bulle nouvellement vide
`Remidesbois/LightOnOCR-3-Poneglyph`. Il exige l'export vérifié et le test complet,
publie une liste fermée de poids/configs/agrégats, puis contrôle le SHA-256 distant.
Le dépôt bbox reste `Remidesbois/LightOnOCR-3-Poneglyph-bbox`.

## Audit et protocole final

Le run retenu est `rtx5090-20261012-bubble`, checkpoint 1175. Consolidation à LR
3e-6 / EOS ×2 puis un lissage FP32 75/25 ont été évalués sur toute la validation
native et rejetés. `average_checkpoints.py` garde les SHA des deux parents et
vérifie les 473 casts BF16 avant la validation ; il ne change aucun split.

L'audit visuel (`visual_audit.py`) a examiné 34 crops locaux : 18 de validation,
six bruitages/répétitions train et dix crops train tirés avec seed 42. Deux
annotations de validation sont manifestement incorrectes ; plusieurs lettrages
stylisés restent à confirmer. Les 16 exemples train inspectés sont cohérents.
Cet échantillon ciblé ne constitue pas une certification de tout le dataset.

Le protocole hérité supprimait les guillemets des prédictions uniquement, même
lorsqu'ils étaient visibles et attendus. Le protocole final **conserve guillemets,
casse, accents et ponctuation**, normalise seulement les espaces et applique le
même calcul aux sorties brutes stockées de Surya. `rescore_validation.py` a
recalculé les trois candidats avant l'ouverture du test ; même checkpoint retenu.
Aucun label n'a été modifié. Anciens scores et courbes conservés et explicitement
identifiés ; scores corrigés et IC de pages figurent dans le rapport final.

`compare_surya.py` recalcule indépendamment CER, WER et exact match, vérifie IDs,
références et pages identiques, puis réalise 5 000 bootstraps appariés par page.
`report_experiment.py` assemble les deux phases, le candidat de lissage, le test,
les courbes PNG/SVG, la recette et la carte du modèle. L'audit visuel et les crops
sont servis seulement en local ; seuls poids natifs, configs, courbes agrégées
et benchmark agrégé sont publiés sur Hugging Face.
