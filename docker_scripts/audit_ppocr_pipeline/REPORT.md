# Audit de la pipeline PP OCR et plan d'amélioration

Date : 3 novembre 2026. Révision du dépôt : `21d9a111f84e3d83c61d7436282de0532b0441d6`.

## Implémentation vérifiée après l'audit

Le réglage candidat est maintenant actif dans `frontend/src/workers/ocr.worker.js` : seuil du détecteur de lignes à `0.10` et reconnaissance avec ratio conservé, padding gris `#808080` et largeur respectant le ratio. Sur un nouvel échantillon déterministe de 100 crops de validation exécuté avec Chrome 156/WASM, le CER passe de **2,73 % à 1,16 %** (92 à 39 erreurs), et les bulles exactes de **76 à 81**. Le JPEG à 0,95 obtient 42 erreurs sur ce même lot, ce qui confirme que le chemin sans perte retenu convient ici. `npm run build` passe ; le worker passe `node --check` et `git diff --check` ne signale pas d'erreur. Ce contrôle complémentaire n'est pas ajouté au `summary.json` historique ; ses résultats sont conservés dans `results/browser_val_100.summary.json` (cache ignoré par Git).

## Ce qui limite actuellement la qualité

**La détection des lignes est le plus gros défaut isolé mesuré en nombre d'erreurs de caractères.** Sur 300 bulles de validation, abaisser son seuil de `0.25` à `0.10`, sans modifier le recognizer, fait passer les erreurs de 314 à 165. Des lignes entières visibles sont actuellement rejetées. Le deuxième problème important est le redimensionnement : les textes courts sont étirés horizontalement jusqu'à une largeur minimale de 960 pixels, alors que le manifeste annonce une conservation du ratio suivie d'un padding.

Un candidat combinant **seuil de lignes à `0.10` et conservation du ratio avec padding gris** a été choisi sur la validation, puis testé sur les 1 219 bulles du test historique. Sans réentraînement, le CER passe de **3,18 % à 1,04 %** sur les crops annotés, soit **67,2 % d'erreurs de caractères en moins**. Sur 129 pages avec détection des bulles, il passe de **3,94 % à 1,62 %**. Ces grands tests exécutent les fonctions des workers avec ONNX Runtime CPU et Canvas Skia ; le contrôle séparé dans Chrome/WASM confirme un gain, mais son ampleur est différente.

Cela ne donne pas encore des pages parfaites. Les causes restantes comprennent des bulles non détectées, des lignes toujours oubliées, des corrections textuelles destructrices, des césures, des annotations discutables et des désaccords sur l'ordre de lecture. Le modèle géométrique ultra simple testé pour router les cas atypiques n'en retrouve que **3 sur 6** dans son test. Il ne faut pas lui confier une suppression automatique.

Les changements livrés concernent l'audit, ses scripts et ses résultats consolidés. Les fichiers de production et les annotations distantes n'ont pas été modifiés.

Source chiffrée : [summary.json](summary.json), notamment `validation`, `selection`, `test`, `page_ocr`, `browser`, `paired` et `router`.

## Protocole et portée des mesures

Le snapshot local provient des annotations validées de Supabase et des images R2, lus sans écriture distante. Il contient 11 670 bulles validées au total. Les splits historiques du modèle permettent de retrouver 2 410 exemples sur 258 pages : 1 191 en validation et 1 219 en test. Aucun identifiant ni crop n'est manquant ou invalide ; 16 textes de référence ont changé depuis les CSV historiques. Les pages de validation et de test ne se chevauchent pas. Le snapshot est daté du `2026-11-03T18:15:44.471Z`.

Les expériences comprennent 300 crops de validation pour les ablations et le choix du candidat, les 1 219 crops de test pour la confirmation, 258 pages pour la détection et l'ordre, puis les 129 pages de test pour l'OCR des bulles prédites. Ces dernières contiennent 1 221 annotations validées : deux bulles « Zzz » supplémentaires par rapport aux 1 219 lignes du CSV. Sur les 258 pages, le total est de 2 414 annotations.

Un échantillon déterministe de 100 crops du test a aussi été exécuté dans un vrai Chrome headless, avec Canvas et ONNX Runtime Web/WASM. Cet échantillon est sélectionné par tri du SHA256 de `ppocr-audit-20261103:<id>`, pas par résultat OCR. Un autre lot de 20 cas difficiles de validation est choisi explicitement pour le diagnostic ; ses résultats ne sont pas représentatifs de la population. Les splits ont déjà servi dans l'historique du projet : ils ne constituent pas un nouveau test aveugle.

Le grand banc d'essai utilise les fonctions réelles de `ocr.worker.js`, `detection.worker.js` et `ocr-utils.js`, chargées avec adaptation en mémoire des imports, constantes et dépendances. Le décodage CTC de l'audit est vérifié contre celui du worker. Il utilise le modèle `ppocrv6_bubble_line_rec_webgpu.onnx`, qui est aussi celui de la configuration PP-OCR par défaut du worker. Le nom du fichier ne signifie pas que le backend WebGPU a été exécuté : **WebGPU n'a pas été mesuré**.

La machine observée est un Ryzen AI 9 365, 10 cœurs et 20 threads. Les versions sont Node 26.11.1, ONNX Runtime Node 1.24.3, ONNX Runtime Web 1.27.0, Canvas 1.1.0 et Chrome 156. Le grand test natif utilise quatre threads intra-op ; le contrôle WASM en utilise deux. Les temps collectés pendant des tâches concurrentes ne constituent pas un benchmark de latence de production ou de RTX 5090.

### Comment lire les scores

Le **CER** est la somme des distances de Levenshtein divisée par le nombre de caractères des références. Une bulle non détectée compte comme un texte vide, donc ses caractères comme des suppressions. Le **taux exact** est la proportion de bulles dont tout le texte est identique à la référence après normalisation légère : NFKC, variantes d'apostrophes, points de suspension et espaces répétés. La casse et l'espacement avant ponctuation restent significatifs.

Les métriques `content_*` ignorent en plus la casse et les espaces avant la ponctuation. Elles servent à distinguer les conventions typographiques des erreurs de transcription ; elles ne remplacent pas le taux exact strict et ne prouvent pas une équivalence sémantique. Une normalisation proche de l'entraînement est également conservée dans le JSON.

L'appariement des bulles est glouton, un à un, avec IoU au moins égal à 0,5. L'exhaustivité des annotations validées n'étant pas établie, les prédictions sans annotation correspondante ne sont pas comptées comme des faux positifs certains. **Les scores sur pages mesurent donc la restitution des annotations connues, sans pénaliser les prédictions supplémentaires.** Même une « page exacte » ci-dessous n'est pas une certification de toute la sortie utilisateur.

Sources : [runtime.cjs](runtime.cjs), [prepare.cjs](prepare.cjs), [pages.cjs](pages.cjs), [page-ocr.cjs](page-ocr.cjs) et [browser-parity.cjs](browser-parity.cjs).

## Résultats du candidat retenu

Le candidat change uniquement le seuil de lignes et le prétraitement de reconnaissance. Il conserve le post-traitement, les poids des modèles et le seuil de détection des bulles à `0.45`.

| Expérience | Configuration actuelle | Candidat |
| --- | ---: | ---: |
| 1 219 crops annotés, natif : CER | 3,185 % — 1 278 erreurs | 1,044 % — 419 erreurs |
| Même test : bulles exactes | 878 / 1 219 — 72,03 % | 1 015 / 1 219 — 83,26 % |
| Même test : sorties vides | 70 | 3 |
| Même test : exact après normalisation `content` | 78,10 % | 89,58 % |
| 129 pages, 1 221 annotations, natif : CER | 3,942 % — 1 582 erreurs | 1,624 % — 652 erreurs |
| Même test : bulles exactes | 859 / 1 221 — 70,35 % | 1 018 / 1 221 — 83,37 % |
| Même test : pages au texte entièrement exact | 9 / 129 | 28 / 129 |
| Même test : texte exact et accord sur l'ordre | 8 / 129 | 24 / 129 |
| Chrome/WASM, 100 crops de test : CER | 3,019 % — 97 erreurs | 1,650 % — 53 erreurs |
| Même contrôle navigateur : bulles exactes | 78 / 100 | 80 / 100 |

Sur les 1 219 crops, le candidat réduit la distance d'édition de 195 bulles, l'augmente pour 28 et la laisse inchangée pour 996. Il rend 155 bulles exactes, mais en fait perdre 18 qui l'étaient : le gain n'est pas uniforme. Dans Chrome sur 100 crops, six bulles progressent et une régresse ; trois deviennent exactes et une cesse de l'être.

Le navigateur produit des textes différents du natif sur 12 des 100 crops avec la configuration actuelle, mais seulement deux avec le candidat. En particulier, plusieurs ponctuations vides dans le test natif sont reconnues par Chrome. **Le passage de 70 sorties vides à 3 est donc un résultat natif, pas une estimation directe des vides dans le navigateur déployé.**

### Les bulles classiques

Une inspection visuelle de 100 crops de validation et 100 de test a produit des labels exploratoires : texte classique, SFX stylisé, disposition spéciale et ambigu. Chaque lot contient 93 bulles classiques. Les ponctuations seules, dialogues courts, noms d'attaques et dialogues en gras ne sont pas automatiquement classés comme SFX. Ces labels sont ceux d'un seul annotateur, l'assistant, et demandent une confirmation humaine.

| 93 bulles classiques du test | Actuel | Candidat |
| --- | ---: | ---: |
| Natif : CER strict | 2,801 % | 1,228 % |
| Natif : bulles exactes | 69 / 93 | 76 / 93 |
| Natif : exact `content` | 77 / 93 | 86 / 93 |
| Natif : sorties vides | 7 | 0 |
| Chrome/WASM : CER strict | 2,455 % | 1,196 % |
| Chrome/WASM : bulles exactes | 76 / 93 | 76 / 93 |
| Chrome/WASM : exact `content` | 85 / 93 | 86 / 93 |

Dans cet échantillon navigateur, le candidat corrige surtout des erreurs longues : il divise approximativement le CER par deux sans augmenter le nombre total de bulles strictement exactes. Il reste donc du travail sur les bulles classiques, indépendamment des SFX.

Sources : `summary.json` → `test`, `page_ocr`, `browser`, `classes.test`, `paired` ; [visual-labels.json](visual-labels.json).

## Où se produisent les erreurs

### Lignes rejetées avant la reconnaissance

Ces ablations utilisent exactement les mêmes 300 bulles de validation et 9 838 caractères de référence. Sauf la dernière ligne, chaque variation est appliquée seule à la configuration actuelle.

| Variation | Erreurs | CER | Bulles exactes | Sorties vides |
| --- | ---: | ---: | ---: | ---: |
| Configuration actuelle | 314 | 3,192 % | 200 / 300 | 17 |
| Seuil de lignes `0.20` | 249 | 2,531 % | 205 / 300 | 17 |
| Seuil de lignes `0.10` | 165 | 1,677 % | 210 / 300 | 17 |
| Seuil de lignes `0.05` | 160 | 1,626 % | 211 / 300 | 17 |
| Ratio préservé, padding gris | 257 | 2,612 % | 236 / 300 | 4 |
| Ratio préservé, padding blanc | 292 | 2,968 % | 214 / 300 | 9 |
| Largeur minimale 480 avec étirement | 258 | 2,622 % | 232 / 300 | 1 |
| Largeur minimale 640 avec étirement | 270 | 2,744 % | 223 / 300 | 7 |
| Marge autour des lignes de 4 px au lieu de 2 | 319 | 3,243 % | 190 / 300 | 16 |
| Seuil `0.10` et padding gris | 107 | 1,088 % | 248 / 300 | 4 |

Le seuil `0.10` modifie les boîtes de 16 bulles : 12 améliorent leur texte et aucune n'augmente sa distance d'édition sur ce lot. Le padding gris améliore davantage de bulles, mais corrige moins de caractères au total et en dégrade neuf. Les deux effets se complètent ; il ne faut pas additionner leurs gains comme s'ils étaient indépendants.

La comparaison combinée donne 107 erreurs pour gris + `0.10`, 107 pour gris + `0.05`, 110 pour largeur 480 + `0.10` et 187 pour gris + `0.20`. Le choix `0.10` + gris conserve un seuil moins permissif que `0.05` et une bulle exacte de plus sur la validation. C'est ce choix, gelé avant le test complet, qui figure dans les tableaux principaux.

Les overlays montrent des lignes réelles avec des scores de détection entre environ `0.11` et `0.23`, rejetées au seuil actuel. Exemples : « Capitaine, il y » dans la bulle 5556, « de quel bois » dans 6171, « n'entendez » dans 5562 et « Cette » dans 4347. Dans 2779, le nombre de lignes passe de trois à cinq et des mots de la phrase sont récupérés. Il reste néanmoins une ponctuation finale non récupérée.

Le candidat manque encore la première ligne « C'EST UN CADEAU » dans la bulle classique 5495. Il renvoie seulement « Pour vous de la part de la patronne. », avec un score CTC moyen de **0,9797**. Le recognizer est confiant sur ce qu'on lui a donné ; il ne sait pas qu'une ligne n'a jamais atteint son entrée.

**Action :** adopter le seuil candidat après validation navigateur étendue, puis ajouter un contrôle de couverture et un second passage ciblé lorsque de l'encre ressemblant à du texte reste hors des boîtes. Ce contrôle doit exclure autant que possible bordures et dessins ; une simple quantité de pixels noirs n'est pas une preuve de texte manqué. Prévoir une revue lorsque les deux passages donnent des résultats différents. La présence d'au moins une boîte ne suffit pas : le fallback actuel sur la bulle entière ne s'active que lorsque le détecteur ne trouve aucune ligne.

La précision et le rappel absolus du détecteur de lignes ne sont pas connus : ce dataset contient des annotations de bulles et de texte, pas une vérité terrain exhaustive des lignes. Les récupérations observées et le gain OCR sont mesurés ; un taux de rappel de lignes ne doit pas être inventé.

Sources : `summary.json` → `validation`, `selection`, `review.validation_line_recovery` ; [worker OCR](../../frontend/src/workers/ocr.worker.js), lignes 261 à 299 ; [overlay des lignes](results/visual/missed_lines_1.png) et [suite](results/visual/missed_lines_7.png).

### Prétraitement et recadrage

Dans `preprocessPpocrCanvas`, le canvas final est au minimum de 960 × 48 et le texte est dessiné sur toute cette largeur. Une courte ponctuation est donc fortement déformée. Le manifeste décrit pourtant `height_48_preserve_aspect_then_pad_to_min_width`. Le candidat dessine le texte à sa largeur proportionnelle, limitée au maximum de 3 200 pixels, puis remplit l'espace restant avec `#808080`.

Le padding gris est le meilleur candidat mesuré pour le CER strict parmi les configurations retenues ; sa correspondance exacte au prétraitement Python d'entraînement n'a pas été vérifiée tensoriellement. Il faut établir cette parité avant le prochain entraînement, avec des crops de contrôle courts, longs, accentués et ponctués. Les deux crops de test atteignant la largeur maximale de 3 200 sont exacts avec le candidat ; la limite de largeur n'est pas une cause démontrée des erreurs de ce test.

Un autre écart se situe avant le worker. `BatchOcrManager` utilise `cropImage`, qui réencode en JPEG qualité 0,95. Le chemin `cropImageBitmap` conserve les pixels lors du recadrage, mais retombe également sur le JPEG si `createImageBitmap` n'est pas disponible. Les chemins n'ont pas exactement la même politique d'arrondi des coordonnées.

Le test navigateur isole un réencodage JPEG à qualité 0,95 sur les mêmes crops. Sur les 20 cas difficiles, le candidat passe de **11 à 28 erreurs**, avec trois textes modifiés ; la bulle 2779 perd notamment « tombe amoureux ». Sur les 100 crops déterministes, le JPEG passe au contraire de **53 à 51 erreurs**, avec trois textes modifiés, dont une amélioration d'accent. **L'expérience démontre une sensibilité à l'encodage, pas une dégradation moyenne systématique du JPEG.**

**Action :** unifier les crops sans perte et leur arrondi pour supprimer une source de variation évitable, puis refaire le test sur les pixels réellement transmis par le batch. Ne pas promettre le gain du lot difficile sur toute la population.

Sources : [worker OCR](../../frontend/src/workers/ocr.worker.js), ligne 328 ; [utilitaires de crop](../../frontend/src/lib/utils.js), lignes 65 et 110 ; [BatchOcrManager](../../frontend/src/components/BatchOcrManager.jsx), lignes 126 et 769 ; `summary.json` → `browser`, `diagnostic_browser`.

### Bulles absentes et géométrie des crops

| Détection sur les pages de test | Seuil actuel `0.45` | Seuil `0.25` |
| --- | ---: | ---: |
| Annotations appariées | 1 209 / 1 221 | 1 217 / 1 221 |
| Rappel sur annotations validées | 99,02 % | 99,67 % |
| Annotations manquées | 12 | 4 |
| Prédictions sans annotation appariée | 6 | 9 |

Sur les 258 pages, on passe de 28 à 11 annotations manquées. Huit des douze bulles absentes du test sont récupérées avec `0.25`, dont des phrases de dialogue ordinaires : ce n'est pas exclusivement un problème de SFX. Les prédictions supplémentaires doivent être inspectées avant de choisir ce seuil ; sans annotations exhaustives, elles ne sont pas toutes des faux positifs démontrés.

Les douze bulles absentes représentent **209 des 652 erreurs** du candidat sur pages, soit **32,1 %**. Après amélioration de l'OCR, la détection des bulles devient donc un chantier proportionnellement important. L'OCR des nouvelles détections au seuil `0.25` n'a pas été exécuté dans ce benchmark : on ne peut pas transformer directement ce meilleur rappel en un nouveau CER de pipeline complète.

La comparaison à identifiants constants isole aussi l'effet du crop sur les 1 207 bulles de test appariées communes aux deux évaluations. Avec la configuration actuelle, passer du crop annoté au crop prédit fait monter les erreurs de 1 261 à 1 367. Avec le candidat, elles passent de 416 à 437. Pour ce dernier, 43 bulles progressent et 39 régressent ; 30 deviennent exactes et 18 cessent de l'être. Un crop prédit peut donc être meilleur qu'un crop annoté sur certains exemples, même si son bilan en caractères est légèrement défavorable.

**Action :** comparer `0.45` et `0.25` sur des pages complètement annotées, revoir les nouvelles détections, puis mesurer à nouveau l'OCR et l'ordre avec ce seuil. Ne pas optimiser uniquement le rappel ni augmenter toutes les marges : le passage systématique de 2 à 4 pixels autour des lignes a dégradé le test de validation.

Sources : `summary.json` → `stage`, `page_detection`, `page_ocr` ; [worker de détection](../../frontend/src/workers/detection.worker.js), ligne 16.

### Corrections textuelles qui détruisent une bonne lecture

Le traitement est évalué à trois états : sortie CTC brute, texte après lexique/capitalisation du worker, puis texte après `fixFrenchPunctuation`.

| 1 219 crops de test, candidat | Erreurs | CER strict | Bulles exactes |
| --- | ---: | ---: | ---: |
| CTC brut | 529 | 1,318 % | 935 |
| Après lexique et capitalisation | 388 | 0,967 % | 1 022 |
| Après traitement final actuel | 419 | 1,044 % | 1 015 |

Le lexique et la capitalisation restent globalement utiles au score strict, mais dégradent 31 bulles et font perdre 24 textes initialement exacts. Le traitement final augmente les erreurs de **31 nettes** : il améliore quatre bulles et en dégrade vingt. Neuf textes exacts sont perdus contre deux gagnés. Il supprime des guillemets sur 19 sorties.

La suppression systématique de `"` est explicite dans `ocr-utils.js`. La bulle 1128 est correctement reconnue avec ses guillemets, puis ceux-ci sont effacés. La capitalisation après chaque point transforme aussi « Même si... en vérité » en « Même si... En vérité », et « Qu... quoi ?! » en « Qu... Quoi ?! ». Ces cas ne demandent pas de nouveaux poids pour être corrigés.

**Action :** conserver le texte brut, distinguer transcription fidèle et présentation, préserver les guillemets, et limiter les corrections de casse aux cas réellement établis. Valider chaque règle sur les cas qu'elle améliore et ceux qu'elle détériore. Supprimer tout le lexique serait également une mauvaise conclusion : son bilan strict reste positif. Un LLM correcteur ne doit pas inventer une ligne absente de l'image fournie au recognizer.

Sources : `summary.json` → `postprocess`, `test.variants.candidate`, `review.ordinary_test_failures` ; [ocr-utils.js](../../frontend/src/lib/ocr-utils.js), lignes 3 et 19 ; [worker OCR](../../frontend/src/workers/ocr.worker.js), ligne 411.

### Césures et qualité des références

Les résidus ordinaires incluent des problèmes différents qu'il faut séparer. Dans 5361, « ÉCARTE- / TOI » devient « Écartetoi » : le trait d'union lexical est perdu. Dans 2617, « ARRI- / VÉS » appelle une réunion de mot, tandis que le crop montre un seul `?` et que la référence contient `?!`. Une règle retirant tous les traits d'union en fin de ligne casserait donc d'autres phrases.

L'image 5750 contient visiblement un seul `!`, mais la référence en attend deux. La référence 2477 contient un espace avant une virgule. Ces désaccords ne doivent pas être présentés comme autant de défaillances certaines du modèle. Les résultats publiés ici conservent les références figées : aucune correction opportuniste du test n'a été faite.

**Action :** établir une convention de transcription pour casse, césures, espaces, ponctuation et citations ; faire revoir les exemples ambigus ; créer une nouvelle version des annotations avec journal des corrections. Conserver la structure par lignes assez longtemps pour traiter les césures selon leur contexte, ou entraîner la transcription sur des exemples cohérents de ces deux types de jonction.

![Exemples résiduels avec les boîtes de lignes du candidat](results/visual/residual_cases.png)

Sources : `summary.json` → `review.ordinary_test_failures` et `review.remaining_test_errors` ; images du snapshot local.

### Cases et ordre de lecture

Sur 256 pages dont les ordres sont exploitables, l'ordre des bulles appariées est entièrement conforme aux annotations pour **225 pages**, avec **44 inversions sur 11 194 paires**. En fournissant les boîtes de bulles annotées tout en conservant les cases prédites, on obtient **222 pages conformes** et **48 inversions sur 11 451 paires**. Les populations de paires diffèrent : ces deux taux ne sont pas une comparaison causale isolée de la qualité des boîtes.

La géométrie correcte des bulles ne suffit donc pas à éliminer les désaccords. La segmentation des cases, l'assignation bulle-case et le ranking restent à contrôler. Sur la page 356, une grande case prédite regroupe des bandes de lecture distinctes et l'ordre initial diffère des annotations. C'est un exemple utile pour inspecter l'ensemble case/assignation/ranking, sans attribuer arbitrairement toute l'erreur au seul ranker.

La vérité terrain d'ordre demande aussi une révision. Sur la page 476, l'ordre prédit paraît suivre les cases de droite à gauche alors que les annotations placent une bulle de gauche avant celles du milieu. Deux autres pages, 272 et 280, ont des ordres absents ou non uniques et sont exclues des statistiques d'ordre. Les chiffres expriment un **accord avec les annotations**, pas une certitude que toutes ces annotations sont correctes.

**Action :** revoir visuellement les 31 pages en désaccord, en distinguant annotation, détection de cases, assignation et ranking ; ensuite seulement constituer les exemples d'entraînement correspondants. Le tri interne des lignes par bandes verticales dans le worker mérite des cas de test de disposition complexe, mais cet audit n'a pas établi qu'il constituait une cause majeure des erreurs observées.

Sources : `summary.json` → `page_detection.order`, `snapshot.invalid_order_pages` ; [page 356](results/visual/order_356.png), [page 476](results/visual/order_476.png) ; [worker de détection](../../frontend/src/workers/detection.worker.js), lignes 886, 924 et 1008.

## Test d'un routeur primitif pour les textes atypiques

Un arbre à une seule décision a été ajusté sur 99 crops de validation visuellement étiquetés, puis gelé et évalué sur 99 crops de test. Les deux crops ambigus sont exclus. Le modèle compare six caractéristiques géométriques, sans texte de référence, sortie OCR ni longueur de texte : ratio du crop, nombre de lignes et ratios des boîtes de lignes. Il recherche le meilleur rappel sous la contrainte de ne router aucune bulle classique du lot d'ajustement.

La règle retenue est simplement : **router si largeur / hauteur du crop < 0,4929**.

| Population | Classiques routées à tort | Atypiques trouvés | Atypiques manqués |
| --- | ---: | ---: | ---: |
| Validation, 93 classiques et 6 atypiques | 0 / 93 | 4 / 6 | 2 / 6 |
| Test, 93 classiques et 6 atypiques | 0 / 93 | 3 / 6 | 3 / 6 |

Ce résultat est encourageant pour une heuristique de verticalité, mais insuffisant pour un filtre SFX. Le test ne contient que cinq SFX stylisés et une disposition spéciale. Le routeur manque trois SFX, dont certains sont pourtant parfaitement lisibles par PP-OCR. Il route aussi le cri vertical 7045, « AAHH !! », correctement reconnu. Zéro fausse route sur 93 bulles classiques ne prouve pas un risque nul à grande échelle.

Le candidat lit exactement deux des cinq SFX stylisés du test, notamment « FRSHH » et « CRAAC ». Il est donc trop fort de dire que ce type de modèle ne pourra jamais lire les SFX. La difficulté pertinente est la forme graphique et la disposition, pas seulement la présence d'une onomatopée. Un « BOUM » imprimé normalement peut rester dans le chemin standard ; un cri de dialogue vertical peut nécessiter un chemin spécial.

### Modèle visuel à développer ensuite

Le prochain essai devrait être un petit CNN sur le crop, avec ratio conservé, par exemple trois blocs convolutionnels de 16, 32 et 64 canaux, pooling global et une tête de classification. C'est une proposition de modèle à entraîner, pas un modèle visuel déjà validé par cet audit. Les trois sorties utiles seraient `standard`, `atypique` et `incertain`, avec des tags d'annotation séparés pour verticalité, courbure, lettres déformées et texte sur dessin.

Comme budget de départ, annoter 1 000 à 3 000 crops, dont plusieurs centaines de cas atypiques, puis ajuster ce volume à la diversité des erreurs. Ce n'est pas un seuil de données garantissant la réussite. Garder le split par page, et réserver aussi un tome ou un style de lettrage pour mesurer la généralisation. Les 200 labels exploratoires actuels servent d'amorce, pas de dataset suffisant pour une promesse de fiabilité.

Le critère principal doit être le taux de dialogues classiques envoyés à tort vers le chemin spécial, accompagné du rappel des atypiques. Éviter l'accuracy globale, qui peut être élevée en prédisant toujours la classe majoritaire. Pour un cas atypique ou incertain, conserver l'image et un statut `needs_review` ou déclencher un second modèle. **Ne pas convertir « ignoré » en transcription réussie, ni supprimer silencieusement la bulle.**

Continuer à entraîner PP-OCR sur les SFX lisibles et les cris dont les cibles sont fiables. Comparer sur le train, puis la validation, les mélanges tout inclus, cas compatibles et pondération par difficulté. Il n'est pas démontré ici que retirer tous les SFX améliore les dialogues ; les exemples incompatibles avec une lecture linéaire doivent être routés ou traités avec une géométrie adaptée.

Sources : [router.cjs](router.cjs), [visual-labels.json](visual-labels.json), `summary.json` → `router`, `classes`.

## Ce qu'il faut corriger avant le prochain entraînement

Le dataset historique a exporté 7 821 bulles sur 7 843 ; **22 sans lignes détectées ont été exclues**. Ses paramètres étaient seuil de lignes `0.20`, IoU `0.45` et taille 1 280, tandis que l'ONNX de lignes chargé prend 1 024 × 1 024 et que le worker applique son propre seuil et NMS. Le score historique après traitement, autour de 1,45 % de CER, n'est donc pas le score de la pipeline de pages actuelle. Il repose également sur d'anciennes références et une normalisation différente.

Le générateur conserve le texte complet de la bulle comme cible de l'image recollée. Si une ligne est oubliée lors de l'export, le couple image/cible peut devenir incohérent. L'incidence exacte n'a pas été comptée sur le train. Il faut annoter la complétude ou revoir ces couples, et journaliser les exclusions au lieu d'effacer les cas difficiles des métriques. Le chemin Supabase effectue un split par page ; le chemin local sans manifeste peut répartir par identifiant de bulle, ce qui expose à un partage de pages entre splits. Ce dernier risque n'a pas été constaté dans le snapshot audité.

Le script de reconnaissance filtre les caractères absents du vocabulaire lors de la conversion des cibles. Il limite leur longueur à `image_width // 8` et utilise `zero_infinity=True`. Pour CTC, il faut vérifier sur la longueur temporelle réelle de la sortie la condition `T >= longueur_cible + repetitions_consecutives`, où `repetitions_consecutives` compte les paires de caractères adjacents identiques dans la cible. Deux caractères identiques consécutifs nécessitent une séparation CTC ; la seule longueur de la cible ne suffit pas. Sans journalisation, une perte impossible peut être masquée. **Ce risque est établi par lecture du code ; le nombre d'exemples effectivement concernés n'a pas été mesuré.**

Le script dispose déjà de pondérations et de suréchantillonnage pour les textes courts, les images à une ligne et les exemples difficiles. L'étape suivante est de les alimenter avec des erreurs correctement diagnostiquées, issues du train uniquement, plutôt que d'ajouter un second mécanisme équivalent. Le défaut de largeur d'entrée doit être corrigé avant de conclure qu'il faut davantage suréchantillonner les ponctuations.

Concrètement, figer une nouvelle version de données, vérifier vocabulaire et faisabilité CTC, assurer la parité du prétraitement export/Python/navigateur, puis entraîner séparément le détecteur sur les lignes manquées et le recognizer sur les erreurs de transcription restantes. Un entraînement du recognizer seul ne récupère pas les pixels d'une ligne que le détecteur a supprimée. Réserver un nouveau jeu final qui ne sert ni au choix des règles ni au minage des erreurs.

Sources : `cache/models/dataset_stats.json` ; [préparation du dataset](../finetune_paddleocr_line_rec/prepare_line_rec_dataset.py), lignes 369, 446 et 567 ; [entraînement](../finetune_paddleocr_line_rec/train_ppocrv6_rec.py), lignes 218, 288, 470, 552 et 718. La compatibilité directe de l'ancien téléchargeur HTTP avec les URL actuelles `r2://` est également à corriger pour une prochaine réexportation.

## Ordre de réalisation recommandé

| Priorité | Travail concret | Critère de validation |
| --- | --- | --- |
| 1 | Harmoniser les crops sans perte et les arrondis ; intégrer derrière un réglage le ratio préservé et le seuil de lignes `0.10`. | Rejouer les vrais chemins navigateur et batch, avec comparaison par bulle, sorties vides et régressions ; mesurer WASM et WebGPU séparément. |
| 2 | Corriger la suppression des guillemets et les règles de casse agressives ; conserver texte brut et texte présenté. | Mesurer les règles séparément et vérifier les 19 sorties contenant des guillemets ainsi que les continuations après points de suspension. |
| 3 | Ajouter couverture de lignes, statut incomplet et second passage ciblé ; annoter les lignes d'un lot de pages complètes. | Faire disparaître les omissions connues sans augmenter les faux textes issus des dessins ; mesurer complétude, rappel et précision des lignes. |
| 4 | Réévaluer le seuil de bulles `0.25`, puis corriger les cas d'ordre et les références douteuses. | Mesurer toute la page, y compris ajouts, doublons et ordre, sur des annotations exhaustives et revues. |
| 5 | Réexporter des données cohérentes, réentraîner détecteur/recognizer selon la cause, puis entraîner le petit routeur visuel. | Test neuf par page/tome ; résultats séparés standard/atypique/incertain, couverture et taux de fausses routes. |

Les gains mesurés justifient de commencer par les paramètres et le code. Aucun gain supplémentaire de fine-tuning, de CNN de routage ou de nouveau modèle de secours n'est présumé dans les chiffres de ce rapport.

## Définir un résultat presque parfait sans masquer les échecs

La bonne unité produit est une bulle complète, puis une page complète : toutes les bulles attendues, toutes leurs lignes, texte fidèle et ordre correct, sans doublon ni texte ajouté. Un CER proche de 1 % ne signifie pas 99 % de pages parfaites. Dans le test natif candidat, seules 24 des 129 pages ont simultanément le texte strictement exact et l'ordre conforme aux annotations, selon la couverture limitée décrite plus haut.

Le score de reconnaissance seul ne permet pas une acceptation sûre. À titre descriptif, sur les 1 219 crops de test, conserver seulement un score CTC moyen d'au moins `0.99` garde 430 bulles, soit 35,3 % de couverture, mais seulement 403 sont strictement exactes : **93,7 %**, et non 99 %. Même `0.999` ne donne que 88 textes exacts sur 92. Ces seuils ne sont ni calibrés ni validés comme règles de décision et ne voient pas les bulles absentes.

Pour viser un résultat exploitable sans correction courante, fixer un objectif explicite sur le sous-ensemble accepté automatiquement, par exemple 99,5 % de bulles classiques entièrement exactes, puis mesurer la couverture obtenue sur un test indépendant et revu. C'est un objectif, pas le résultat actuel. Combiner complétude, confiance calibrée, désaccord entre chemins et contrôle du post-traitement ; orienter le reste vers relecture ou modèle de secours. Une garantie universelle de 100 % n'est pas établie par ces tests.

Sources : `summary.json` → `review.confidence_diagnostic`, `page_ocr`.

## Reproduction et fichiers livrés

Le [résumé JSON](summary.json) conserve les métriques, comparaisons appariées, exemples, révisions de modèles et SHA256 des sources et des résultats. Les poids utilisés sont épinglés à :

| Artefact | Révision |
| --- | --- |
| `Remidesbois/pp-ocrv6-one-piece-bubble-line-rec` | `10b932d4aadca2830850ccf5951116597404bef8` |
| `Remidesbois/Poneglyph-ReaderNet` | `d97d4cd2903a7ebe49276a5269c4f3b7df608be7` |

Les dimensions observées sont 1 024 × 1 024 pour le détecteur de lignes, hauteur 48 avec largeur dynamique pour la reconnaissance, 800 pour le détecteur de bulles et 1 504 pour les cases. Il faut privilégier les métadonnées ONNX effectivement chargées aux anciennes valeurs documentées.

Les sources et résultats légers à conserver dans le dépôt sont les scripts de ce dossier, `visual-labels.json`, `summary.json` et ce rapport. `cache/` et `results/` restent locaux et ignorés par Git : ils contiennent les modèles, images et sorties détaillées. Les illustrations du rapport y sont consultables dans cet environnement ; elles ne sont pas embarquées dans un simple clone du dépôt.

Prérequis observés à la racine : `onnxruntime-node@1.24.3`, `onnxruntime-web@1.27.0`, `sharp@0.34.5`, `@supabase/supabase-js@2.115.0`, `@aws-sdk/client-s3@3.1127.0` et `dotenv@16.6.1`. Canvas est installé dans `cache/runtime/node_modules`. Chrome est attendu dans `C:/Program Files/Google/Chrome/Application/chrome.exe`. Pour reconstruire ce cache Canvas :

```powershell
npm install --prefix docker_scripts/audit_ppocr_pipeline/cache/runtime --no-save --package-lock=false @napi-rs/canvas@1.1.0
```

Les commandes ci-dessous sont à lancer depuis la racine du dépôt. Le snapshot et les résultats présents sont réutilisés ; il n'est pas nécessaire de relancer l'inférence pour lire ou reconsolider cet audit.

```powershell
# Reconstruction initiale seulement : lectures Supabase/R2 avec backend/.env.
node docker_scripts/audit_ppocr_pipeline/prepare.cjs

# Ablations et choix sur validation, puis confirmation du candidat.
node docker_scripts/audit_ppocr_pipeline/benchmark.cjs --split val --limit 300 --mode ablation
node docker_scripts/audit_ppocr_pipeline/compare.cjs val 300 select
node docker_scripts/audit_ppocr_pipeline/compare.cjs test 0 confirm

# Detection et ordre sur 258 pages, OCR des 129 pages de test.
node docker_scripts/audit_ppocr_pipeline/pages.cjs
node docker_scripts/audit_ppocr_pipeline/page-ocr.cjs

# Vrai navigateur : diagnostic et echantillon deterministe de test.
node docker_scripts/audit_ppocr_pipeline/browser-parity.cjs targeted
node docker_scripts/audit_ppocr_pipeline/browser-parity.cjs test 100

# Routeur avec les labels visuels conserves, puis consolidation sans inference.
node docker_scripts/audit_ppocr_pipeline/router.cjs
node docker_scripts/audit_ppocr_pipeline/consolidate.cjs
```

Les scripts d'inférence reprennent les JSONL existants. Pour tester une autre version du code ou des modèles, archiver les résultats et le snapshot précédents avant de créer un nouveau run ; ne pas mélanger des lignes issues de versions différentes. `consolidate.cjs` vérifie les effectifs attendus et l'absence de doublons. Les fichiers bruts restent la source de détail pour chaque mesure.
