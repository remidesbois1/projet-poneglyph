# Surya bulles : publication et mesures d'inférence

Le checkpoint choisi est exporté en BF16 et publié avec son export WebGPU issu
des mêmes poids. SHA-256 du modèle natif :
`8fa50e74b2ae134e5a8f0269cee2f8a41cfd586096bb8891eb1f3bd676b1dca7`.
Les 473 tenseurs ont été comparés à l'arrondi BF16 des poids FP32 d'entraînement.

L'évaluation native du modèle entraîné porte sur 1 691 bulles de test séparées
par page : CER 0,465 %, WER 1,823 %, transcription exacte 91,19 %, sans sortie
atteignant la limite de 256 tokens. Les métriques et intervalles de confiance
agrégés sont publiés dans `benchmark_test.json`. Ces chiffres ne sont pas ceux
du moteur WebGPU.

## Moteur desktop CUDA

Sur une RTX 5090 sous Windows, avec le vrai endpoint `/ocr/text`, 128 bulles
réparties sur les longueurs du jeu de test ont été mesurées après échauffement.
Les kernels Flash Linear Attention réduisent la latence médiane de 230,35 à
106,75 ms (2,16 fois plus rapide) et la moyenne de 232,74 à 116,62 ms. Les
128 transcriptions normalisées sont identiques ; les deux exécutions obtiennent
121 transcriptions exactes et un CER de 0,191 % sur ce sous-ensemble. Ce
sous-ensemble stratifié n'est pas une estimation du CER du jeu complet.

Le temps de chargement et de compilation n'est pas inclus dans ces mesures.
Les dépendances sont épinglées ; les machines sans CUDA BF16, Surya-BBox et les
GPU incompatibles utilisent les opérations de référence. Le banc reproductible
est `desktop_backend/benchmark_surya_bubbles.py` et les résultats détaillés sont
conservés localement dans le dossier ignoré des sorties d'entraînement.

Un essai des intervalles de détection de fin 1, 2, puis 1 dans le même processus
donne respectivement 119,30, 117,00 et 118,23 ms de médiane sur ces 128 bulles,
avec des sorties identiques. Le gain observé est trop faible pour changer le
réglage. Une autre convolution Triton a également été essayée sans gain net.

## Moteur WebGPU

Sur les mêmes 128 bulles, avec le worker utilisé par Poneglyph, la fusion des
projections MLP préserve les arrondis intermédiaires FP16. Le préremplissage en
blocs de 32 tokens obtient 128/128 sorties identiques à la référence native et
une médiane de 330 ms. Sur les 24 bulles normales d'un autre jeu de validation,
trois passages avant/après fusion ont donné 354 puis 337 ms de médiane, avec
72/72 sorties identiques.

Avec le réglage retenu et les optimisations de prétraitement, l'exécution
WebGPU sur les 1 691 bulles donne un CER de 0,474 %, un WER de 1,851 % et
1 539 transcriptions exactes sur 1 691 (91,01 %). Il y a 17 sorties
différentes de la référence native, aucune de plus de quatre caractères,
et aucune sortie tronquée. La médiane observée est de 392 ms ; d'autres
processus et tests étaient présents pendant ce contrôle, contrairement aux
comparaisons appariées sur 128 bulles ci-dessus.

La conversion des pixels utilise une table de correspondance exacte et copie
la seconde frame au lieu de recalculer chaque valeur. Sur une image synthétique
224 × 320, le microbenchmark CPU `frontend/bench/surya-preprocessing.mjs`
mesure 2,31 ms avant et 0,48 ms après, avec les mêmes valeurs FP16. Ce gain
concerne le prétraitement seul ; il ne multiplie pas la vitesse totale par 4,8.

Des blocs de 64 tokens ont mesuré 241 ms de médiane sur les 128 bulles. Sur
les 1 691 bulles de test, ils donnent 0,816 % de CER et 87,94 % de
transcriptions exactes, contre 0,465 % et 91,19 % pour la référence native :
92 sorties diffèrent, dont cinq de plus de cinq caractères. Une taille
intermédiaire de 48 a elle aussi provoqué une erreur longue. Ces variantes
ont été écartées : le modèle publié conserve les blocs de 32 tokens. Le jeu
complet a servi à ce choix d'implémentation et ne constitue donc plus une
évaluation indépendante du réglage WebGPU choisi à partir de ces résultats.

Les deux moteurs ont aussi été testés dans la Sandbox de l'application Windows
sur une page de démonstration, avec le frontend local du projet. Les images et
transcriptions privées des bancs ont été déplacées hors du répertoire public,
dans les sorties d'entraînement ignorées par Git.
