# Falcon OCR : accélération WebGPU

Le modèle reste `Remidesbois/Falcon-OCR-Poneglyph`, révision
`abd33103ab9fef627d89ef430e1ecb1c15fb7194`. Le nouveau décodeur utilise les
poids et le préremplissage du graphe ONNX existant, puis réalise la génération
token par token avec des kernels WebGPU et un cache KV de taille fixe. Les
projections FP32 et les arrondis intermédiaires FP16 sont conservés.

L'index des poids est intégré au frontend et n'est utilisé que si les
empreintes du checkpoint, du graphe et des données ONNX correspondent. Les
GPU sans sous-groupes ou avec des limites de buffers insuffisantes utilisent
le moteur ONNX. Si le décodeur accéléré échoue sur une image et que le GPU
reste disponible, la transcription est recommencée avec ONNX.

Sur RTX 5090 sous Windows, dans le worker de l'application, les 1 691 bulles
du corpus ont donné les mesures suivantes après échauffement :

| Mesure | ONNX | Décodeur fusionné |
| --- | ---: | ---: |
| Latence médiane | 319 ms | 63 ms |
| Latence moyenne | 356,6 ms | 68,3 ms |
| 95e centile | 802 ms | 111 ms |
| Erreur de caractères (CER) | 0,143 % | 0,141 % |
| Sorties exactes après normalisation | 1 654/1 691 | 1 655/1 691 |
| Sorties atteignant la limite | 0 | 0 |

Les transcriptions brutes des deux moteurs sont identiques sur 1 690 bulles sur
1 691 ; la différence restante est d'un caractère et correspond mieux à la
référence. Le rapport des médianes est de 5,06 ; la médiane des accélérations
calculées bulle par bulle est de 4,86. Sur un sous-ensemble de 128 bulles
réparties selon leur longueur, les sorties brutes sont identiques sur 128/128 et
les médianes sont de 343,5 ms et 63 ms respectivement.

Sur sept images synthétiques extrêmes, cinq sorties sont identiques et deux
diffèrent, de deux caractères au maximum ; aucune ne change de condition
d'arrêt. Ces cas artificiels ne sont pas représentatifs des bulles du corpus.

La première initialisation du décodeur ajoute un transfert et une transposition
des poids sur le GPU : la copie persistante occupe environ 617 Mo supplémentaires,
avec un pic temporaire supérieur. Le cache KV utilise uniquement les huit têtes
distinctes et occupe environ 23 Mo au lieu de 46 Mo pour une bulle représentative.
Les temps ci-dessus ne comprennent pas le téléchargement ou l'initialisation.
Ils ne préjugent pas des performances sur d'autres cartes ou navigateurs.

Ces bulles servent à comparer deux implémentations du même modèle, non à
mesurer sa généralisation : le checkpoint Falcon publié a été réentraîné sur
toutes les données de son corpus. Les images et transcriptions de ces essais
restent dans des sorties locales ignorées par Git.
