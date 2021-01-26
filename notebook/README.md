# ros4pro_perception/notebook

Hand-made digits recognition with tensorflow2

Cette activité propose la programmation, l'entraînement et l'évaluation de réseaux de neurones dédiés à la reconnaissance de chiffres écrits à la main provenant de la banque de données MNIST.

Deux TP sont proposés sous forme de notebooks jupyter 'à trous' :

- `TP1_MNIST_dense.ipynb` : présente des rappels sur des concepts de base utiles en ML (neurone artificiel, fonctions d'activations, *one-hot coding*, catégorisation des labels, entraînement du réseau, affichage de la matrice de confusion...).
Le notebook aborde ensuite la construction d'un réseau dense à 2 couches, permettant de reconnaissance les chifres MNIST avec une précision qui voisine de 98 %.

- `TP2_MNIST_convol.ipynb` : aborde la construction d'un réseau convolutionnel pour la reconnaissance des chifres MNIST qui peut atteindre 99 % de réusssite.

À la suite de ces TP, tu pourras :
- finaliser les programmes Python qui sont dans le répertoire `ros4pro_perception/src/`, pour entraîner un réseau de neurones convolutif à reconnaître des images de '1' et de '2' extraites d'images existantes,
- puis créer un service ROS qui utilise un réseau le neurones entraîné, pour reconnaître les chiffres ('1' ou '2') écrits sur les cubes pris en photo par la caméra du bras manipulateur utilisé.


## Principaux points abordés dans les TP

- neurone artificiel,
- réseau de neurones,
- fonction d'activation,
- téléchargement et visualisation des images MNIST (*handwritten digits*),
- préparation des images et des labels pour entraÎner le réseau de neurones,
- programmation d'un réseau de neurones dense puis convolutionnel avec tensorFlow-keras,
- entraînement des réseaux,
- courbes de précision et de perte,
- matrice de confusion,
- `callback` tensorflow `Early Stopping` pour éviter le sur-entraînement.
- exploitation des réseaux avec des chiffres manuscrits hors banque MNIST.
