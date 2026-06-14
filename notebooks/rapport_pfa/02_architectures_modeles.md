# 🧠 Chapitre 2 : Architectures des Modèles

Ce chapitre propose une analyse approfondie des fondations théoriques et structurelles des modèles de vérification du locuteur étudiés. Il inclut le détail des équations de propagation et des tableaux de configuration complets des couches.

---

## 1. Modèle X-Vector : Réseau à Retard Temporel (TDNN)

Le modèle **X-Vector** s'appuie sur une architecture TDNN (Time Delay Neural Network). L'originalité du TDNN réside dans sa capacité à modéliser des dépendances temporelles à long terme en empilant des convolutions 1D dilatées opérant au niveau de la trame.

### A. Calcul du Champ de Réception Temporel (Receptive Field)
Chaque couche TDNN traite une trame à l'instant $t$ en considérant un contexte temporel symétrique autour de celle-ci. Si l'on note $d$ la dilatation et $k$ la taille du noyau, le champ de réception effectif d'une couche est donné par la formule :
$$R_{\text{couche}} = (k - 1)d + 1$$
Lors de l'empilement de $L$ couches, le champ de réception cumulé s'ajoute selon la formule :
$$R_{\text{cumulé}} = R_{\text{précédent}} + (k_L - 1)d_L$$

Calculons le champ de réception trame par trame pour notre modèle X-Vector :

| Couche | Kernel ($k$) | Dilatation ($d$) | Contexte Temporel à l'instant $t$ | Receptive Field (Trames) | Receptive Field Cumulé |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **tdnn1** | $5$ | $1$ | $[t-2, t-1, t, t+1, t+2]$ | $5$ trames ($50\text{ ms}$) | $5$ trames ($50\text{ ms}$) |
| **tdnn2** | $3$ | $2$ | $[t-2, t, t+2]$ | $5$ trames | $9$ trames ($90\text{ ms}$) |
| **tdnn3** | $3$ | $3$ | $[t-3, t, t+3]$ | $7$ trames | $15$ trames ($150\text{ ms}$) |
| **tdnn4** | $1$ | $1$ | $[t]$ (convolution 1x1) | $1$ trame | $15$ trames ($150\text{ ms}$) |
| **tdnn5** | $1$ | $1$ | $[t]$ (convolution 1x1) | $1$ trame | $15$ trames ($150\text{ ms}$) |

Grâce à ces dilatations, le modèle agrège un contexte acoustique de 15 trames (soit 150 ms de parole) avant de projeter les caractéristiques vers les couches globales.

### B. Équations du Statistics Pooling
Soit $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T] \in \mathbb{R}^{C \times T}$ la matrice des caractéristiques de trames issue de la couche **tdnn5** ($C = 1500$, $T$ étant la longueur temporelle). La couche de pooling statistique écrase la dimension temporelle $T$ en calculant deux statistiques globales :

1.  **La moyenne temporelle ($\boldsymbol{\mu} \in \mathbb{R}^C$)** :
    $$\boldsymbol{\mu} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t$$
2.  **L'écart-type temporel ($\boldsymbol{\sigma} \in \mathbb{R}^C$)** :
    $$\boldsymbol{\sigma} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (\mathbf{h}_t - \boldsymbol{\mu})^2 + \epsilon}$$
    *(où $\epsilon = 10^{-9}$ est un facteur de régularisation empêchant l'annulation du gradient).*

Les deux vecteurs sont concaténés pour former le vecteur statistique global $\mathbf{v}_{\text{pool}}$ :
$$\mathbf{v}_{\text{pool}} = \left[\begin{array}{c} \boldsymbol{\mu} \\ \boldsymbol{\sigma} \end{array}\right] \in \mathbb{R}^{2C \times 1} \quad (\text{dimension } 3000)$$

---

## 2. Modèle ECAPA-TDNN : Attention, Propagation et Agrégation Emphasées

Le modèle **ECAPA-TDNN** perfectionne le traitement temporel en introduisant des mécanismes d'attention et des convolutions multi-échelles.

```
                  ┌───────────────┐
                  │ Log-Mel (80)  │
                  └───────┬───────┘
                          │ (Conv1D + BN + ReLU)
                  ┌───────▼───────┐
            ┌────>│  x0 (512, T)  ├───────────────────────────────┐
            │     └───────┬───────┘                               │
            │             │ (SE-Res2Net, Dilation=2)              │
            │     ┌───────▼───────┐                               │
            ├────>│  x1 (512, T)  ├─────────────────────────┐     │
            │     └───────┬───────┘                         │     │
            │             │ (SE-Res2Net, Dilation=3)        │     │
            │     ┌───────▼───────┐                         │     │
            ├────>│  x2 (512, T)  ├───────────────────┐     │     │
            │     └───────┬───────┘                   │     │     │
            │             │ (SE-Res2Net, Dilation=4)  │     │     │
            │     ┌───────▼───────┐                   │     │     │
            ├────>│  x3 (512, T)  │                   │     │     │
            │     └───────┬───────┘                   │     │     │
            │             └───────────┐               │     │     │
            │                         ▼               ▼     ▼     ▼
            │                   ┌─────────────────────────────────┐
            │                   │ Concaténation [x0, x1, x2, x3]  │ (2048, T)
            │                   └────────────────┬────────────────┘
            │                                    │ (Conv1x1)
            │                           ┌────────▼────────┐
            │                           │   (1536, T)     │
            │                           └────────┬────────┘
            │                                    │ (Attentive Stats Pooling)
            │                           ┌────────▼────────┐
            │                           │   (3072, 1)     │
            │                           └────────┬────────┘
            │                                    │ (Linear)
            │                           ┌────────▼────────┐
            │                           │ Embedding (192) │
            │                           └─────────────────┘
```

### A. Le Bloc SE-Res2Net
Le bloc SE-Res2Net est le cœur du modèle. Il combine le traitement hiérarchique multi-échelle de Res2Net et le recalibrage de canaux de Squeeze-and-Excitation.

#### 1. Propagation Res2Net (Échelle $S = 4$)
Le vecteur de caractéristiques d'entrée $\mathbf{x} \in \mathbb{R}^{C \times T}$ ($C=512$) subit d'abord une convolution 1x1 qui projette les canaux de $512$ à $512$. La sortie $\mathbf{z}$ est divisée équitablement en $S = 4$ sous-vecteurs le long de l'axe des canaux :
$$\mathbf{z} = [\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3, \mathbf{z}_4] \quad \text{avec } \mathbf{z}_i \in \mathbb{R}^{\frac{C}{S} \times T} \quad \left(\frac{C}{S} = 128\right)$$

Le premier canal $\mathbf{z}_1$ sert de connexion directe. Les autres canaux subissent des convolutions 1D de noyau $3$ et de dilatation $d$ cumulativement :
$$\mathbf{y}_1 = \mathbf{z}_1$$
$$\mathbf{y}_2 = \text{Conv}_{3\text{x}3}(\mathbf{z}_2)$$
$$\mathbf{y}_3 = \text{Conv}_{3\text{x}3}(\mathbf{z}_3 + \mathbf{y}_2)$$
$$\mathbf{y}_4 = \text{Conv}_{3\text{x}3}(\mathbf{z}_4 + \mathbf{y}_3)$$

Les sous-vecteurs de sortie $\mathbf{y}_i$ sont ensuite concaténés pour reformer le vecteur $\mathbf{y} \in \mathbb{R}^{C \times T}$ de dimension $512$ :
$$\mathbf{y} = [\mathbf{y}_1, \mathbf{y}_2, \mathbf{y}_3, \mathbf{y}_4]$$

Cette structure augmente exponentiellement le nombre de chemins de traitement internes, permettant de capturer des formants à différentes échelles de résolutions temporelles.

#### 2. Squeeze-and-Excitation (SE-Block)
Le Squeeze-and-Excitation applique une attention sur les canaux pour filtrer le bruit :
*   **Squeeze (Séchage)** : Un vecteur statistique global $\mathbf{g} \in \mathbb{R}^{C \times 1}$ est calculé par moyenne temporelle globale :
    $$\mathbf{g}_c = \frac{1}{T}\sum_{t=1}^{T} \mathbf{y}_c(t)$$
*   **Excitation (Excitation)** : Deux couches linéaires calculent un vecteur de poids d'attention de canaux $\mathbf{s} \in \mathbb{R}^{C \times 1}$ avec un facteur de réduction $r = 8$ (bottleneck) :
    $$\mathbf{s} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{g}))$$
    *(où $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ et $\sigma(x) = \frac{1}{1 + e^{-x}}$ est la fonction Sigmoïde).*
*   **Mise à l'échelle** : La sortie du bloc SE est multipliée élément par élément avec l'entrée :
    $$\tilde{\mathbf{y}}_c(t) = \mathbf{s}_c \cdot \mathbf{y}_c(t)$$

Enfin, la connexion résiduelle externe réinjecte l'entrée du bloc complet :
$$\text{Sortie} = \text{ReLU}(\tilde{\mathbf{y}} + \mathbf{x})$$

### B. Attentive Statistics Pooling (ASP)
L'ASP estime de manière dynamique l'importance de chaque trame de parole pour le locuteur :

1.  Soit $\mathbf{h}_t \in \mathbb{R}^{C \times 1}$ le vecteur de caractéristiques à l'instant $t$ ($C=1536$ après projection multi-couches MFA).
2.  Le score d'attention scalaire $e_t$ est calculé par projection linéaire :
    $$e_t = \mathbf{v}^T \tanh(\mathbf{W} \mathbf{h}_t + \mathbf{b})$$
    *(où $\mathbf{W} \in \mathbb{R}^{d_{\text{att}} \times C}$ avec $d_{\text{att}}=128$, $\mathbf{b} \in \mathbb{R}^{d_{\text{att}} \times 1}$ et $\mathbf{v} \in \mathbb{R}^{d_{\text{att}} \times 1}$).*
3.  Les poids d'attention normalisés $\alpha_t$ sont obtenus par une opération Softmax temporelle :
    $$\alpha_t = \frac{e^{e_t}}{\sum_{\tau=1}^{T} e^{e_\tau}}$$
4.  Les statistiques globales pondérées sont alors calculées :
    *   **Moyenne attentive ($\tilde{\boldsymbol{\mu}}$)** :
        $$\tilde{\boldsymbol{\mu}} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t$$
    *   **Écart-type attentif ($\tilde{\boldsymbol{\sigma}}$)** :
        $$\tilde{\boldsymbol{\sigma}} = \sqrt{\sum_{t=1}^{T} \alpha_t (\mathbf{h}_t - \tilde{\boldsymbol{\mu}})^2 + \epsilon}$$

Le vecteur résultant concaténé $\mathbf{v}_{\text{ASP}} = [\tilde{\boldsymbol{\mu}}, \tilde{\boldsymbol{\sigma}}] \in \mathbb{R}^{3072 \times 1}$ est projeté par une couche dense finale pour obtenir l'embedding de dimension **192**.

---

## 3. Modélisation de la Marge Angulaire : AAM-Softmax

La fonction de perte **AAM-Softmax** (Additive Angular Margin Softmax) est formulée mathématiquement comme suit.

### Équations de la Transformation Angulaire
En Softmax classique, le terme d'activation pour une classe $j$ s'écrit $\mathbf{w}_j^T \mathbf{x}_i + b_j$. En AAM-Softmax, nous posons les contraintes suivantes :
1.  Les biais sont annulés : $b_j = 0$.
2.  Les embeddings $\mathbf{x}_i$ et les poids de classes $\mathbf{w}_j$ sont normalisés par leur norme L2 :
    $$\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}, \quad \tilde{\mathbf{w}}_j = \frac{\mathbf{w}_j}{\|\mathbf{w}_j\|_2}$$
    Le produit scalaire devient :
    $$\tilde{\mathbf{w}}_j^T \tilde{\mathbf{x}}_i = \cos(\theta_{j, i})$$
    *(où $\theta_{j, i}$ est l'angle entre le vecteur d'embedding et le vecteur représentatif de la classe $j$).*

Pour la classe cible $y_i$, on ajoute une marge angulaire constante $m = 0{,}2$ :
$$\theta_{y_i, i} \leftarrow \theta_{y_i, i} + m$$
Comme la fonction cosinus est décroissante sur $[0, \pi]$, ajouter $m$ à l'angle diminue la valeur du cosinus :
$$\cos(\theta_{y_i, i} + m) = \cos(\theta_{y_i, i})\cos(m) - \sin(\theta_{y_i, i})\sin(m) < \cos(\theta_{y_i, i})$$

Le réseau de neurones doit faire des efforts d'optimisation supplémentaires pour compenser cette pénalité angulaire artificielle sur la classe cible. La perte globale est multipliée par un paramètre d'échelle $s=30$ pour faciliter la convergence numérique :
$$\mathcal{L}_{\text{AAM}} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s \cdot \cos(\theta_{y_i, i} + m)}}{e^{s \cdot \cos(\theta_{y_i, i} + m)} + \sum_{j \neq y_i}^{C} e^{s \cdot \cos(\theta_{j, i})}}$$

### Rôle géométrique de la Marge Angulaire
L'application de cette marge angulaire modifie radicalement les frontières de décision dans l'espace d'embedding.

```
   Frontière Linéaire (Cross-Entropy)            Frontière Angulaire (AAM-Softmax)
           
              Classe 1                                      Classe 1
                 |                                             /
                 |                                            /  Zone de marge m
                 |                                           /  (Géométriquement
                 |                                          /     interdite)
                 |                                         /
              Classe 2                                    /  Classe 2
```

1.  **Frontière de Décision stricte** : Pour que le modèle classe correctement l'échantillon, il doit valider l'inégalité $\cos(\theta_{1} + m) > \cos(\theta_{2})$, ce qui signifie que l'angle avec le vrai locuteur doit être nettement plus petit que l'angle avec les autres locuteurs.
2.  **Hypersphère unitaire** : Tous les locuteurs sont projetés sur une sphère. La distance entre deux points sur cette sphère est directement proportionnelle à l'angle, ce qui rend la **similarité cosinus** mathématiquement cohérente lors de la phase d'évaluation.
