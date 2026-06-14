# 📂 Chapitre 1 : Introduction et Méthodologie

---

## 1. Contexte de la Vérification Biométrique Vocale

La reconnaissance vocale biométrique se divise en deux tâches principales aux objectifs et aux complexités bien distincts : **l'identification du locuteur** et **la vérification du locuteur**. 

*   **L'identification (1-contre-N)** consiste à comparer une empreinte vocale inconnue à une base de données de $N$ locuteurs enregistrés afin de déterminer l'identité de l'orateur. La complexité algorithmique croît linéairement avec le nombre de locuteurs $N$, et le taux d'erreur augmente au fur et à mesure que la base s'élargit.
*   **La vérification (1-contre-1)**, également appelée authentification vocale, consiste à valider l'identité revendiquée par un utilisateur. Le système compare la voix d'entrée uniquement avec le modèle de référence de l'identité revendiquée. La complexité reste constante quel que soit le nombre d'utilisateurs inscrits. C'est cette tâche, hautement critique pour la cybersécurité (contrôle d'accès, validation de transactions bancaires), qui est développée dans ce projet.

```mermaid
graph TD
    subgraph Vérification Binaire (1-vs-1)
        A[Revendication d'Identité ID] --> B[Charger Modèle de Référence de ID]
        C[Signal Audio de Test] --> D[Extracteur d'Embeddings]
        B --> E[Calcul de Similarité Cosinus]
        D --> E
        E --> F{Similarité >= Seuil de Décision ?}
        F -- Oui (Score élevé) --> G[Locuteur Authentique]
        F -- Non (Score faible) --> H[Imposteur Rejeté]
    end
```

### Classification des Systèmes de Vérification
Les systèmes de Vérification Automatique du Locuteur (VAL) se séparent également en deux catégories :
1.  **Dépendant du texte (Text-Dependent)** : Le locuteur doit prononcer une phrase fixe prédéfinie (ex: *"Ok Google"*). Ces systèmes sont plus simples car le contenu phonétique est identique entre la référence et le test, mais ils sont vulnérables aux attaques par rejeu.
2.  **Indépendant du texte (Text-Independent)** : Le locuteur peut prononcer n'importe quel énoncé. Le système doit faire abstraction du contenu linguistique pour n'extraire que les caractéristiques physiologiques (cordes vocales, conduit vocal) et comportementales (prosodie, accent). **C'est cette approche robuste et complexe qui est implémentée dans nos modèles X-Vector et ECAPA-TDNN.**

---

## 2. Présentation Exhaustive des Bases de Données

L'entraînement de réseaux de neurones profonds robustes *from scratch* requiert des données acoustiques massives, variées et documentées.

### A. VoxCeleb1 : Parole en conditions réelles
Le jeu de données **VoxCeleb1** (Nagrani et al., 2017) a marqué un tournant dans la recherche en VAL. Contrairement aux bases de données historiques enregistrées en studio insonorisé, VoxCeleb1 contient de la parole extraite de vidéos YouTube (interviews, conférences de presse, débats) dans des environnements non contrôlés.

*   **Caractéristiques physiques** :
    *   **Nombre de locuteurs** : **1 111 locuteurs uniques** indexés dans notre partition d'entraînement Kaggle.
    *   **Variabilité acoustique** : Présence de bruits de fond (musique de fond, rires, applaudissements), réverbération de salle, artefacts de compression audio YouTube, et microphones de qualités diverses.
    *   **Variabilité biologique** : Diversité d'âges, d'accents (anglais américain, britannique, européen, asiatique) et d'états émotionnels.
*   **Traitement temporel** : Pour chaque enregistrement, nous découpons dynamiquement un segment de **3 secondes** (soit $48\,000$ échantillons à $16\text{ kHz}$). Si l'audio d'origine est plus court, un rembourrage par zéros (zero-padding) est appliqué.

### B. MUSAN : Bruitages pour la robustesse acoustique
Le dataset **MUSAN** (Snyder et al., 2015) est conçu pour l'évaluation et l'entraînement de systèmes de traitement de la parole. Il contient environ 109 heures d'audio réparties en trois sous-ensembles :
1.  **Speech** : Voix de lecture dans plusieurs langues (utilisé pour simuler du bruit de type "brouhaha" ou "babble noise").
2.  **Music** : Morceaux de musique instrumentale et vocale de genres variés.
3.  **Noise** : Bruits environnementaux et techniques (bruit blanc, trafic, climatisation, tapotement de clavier). 
Dans notre pipeline, nous exploitons la catégorie **Noise** pour corrompre nos signaux de parole propre lors de l'entraînement et des tests de robustesse.

---

## 3. Extraction de Caractéristiques : Log-Mel Spectrogrammes

L'onde acoustique brute $x(t)$ subit une chaîne de traitement numérique pour être convertie en une représentation temps-fréquence adaptée aux convolutions 1D :

```
Signal Brut (16 kHz) ──> [Pre-emphasis] ──> [STFT (25ms / 10ms)] ──> [Mel Filterbank (80 bins)] ──> [Logarithme] ──> [CMS] ──> Log-Mel [80, T]
```

1.  **Pre-emphasis** (Pré-accentuation) : Optionnelle mais souvent intégrée pour accentuer les hautes fréquences contenant les informations de formants fins.
2.  **Transformée de Fourier à Court Terme (STFT)** :
    *   **Taille de fenêtre** : $400$ échantillons ($25\text{ ms}$ à $16\text{ kHz}$), permettant d'assurer la stationnarité du signal de parole sur chaque trame.
    *   **Pas (Hop length)** : $160$ échantillons ($10\text{ ms}$), assurant un recouvrement de $60\%$ pour éviter la perte d'informations aux frontières des fenêtres.
    *   **Fenêtrage** : Application d'une fenêtre de Hamming pour réduire le repliement spectral (spectral leakage).
3.  **Banc de Filtres Mel (Mel Filterbank)** :
    *   Application de **80 filtres triangulaires** espacés selon l'échelle perceptive de Mel, modélisant la perception fréquentielle non linéaire de l'oreille humaine (très précise dans les basses fréquences, moins sensible dans les hautes fréquences).
4.  **Logarithme et Normalisation (CMS)** :
    *   Le logarithme de l'énergie Mel est calculé : $\log(P_{\text{Mel}} + 10^{-6})$.
    *   **Cepstral Mean Subtraction (CMS)** : La moyenne de chaque canal fréquentiel sur l'ensemble des trames de l'audio est soustraite :
        $$\tilde{x}_c(t) = x_c(t) - \frac{1}{T}\sum_{\tau=1}^{T} x_c(\tau)$$
        Cette normalisation temporelle élimine les distorsions convolutives introduites par le canal de transmission ou le microphone (effet de filtre linéaire invariant), améliorant de manière critique la robustesse du modèle.

---

## 4. Stratégie d'Augmentation de Données et Formulation Mathématique du SNR

L'augmentation de données par mixage de bruit simule une corruption acoustique additive. Pour mélanger un signal propre $s(t)$ et un bruit de fond $n(t)$ à un rapport signal-sur-bruit (SNR) précis en décibels (dB), la méthodologie suivante est appliquée :

1.  **Calcul de la puissance du signal propre ($P_{s}$)** et du bruit ($P_{n}$) par valeur efficace (Root Mean Square - RMS) :
    $$P_{s} = \frac{1}{T}\sum_{t=1}^{T} s(t)^2 + \epsilon$$
    $$P_{n} = \frac{1}{T}\sum_{t=1}^{T} n(t)^2 + \epsilon$$
    *(où $\epsilon = 10^{-8}$ garantit la stabilité numérique).*

2.  **Détermination du facteur d'échelle du bruit ($\alpha$)** :
    Le SNR en dB est défini par la relation logarithmique :
    $$\text{SNR} = 10 \log_{10} \left( \frac{P_{s}}{\alpha^2 P_{n}} \right)$$
    En isolant le facteur d'atténuation ou d'amplification $\alpha$ du bruit, on obtient :
    $$\log_{10} \left( \frac{P_{s}}{\alpha^2 P_{n}} \right) = \frac{\text{SNR}}{10} \implies \frac{P_{s}}{\alpha^2 P_{n}} = 10^{\frac{\text{SNR}}{10}}$$
    $$\alpha^2 = \frac{P_{s}}{P_{n} \cdot 10^{\frac{\text{SNR}}{10}}} \implies \alpha = \sqrt{\frac{P_{s}}{P_{n} \cdot 10^{-\frac{\text{SNR}}{10}}}}$$

3.  **Mixage et écrêtage (Clipping Prevention)** :
    Le signal bruité mixte $y(t)$ est généré par sommation temporelle :
    $$y(t) = s(t) + \alpha \cdot n(t)$$
    Pour éviter que la somme des deux signaux ne dépasse la dynamique numérique autorisée (écrêtage provoquant des distorsions harmoniques sévères), le signal est renormalisé si son amplitude maximale absolue excède $1.0$ :
    $$\text{si } \max(|y(t)|) > 1.0, \quad y(t) = \frac{y(t)}{\max(|y(t)|)}$$

---

## 5. Protocole de Test et de Partitionnement

Pour valider scientifiquement les modèles, nous avons séparé les locuteurs de manière hermétique :

*   **Split Train/Val** : 
    Pour chaque locuteur possédant $\ge 4$ fichiers audio, $75\%$ de ses fichiers sont affectés à la base d'entraînement et $25\%$ à la base de validation. Cela garantit que le modèle apprend la variabilité intra-locuteur tout en disposant de données de test inédites pour ce même locuteur.
*   **Génération des Paires de Benchmark** :
    Pour l'évaluation, un ensemble fixe de **1 500 paires** est assemblé à partir des données de validation :
    *   **750 paires positives (même identité)** : Deux enregistrements différents du locuteur $A$.
    *   **750 paires négatives (identités différentes)** : Un enregistrement du locuteur $A$ comparé à un enregistrement du locuteur $B$.
    La graine aléatoire de partitionnement (`seed=8888`) assure que les paires évaluées sous stress de bruit sont rigoureusement identiques pour tous les modèles.

### Description des Niveaux de Stress Test au Bruit
Les 1 500 paires de test subissent des mixages de bruits MUSAN selon quatre niveaux de difficulté contrôlés :
1.  **Clean (Baseline)** : Signal original (EER minimal théorique).
2.  **SNR 20 dB (Bruit léger)** : Environnement calme. Modélise une utilisation en bureau fermé.
3.  **SNR 10 dB (Bruit modéré)** : Environnement de travail standard ou café en arrière-plan.
4.  **SNR 0 dB (Bruit sévère)** : La puissance acoustique du bruit est identique à celle de la parole. Modélise un appel téléphonique en extérieur venteux ou dans les transports en commun. Ce test mesure les limites absolues de la robustesse de l'extraction des caractéristiques vocales.
