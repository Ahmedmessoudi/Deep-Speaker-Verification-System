# 📈 Chapitre 4 : Résultats Expérimentaux et Analyses

Ce chapitre détaille les résultats chiffrés des benchmarks, présente les mesures d'efficacité matérielle et fournit une interprétation approfondie des courbes d'évaluation générées.

---

## 1. Synthèse des Résultats Expérimentaux (Clean vs Noisy)

L'évaluation de la robustesse des modèles X-Vector et ECAPA-TDNN sous différents niveaux de bruit a donné les résultats suivants :

| Modèle Évalué | SNR (dB) | F1-Score | EER (%) | Accuracy (%) | Precision (%) | Recall (%) | Seuil Cosinus Calibré ($\theta_{\text{opt}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **X-Vector** | **Clean** | $0{,}8520$ | $14{,}40\%$ | $84{,}60\%$ | $82{,}00\%$ | $88{,}67\%$ | $+0{,}320$ |
| | **20 dB** | $0{,}8531$ | $14{,}40\%$ | $84{,}73\%$ | $82{,}21\%$ | $88{,}67\%$ | $+0{,}320$ |
| | **10 dB** | $0{,}8407$ | $16{,}13\%$ | $83{,}53\%$ | $81{,}17\%$ | $87{,}33\%$ | $+0{,}320$ |
| | **0 dB** | $0{,}7997$ | $20{,}80\%$ | $79{,}07\%$ | $76{,}94\%$ | $83{,}33\%$ | $+0{,}320$ |
| **ECAPA-TDNN**| **Clean** | $\mathbf{0{,}9537}$ | $\mathbf{4{,}53\%}$ | $\mathbf{95{,}40\%}$ | $\mathbf{96{,}08\%}$ | $\mathbf{94{,}67\%}$ | $\mathbf{+0{,}350}$ |
| | **20 dB** | $\mathbf{0{,}9502}$ | $\mathbf{5{,}20\%}$ | $\mathbf{95{,}07\%}$ | $\mathbf{95{,}69\%}$ | $\mathbf{94{,}40\%}$ | $\mathbf{+0{,}350}$ |
| | **10 dB** | $\mathbf{0{,}9382}$ | $\mathbf{5{,}87\%}$ | $\mathbf{93{,}87\%}$ | $\mathbf{94{,}65\%}$ | $\mathbf{93{,}07\%}$ | $\mathbf{+0{,}350}$ |
| | **0 dB** | $\mathbf{0{,}8559}$ | $\mathbf{10{,}67\%}$ | $\mathbf{85{,}80\%}$ | $\mathbf{87{,}22\%}$ | $\mathbf{84{,}00\%}$ | $\mathbf{+0{,}350}$ |

### Analyse Comparative de la Précision :
*   **Séparation des performances (Clean)** : ECAPA-TDNN surpasse nettement le X-Vector avec un gain de $10{,}17\%$ sur l'Accuracy globale et un EER divisé par plus de trois ($4{,}53\%$ vs $14{,}40\%$). La précision d'ECAPA ($96{,}08\%$) confirme sa grande sécurité biométrique.
*   **Résistance aux environnements dégradés** : 
    *   À 10 dB (bruit ambiant moyen), ECAPA-TDNN conserve d'excellentes performances (F1 de $0{,}9382$, EER de $5{,}87\%$), montrant que l'architecture tolère très bien le bruit de fond.
    *   À 0 dB (bruit extrême), le déclin est inévitable pour les deux modèles. Cependant, **ECAPA-TDNN préserve un F1-score de $0{,}8559$ et un EER de $10{,}67\%$**, ce qui reste supérieur aux performances maximales du X-Vector en conditions propres sans aucun bruit (Clean : F1 de $0{,}8520$, EER de $14{,}40\%$). Cela valide de façon éclatante la robustesse d'ECAPA-TDNN en conditions réelles difficiles.

---

## 2. Analyse Matérielle et Profiling d'Inférence

Le tableau suivant montre les caractéristiques computationnelles de chaque modèle :

| Indicateur Computationnel | X-Vector | ECAPA-TDNN | Différence / Impact |
| :--- | :---: | :---: | :--- |
| **Nombre de paramètres** | $\mathbf{5\,240\,911}$ | $6\,801\,915$ | ECAPA-TDNN est $30\%$ plus grand. |
| **Poids du checkpoint (.pt)** | $\mathbf{20{,}1\text{ Mo}}$ | $26{,}1\text{ Mo}$ | Différence mineure pour les architectures modernes. |
| **Latence moyenne d'inférence** | $112{,}5\text{ ms}$ | $\mathbf{6{,}5\text{ ms}}$ | **ECAPA-TDNN est 17 fois plus rapide**. |
| **Consommation VRAM Max** | $\mathbf{242\text{ Mo}}$ | $426\text{ Mo}$ | ECAPA-TDNN consomme $76\%$ de VRAM en plus. |

### Explications techniques des résultats d'efficacité :
1.  **L'impact de la VRAM** : La consommation mémoire d'ECAPA-TDNN ($426\text{ Mo}$ vs $242\text{ Mo}$) est induite par l'agrégation MFA (Multi-layer Feature Aggregation). Concaténer les cartes de caractéristiques intermédiaires de dimension 512 crée un tenseur de dimension 2048 qui doit être stocké en mémoire avant la convolution de projection. 
2.  **L'explication du goulot d'étranglement (Bottleneck) de la latence du X-Vector** :
    Bien que le modèle X-Vector possède moins de paramètres, sa latence moyenne par clip est très élevée ($112{,}5\text{ ms}$). L'analyse du code montre que cela est dû à la structure interne et au pipeline d'extraction des caractéristiques temporelles :
    *   L'architecture X-Vector utilise un traitement séquentiel trame par trame avec un contexte temporel plus rigide. L'implémentation de la boucle d'inférence dans le notebook applique des transformations Mel et Log-Mel séparées pour le X-Vector avec des étapes de synchronisation CPU-GPU supplémentaires, ce qui bloque le parallélisme.
    *   ECAPA-TDNN est codé sous forme de blocs convolutifs 1D hautement intégrés en PyTorch. Ces blocs profitent pleinement des optimisations matérielles CUDA de bas niveau (fusions d'opérations Conv-BatchNorm-Activation), permettant un passage ultra-rapide des tenseurs ($6{,}5\text{ ms}$ par clip).

---

## 3. Analyse et Interprétation des Graphiques d'Évaluation

Les graphiques exportés dans `results/plots/` fournissent des détails visuels majeurs sur le comportement des modèles :

### A. Graphiques individuels de performance (`xvector_evaluation_plots (1).png` et `ecapa_evaluation_plots (1).png`)
Chaque fichier contient 4 graphiques clés :
1.  **Classification Loss** : Affiche les pertes d'entropie croisée. La courbe montre une décroissance régulière sans surapprentissage (overfitting) sur les 30 époques d'apprentissage.
2.  **Verification Metrics per Epoch** : Illustre l'évolution du F1-Score et de l'EER au fil de l'apprentissage. La courbe confirme une corrélation forte : au fur et à mesure que la perte décroît, l'EER diminue et le F1-score s'améliore, se stabilisant à partir de la 20ème époque.
3.  **Histogramme de Distribution des Similarités** :
    *   *X-Vector* : Les scores des imposteurs (rouge) et des locuteurs identiques (vert) se chevauchent de manière importante dans la zone $[0{,}2 \,;\, 0{,}5]$. Le seuil optimal ($\theta = 0{,}32$) coupe ce chevauchement, générant $14{,}40\%$ d'erreurs (EER).
    *   *ECAPA-TDNN* : La distribution des scores de paires identiques (vert) est concentrée entre $0{,}6$ et $0{,}9$. La distribution des imposteurs (rouge) forme un pic très étroit centré sur $0{,}0$, ce qui démontre que la perte AAM-Softmax a rendu les caractéristiques de locuteurs différents presque orthogonales dans l'espace d'embedding. Les deux distributions sont séparées par une large vallée, permettant au seuil optimal ($\theta = 0{,}35$) de trier les locuteurs avec une erreur minime ($4{,}53\%$).
4.  **ROC Curve** : La courbe ROC d'ECAPA-TDNN colle presque parfaitement à l'angle supérieur gauche, affichant une aire sous la courbe (AUC) de $0{,}992$, ce qui confirme la séparabilité presque parfaite des locuteurs.

---

### B. Graphique Comparatif Général (`model_comparison_report.png`)

```
      Verification Accuracy (Haut-Gauche)               EER Comparison (Haut-Droite)
      [Barres comparatives des métriques]              [Barres comparatives des EER]
      ECAPA domine largement tous les axes             ECAPA réduit l'EER de 14.4% à 4.5%
      
      Combined ROC Curve (Bas-Gauche)                  DET Curves (Bas-Droite - Log-Log)
      [ROC de ECAPA vs X-Vector]                       [FAR vs FRR en coordonnées log]
      ECAPA montre un AUC bien supérieur               ECAPA est sous le X-Vector partout
```

Le quadrant le plus instructif pour l'évaluation biométrique de sécurité est celui des **courbes DET (Detection Error Tradeoff)** :
*   **Analyse du tracé** : Les axes logarithmiques révèlent que pour n'importe quel Taux de Fausse Acceptation (FAR) visé, le Taux de Faux Rejet (FRR) d'ECAPA-TDNN est inférieur à celui du X-Vector.
*   **Cas opérationnel** : Si un concepteur de système exige un niveau de sécurité élevé avec un taux de fausse alarme toléré de seulement $1\%$ ($\text{FAR} = 0{,}01$), la courbe DET montre que le modèle X-Vector subira un taux de faux rejet d'environ $30\%$ (rendant le système inutilisable en pratique car 1 utilisateur légitime sur 3 serait rejeté). Dans les mêmes conditions, ECAPA-TDNN ne subit qu'un taux de faux rejet d'environ $7\%$, prouvant sa viabilité opérationnelle.

---

### C. Analyse de la courbe de dégradation du F1-Score sous bruit (`noise_stress_degradation_comparison.png`)
Ce graphique montre l'évolution du F1-Score en fonction du niveau de bruit injecté (Clean $\rightarrow$ 20 dB $\rightarrow$ 10 dB $\rightarrow$ 0 dB) :

*   **Pente X-Vector** : Affiche une dégradation linéaire entre Clean et 10 dB, suivie d'une rupture de pente abrupte vers 0 dB ($0{,}8520 \rightarrow 0{,}7997$), indiquant une perte de repères fréquentiels critiques lorsque le bruit masque le signal utile.
*   **Pente ECAPA-TDNN** : La courbe reste presque plane entre Clean ($0{,}9537$), 20 dB ($0{,}9502$) et 10 dB ($0{,}9382$). Cela démontre l'efficacité du bloc Squeeze-and-Excitation (SE), qui parvient à identifier et isoler les canaux contenant la voix propre même en présence de bruit modéré. À 0 dB, la dégradation est visible ($0{,}8559$), mais le modèle conserve sa supériorité.
