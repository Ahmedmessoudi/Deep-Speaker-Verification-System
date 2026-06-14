# 📊 Chapitre 3 : Métriques d'Évaluation et Choix de Décision

Ce chapitre décrit les aspects théoriques et opérationnels de l'évaluation en vérification du locuteur. Il explique le compromis de seuillage et justifie la focalisation méthodologique sur le F1-Score pour le calibrage de nos modèles.

---

## 1. Formulation Mathématique Complète des Métriques

La vérification du locuteur est une tâche de classification binaire où le système compare deux voix et émet une hypothèse sur l'identité (identique ou différente).

### A. Taux d'Erreurs Biométriques (FAR et FRR)
L'évaluation repose sur la quantification des erreurs commises par le système.

1.  **False Alarm Rate (FAR - Taux de Fausse Acceptation)** :
    Le FAR est la probabilité que le système accepte par erreur une paire d'enregistrements appartenant à deux personnes différentes (imposteurs). Il est défini par :
    $$\text{FAR}(\theta) = P(\text{Score}(a, b) \ge \theta \mid \text{Locuteur}_a \neq \text{Locuteur}_b) = \frac{FP}{VN + FP}$$
    *Impact opérationnel* : Un FAR élevé compromet directement la sécurité du système en laissant passer des personnes non autorisées.

2.  **False Reject Rate (FRR - Taux de Faux Rejet)** :
    Le FRR est la probabilité que le système rejette par erreur une paire d'enregistrements appartenant au même locuteur légitime. Il est défini par :
    $$\text{FRR}(\theta) = P(\text{Score}(a, b) < \theta \mid \text{Locuteur}_a = \text{Locuteur}_b) = \frac{FN}{VP + FN}$$
    *Impact opérationnel* : Un FRR élevé nuit à l'utilisabilité du système (l'utilisateur doit répéter sa phrase plusieurs fois pour être authentifié).

### B. Equal Error Rate (EER) et point d'équilibre
L'EER (Taux d'Erreur Égal) représente le point de fonctionnement unique où les taux de fausse acceptation et de faux rejet sont identiques :
$$\text{EER} = \text{FAR}(\theta_{\text{EER}}) = \text{FRR}(\theta_{\text{EER}})$$
L'EER sert de mesure de performance globale intrinsèque car il ne dépend pas d'une préférence métier pour la sécurité ou le confort. C'est la métrique standard de comparaison dans la littérature académique.

### C. Métriques Classiques de Classification (Accuracy, Precision, Recall)
Ces métriques sont calculées pour un seuil opérationnel donné $\theta$ :

*   **Accuracy (Exactitude globale)** :
    $$\text{Accuracy}(\theta) = \frac{VP + VN}{VP + VN + FP + FN}$$
*   **Precision (Précision de la détection de l'utilisateur)** :
    $$\text{Precision}(\theta) = \frac{VP}{VP + FP}$$
*   **Recall (Rappel ou Taux de Vrais Positifs - TPR)** :
    $$\text{Recall}(\theta) = \frac{VP}{VP + FN} = 1 - \text{FRR}(\theta)$$

---

## 2. Intérêt de la Courbe DET (Detection Error Tradeoff)

La courbe **DET** est une représentation graphique spécifique à la biométrie (norme ISO/IEC 19795). Elle trace le taux de faux rejet (FRR) en fonction du taux de fausse alarme (FAR).

```
  Taux de Faux Rejet (FRR) %
   100 +--------------------------------------------------+
       | X-Vector                                         |
       |  \                                               |
    10 |   \                                              |
       |    \   ECAPA-TDNN                                |
     1 |     \__\                                         |
       |        \                                         |
   0.1 |                                                  |
       +--------------------------------------------------+
      0.1        1         10        100
           Taux de Fausse Alarme (FAR) %  [Échelle Logarithmique]
```

### Pourquoi la courbe DET est supérieure à la courbe ROC pour la VAL :
1.  **Échelle Logarithmique (Log-Log Scale)** : La courbe ROC classique (TPR en fonction du FPR) compresse la région la plus importante (les taux d'erreur très faibles, compris entre 0% et 1%) dans un espace microscopique près de l'origine ou du bord supérieur. La courbe DET utilise des axes à échelle logarithmique, ce qui étire visuellement cette zone critique.
2.  **Analyse des cas extrêmes de sécurité** : Dans un scénario bancaire, le concepteur du système exige un FAR extrêmement bas, par exemple $\text{FAR} \le 0{,}1\%$. La courbe DET permet de lire directement et avec précision quel sera le taux de faux rejet (FRR) correspondant à ce niveau de sécurité pour chaque modèle.
3.  **Comparaison visuelle claire** : Plus la courbe DET d'un modèle est proche de l'origine (en bas à gauche), plus le modèle est performant. Une courbe DET qui se situe entièrement sous une autre indique une supériorité absolue sur toute la plage de décision.

---

## 3. Justification Mathématique du Focus sur le F1-Score

Bien que l'EER soit indispensable pour comparer les modèles sur le plan théorique, il ne fournit pas de réponse à la question de mise en production : *« Quel seuil $\theta$ devons-nous appliquer pour trier nos utilisateurs ? »*. 

Nous avons focalisé notre protocole de validation et de sauvegarde des meilleurs checkpoints sur la maximisation du **F1-Score**. Ce choix repose sur plusieurs propriétés mathématiques majeures :

### A. Propriété de la Moyenne Harmonique
Le F1-Score est la moyenne harmonique de la Précision et du Rappel :
$$F_1(\theta) = \frac{2}{\frac{1}{\text{Precision}(\theta)} + \frac{1}{\text{Recall}(\theta)}} = 2 \cdot \frac{\text{Precision}(\theta) \cdot \text{Recall}(\theta)}{\text{Precision}(\theta) + \text{Recall}(\theta)}$$

Contrairement à la moyenne arithmétique simple, la moyenne harmonique pénalise de manière disproportionnée les déséquilibres extrêmes entre ses composantes :
*   Si le système a un Rappel de $100\%$ (il accepte absolument tout le monde) mais une Précision de $5\%$ (il accepte tous les imposteurs), la moyenne arithmétique serait de $52{,}5\%$, ce qui pourrait sembler acceptable.
*   La moyenne harmonique (F1-score) est de :
    $$F_1 = 2 \cdot \frac{0{,}05 \cdot 1{,}0}{0{,}05 + 1{,}0} = 9{,}5\%$$
    Le F1-score s'effondre, indiquant immédiatement que le modèle est inutilisable. Maximiser le F1-score garantit donc l'obtention d'un seuil équilibré.

### B. Calibrage Algorithmique par Recherche par Grille (Grid Search)
Pendant la phase de validation à la fin de chaque époque d'entraînement, le script exécute une recherche par grille pour calibrer le seuil optimal $\theta_{\text{opt}}$ :
1.  Le score de similarité cosinus est calculé pour les 1000 paires de validation.
2.  Le seuil $\theta$ varie sur 200 valeurs intermédiaires réparties uniformément dans l'intervalle $[-1{,}0 \,;\, +1{,}0]$.
3.  Pour chaque valeur de seuil, les prédictions sont générées et le F1-Score est calculé.
4.  Le seuil maximisant le F1-score est sélectionné :
    $$\theta_{\text{opt}} = \arg\max_{\theta \in [-1, 1]} F_1(\theta)$$
5.  Ce seuil $\theta_{\text{opt}}$ est sauvegardé avec les poids du modèle. Lors de la phase de test ou de déploiement, le modèle utilise ce seuil pour prendre ses décisions binaires en temps réel.

### C. Robustesse face au déséquilibre de classes réel
Dans une application réelle de contrôle d'accès, la proportion d'imposteurs qui tentent de forcer l'entrée est largement supérieure à celle de l'utilisateur légitime (scénario asymétrique).
*   L'Accuracy classique est biaisée par le VN (Vrais Négatifs) qui domine la formule. Un système stupide qui rejetterait tout le monde obtiendrait une Accuracy de $99\%$ si 99% des tentatives proviennent d'imposteurs.
*   Le F1-Score n'inclut pas le nombre de Vrais Négatifs ($VN$) dans sa formule. Il se focalise sur les Vrais Positifs ($VP$), les Faux Positifs ($FP$, fausses acceptations) et les Faux Négatifs ($FN$, faux rejets). Il est donc structurellement insensible à la prolifération de paires négatives d'imposteurs dans le flux de test, assurant la stabilité opérationnelle du seuil de décision au fil du temps.
