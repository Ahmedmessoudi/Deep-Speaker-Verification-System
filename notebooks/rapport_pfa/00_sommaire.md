# 🎤 Rapport de Projet de Fin d'Études (PFA) : Vérification du Locuteur

> **Projet** : Étude Comparative et Implémentation de Modèles de Vérification du Locuteur (X-Vector vs ECAPA-TDNN)
> **Environnement de développement** : PyTorch / Kaggle GPU
> **Auteur** : AHMED

---

## 📌 Sommaire Général

Le rapport est divisé en plusieurs chapitres détaillés, accessibles via les liens ci-dessous :

1. 📂 **[Chapitre 1 : Introduction et Méthodologie](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/01_introduction_et_methodologie.md)**
   * Contexte de la vérification biométrique vocale
   * Présentation des jeux de données (VoxCeleb1 et MUSAN)
   * Protocole de partitionnement (train/val/benchmark)
   * Stratégie d'augmentation et protocole de stress test de bruit (0dB, 10dB, 20dB)
2. 🧠 **[Chapitre 2 : Architectures des Modèles](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/02_architectures_modeles.md)**
   * Architecture X-Vector (Convolutions TDNN et Pooling Statistique)
   * Architecture ECAPA-TDNN (Blocs SE-Res2Net et Attentive Pooling)
   * Fonctions de perte : Cross-Entropy vs Additive Angular Margin Loss (AAM-Softmax)
   * Tableaux complets des couches et paramètres
3. 📊 **[Chapitre 3 : Métriques d'Évaluation](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/03_metriques_evaluation.md)**
   * Métriques clés : Accuracy, Precision, Recall, EER, ROC AUC
   * Justification approfondie de la focalisation sur le F1-Score pour le calibrage du seuil de décision
4. 📈 **[Chapitre 4 : Résultats et Analyse Comparative](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/04_resultats_et_analyse.md)**
   * Résultats chiffrés en conditions propres (Clean)
   * Robustesse au bruit sous différents niveaux de SNR (0 dB, 10 dB, 20 dB)
   * Analyse de l'efficacité matérielle (paramètres, taille, mémoire, latence)
   * Description et interprétation des graphiques d'évaluation (ROC, DET, distributions de similarité)
5. 🔮 **[Chapitre 5 : Conclusion et Perspectives](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/05_conclusion_et_perspectives.md)**
   * Synthèse des conclusions et désignation du meilleur modèle
   * Analyse technique des raisons du succès d'ECAPA-TDNN
   * Limites du projet et axes de recherche futurs (finetuning, bruits hors-domaine)
6. 📈 **[Chapitre 6 : Stratégie d'Entraînement — From Scratch vs Fine-Tuning](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/06_strategie_entrainement.md)**
   * Comparaison conceptuelle : From Scratch vs Fine-Tuning
   * Avantages, inconvénients et matrice de décision
   * Justification académique du choix d'entraînement *from scratch*
   * Guide technique détaillé pour le passage au fine-tuning en PyTorch
7. 💾 **[Chapitre 7 : Gestion de la Mémoire et Passage à l'Échelle](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/07_gestion_memoire_scalabilite.md)**
   * Problématique de volumétrie sur GPU Kaggle (44 Go de données)
   * Mécanisme de chargement à la demande (*Lazy Loading*)
   * Pré-chargement multi-processus (*Pre-Fetching Workers*)
   * Sécurisation de la VRAM (Nettoyage de cache, Caching Allocator)
   * Analyse détaillée de l'empreinte mémoire RAM/VRAM
8. 🛠️ **[Chapitre 8 : Analyse et Résolution des Faux Positifs en Conditions Réelles](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/rapport_pfa/08_resolution_faux_positifs.md)**
   * Problématique des Faux Positifs hors-dataset
   * Analyse des biais (canal, silences) et normalisation L2
   * Bloc de code d'inférence sécurisé (VAD, CMS, L2 Normalization)
   * Guide de calibrage pratique du seuil de décision ($\theta \approx 0.55$)

---

## 📝 Résumé du Projet (Abstract)

Ce projet de fin d'études porte sur la conception, l'implémentation *from scratch* et l'évaluation comparative de deux architectures de pointe en traitement de la parole pour la **vérification automatique du locuteur (VAL)** : le modèle classique **X-Vector** et le modèle état de l'art **ECAPA-TDNN**. 

En utilisant le jeu de données de référence **VoxCeleb1** (contenant de la parole en milieu sauvage pour plus de 1100 locuteurs) et le jeu de données **MUSAN** pour l'augmentation acoustique, nous avons développé une chaîne de traitement complète en PyTorch. Les modèles ont été entraînés sous des contraintes GPU spécifiques (Kaggle P100/T4) pendant 30 époques avec des mécanismes rigoureux de gestion de la mémoire.

Les résultats de nos benchmarks montrent la nette supériorité de l'architecture **ECAPA-TDNN**, qui obtient un **EER de 4,53%** (contre **14,40%** pour le X-Vector) et un **F1-score de 0,9537** en conditions de test optimales (Clean). Face au bruit, l'intégration de blocs de convolution multi-échelles (Res2Net), de l'attention sur les canaux (Squeeze-and-Excitation), de la mise en commun par attention (Attentive Statistics Pooling) et de la perte angulaire **AAM-Softmax** confère à ECAPA-TDNN une robustesse accrue. Même sous un bruit extrême à **0 dB de SNR**, le modèle ECAPA-TDNN préserve un F1-score de **0,8559** là où le modèle X-Vector chute à **0,7997**. 

Ce rapport présente de manière exhaustive les détails méthodologiques, structurels et expérimentaux de ces travaux.
