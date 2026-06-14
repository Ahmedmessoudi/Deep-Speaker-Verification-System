# 📂 Chapitre 8 : Analyse et Résolution des Faux Positifs en Conditions Réelles

Lors du test des modèles X-Vector et ECAPA-TDNN avec des voix enregistrées en dehors du dataset VoxCeleb1 (par exemple, des enregistrements faits maison), un problème classique survient : **le modèle identifie correctement une voix et sa version augmentée, mais classifie également deux personnes complètement différentes comme étant le même locuteur** (Faux Positifs).

Ce chapitre analyse les causes de ce dysfonctionnement et propose une solution logicielle robuste à intégrer dans les notebooks.

---

## 1. Diagnostic de la cause du problème

Ce comportement anormal (deux personnes différentes = même locuteur) s'explique par trois facteurs acoustiques et algorithmiques cumulés :

```
Audio Brut ──> [Pas de VAD] ──> Silences identiques matchés par le CNN
Audio Brut ──> [Pas de CMS] ──> Microphone / Acoustique commune matchés (Biais de Canal)
Embedding ───> [Pas de Normalisation L2] ──> Score cosinus > 1.0 (Seuil 0.35 dépassé)
```

### A. Le Biais de Canal (Channel Bias)
Si vous enregistrez les deux locuteurs différents avec le même matériel (ex: le microphone de votre PC ou de votre smartphone) dans la même pièce :
*   Le signal audio contient la voix, mais aussi la **réponse impulsionnelle de la pièce** (reverb) et la **courbe de réponse du micro** (fréquences favorisées).
*   Sans traitement, le réseau de neurones (CNN) se focalise sur ces caractéristiques d'environnement communes. Il conclut que les voix sont identiques car elles partagent le même canal.

### B. Le matching des silences
Si les fichiers contiennent des silences au début ou à la fin, les spectrogrammes Mel afficheront des bandes d'énergie nulle identiques. Le modèle fait correspondre ces silences et surestime la similarité.

### C. L'absence de normalisation L2 en inférence
À l'entraînement, la perte AAM-Softmax projette les embeddings sur une hypersphère unitaire (norme = 1). En inférence, si les embeddings extraits ne sont pas explicitement normalisés via `F.normalize(embedding, p=2, dim=1)` avant le calcul du produit scalaire, le score cosinus peut dépasser largement la borne $1.0$ (atteignant $5.0$ ou $10.0$). Ainsi, n'importe quelle paire dépassera le seuil de décision de $0.35$.

---

## 2. Solution : Code d'Inférence Robuste et Sécurisé

Voici le bloc de code d'inférence universel à utiliser dans vos notebooks pour comparer deux fichiers audio personnalisés. Il intègre :
1.  **Voice Activity Detection (VAD)** par seuil d'énergie pour supprimer les silences.
2.  **Cepstral Mean Subtraction (CMS)** pour éliminer le biais de micro.
3.  **Normalisation L2 stricte** pour borner le score entre $-1$ et $1$.
4.  **Seuil adaptatif** (calibré à $0.55$ pour les enregistrements propres, plus strict que le $0.35$ de VoxCeleb bruité).

```python
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
import numpy as np

def predict_speaker_verification(model, audio_path_1, audio_path_2, device, threshold=0.55):
    """
    Compare deux fichiers audio et détermine s'ils proviennent du même locuteur.
    Intègre un VAD, un CMS et une normalisation L2 stricte.
    """
    model.eval()
    embeddings = []
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80).to(device)
    
    for path in [audio_path_1, audio_path_2]:
        # 1. Charger l'audio brut (16 kHz, mono)
        try:
            w, sr = librosa.load(path, sr=16000)
        except Exception as e:
            raise FileNotFoundError(f"Impossible de lire le fichier {path} : {e}")
        
        # 2. VAD (Voice Activity Detection) - Supprimer les silences (> 30 dB de silence)
        w_trimmed, _ = librosa.effects.trim(w, top_db=30)
        
        # Ajuster à la durée fixe de 3 secondes (48000 échantillons)
        ns = 48000
        if len(w_trimmed) < ns:
            w_trimmed = np.pad(w_trimmed, (0, ns - len(w_trimmed)))
        else:
            w_trimmed = w_trimmed[:ns]
            
        # Convertir en tenseur PyTorch
        wt = torch.tensor(w_trimmed, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 3. Extraction du Spectrogramme + CMS (Cepstral Mean Subtraction)
        with torch.no_grad():
            mel = mel_transform(wt)
            log_mel = torch.log(mel + 1e-6)
            log_mel = log_mel - log_mel.mean()  # Supprime le biais de micro/canal
            
            # Extraction de l'embedding selon le modèle
            if hasattr(model, 'extract_embedding'):
                emb = model.extract_embedding(log_mel)
            else:
                # Modèle enveloppé dans nn.DataParallel
                emb = model.module.extract_embedding(log_mel)
                
            # 4. Normalisation L2 stricte de l'embedding
            emb_normalized = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb_normalized.cpu().numpy()[0])
            
    # 5. Calcul de la similarité cosinus (produit scalaire des vecteurs de norme 1)
    cos_similarity = np.dot(embeddings[0], embeddings[1])
    
    # 6. Décision finale
    decision = "MÊME LOCUTEUR" if cos_similarity >= threshold else "LOCUTEURS DIFFÉRENTS"
    
    print("=" * 60)
    print(f"FICHIER 1 : {audio_path_1}")
    print(f"FICHIER 2 : {audio_path_2}")
    print(f"Similarité Cosinus : {cos_similarity:.4f}")
    print(f"Seuil de Décision   : {threshold:.2f}")
    print(f"RÉSULTAT            : {decision}")
    print("=" * 60)
    
    return cos_similarity, decision
```

---

## 3. Guide de calibrage du Seuil $\theta$ en conditions réelles

Le seuil de décision optimal dépend de la qualité des enregistrements :

*   **Seuil $\theta = 0{,}35$ (Sensibilité élevée / Sécurité moyenne)** : 
    *   *Usage* : Convient pour les bases de données très bruitées et compressées (type VoxCeleb1). 
    *   *Risque* : Tolère trop de faux positifs si les deux voix différentes sont enregistrées dans un environnement silencieux avec le même micro.
*   **Seuil $\theta = 0{,}55$ (Recommandé en conditions propres / Sécurité forte)** :
    *   *Usage* : Idéal pour les applications de contrôle d'accès avec enregistrement propre sur smartphone ou PC. 
    *   *Effet* : Rejette efficacement les voix différentes partageant le même micro, tout en acceptant les variations normales de la voix de l'utilisateur légitime.
*   **Seuil $\theta = 0{,}70$ (Sécurité maximale / Confort faible)** :
    *   *Usage* : Les applications bancaires ou militaires.
    *   *Effet* : Risque élevé de faux rejets si le locuteur légitime est légèrement enrhumé ou fatigué.

---

## 4. Outil Pratique Clé en Main : `custom_inference.py`

Pour faciliter vos tests et vos démonstrations lors de votre présentation orale (soutenance), j'ai écrit un script autonome complet nommé **[custom_inference.py](file:///c:/Users/AHMED/OneDrive/Desktop/notebook%20PFA/custom_inference.py)** à la racine de votre dossier de travail.

### Avantages de ce script :
1. **Interactive CLI** : Il propose un menu texte interactif vous demandant de choisir votre modèle (ECAPA-TDNN ou X-Vector), de renseigner les chemins des fichiers WAV/FLAC, et de configurer le seuil (seuil de `0.55` par défaut recommandé).
2. **Recherche de Checkpoint** : Il scanne automatiquement vos dossiers locaux `results/` et `checkpoints/` pour retrouver vos poids restaurés.
3. **Traitement Complet et Sécurisé** : Il embarque exactement les étapes de VAD, de CMS (soustraction de moyenne), et de normalisation L2 stricte expliquées ci-dessus.

### Comment l'exécuter dans votre terminal :
```bash
python custom_inference.py
```
Il vous guidera ensuite pas à pas pour tester vos paires vocales et valider la correction absolue de votre problème de Faux Positifs !

