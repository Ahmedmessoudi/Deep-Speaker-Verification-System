# 🎙️ Voice Biometrics Robustness Evaluation Platform

## Contexte

Je souhaite développer une plateforme complète de biométrie vocale permettant d'évaluer la robustesse de modèles de reconnaissance du locuteur (Speaker Verification) face à différents environnements sonores.

L'objectif est de permettre à un utilisateur de charger ou enregistrer sa voix, puis de générer automatiquement plusieurs versions bruitées à partir du dataset MUSAN afin d'évaluer les performances de deux modèles de reconnaissance vocale :

* ECAPA-TDNN
* X-Vector

L'application doit être moderne, professionnelle, modulaire et prête pour un déploiement en production.

---

# Stack Technique

## Frontend

* Streamlit
* Plotly
* Streamlit Components
* HTML/CSS personnalisés

## Backend

* FastAPI
* Uvicorn
* Pydantic
* SQLAlchemy
* Alembic

## Machine Learning

* PyTorch
* SpeechBrain
* Torchaudio
* Librosa
* NumPy
* Scikit-Learn

## Base de Données

* PostgreSQL

## DevOps

* Docker
* Docker Compose
* Nginx (optionnel)

---

# Architecture Attendue

```text
project/

├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── database/
│   │   ├── utils/
│   │   └── main.py
│   │
│   ├── models_weights/
│   │   ├── ecapa.pth
│   │   └── xvector.pth
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── pages/
│   ├── components/
│   ├── assets/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── database/
│
├── musan/
│
├── docker-compose.yml
│
└── README.md
```

---

# Fonctionnalités Principales

## 1. Upload Audio

Créer une page permettant :

* Upload WAV
* Upload MP3
* Upload FLAC

ou

* Enregistrement direct via microphone

Afficher :

* Nom du fichier
* Taille
* Durée
* Sample Rate

---

## 2. Prétraitement Audio

Créer un pipeline automatique :

### Étapes

* Conversion vers WAV
* Conversion vers 16 kHz
* Normalisation
* Détection du silence
* Suppression optionnelle des silences

Retourner :

```json
{
  "duration": 8.4,
  "sample_rate": 16000,
  "energy": 0.87
}
```

---

## 3. Génération Automatique de Bruit (MUSAN)

Utiliser le dataset MUSAN.

Créer automatiquement plusieurs versions :

### Types de bruit

* Café
* Rue
* Bureau
* Musique

### Niveaux de bruit

* 20 dB
* 15 dB
* 10 dB
* 5 dB

Exemples :

```text
original.wav

cafe_20.wav
street_20.wav
office_20.wav
music_20.wav

cafe_15.wav
street_15.wav
...
```

Créer un service dédié :

```python
AudioAugmentationService
```

---

## 4. Visualisation Audio

Créer une page dédiée.

Afficher pour chaque audio :

### Waveform

* Signal temporel

### Spectrogram

* STFT

### Mel Spectrogram

* Échelle Mel

Utiliser Plotly pour toutes les visualisations.

Permettre la comparaison :

```text
Original VS Street Noise
Original VS Cafe Noise
Original VS Music Noise
```

---

## 5. Audio Gallery

Créer une galerie affichant :

### Original

* Lecteur audio
* Téléchargement

### Café

* Lecteur audio
* Téléchargement

### Rue

* Lecteur audio
* Téléchargement

### Bureau

* Lecteur audio
* Téléchargement

### Musique

* Lecteur audio
* Téléchargement

---

## 6. Speaker Verification

Deux modèles sont déjà entraînés :

```text
ECAPA-TDNN
X-Vector
```

Créer un service :

```python
SpeakerVerificationService
```

Fonctionnalités :

### Chargement des modèles

Au démarrage de FastAPI.

### Extraction d'embeddings

Pour chaque audio.

### Similarité Cosinus

Comparer :

```text
Original
VS
Version bruitée
```

Retourner :

```json
{
  "model": "ECAPA",
  "similarity": 0.94
}
```

---

## 7. Évaluation de Robustesse

Comparer :

* Original
* Café
* Rue
* Bureau
* Musique

Pour chaque niveau SNR.

Calculer :

### ECAPA

* Score moyen
* Score max
* Score min

### X-Vector

* Score moyen
* Score max
* Score min

---

## 8. Dashboard Analytics

Créer une page Dashboard.

Afficher :

### Comparaison des modèles

Graphique :

```text
ECAPA vs X-Vector
```

### Impact du bruit

Graphique :

```text
Noise Type vs Similarity Score
```

### Impact du SNR

Graphique :

```text
SNR vs Performance
```

### Historique des tests

Graphique temporel.

Utiliser Plotly.

---

## 9. Voice Quality Score

Créer un score global :

```text
0 → 100
```

Basé sur :

### Durée

20 %

### Énergie

20 %

### SNR

20 %

### Robustesse ECAPA

20 %

### Robustesse X-Vector

20 %

Retour :

```json
{
  "voice_quality_score": 88
}
```

Afficher :

| Score  | Qualité   |
| ------ | --------- |
| 90-100 | Excellent |
| 70-89  | Good      |
| 50-69  | Medium    |
| 0-49   | Poor      |

---

## 10. Génération de Rapport

Créer un endpoint :

```http
POST /generate-report
```

Retour :

```json
{
  "voice_quality_score": 88,
  "best_model": "ECAPA",
  "average_ecapa_score": 95,
  "average_xvector_score": 87,
  "generated_samples": 16
}
```

Générer également :

* PDF
* JSON

---

## Base de Données

Créer les tables :

### users

### audio_files

### generated_samples

### verification_results

### reports

Utiliser :

* SQLAlchemy
* Alembic

---

## API REST

Créer les endpoints :

```http
POST /upload

POST /augment

POST /verify

POST /evaluate

POST /generate-report

GET /results

GET /health
```

Documenter automatiquement via Swagger.

---

## Docker

Créer :

### Backend Dockerfile

### Frontend Dockerfile

### Docker Compose

Services :

```yaml
frontend
backend
postgres
```

Volumes persistants obligatoires.

---

## Qualité Logicielle

Le projet doit respecter :

* Architecture propre
* SOLID
* Typage Python
* Logging
* Gestion des exceptions
* Variables d'environnement
* Fichiers .env
* Tests unitaires

---

# Livrables Attendus

Générer automatiquement :

1. Structure complète du projet
2. Backend FastAPI
3. Frontend Streamlit
4. Modèles SQLAlchemy
5. Dockerfiles
6. Docker Compose
7. README.md complet
8. Scripts d'installation
9. Scripts de lancement
10. Exemple de données de test

Le code doit être directement exécutable et prêt pour un déploiement local ou cloud.

---

## Déploiement (Docker Compose)

Instructions rapides pour déployer localement avec Docker Compose :

1. Construire et démarrer les services :

```bash
docker compose up --build -d
```

2. Services exposés :

- API FastAPI : http://localhost:8000 (endpoint `/health`, `/verify`, `/embed`, ...)
- Frontend Streamlit : http://localhost:8501
- PostgreSQL : 5432 (utilisateur : `sv_user`, mot de passe : `sv_pass`, db : `speaker_verification`)

3. Variables d'environnement importantes (exemple `.env`):

```env
API_MODEL_TYPE=ecapa_tdnn
POSTGRES_USER=sv_user
POSTGRES_PASSWORD=sv_pass
POSTGRES_DB=speaker_verification
```

4. Vérifications après démarrage :

```bash
# Vérifier que les conteneurs tournent
docker compose ps

# Vérifier santé API
curl http://localhost:8000/health

# Accéder à l'interface Streamlit
open http://localhost:8501
```

5. Déploiement cloud (synthèse) :

- Construire et pousser les images vers un registre (`ghcr.io`, `docker.io`, `aws ECR`).
- Utiliser un orchestrateur (Docker Swarm, Kubernetes) pour production.
- Prévoir un reverse-proxy (Nginx) et certificats TLS (Let's Encrypt).
- Utiliser variables d'environnement secrètes et un stockage persistant pour Postgres.

6. Migrations DB et initialisation

- Si vous utilisez SQLAlchemy + Alembic, exécuter les migrations au démarrage ou via job CI/CD :

```bash
docker compose run --rm api alembic upgrade head
```

7. Astuces et sécurité

- Ne mappez pas les volumes de modèles sensibles vers un dépôt public.
- Activez un mécanisme d'authentification pour l'API en production.
- Surveillez l'utilisation GPU / CPU selon l'hébergement.

---

Si vous voulez, je peux :

- générer des fichiers `alembic` et des modèles SQLAlchemy de base;
- ajouter des variables d'environnement dans un `.env.example`;
- construire et tester les images Docker localement (si vous me l'autorisez).

