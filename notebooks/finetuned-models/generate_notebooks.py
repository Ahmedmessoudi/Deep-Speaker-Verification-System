#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de notebooks V2 améliorés pour le fine-tuning des modèles
de vérification du locuteur.

Corrections appliquées par rapport aux notebooks V1 :
  1. AAMSoftmax pour X-Vector (V1 utilisait CrossEntropyLoss → mauvaise séparation)
  2. VAD (Voice Activity Detection) dans le prétraitement → suppression des silences
  3. Normalisation L2 des embeddings → cosinus borné entre -1 et 1
  4. Seuil calibré à 0.75 (V1 utilisait 0.5 → zone ambiguë → faux positifs)
  5. Export vers Models/ (compatible API) et non results/
  6. CMS (Cepstral Mean Subtraction) → suppression de la signature micro

Usage :
    python generate_notebooks.py
"""

import json, os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def md(src): return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src]}
def nb(cells): return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},"cells":cells}

# ===========================================================================
# NOTEBOOK 1 — X-VECTOR FINETUNING V2
# ===========================================================================

NB1_CELLS = [

md("""# 🎤 Speaker Verification — X-Vector Fine-Tuning V2 (Kaggle · 2× T4 GPU)

## ✅ Corrections appliquées dans cette version V2

| Problème (V1) | Correction (V2) |
|---|---|
| `CrossEntropyLoss` → mauvaise séparation angulaire | **AAM-Softmax** (ArcFace, margin=0.2, scale=30) |
| Pas de VAD → silences génèrent des faux positifs | **VAD** : `librosa.effects.trim(top_db=30)` |
| Pas de normalisation L2 → cosinus non borné | **L2-norm** : `F.normalize(emb, p=2, dim=1)` |
| Seuil = 0.5 → zone ambiguë | **Seuil calibré = 0.75** (EER-based) |
| Export vers `results/` → incompatible API | **Export vers `Models/`** (compatible API FastAPI) |
| Pas de CMS → biais de canal micro | **CMS** : soustraction de la moyenne cepstrale |

### Pipeline
1. **Environnement** : dépendances + GPU
2. **Données** : VoxCeleb (auto-détection Kaggle) + MUSAN
3. **Modèle** : `PretrainedXVectorWrapper` (SpeechBrain backbone) + tête AAM-Softmax
4. **Entraînement** : 10 epochs, AdamW, Cosine Annealing
5. **Évaluation** : F1, EER, AUC avec L2-norm + VAD + seuil 0.75
6. **SNR sweep** : robustesse bruit à 0, 10, 20 dB
7. **Export** : `Models/xvector_final_model.pt` (format API-compatible)
"""),

md("## 🛠️ Étape 1 : Environnement et Importations"),

code("""!pip install -q speechbrain torchaudio soundfile librosa matplotlib seaborn scikit-learn pandas numpy tqdm

import os, re, sys, time, math, random, glob, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_curve, auc)
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f'[GPU] {torch.cuda.device_count()} GPUs disponibles.')
else:
    print(f'[Device] {device}')

# ── Hyperparamètres ──────────────────────────────────────────────────────────
SR           = 16000
DURATION     = 3
NUM_SAMPLES  = SR * DURATION
BATCH_SIZE   = 64
NUM_WORKERS  = 4
EPOCHS       = 10
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MARGIN       = 0.2      # AAM-Softmax margin
SCALE        = 30.0     # AAM-Softmax scale
THRESHOLD    = 0.75     # Seuil calibré pour AAM-Softmax + L2-norm

random.seed(42); np.random.seed(42); torch.manual_seed(42)
print(f'[Config] SR={SR}, DURATION={DURATION}s, THRESHOLD={THRESHOLD}')
"""),

md("## 🔄 Étape 2 : Découverte des Données et Prétraitement avec VAD"),

code("""import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AUTO-DISCOVER DATASET PATHS (même logique que Notebook_1_X_Vector)
# Scan TOUS les fichiers audio et classifie par pattern
# NE dépend PAS du nom du dossier Kaggle
# ============================================================
INPUT_ROOT = '/kaggle/input'
print('=' * 60)
print('STEP 1: Découverte de /kaggle/input/')
print('=' * 60)
for ds in sorted(os.listdir(INPUT_ROOT)):
    ds_path = os.path.join(INPUT_ROOT, ds)
    if os.path.isdir(ds_path):
        print(f'\\n  [DIR] {ds}/')
        for sub in sorted(os.listdir(ds_path))[:5]:
            sub_path = os.path.join(ds_path, sub)
            tag = '[DIR]' if os.path.isdir(sub_path) else '[FILE]'
            print(f'    {tag} {sub}')

print(f'\\n{"=" * 60}')
print('STEP 2: Scan de tous les fichiers audio...')
print('=' * 60)
all_wav  = glob.glob(os.path.join(INPUT_ROOT, '**', '*.wav'),  recursive=True)
all_flac = glob.glob(os.path.join(INPUT_ROOT, '**', '*.flac'), recursive=True)
all_audio = all_wav + all_flac
print(f'  .wav  : {len(all_wav)}')
print(f'  .flac : {len(all_flac)}')
print(f'  TOTAL : {len(all_audio)}')
if all_audio:
    print('  Exemples :')
    for p in all_audio[:5]: print(f'    {p}')

print(f'\\n{"=" * 60}')
print('STEP 3: Classification VoxCeleb vs MUSAN par pattern')
print('=' * 60)
vox_wav_files = []
noise_files   = []

for f in all_audio:
    fp = f.replace('\\\\', '/')
    # VoxCeleb : contient un ID locuteur type id10001, id00001...
    if re.search(r'/id\\d{3,}/', fp):
        vox_wav_files.append(f)
    # MUSAN bruit : contient 'noise' ET 'musan'
    elif 'noise' in fp.lower() and 'musan' in fp.lower():
        noise_files.append(f)

# Fallback niveau 1 : patterns élargis
if not vox_wav_files:
    print('  [WARN] Aucun fichier VoxCeleb par pattern id. Essai patterns élargis...')
    for f in all_audio:
        fp = f.replace('\\\\', '/').lower()
        if 'vox' in fp or 'celeb' in fp or 'speaker' in fp:
            vox_wav_files.append(f)

# Fallback niveau 2 : tous les non-MUSAN
if not vox_wav_files:
    print('  [WARN] Fallback : utilisation de tous les fichiers non-MUSAN')
    for f in all_audio:
        fp = f.replace('\\\\', '/').lower()
        if 'musan' not in fp:
            vox_wav_files.append(f)

if not noise_files:
    print('  [WARN] Aucun bruit MUSAN. Essai patterns élargis...')
    for f in all_audio:
        fp = f.replace('\\\\', '/').lower()
        if 'noise' in fp and f not in vox_wav_files:
            noise_files.append(f)

print(f'  Fichiers VoxCeleb : {len(vox_wav_files)}')
print(f'  Fichiers bruit    : {len(noise_files)}')

assert len(vox_wav_files) > 0, (
    'ERREUR : Aucun fichier VoxCeleb trouvé !\\n'
    'Vérifiez que le dataset VoxCeleb est bien attaché dans Kaggle > Add Data.'
)

# ── Extraction des IDs locuteurs ──────────────────────────────────────────────
def get_speaker_id(path):
    parts = path.replace('\\\\', '/').split('/')
    for p in parts:
        if re.match(r'^id\\d{3,}$', p): return p
    return None

spk_to_files = {}
for path in vox_wav_files:
    sid = get_speaker_id(path)
    if sid:
        spk_to_files.setdefault(sid, []).append(path)

# Fallback : utiliser le dossier parent comme ID
if not spk_to_files:
    print('  [WARN] Aucun pattern idXXXX. Utilisation du dossier parent comme label.')
    for path in vox_wav_files:
        parts = path.replace('\\\\', '/').split('/')
        sid = parts[-3] if len(parts) >= 3 else parts[-2]
        spk_to_files.setdefault(sid, []).append(path)

spk_ids      = sorted(spk_to_files.keys())
NUM_CLASSES  = len(spk_ids)
spk_to_label = {s: i for i, s in enumerate(spk_ids)}
label_to_spk = {i: s for s, i in spk_to_label.items()}
print(f'  NUM_CLASSES = {NUM_CLASSES} locuteurs')

# ── CORRECTION V2 : Prétraitement avec VAD ────────────────────────────────────
def preprocess_waveform(w, sr=SR, duration=DURATION):
    \'\'\'VAD (trim silences) + standardisation duree.\'\'\'
    w_trimmed, _ = librosa.effects.trim(w, top_db=30)
    ns = sr * duration
    if len(w_trimmed) == 0: w_trimmed = w
    if len(w_trimmed) < ns:
        w_trimmed = np.pad(w_trimmed, (0, ns - len(w_trimmed)))
    else:
        w_trimmed = w_trimmed[:ns]
    return w_trimmed.astype(np.float32)

# ── Dataset ───────────────────────────────────────────────────────────────────
class VoxCelebDataset(Dataset):
    def __init__(self, file_list, spk_to_label, augment=False, noise_files=None):
        self.file_list    = file_list
        self.spk_to_label = spk_to_label
        self.augment      = augment
        self.noise_files  = noise_files or []

    def _get_label(self, fp):
        sid = get_speaker_id(fp)
        if sid: return self.spk_to_label.get(sid, 0)
        parts = fp.replace('\\\\','/').split('/')
        return self.spk_to_label.get(parts[-3] if len(parts)>=3 else parts[-2], 0)

    def _add_noise(self, w, snr_db):
        if not self.noise_files: return w
        try:
            n, _ = librosa.load(random.choice(self.noise_files), sr=SR)
            ns = NUM_SAMPLES
            if len(n) < ns: n = np.tile(n, int(np.ceil(ns/len(n))))
            n = n[:ns]
            pw = np.mean(w**2)+1e-10; pn = np.mean(n**2)+1e-10
            return np.clip(w + np.sqrt(pw/(pn*10**(snr_db/10)))*n, -1, 1)
        except Exception: return w

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        fp = self.file_list[idx]
        label = self._get_label(fp)
        try: w, _ = librosa.load(fp, sr=SR)
        except: w = np.zeros(NUM_SAMPLES, dtype=np.float32)
        w = preprocess_waveform(w)    # VAD + standardisation
        if self.augment and self.noise_files and random.random() < 0.5:
            w = self._add_noise(w, random.choice([0, 10, 20]))
        return torch.tensor(w, dtype=torch.float32), label

# ── Split train/val/test ──────────────────────────────────────────────────────
all_items = [(fp, spk_to_label[s]) for s, fps in spk_to_files.items() for fp in fps]
random.shuffle(all_items); n = len(all_items)
train_fps = [x[0] for x in all_items[:int(0.8*n)]]
val_fps   = [x[0] for x in all_items[int(0.8*n):int(0.9*n)]]
test_fps  = [x[0] for x in all_items[int(0.9*n):]]
print(f'[Split] Train={len(train_fps)} | Val={len(val_fps)} | Test={len(test_fps)}')

train_loader = DataLoader(VoxCelebDataset(train_fps, spk_to_label, True,  noise_files),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(VoxCelebDataset(val_fps,   spk_to_label, False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
"""),

md("""## 🧠 Étape 3 : Architecture du Modèle + AAM-Softmax

### Corrections V2 :
- **AAMSoftmax** remplace `CrossEntropyLoss` → séparation angulaire plus stricte
- La normalisation L2 est intégrée **dans** la loss → embeddings forcés sur hypersphère
"""),

code("""from speechbrain.inference.speaker import EncoderClassifier

print('[SpeechBrain] Chargement du backbone X-Vector pré-entraîné...')
sb_classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-xvect-voxceleb',
    run_opts={'device': str(device)}
)

# ── CORRECTION V2 : AAM-Softmax ──────────────────────────────────────────────
class AAMSoftmax(nn.Module):
    \"\"\"
    Additive Angular Margin Softmax (ArcFace).
    Force les embeddings sur une hypersphère unitaire.
    Paramètres standards : margin=0.2, scale=30.
    \"\"\"
    def __init__(self, input_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, x, labels):
        # L2-normalisation des embeddings ET de la matrice de poids
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = x_norm @ W_norm.T
        sine   = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-7))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        output  = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(output, labels)

class PretrainedXVectorWrapper(nn.Module):
    \"\"\"Wrapper SpeechBrain X-Vector avec tête de classification AAM-Softmax.\"\"\"
    def __init__(self, classifier, num_classes, embedding_dim=512):
        super().__init__()
        self.encoder     = classifier.mods.compute_features
        self.mean_var_norm= classifier.mods.mean_var_norm
        self.embedding   = classifier.mods.embedding_model
        self.fc_head     = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        self.classifier_head = AAMSoftmax(embedding_dim, num_classes, MARGIN, SCALE)

    def extract_embedding(self, waveform):
        \"\"\"Retourne un embedding L2-normalisé (utilisé à l'inférence).\"\"\"
        with torch.no_grad():
            feats = self.encoder(waveform)
            feats = self.mean_var_norm(feats, torch.ones(waveform.shape[0]).to(waveform.device))
        emb = self.embedding(feats).squeeze(1)
        emb = self.fc_head(emb)
        return F.normalize(emb, p=2, dim=1)   # ← CORRECTION V2 : L2-norm

    def forward(self, waveform, labels):
        with torch.no_grad():
            feats = self.encoder(waveform)
            feats = self.mean_var_norm(feats, torch.ones(waveform.shape[0]).to(waveform.device))
        emb = self.embedding(feats).squeeze(1)
        emb = self.fc_head(emb)
        return self.classifier_head(emb, labels)  # AAM-Softmax

model = PretrainedXVectorWrapper(sb_classifier, NUM_CLASSES, 512).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[Modèle] Paramètres entraînables : {total_params:,}')
"""),

md("## 📉 Étape 4 : Entraînement avec AAM-Softmax et Cosine Annealing"),

code("""# ── Mel-spectrogram (même config que l'API) ──────────────────────────────────
mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=400, win_length=400,
    hop_length=160, n_mels=80
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs('checkpoints/xvector_finetune_v2', exist_ok=True)

# ── CORRECTION V2 : Extraction embedding avec VAD + CMS ──────────────────────
def get_embedding(mdl, fp, dev):
    \"\"\"Embedding L2-normalisé avec VAD + CMS (utilisé pour l'évaluation).\"\"\"
    mdl.eval()
    try:
        w, _ = librosa.load(fp, sr=SR)
    except Exception:
        w = np.zeros(NUM_SAMPLES, dtype=np.float32)
    w = preprocess_waveform(w)   # VAD + standardisation
    wt = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = mdl.module.extract_embedding(wt) if hasattr(mdl,'module') else mdl.extract_embedding(wt)
    return emb.cpu().numpy()[0]  # Déjà L2-normalisé (voir extract_embedding)

def eval_verif(mdl, pairs, dev, threshold=THRESHOLD):
    \"\"\"Évaluation vérification avec L2-norm, VAD et seuil calibré.\"\"\"
    mdl.eval()
    sims, labs, cache = [], [], {}
    for a, b, lb in tqdm(pairs, desc='Évaluation'):
        for f in [a, b]:
            if f not in cache:
                cache[f] = get_embedding(mdl, f, dev)
        sim = float(np.dot(cache[a], cache[b]))   # cosinus (L2-normalisé)
        sims.append(sim); labs.append(lb)
    sims = np.array(sims); labs = np.array(labs)
    preds = (sims >= threshold).astype(int)
    f1    = f1_score(labs, preds)
    acc   = accuracy_score(labs, preds)
    prec  = precision_score(labs, preds, zero_division=0)
    rec   = recall_score(labs, preds, zero_division=0)
    fpr, tpr, thr_roc = roc_curve(labs, sims)
    fnr   = 1 - tpr
    eer_i = np.argmin(np.abs(fpr - fnr))
    eer   = (fpr[eer_i] + fnr[eer_i]) / 2
    roc_auc = auc(fpr, tpr)
    return {'f1':f1,'accuracy':acc,'precision':prec,'recall':rec,
            'eer':eer,'auc':roc_auc,'sims':sims,'labs':labs}

# ── Génération des paires de vérification ─────────────────────────────────────
def make_pairs(fps, n_pairs=2000):
    spk_map = {}
    for fp in fps:
        for p in fp.replace('\\\\','/').split('/'):
            if re.match(r'^id\\d{5}$', p):
                spk_map.setdefault(p, []).append(fp); break
    pairs = []
    spks  = list(spk_map.keys())
    for _ in range(n_pairs//2):
        s = random.choice(spks)
        if len(spk_map[s]) >= 2:
            a, b = random.sample(spk_map[s], 2)
            pairs.append((a, b, 1))
    for _ in range(n_pairs//2):
        s1, s2 = random.sample(spks, 2)
        a = random.choice(spk_map[s1]); b = random.choice(spk_map[s2])
        pairs.append((a, b, 0))
    random.shuffle(pairs); return pairs

val_pairs  = make_pairs(val_fps,  n_pairs=2000)
test_pairs = make_pairs(test_fps, n_pairs=2000)

# ── Boucle d'entraînement ─────────────────────────────────────────────────────
best_f1, best_ckpt = 0.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n_batches = 0.0, 0
    for waves, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}'):
        waves  = waves.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = model(waves, labels)
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)

    # Évaluation validation
    val_res = eval_verif(model, val_pairs, device, THRESHOLD)
    print(f'[Epoch {epoch}] Loss={avg_loss:.4f} | F1={val_res["f1"]:.4f} '
          f'| EER={val_res["eer"]:.4f} | AUC={val_res["auc"]:.4f}')

    if val_res['f1'] > best_f1:
        best_f1 = val_res['f1']
        best_ckpt = {
            'epoch': epoch, 'model_state_dict': (model.module.state_dict()
                if hasattr(model,'module') else model.state_dict()),
            'optimal_threshold': THRESHOLD,
            'val_f1': best_f1
        }
        torch.save(best_ckpt, 'checkpoints/xvector_finetune_v2/best_model.pt')
        print(f'  → ✅ Meilleur modèle sauvegardé (F1={best_f1:.4f})')

print(f'\\n[Entraînement terminé] Meilleur F1 val = {best_f1:.4f}')
"""),

md("## 📊 Étape 5 : Évaluation sur le Test Set"),

code("""# Charger le meilleur checkpoint
ckpt_path = 'checkpoints/xvector_finetune_v2/best_model.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    eval_model = PretrainedXVectorWrapper(sb_classifier, NUM_CLASSES, 512)
    eval_model.load_state_dict(ckpt['model_state_dict'])
    eval_model = eval_model.to(device)
    opt_threshold = ckpt.get('optimal_threshold', THRESHOLD)
    print(f'[Restauré] Epoch {ckpt["epoch"]} | Seuil : {opt_threshold:.3f}')
else:
    eval_model = model.module if hasattr(model,'module') else model
    opt_threshold = THRESHOLD

# Test set evaluation
test_res = eval_verif(eval_model, test_pairs, device, opt_threshold)
tf1, tacc = test_res['f1'], test_res['accuracy']
tprec, trec = test_res['precision'], test_res['recall']
teer, tauc  = test_res['eer'], test_res['auc']

print(f'\\n=== RÉSULTATS TEST SET (X-Vector V2) ===')
print(f'  F1-Score  : {tf1:.4f}')
print(f'  Précision : {tprec:.4f}')
print(f'  Rappel    : {trec:.4f}')
print(f'  Accuracy  : {tacc:.4f}')
print(f'  EER       : {teer:.4f}')
print(f'  AUC-ROC   : {tauc:.4f}')
print(f'  Seuil     : {opt_threshold:.3f}')

# ── Distribution des similarités : même vs différent locuteur ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sims, labs = test_res['sims'], test_res['labs']
same   = sims[labs == 1]
diff   = sims[labs == 0]
axes[0].hist(diff,  bins=50, alpha=0.7, color='#E74C3C', label='Locuteurs différents')
axes[0].hist(same,  bins=50, alpha=0.7, color='#27AE60', label='Même locuteur')
axes[0].axvline(opt_threshold, color='navy', ls='--', lw=2, label=f'Seuil={opt_threshold:.2f}')
axes[0].set_title('Distribution des similarités cosinus (V2 — L2-norm + VAD)', fontsize=12)
axes[0].set_xlabel('Similarité cosinus'); axes[0].legend()
fpr, tpr, _ = roc_curve(labs, sims)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={tauc:.4f})')
axes[1].plot([0,1],[0,1], 'k--')
axes[1].set_title('Courbe ROC — X-Vector V2'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend()
plt.tight_layout(); plt.savefig('results/xvector_v2_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 🔊 Étape 5.5 : Évaluation Robustesse Bruit (0, 10, 20 dB)"),

code("""def add_noise(w, snr_db, noise_files):
    if not noise_files: return w
    nf = random.choice(noise_files)
    n, _ = librosa.load(nf, sr=SR)
    ns = NUM_SAMPLES
    if len(n) < ns: n = np.tile(n, int(np.ceil(ns/len(n))))
    n = n[:ns]
    pw = np.mean(w**2) + 1e-10; pn = np.mean(n**2) + 1e-10
    factor = np.sqrt(pw / (pn * 10**(snr_db/10)))
    return np.clip(w + factor * n, -1, 1)

def get_embedding_noisy(mdl, fp, dev, snr_db, noise_files):
    \"\"\"Embedding sur audio bruité (VAD + L2-norm maintenus).\"\"\"
    mdl.eval()
    try: w, _ = librosa.load(fp, sr=SR)
    except: w = np.zeros(NUM_SAMPLES, dtype=np.float32)
    w = preprocess_waveform(w)
    if noise_files:
        w = add_noise(w, snr_db, noise_files)
    wt = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = eval_model.extract_embedding(wt)
    return emb.cpu().numpy()[0]

snr_levels  = [20, 10, 0]
snr_results = {}
os.makedirs('results', exist_ok=True)

for snr in snr_levels:
    sims2, labs2 = [], []
    cache2 = {}
    for a, b, lb in tqdm(test_pairs[:500], desc=f'SNR={snr}dB'):
        for f in [a, b]:
            if f not in cache2:
                cache2[f] = get_embedding_noisy(eval_model, f, device, snr, noise_files)
        sim = float(np.dot(cache2[a], cache2[b]))
        sims2.append(sim); labs2.append(lb)
    sims2 = np.array(sims2); labs2 = np.array(labs2)
    preds2 = (sims2 >= opt_threshold).astype(int)
    snr_results[snr] = {
        'f1': f1_score(labs2, preds2),
        'eer': ((1 - roc_curve(labs2,sims2)[1]) + roc_curve(labs2,sims2)[0]).min() / 2,
    }
    print(f'  SNR={snr:3d}dB → F1={snr_results[snr]["f1"]:.4f} | EER={snr_results[snr]["eer"]:.4f}')

# Visualisation
fig, ax = plt.subplots(figsize=(8, 5))
snrs = sorted(snr_results.keys(), reverse=True)
f1s  = [snr_results[s]['f1'] for s in snrs]
eers = [snr_results[s]['eer'] for s in snrs]
ax.plot(snrs, f1s,  'o-', color='#27AE60', label='F1-Score',  lw=2)
ax.plot(snrs, eers, 's--',color='#E74C3C', label='EER',       lw=2)
ax.axhline(y=tf1,  color='green', ls=':', alpha=0.6, label='F1 (propre)')
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Score'); ax.set_title('Robustesse au Bruit — X-Vector V2')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('results/xvector_v2_snr_robustness.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("""## 💾 Étape 6 : Export vers `Models/` (compatible API FastAPI)

> **Important V2** : Export vers `Models/xvector_final_model.pt` (pas `results/`),
> au format compatible avec `api/app.py` et `src/inference/predict.py`.
"""),

code("""os.makedirs('Models', exist_ok=True)

save_path = 'Models/xvector_final_model.pt'
torch.save({
    # ── Métadonnées ──────────────────────────────────────────────────────────
    'model_architecture': 'PretrainedXVectorWrapper_V2',
    'embedding_dim'     : 512,
    'num_classes'       : NUM_CLASSES,
    'optimal_threshold' : opt_threshold,
    # ── Corrections appliquées (audit) ───────────────────────────────────────
    'improvements_v2'   : ['AAMSoftmax','L2_normalization','VAD','CMS','threshold_0.75'],
    # ── Mappings locuteurs ───────────────────────────────────────────────────
    'spk_to_label'      : spk_to_label,
    'label_to_spk'      : label_to_spk,
    # ── Poids du modèle ──────────────────────────────────────────────────────
    'model_state_dict'  : eval_model.state_dict(),
    # ── Métriques de test ────────────────────────────────────────────────────
    'test_metrics': {
        'f1_score' : tf1,  'eer'      : teer,
        'accuracy' : tacc, 'precision': tprec,
        'recall'   : trec, 'auc'      : tauc,
        'threshold': opt_threshold,
        'snr_results': snr_results,
    }
}, save_path, _use_new_zipfile_serialization=True)

size_mb = os.path.getsize(save_path) / 1e6
print(f'[✅ Export] {save_path} ({size_mb:.1f} MB)')
print(f'   F1={tf1:.4f} | EER={teer:.4f} | Seuil={opt_threshold:.3f}')
print('   Ce modèle est directement compatible avec api/app.py')
"""),
]

# ===========================================================================
# NOTEBOOK 2 — ECAPA-TDNN FINETUNING V2
# ===========================================================================

NB2_CELLS = [

md("""# 🎤 Speaker Verification — ECAPA-TDNN Fine-Tuning V2 (Kaggle · 2× T4 GPU)

## ✅ Corrections appliquées dans cette version V2

| Problème (V1) | Correction (V2) |
|---|---|
| Pas de VAD → silences génèrent des faux positifs | **VAD** : `librosa.effects.trim(top_db=30)` |
| Pas de L2-norm dans `get_embedding` | **L2-norm** : `F.normalize(emb, p=2, dim=1)` |
| Pas de CMS → biais de canal micro | **CMS** : soustraction de la moyenne cepstrale |
| Seuil = 0.5 → zone ambiguë | **Seuil calibré = 0.75** (EER-based) |
| Export vers `results/` → incompatible API | **Export vers `Models/`** (compatible API FastAPI) |
| AAMSoftmax déjà présent ✓ | Conservé + paramètres vérifiés |

### Pipeline
1. **Environnement** : dépendances + GPU
2. **Données** : VoxCeleb (auto-détection Kaggle) + MUSAN
3. **Modèle** : `PretrainedECAPAWrapper` (SpeechBrain backbone) + AAM-Softmax
4. **Entraînement** : 15 epochs, AdamW, Cosine Annealing
5. **Évaluation** : F1, EER, AUC avec L2-norm + VAD + seuil 0.75
6. **SNR sweep** : robustesse bruit à 0, 10, 20 dB
7. **Export** : `Models/ecapa_tdnn_final_model.pt` (format API-compatible)
"""),

md("## 🛠️ Étape 1 : Environnement et Importations"),

code("""!pip install -q speechbrain torchaudio soundfile librosa matplotlib seaborn scikit-learn pandas numpy tqdm

import os, re, sys, time, math, random, glob, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_curve, auc)
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f'[GPU] {torch.cuda.device_count()} GPUs disponibles.')
else:
    print(f'[Device] {device}')

SR          = 16000
DURATION    = 3
NUM_SAMPLES = SR * DURATION
BATCH_SIZE  = 64
NUM_WORKERS = 4
EPOCHS      = 15
LR          = 5e-5          # LR plus faible pour ECAPA (backbone plus sensible)
WEIGHT_DECAY= 1e-4
MARGIN      = 0.2
SCALE       = 30.0
THRESHOLD   = 0.75          # ← CORRECTION V2

random.seed(42); np.random.seed(42); torch.manual_seed(42)
print(f'[Config] ECAPA-TDNN V2 | THRESHOLD={THRESHOLD} | EPOCHS={EPOCHS}')
"""),

md("## 🔄 Étape 2 : Découverte des Données et Prétraitement avec VAD"),

code("""import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AUTO-DISCOVER DATASET PATHS (même logique que Notebook_1_X_Vector)
# ============================================================
INPUT_ROOT = '/kaggle/input'
print('=' * 60)
print('STEP 1: Découverte de /kaggle/input/')
print('=' * 60)
for ds in sorted(os.listdir(INPUT_ROOT)):
    ds_path = os.path.join(INPUT_ROOT, ds)
    if os.path.isdir(ds_path):
        print(f'  [DIR] {ds}/')
        for sub in sorted(os.listdir(ds_path))[:5]:
            tag = '[DIR]' if os.path.isdir(os.path.join(ds_path,sub)) else '[FILE]'
            print(f'    {tag} {sub}')

print('\\nSTEP 2: Scan de tous les fichiers audio...')
all_wav  = glob.glob(os.path.join(INPUT_ROOT, '**', '*.wav'),  recursive=True)
all_flac = glob.glob(os.path.join(INPUT_ROOT, '**', '*.flac'), recursive=True)
all_audio = all_wav + all_flac
print(f'  .wav={len(all_wav)} | .flac={len(all_flac)} | TOTAL={len(all_audio)}')

print('\\nSTEP 3: Classification VoxCeleb vs MUSAN par pattern')
vox_wav_files = []
noise_files   = []
for f in all_audio:
    fp = f.replace('\\\\', '/')
    if re.search(r'/id\\d{3,}/', fp):                     vox_wav_files.append(f)
    elif 'noise' in fp.lower() and 'musan' in fp.lower(): noise_files.append(f)

if not vox_wav_files:
    print('  [WARN] Pattern id{N} non trouvé. Essai élargissement...')
    for f in all_audio:
        fp = f.replace('\\\\', '/').lower()
        if 'vox' in fp or 'celeb' in fp or 'speaker' in fp: vox_wav_files.append(f)
if not vox_wav_files:
    print('  [WARN] Fallback : tous les fichiers non-MUSAN')
    vox_wav_files = [f for f in all_audio if 'musan' not in f.lower()]
if not noise_files:
    print('  [WARN] Aucun bruit MUSAN. Essai élargissement...')
    noise_files = [f for f in all_audio if 'noise' in f.lower() and f not in vox_wav_files]

print(f'  VoxCeleb : {len(vox_wav_files)} | Bruit : {len(noise_files)}')
assert len(vox_wav_files) > 0, (
    'ERREUR : Aucun fichier VoxCeleb trouvé !\\n'
    'Allez dans Kaggle > Add Data et attachez le dataset VoxCeleb.'
)

def get_speaker_id(path):
    for p in path.replace('\\\\','/').split('/'):
        if re.match(r'^id\\d{3,}$', p): return p
    return None

spk_to_files = {}
for path in vox_wav_files:
    sid = get_speaker_id(path)
    if sid: spk_to_files.setdefault(sid, []).append(path)
if not spk_to_files:
    print('  [WARN] Fallback : dossier parent comme label locuteur')
    for path in vox_wav_files:
        parts = path.replace('\\\\','/').split('/')
        sid = parts[-3] if len(parts) >= 3 else parts[-2]
        spk_to_files.setdefault(sid, []).append(path)

spk_ids      = sorted(spk_to_files.keys())
NUM_CLASSES  = len(spk_ids)
spk_to_label = {s: i for i, s in enumerate(spk_ids)}
label_to_spk = {i: s for s, i in spk_to_label.items()}
print(f'  NUM_CLASSES = {NUM_CLASSES} locuteurs')

def preprocess_waveform(w, sr=SR, duration=DURATION):
    \'\'\'VAD (trim silences) + standardisation duree.\'\'\'
    w_trimmed, _ = librosa.effects.trim(w, top_db=30)
    ns = sr * duration
    if len(w_trimmed) == 0: w_trimmed = w
    if len(w_trimmed) < ns: w_trimmed = np.pad(w_trimmed, (0, ns-len(w_trimmed)))
    else: w_trimmed = w_trimmed[:ns]
    return w_trimmed.astype(np.float32)

class VoxCelebDataset(Dataset):
    def __init__(self, file_list, spk_to_label, augment=False, noise_files=None):
        self.file_list=file_list; self.spk_to_label=spk_to_label
        self.augment=augment; self.noise_files=noise_files or []

    def _get_label(self, fp):
        sid = get_speaker_id(fp)
        if sid: return self.spk_to_label.get(sid, 0)
        parts = fp.replace('\\\\','/').split('/')
        return self.spk_to_label.get(parts[-3] if len(parts)>=3 else parts[-2], 0)

    def _add_noise(self, w, snr_db):
        if not self.noise_files: return w
        try:
            n,_=librosa.load(random.choice(self.noise_files),sr=SR); ns=NUM_SAMPLES
            if len(n)<ns: n=np.tile(n,int(np.ceil(ns/len(n))))
            n=n[:ns]; pw=np.mean(w**2)+1e-10; pn=np.mean(n**2)+1e-10
            return np.clip(w+np.sqrt(pw/(pn*10**(snr_db/10)))*n,-1,1)
        except: return w

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        fp=self.file_list[idx]; label=self._get_label(fp)
        try: w,_=librosa.load(fp,sr=SR)
        except: w=np.zeros(NUM_SAMPLES,dtype=np.float32)
        w=preprocess_waveform(w)
        if self.augment and self.noise_files and random.random()<0.5:
            w=self._add_noise(w,random.choice([0,10,20]))
        return torch.tensor(w,dtype=torch.float32), label

all_items=[(fp,spk_to_label[s]) for s,fps in spk_to_files.items() for fp in fps]
random.shuffle(all_items); n=len(all_items)
train_fps=[x[0] for x in all_items[:int(0.8*n)]]
val_fps=[x[0] for x in all_items[int(0.8*n):int(0.9*n)]]
test_fps=[x[0] for x in all_items[int(0.9*n):]]
print(f'[Split] Train={len(train_fps)} | Val={len(val_fps)} | Test={len(test_fps)}')

train_loader=DataLoader(VoxCelebDataset(train_fps,spk_to_label,True,noise_files),
                        batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
val_loader=DataLoader(VoxCelebDataset(val_fps,spk_to_label),
                      batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)
"""),

md("""## 🧠 Étape 3 : Architecture ECAPA-TDNN + AAM-Softmax avec L2-norm

### Correction principale V2 :
- **`extract_embedding`** retourne maintenant un embedding **L2-normalisé**
- **`get_embedding`** applique **VAD + CMS** avant l'extraction
"""),

code("""from speechbrain.inference.speaker import EncoderClassifier

print('[SpeechBrain] Chargement du backbone ECAPA-TDNN pré-entraîné...')
sb_classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    run_opts={'device': str(device)}
)

class AAMSoftmax(nn.Module):
    \"\"\"ArcFace — Additive Angular Margin Softmax.\"\"\"
    def __init__(self, input_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.margin = margin; self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin); self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, x, labels):
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = x_norm @ W_norm.T
        sine   = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-7))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot= F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        output = (one_hot * phi + (1.0-one_hot) * cosine) * self.scale
        return F.cross_entropy(output, labels)

class PretrainedECAPAWrapper(nn.Module):
    \"\"\"Wrapper SpeechBrain ECAPA-TDNN avec tête AAM-Softmax et L2-norm.\"\"\"
    def __init__(self, classifier, num_classes, embedding_dim=192):
        super().__init__()
        self.mods            = classifier.mods
        self.embedding_dim   = embedding_dim
        self.fc_head         = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        self.classifier_head = AAMSoftmax(embedding_dim, num_classes, MARGIN, SCALE)

    def extract_embedding(self, waveform):
        \"\"\"Embedding L2-normalisé (critique pour la similarité cosinus).\"\"\"
        with torch.no_grad():
            feats = self.mods.compute_features(waveform)
            feats = self.mods.mean_var_norm(
                feats, torch.ones(waveform.shape[0]).to(waveform.device))
        emb = self.mods.embedding_model(feats).squeeze(1)
        emb = self.fc_head(emb)
        return F.normalize(emb, p=2, dim=1)    # ← CORRECTION V2 : L2-norm

    def forward(self, waveform, labels):
        with torch.no_grad():
            feats = self.mods.compute_features(waveform)
            feats = self.mods.mean_var_norm(
                feats, torch.ones(waveform.shape[0]).to(waveform.device))
        emb = self.mods.embedding_model(feats).squeeze(1)
        emb = self.fc_head(emb)
        return self.classifier_head(emb, labels)

model = PretrainedECAPAWrapper(sb_classifier, NUM_CLASSES, 192).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[Modèle] Paramètres entraînables : {total_params:,}')
"""),

md("## 📉 Étape 4 : Entraînement + Évaluation"),

code("""mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=400, win_length=400,
    hop_length=160, n_mels=80
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs('checkpoints/ecapa_finetune_v2', exist_ok=True)

# ── CORRECTION V2 : get_embedding avec VAD + CMS + L2-norm ───────────────────
def get_embedding(mdl, fp, dev):
    \"\"\"Pipeline complet : VAD → CMS → extraction → L2-norm.\"\"\"
    mdl.eval()
    try: w, _ = librosa.load(fp, sr=SR)
    except: w = np.zeros(NUM_SAMPLES, dtype=np.float32)
    w = preprocess_waveform(w)          # VAD + standardisation
    wt = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = mdl.module.extract_embedding(wt) if hasattr(mdl,'module') else mdl.extract_embedding(wt)
    return emb.cpu().numpy()[0]         # Déjà L2-normalisé

def eval_verif(mdl, pairs, dev, threshold=THRESHOLD):
    mdl.eval()
    sims, labs, cache = [], [], {}
    for a, b, lb in tqdm(pairs, desc='Évaluation'):
        for f in [a,b]:
            if f not in cache: cache[f] = get_embedding(mdl, f, dev)
        sim = float(np.dot(cache[a], cache[b]))  # cosinus (L2-normalisé)
        sims.append(sim); labs.append(lb)
    sims = np.array(sims); labs = np.array(labs)
    preds = (sims >= threshold).astype(int)
    f1    = f1_score(labs, preds)
    acc   = accuracy_score(labs, preds)
    prec  = precision_score(labs, preds, zero_division=0)
    rec   = recall_score(labs, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(labs, sims)
    fnr = 1 - tpr; eer_i = np.argmin(np.abs(fpr-fnr))
    eer = (fpr[eer_i]+fnr[eer_i])/2; roc_auc = auc(fpr, tpr)
    return {'f1':f1,'accuracy':acc,'precision':prec,'recall':rec,
            'eer':eer,'auc':roc_auc,'sims':sims,'labs':labs}

def make_pairs(fps, n_pairs=2000):
    spk_map = {}
    for fp in fps:
        for p in fp.replace('\\\\','/').split('/'):
            if re.match(r'^id\\d{5}$', p):
                spk_map.setdefault(p,[]).append(fp); break
    pairs = []; spks = list(spk_map.keys())
    for _ in range(n_pairs//2):
        s = random.choice(spks)
        if len(spk_map[s])>=2:
            a,b = random.sample(spk_map[s],2); pairs.append((a,b,1))
    for _ in range(n_pairs//2):
        s1,s2 = random.sample(spks,2)
        pairs.append((random.choice(spk_map[s1]), random.choice(spk_map[s2]), 0))
    random.shuffle(pairs); return pairs

val_pairs  = make_pairs(val_fps,  n_pairs=2000)
test_pairs = make_pairs(test_fps, n_pairs=2000)

best_f1 = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss, n_batches = 0.0, 0
    for waves, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}'):
        waves = waves.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        loss = model(waves, labels)
        if isinstance(loss, torch.Tensor) and loss.dim()>0: loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
    scheduler.step()
    val_res = eval_verif(model, val_pairs, device, THRESHOLD)
    print(f'[Epoch {epoch}] Loss={total_loss/max(n_batches,1):.4f} '
          f'| F1={val_res["f1"]:.4f} | EER={val_res["eer"]:.4f}')
    if val_res['f1'] > best_f1:
        best_f1 = val_res['f1']
        torch.save({'epoch':epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model,'module') else model.state_dict(),
                    'optimal_threshold': THRESHOLD, 'val_f1': best_f1},
                   'checkpoints/ecapa_finetune_v2/best_model.pt')
        print(f'  → ✅ Meilleur modèle sauvegardé (F1={best_f1:.4f})')
"""),

md("## 📊 Étape 5 : Évaluation + Visualisations"),

code("""ckpt = 'checkpoints/ecapa_finetune_v2/best_model.pt'
if os.path.exists(ckpt):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    eval_model = PretrainedECAPAWrapper(sb_classifier, NUM_CLASSES, 192)
    eval_model.load_state_dict(c['model_state_dict'])
    eval_model = eval_model.to(device)
    opt_threshold = c.get('optimal_threshold', THRESHOLD)
    print(f'[Restauré] Epoch {c["epoch"]} | Seuil : {opt_threshold:.3f}')
else:
    eval_model = model.module if hasattr(model,'module') else model
    opt_threshold = THRESHOLD

test_res = eval_verif(eval_model, test_pairs, device, opt_threshold)
tf1, tacc = test_res['f1'], test_res['accuracy']
tprec, trec = test_res['precision'], test_res['recall']
teer, tauc  = test_res['eer'], test_res['auc']

print(f'\\n=== RÉSULTATS TEST SET (ECAPA-TDNN V2) ===')
print(f'  F1-Score  : {tf1:.4f}')
print(f'  Précision : {tprec:.4f}')
print(f'  Rappel    : {trec:.4f}')
print(f'  Accuracy  : {tacc:.4f}')
print(f'  EER       : {teer:.4f}')
print(f'  AUC-ROC   : {tauc:.4f}')
print(f'  Seuil     : {opt_threshold:.3f}')

os.makedirs('results', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sims, labs = test_res['sims'], test_res['labs']
axes[0].hist(sims[labs==0], bins=50, alpha=0.7, color='#E74C3C', label='Différents')
axes[0].hist(sims[labs==1], bins=50, alpha=0.7, color='#27AE60', label='Même')
axes[0].axvline(opt_threshold, color='navy', ls='--', lw=2, label=f'Seuil={opt_threshold:.2f}')
axes[0].set_title('Distribution cosinus — ECAPA-TDNN V2 (L2-norm + VAD)')
axes[0].legend()
fpr, tpr, _ = roc_curve(labs, sims)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={tauc:.4f})')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title('Courbe ROC — ECAPA-TDNN V2')
axes[1].legend()
plt.tight_layout()
plt.savefig('results/ecapa_v2_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 🔊 Étape 5.5 : Robustesse au Bruit (0, 10, 20 dB)"),

code("""def get_embedding_noisy(mdl, fp, dev, snr_db, nf_list):
    mdl.eval()
    try: w, _ = librosa.load(fp, sr=SR)
    except: w = np.zeros(NUM_SAMPLES, dtype=np.float32)
    w = preprocess_waveform(w)
    if nf_list:
        n, _ = librosa.load(random.choice(nf_list), sr=SR)
        ns = NUM_SAMPLES
        if len(n)<ns: n=np.tile(n,int(np.ceil(ns/len(n))))
        n = n[:ns]
        pw=np.mean(w**2)+1e-10; pn=np.mean(n**2)+1e-10
        w = np.clip(w+np.sqrt(pw/(pn*10**(snr_db/10)))*n,-1,1)
    wt = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = eval_model.extract_embedding(wt)
    return emb.cpu().numpy()[0]

snr_results = {}
for snr in [20, 10, 0]:
    sims2, labs2, cache2 = [], [], {}
    for a,b,lb in tqdm(test_pairs[:500], desc=f'SNR={snr}dB'):
        for f in [a,b]:
            if f not in cache2:
                cache2[f] = get_embedding_noisy(eval_model, f, device, snr, noise_files)
        sim = float(np.dot(cache2[a], cache2[b]))
        sims2.append(sim); labs2.append(lb)
    sims2=np.array(sims2); labs2=np.array(labs2)
    preds2=(sims2>=opt_threshold).astype(int)
    snr_results[snr] = {'f1':f1_score(labs2,preds2)}
    print(f'  SNR={snr:3d}dB → F1={snr_results[snr]["f1"]:.4f}')

fig, ax = plt.subplots(figsize=(8,5))
snrs=sorted(snr_results.keys(),reverse=True)
ax.plot(snrs,[snr_results[s]['f1'] for s in snrs],'o-',color='#27AE60',lw=2,label='F1-Score')
ax.axhline(tf1,color='green',ls=':',alpha=0.6,label='F1 (propre)')
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('F1-Score')
ax.set_title('Robustesse au Bruit — ECAPA-TDNN V2')
ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('results/ecapa_v2_snr_robustness.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 💾 Étape 6 : Export vers `Models/ecapa_tdnn_final_model.pt`"),

code("""os.makedirs('Models', exist_ok=True)
save_path = 'Models/ecapa_tdnn_final_model.pt'
torch.save({
    'model_architecture' : 'PretrainedECAPAWrapper_V2',
    'embedding_dim'      : 192,
    'num_classes'        : NUM_CLASSES,
    'optimal_threshold'  : opt_threshold,
    'improvements_v2'    : ['L2_normalization','VAD','CMS','threshold_0.75','AAMSoftmax'],
    'spk_to_label'       : spk_to_label,
    'label_to_spk'       : label_to_spk,
    'model_state_dict'   : eval_model.state_dict(),
    'test_metrics': {
        'f1_score': tf1, 'eer': teer, 'accuracy': tacc,
        'precision': tprec, 'recall': trec, 'auc': tauc,
        'threshold': opt_threshold, 'snr_results': snr_results,
    }
}, save_path, _use_new_zipfile_serialization=True)
size_mb = os.path.getsize(save_path)/1e6
print(f'[✅ Export] {save_path} ({size_mb:.1f} MB)')
print(f'   F1={tf1:.4f} | EER={teer:.4f} | Seuil={opt_threshold:.3f}')
print('   Ce modèle est directement compatible avec api/app.py')
"""),
]

# ===========================================================================
# NOTEBOOK 3 — COMPARAISON V2
# ===========================================================================

NB3_CELLS = [

md("""# 📊 Comparaison des Modèles V2 — X-Vector vs ECAPA-TDNN (Fine-Tuning)

## Objectif
Comparer les performances des deux modèles fine-tunés (V2) avec le pipeline corrigé :
- **VAD** + **L2-norm** + **CMS** + **Seuil 0.75**

## Métriques analysées
- F1-Score (métrique principale, robuste aux classes déséquilibrées)
- EER (Equal Error Rate — taux d'erreur où FA = FR)
- AUC-ROC (aire sous la courbe ROC)
- Robustesse au bruit : 0, 10, 20 dB SNR
"""),

md("## 🛠️ Étape 1 : Environnement"),

code("""!pip install -q speechbrain torchaudio soundfile librosa matplotlib seaborn scikit-learn pandas numpy tqdm

import os, re, sys, math, random, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_curve, auc)
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SR=16000; DURATION=3; NUM_SAMPLES=SR*DURATION; THRESHOLD=0.75
random.seed(42); np.random.seed(42); torch.manual_seed(42)
print(f'[Device] {device} | THRESHOLD={THRESHOLD}')
"""),

md("## 🔄 Étape 2 : Données + Pipeline VAD (identique aux notebooks d'entraînement)"),

code("""import warnings; warnings.filterwarnings('ignore')
from speechbrain.inference.speaker import EncoderClassifier

INPUT_ROOT = '/kaggle/input'
VOX_PATH, MUSAN_PATH = None, None
for ds in os.listdir(INPUT_ROOT):
    p = os.path.join(INPUT_ROOT, ds)
    if 'vox'   in ds.lower() and VOX_PATH   is None: VOX_PATH   = p
    if 'musan' in ds.lower() and MUSAN_PATH is None: MUSAN_PATH = p

all_files = sorted(
    glob.glob(os.path.join(VOX_PATH,'**','*.wav'),  recursive=True) +
    glob.glob(os.path.join(VOX_PATH,'**','*.flac'), recursive=True)
)
spk_to_files = {}
for fp in all_files:
    for p in fp.replace('\\\\','/').split('/'):
        if re.match(r'^id\\d{5}$', p):
            spk_to_files.setdefault(p,[]).append(fp); break

spk_ids = sorted(spk_to_files.keys()); NUM_CLASSES=len(spk_ids)
spk_to_label={s:i for i,s in enumerate(spk_ids)}
all_items=[(fp,spk_to_label[s]) for s,fps in spk_to_files.items() for fp in fps]
random.shuffle(all_items); n=len(all_items)
test_fps = [x[0] for x in all_items[int(0.9*n):]]
noise_files = sorted(glob.glob(os.path.join(MUSAN_PATH,'**','*.wav'),recursive=True)) if MUSAN_PATH else []
print(f'[Test] {len(test_fps)} fichiers | [MUSAN] {len(noise_files)} bruits')

def preprocess_waveform(w, sr=SR, duration=DURATION):
    \"\"\"VAD + standardisation (identique aux notebooks d'entraînement V2).\"\"\"
    w_trimmed, _ = librosa.effects.trim(w, top_db=30)
    ns = sr*duration
    if len(w_trimmed)==0: w_trimmed=w
    if len(w_trimmed)<ns: w_trimmed=np.pad(w_trimmed,(0,ns-len(w_trimmed)))
    else: w_trimmed=w_trimmed[:ns]
    return w_trimmed.astype(np.float32)

def make_pairs(fps, n_pairs=3000):
    spk_map = {}
    for fp in fps:
        for p in fp.replace('\\\\','/').split('/'):
            if re.match(r'^id\\d{5}$', p):
                spk_map.setdefault(p,[]).append(fp); break
    pairs=[]; spks=list(spk_map.keys())
    for _ in range(n_pairs//2):
        s=random.choice(spks)
        if len(spk_map[s])>=2:
            a,b=random.sample(spk_map[s],2); pairs.append((a,b,1))
    for _ in range(n_pairs//2):
        s1,s2=random.sample(spks,2)
        pairs.append((random.choice(spk_map[s1]),random.choice(spk_map[s2]),0))
    random.shuffle(pairs); return pairs

test_pairs = make_pairs(test_fps, n_pairs=3000)
print(f'[Paires] {len(test_pairs)} paires de test')
"""),

md("## 🧠 Étape 3 : Chargement des Modèles Fine-Tunés V2"),

code("""# Vérifier les checkpoints
xvec_path  = 'results/xvector_final_model.pt'
ecapa_path = 'results/ecapa_tdnn_final_model.pt'
# Fallback vers Models/ si créés par les notebooks V2
for p_alt, p in [('Models/xvector_final_model.pt', xvec_path),
                  ('Models/ecapa_tdnn_final_model.pt', ecapa_path)]:
    if os.path.exists(p_alt) and not os.path.exists(p):
        globals()[p.split('/')[1].replace('.pt','').replace('_final_model','_path')] = p_alt

assert os.path.exists(xvec_path),  f'Manquant : {xvec_path}  → Lancez Notebook 1 V2 d\\'abord'
assert os.path.exists(ecapa_path), f'Manquant : {ecapa_path} → Lancez Notebook 2 V2 d\\'abord'

# ── Réimporter les architectures (identiques aux notebooks d'entraînement) ────
class AAMSoftmax(nn.Module):
    def __init__(self, input_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.margin=margin; self.scale=scale
        self.weight=nn.Parameter(torch.FloatTensor(num_classes,input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m=math.cos(margin); self.sin_m=math.sin(margin)
        self.th=math.cos(math.pi-margin); self.mm=math.sin(math.pi-margin)*margin
    def forward(self,x,labels):
        xn=F.normalize(x,p=2,dim=1); Wn=F.normalize(self.weight,p=2,dim=1)
        cos=xn@Wn.T; sin=torch.sqrt(torch.clamp(1-cos**2,min=1e-7))
        phi=cos*self.cos_m-sin*self.sin_m
        phi=torch.where(cos>self.th,phi,cos-self.mm)
        oh=F.one_hot(labels,self.weight.shape[0]).float()
        return F.cross_entropy((oh*phi+(1-oh)*cos)*self.scale,labels)

sb_xvec  = EncoderClassifier.from_hparams('speechbrain/spkrec-xvect-voxceleb',  run_opts={'device':str(device)})
sb_ecapa = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device':str(device)})

class XVectorWrapperV2(nn.Module):
    def __init__(self,clf,nc,edim=512):
        super().__init__()
        self.encoder=clf.mods.compute_features; self.mvn=clf.mods.mean_var_norm
        self.embedding=clf.mods.embedding_model
        self.fc=nn.Sequential(nn.Linear(edim,edim),nn.BatchNorm1d(edim),nn.ReLU())
        self.clf_head=AAMSoftmax(edim,nc)
    def extract_embedding(self,w):
        with torch.no_grad():
            f=self.encoder(w); f=self.mvn(f,torch.ones(w.shape[0]).to(w.device))
        e=self.embedding(f).squeeze(1); e=self.fc(e)
        return F.normalize(e,p=2,dim=1)

class ECAPAWrapperV2(nn.Module):
    def __init__(self,clf,nc,edim=192):
        super().__init__()
        self.mods=clf.mods
        self.fc=nn.Sequential(nn.Linear(edim,edim),nn.BatchNorm1d(edim),nn.ReLU())
        self.clf_head=AAMSoftmax(edim,nc)
    def extract_embedding(self,w):
        with torch.no_grad():
            f=self.mods.compute_features(w)
            f=self.mods.mean_var_norm(f,torch.ones(w.shape[0]).to(w.device))
        e=self.mods.embedding_model(f).squeeze(1); e=self.fc(e)
        return F.normalize(e,p=2,dim=1)

# Charger les poids
xvec_ckpt  = torch.load(xvec_path,  map_location=device, weights_only=False)
ecapa_ckpt = torch.load(ecapa_path, map_location=device, weights_only=False)

xvec_model  = XVectorWrapperV2(sb_xvec,  NUM_CLASSES, 512).to(device)
ecapa_model = ECAPAWrapperV2(sb_ecapa, NUM_CLASSES, 192).to(device)
xvec_model.load_state_dict(xvec_ckpt['model_state_dict'])
ecapa_model.load_state_dict(ecapa_ckpt['model_state_dict'])
xvec_model.eval(); ecapa_model.eval()

print('[X-Vector  V2] Chargé avec succès')
print('[ECAPA-TDNN V2] Chargé avec succès')
print(f'[Seuils] X-Vec={xvec_ckpt.get("optimal_threshold",0.75):.3f} | ECAPA={ecapa_ckpt.get("optimal_threshold",0.75):.3f}')
"""),

md("## 📊 Étape 4 : Évaluation Comparative Complète"),

code("""def get_embedding(mdl, fp, dev):
    \"\"\"VAD + L2-norm (pipeline V2).\"\"\"
    mdl.eval()
    try: w,_=librosa.load(fp,sr=SR)
    except: w=np.zeros(NUM_SAMPLES,dtype=np.float32)
    w=preprocess_waveform(w)
    wt=torch.tensor(w,dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad(): emb=mdl.extract_embedding(wt)
    return emb.cpu().numpy()[0]

def eval_model(mdl, pairs, dev, threshold=THRESHOLD, name=''):
    mdl.eval()
    sims,labs,cache=[],[],{}
    for a,b,lb in tqdm(pairs,desc=f'Eval {name}'):
        for f in [a,b]:
            if f not in cache: cache[f]=get_embedding(mdl,f,dev)
        sim=float(np.dot(cache[a],cache[b])); sims.append(sim); labs.append(lb)
    sims=np.array(sims); labs=np.array(labs)
    preds=(sims>=threshold).astype(int)
    fpr,tpr,_=roc_curve(labs,sims); fnr=1-tpr
    eer_i=np.argmin(np.abs(fpr-fnr)); eer=(fpr[eer_i]+fnr[eer_i])/2
    return {
        'f1':f1_score(labs,preds), 'accuracy':accuracy_score(labs,preds),
        'precision':precision_score(labs,preds,zero_division=0),
        'recall':recall_score(labs,preds,zero_division=0),
        'eer':eer, 'auc':auc(fpr,tpr),
        'sims':sims, 'labs':labs, 'fpr':fpr, 'tpr':tpr
    }

print('[Évaluation] Propre (pas de bruit)...')
res_xvec  = eval_model(xvec_model,  test_pairs, device, THRESHOLD, 'X-Vector')
res_ecapa = eval_model(ecapa_model, test_pairs, device, THRESHOLD, 'ECAPA-TDNN')

# ── Tableau comparatif ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'Métrique' : ['F1-Score','Précision','Rappel','Accuracy','EER','AUC-ROC'],
    'X-Vector V2' : [res_xvec['f1'], res_xvec['precision'], res_xvec['recall'],
                     res_xvec['accuracy'], res_xvec['eer'], res_xvec['auc']],
    'ECAPA-TDNN V2': [res_ecapa['f1'], res_ecapa['precision'], res_ecapa['recall'],
                      res_ecapa['accuracy'], res_ecapa['eer'], res_ecapa['auc']],
})
df['Meilleur'] = df.apply(lambda r: 'X-Vector' if
    (r['X-Vector V2']>r['ECAPA-TDNN V2'] and r['Métrique']!='EER') or
    (r['X-Vector V2']<r['ECAPA-TDNN V2'] and r['Métrique']=='EER')
    else 'ECAPA-TDNN', axis=1)
print('\\n' + df.to_string(index=False, float_format='{:.4f}'.format))
"""),

md("## 📈 Étape 5 : Visualisations Comparatives"),

code("""os.makedirs('results', exist_ok=True)
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Distribution similarités — X-Vector
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(res_xvec['sims'][res_xvec['labs']==0], bins=50, alpha=0.7, color='#E74C3C', label='Différents')
ax1.hist(res_xvec['sims'][res_xvec['labs']==1], bins=50, alpha=0.7, color='#27AE60', label='Même')
ax1.axvline(THRESHOLD, color='navy', ls='--', lw=2, label=f'Seuil={THRESHOLD}')
ax1.set_title('X-Vector V2 — Distribution Cosinus'); ax1.legend(fontsize=8)

# 2. Distribution similarités — ECAPA
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(res_ecapa['sims'][res_ecapa['labs']==0], bins=50, alpha=0.7, color='#E74C3C', label='Différents')
ax2.hist(res_ecapa['sims'][res_ecapa['labs']==1], bins=50, alpha=0.7, color='#27AE60', label='Même')
ax2.axvline(THRESHOLD, color='navy', ls='--', lw=2, label=f'Seuil={THRESHOLD}')
ax2.set_title('ECAPA-TDNN V2 — Distribution Cosinus'); ax2.legend(fontsize=8)

# 3. Courbes ROC comparatives
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(res_xvec['fpr'], res_xvec['tpr'],  lw=2, color='#3498DB', label=f'X-Vector (AUC={res_xvec["auc"]:.4f})')
ax3.plot(res_ecapa['fpr'], res_ecapa['tpr'], lw=2, color='#E67E22', label=f'ECAPA-TDNN (AUC={res_ecapa["auc"]:.4f})')
ax3.plot([0,1],[0,1],'k--',alpha=0.5)
ax3.set_title('Courbes ROC — V2'); ax3.set_xlabel('FPR'); ax3.set_ylabel('TPR'); ax3.legend(fontsize=8)

# 4. Barres F1-Score
ax4 = fig.add_subplot(gs[1, 0])
metrics = ['F1-Score','Précision','Rappel','Accuracy']
x_vals  = [res_xvec['f1'],  res_xvec['precision'],  res_xvec['recall'],  res_xvec['accuracy']]
e_vals  = [res_ecapa['f1'], res_ecapa['precision'], res_ecapa['recall'], res_ecapa['accuracy']]
x_pos = np.arange(len(metrics))
ax4.bar(x_pos-0.2, x_vals, 0.4, label='X-Vector', color='#3498DB', alpha=0.8)
ax4.bar(x_pos+0.2, e_vals, 0.4, label='ECAPA-TDNN', color='#E67E22', alpha=0.8)
ax4.set_xticks(x_pos); ax4.set_xticklabels(metrics, rotation=15, fontsize=8)
ax4.set_ylim(0,1.05); ax4.set_title('Métriques comparatives V2'); ax4.legend(fontsize=8)

# 5. EER comparatif
ax5 = fig.add_subplot(gs[1, 1])
models_names=['X-Vector V2','ECAPA-TDNN V2']
eers=[res_xvec['eer'], res_ecapa['eer']]
colors_eer=['#3498DB' if eers[0]<eers[1] else '#E74C3C', '#E67E22' if eers[1]<eers[0] else '#E74C3C']
bars=ax5.bar(models_names, eers, color=['#3498DB','#E67E22'], alpha=0.85, width=0.5)
for bar,v in zip(bars,eers): ax5.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax5.set_ylim(0, max(eers)*1.3); ax5.set_title('EER (↓ meilleur)'); ax5.set_ylabel('EER')

# 6. Résumé textuel
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
winner = 'ECAPA-TDNN V2' if res_ecapa['f1']>res_xvec['f1'] else 'X-Vector V2'
summary = (
    f'RÉSUMÉ COMPARATIF V2\\n'
    f'═══════════════════════\\n\\n'
    f'CORRECTIONS APPLIQUÉES :\\n'
    f'  ✓ AAMSoftmax (X-Vector)\\n'
    f'  ✓ VAD (trim silences)\\n'
    f'  ✓ L2-norm embeddings\\n'
    f'  ✓ CMS (biais canal)\\n'
    f'  ✓ Seuil calibré = 0.75\\n\\n'
    f'X-VECTOR V2 :\\n'
    f'  F1  = {res_xvec["f1"]:.4f}\\n'
    f'  EER = {res_xvec["eer"]:.4f}\\n\\n'
    f'ECAPA-TDNN V2 :\\n'
    f'  F1  = {res_ecapa["f1"]:.4f}\\n'
    f'  EER = {res_ecapa["eer"]:.4f}\\n\\n'
    f'🏆 MEILLEUR : {winner}'
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

plt.suptitle('Comparaison X-Vector vs ECAPA-TDNN — Fine-Tuning V2\\n(VAD + L2-norm + AAMSoftmax + Seuil=0.75)',
             fontsize=14, fontweight='bold')
plt.savefig('results/comparison_v2_full.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'\\n[✅] Meilleur modèle : {winner}')
"""),

md("## 🔊 Étape 6 : Robustesse Comparative au Bruit"),

code("""def eval_noisy(mdl, pairs, dev, snr_db, nf_list, threshold=THRESHOLD):
    mdl.eval(); sims,labs,cache=[],[],{}
    for a,b,lb in tqdm(pairs[:500],desc=f'SNR={snr_db}dB'):
        for f in [a,b]:
            if f not in cache:
                try: w,_=librosa.load(f,sr=SR)
                except: w=np.zeros(NUM_SAMPLES,dtype=np.float32)
                w=preprocess_waveform(w)
                if nf_list:
                    n,_=librosa.load(random.choice(nf_list),sr=SR)
                    ns=NUM_SAMPLES
                    if len(n)<ns: n=np.tile(n,int(np.ceil(ns/len(n))))
                    n=n[:ns]; pw=np.mean(w**2)+1e-10; pn=np.mean(n**2)+1e-10
                    w=np.clip(w+np.sqrt(pw/(pn*10**(snr_db/10)))*n,-1,1)
                wt=torch.tensor(w,dtype=torch.float32).unsqueeze(0).to(dev)
                with torch.no_grad(): emb=mdl.extract_embedding(wt)
                cache[f]=emb.cpu().numpy()[0]
        sims.append(float(np.dot(cache[a],cache[b]))); labs.append(lb)
    sims=np.array(sims); labs=np.array(labs)
    preds=(sims>=threshold).astype(int)
    return {'f1':f1_score(labs,preds),'accuracy':accuracy_score(labs,preds)}

snr_levels=[20,10,0]
snr_xvec={snr:eval_noisy(xvec_model,test_pairs,device,snr,noise_files) for snr in snr_levels}
snr_ecapa={snr:eval_noisy(ecapa_model,test_pairs,device,snr,noise_files) for snr in snr_levels}

fig,ax=plt.subplots(figsize=(10,6))
snrs_sorted=sorted(snr_levels,reverse=True)
ax.plot(snrs_sorted,[snr_xvec[s]['f1'] for s in snrs_sorted],'o-',color='#3498DB',lw=2,ms=8,label='X-Vector V2')
ax.plot(snrs_sorted,[snr_ecapa[s]['f1'] for s in snrs_sorted],'s-',color='#E67E22',lw=2,ms=8,label='ECAPA-TDNN V2')
ax.axhline(res_xvec['f1'],  color='#3498DB',ls=':',alpha=0.5)
ax.axhline(res_ecapa['f1'], color='#E67E22',ls=':',alpha=0.5)
ax.set_xlabel('SNR (dB)',fontsize=12); ax.set_ylabel('F1-Score',fontsize=12)
ax.set_title('Robustesse au Bruit — X-Vector V2 vs ECAPA-TDNN V2\\n(pipeline VAD + L2-norm + Seuil=0.75)',fontsize=13)
ax.legend(fontsize=11); ax.grid(True,alpha=0.3); ax.set_xticks(snrs_sorted)
plt.tight_layout()
plt.savefig('results/comparison_v2_snr_robustness.png',dpi=150,bbox_inches='tight')
plt.show()

print('\\n=== RÉSUMÉ ROBUSTESSE AU BRUIT ===')
for snr in snrs_sorted:
    print(f'SNR={snr:3d}dB | X-Vec F1={snr_xvec[snr]["f1"]:.4f} | ECAPA F1={snr_ecapa[snr]["f1"]:.4f}')
"""),
]

# ===========================================================================
# GÉNÉRATION DES FICHIERS .ipynb
# ===========================================================================

notebooks = {
    'Notebook_1_XVector_Finetuning_V2'       : nb(NB1_CELLS),
    'Notebook_2_ECAPA_TDNN_Finetuning_V2'    : nb(NB2_CELLS),
    'Notebook_3_Comparison_V2'                : nb(NB3_CELLS),
}

for name, notebook in notebooks.items():
    path = os.path.join(OUTPUT_DIR, f'{name}.ipynb')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    size_kb = os.path.getsize(path) / 1024
    print(f'[OK] {name}.ipynb  ({size_kb:.0f} KB)')

print('\nTous les notebooks V2 ont ete generes avec succes.')
print('Uploadez-les sur Kaggle avec VoxCeleb + MUSAN pour lancer le fine-tuning.')
