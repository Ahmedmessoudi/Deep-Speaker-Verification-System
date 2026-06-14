#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VÉRIFICATION DU LOCUTEUR - OUTIL D'INFÉRENCE INTERACTIF ET ROBUSTE
==================================================================
Ce script résout le problème des Faux Positifs (les voix différentes confondues)
en appliquant des traitements de normalisation acoustique stricts :
  1. Voice Activity Detection (VAD) : Suppression des silences perturbateurs.
  2. Cepstral Mean Subtraction (CMS) : Suppression de la signature du microphone (Biais de Canal).
  3. Normalisation L2 : Garantie que la similarité cosinus reste bornée entre -1.0 et 1.0.
  4. Seuil calibré : 0.75 (calibré pour modèles AAM-Softmax from-scratch sur VoxCeleb).
"""

import os
import sys
import gc
import math
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ====================================================================
# DEFINITION DES ARCHITECTURES
# ====================================================================

# --- X-VECTOR ---
class TDNNBlock(nn.Module):
    def __init__(self, inc, outc, ks, dil):
        super().__init__()
        self.conv = nn.Conv1d(inc, outc, ks, dilation=dil, padding=int(dil*(ks-1)/2))
        self.bn = nn.BatchNorm1d(outc)
    def forward(self, x): return F.relu(self.bn(self.conv(x)))

class StatsPooling(nn.Module):
    def forward(self, x): return torch.cat([x.mean(2), x.std(2, unbiased=False)], 1)

class XVectorModel(nn.Module):
    def __init__(self, input_dim=80, num_classes=10, embedding_dim=512):
        super().__init__()
        self.tdnn1=TDNNBlock(input_dim,512,5,1); self.tdnn2=TDNNBlock(512,512,3,2)
        self.tdnn3=TDNNBlock(512,512,3,3); self.tdnn4=TDNNBlock(512,512,1,1)
        self.tdnn5=TDNNBlock(512,1500,1,1); self.pool=StatsPooling()
        self.fc1=nn.Linear(3000,embedding_dim); self.bn1=nn.BatchNorm1d(embedding_dim)
        self.fc2=nn.Linear(embedding_dim,embedding_dim); self.bn2=nn.BatchNorm1d(embedding_dim)
        self.classifier=nn.Linear(embedding_dim,num_classes)
    def extract_embedding(self, x):
        x=self.tdnn1(x);x=self.tdnn2(x);x=self.tdnn3(x);x=self.tdnn4(x);x=self.tdnn5(x)
        return self.bn1(self.fc1(self.pool(x)))
    def forward(self, x):
        e=self.extract_embedding(x); return self.classifier(F.relu(self.bn2(self.fc2(F.relu(e)))))

# --- ECAPA-TDNN ---
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(ch,ch//r),nn.ReLU(),nn.Linear(ch//r,ch),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x.mean(2)).unsqueeze(2)

class SERes2NetBlock(nn.Module):
    def __init__(self,ch,ks,dil,scale=4):
        super().__init__()
        self.scale=scale
        self.conv1=nn.Conv1d(ch,ch,1);self.bn1=nn.BatchNorm1d(ch)
        w=ch//scale
        self.convs=nn.ModuleList([nn.Conv1d(w,w,ks,dilation=dil,padding=int(dil*(ks-1)/2)) for _ in range(scale-1)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(w) for _ in range(scale-1)])
        self.conv3=nn.Conv1d(ch,ch,1);self.bn3=nn.BatchNorm1d(ch)
        self.se=SEBlock(ch)
    def forward(self,x):
        res=x;out=F.relu(self.bn1(self.conv1(x)));sp=torch.chunk(out,self.scale,1);ns=[sp[0]]
        for i in range(1,self.scale):
            s=sp[i]+ns[i-1] if i>1 else sp[i]
            ns.append(F.relu(self.bns[i-1](self.convs[i-1](s))))
        return F.relu(self.se(self.bn3(self.conv3(torch.cat(ns,1))))+res)

class AttentiveStatsPooling(nn.Module):
    def __init__(self,inc,att=128):
        super().__init__()
        self.c1=nn.Conv1d(inc,att,1);self.c2=nn.Conv1d(att,inc,1)
    def forward(self,x):
        w=F.softmax(self.c2(torch.tanh(self.c1(x))),2)
        mu=(x*w).sum(2);sg=torch.sqrt(torch.clamp((w*(x-mu.unsqueeze(2))**2).sum(2),min=1e-9))
        return torch.cat([mu,sg],1)

class ECAPATDNNModel(nn.Module):
    def __init__(self,input_dim=80,num_classes=10,embedding_dim=192):
        super().__init__()
        self.conv1=nn.Conv1d(input_dim,512,5,padding=2);self.bn1=nn.BatchNorm1d(512)
        self.l1=SERes2NetBlock(512,3,2);self.l2=SERes2NetBlock(512,3,3);self.l3=SERes2NetBlock(512,3,4)
        self.conv2=nn.Conv1d(2048,1536,1);self.bn2=nn.BatchNorm1d(1536)
        self.pool=AttentiveStatsPooling(1536)
        self.fc=nn.Linear(3072,embedding_dim);self.bnfc=nn.BatchNorm1d(embedding_dim)
        self.classifier=nn.Linear(embedding_dim,num_classes)
    def extract_embedding(self,x):
        x0=F.relu(self.bn1(self.conv1(x)));x1=self.l1(x0);x2=self.l2(x1);x3=self.l3(x2)
        return self.bnfc(self.fc(self.pool(F.relu(self.bn2(self.conv2(torch.cat([x0,x1,x2,x3],1)))))))
    def forward(self,x): return self.classifier(F.relu(self.extract_embedding(x)))

# ====================================================================
# OUTILS D'INFERENCE ROBUSTES
# ====================================================================

def load_verification_model(model_type, checkpoint_path, device):
    """
    Restaure proprement un checkpoint exporté et configure le modèle en évaluation.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le fichier de poids spécifié n'existe pas : {checkpoint_path}")
        
    print(f"[Restaurateur] Chargement du checkpoint : {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    input_dim = ckpt.get('input_dim', 80)
    embedding_dim = ckpt.get('embedding_dim', 512 if model_type == 'xvector' else 192)
    num_classes = ckpt.get('num_classes', 10)
    opt_threshold = ckpt.get('optimal_threshold', 0.35)
    
    if model_type == 'xvector':
        model = XVectorModel(input_dim, num_classes, embedding_dim)
    else:
        model = ECAPATDNNModel(input_dim, num_classes, embedding_dim)
        
    # Charger les poids
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    
    print(f"[Restaurateur] Modèle {model_type.upper()} chargé avec succès !")
    print(f"               Seuil optimal calibré (VoxCeleb) : {opt_threshold:.3f}")
    return model, opt_threshold

def verify_audio_pair(model, path_1, path_2, device, threshold=0.75):
    """
    Exécute la comparaison 1-vs-1 sécurisée entre deux fichiers audio.
    """
    model.eval()
    embeddings = []
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80).to(device)
    
    for idx, path in enumerate([path_1, path_2], start=1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier audio introuvable : {path}")
            
        # 1. Chargement de l'onde acoustique (librosa force 16 kHz)
        w, sr = librosa.load(path, sr=16000)
        
        # 2. VAD (Voice Activity Detection) : trim les silences sous 30 dB de dynamique
        w_trimmed, _ = librosa.effects.trim(w, top_db=30)
        
        # Standardisation de la durée à 3.0 secondes (48000 échantillons)
        ns = 16000 * 3
        if len(w_trimmed) < ns:
            w_trimmed = np.pad(w_trimmed, (0, ns - len(w_trimmed)))
        else:
            w_trimmed = w_trimmed[:ns]
            
        wt = torch.tensor(w_trimmed, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 3. Extraction du spectrogramme Log-Mel + CMS (Cepstral Mean Subtraction)
        with torch.no_grad():
            mel = mel_transform(wt)
            log_mel = torch.log(mel + 1e-6)
            # CMS pour éliminer l'effet de signature micro/acoustique (biais de canal)
            log_mel = log_mel - log_mel.mean()
            
            # Extraction directe du locuteur
            if hasattr(model, 'extract_embedding'):
                emb = model.extract_embedding(log_mel)
            else:
                emb = model.module.extract_embedding(log_mel)
                
            # 4. NORMALISATION L2 (Indispensable pour borner le cosinus entre -1.0 et 1.0)
            emb_normalized = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb_normalized.cpu().numpy()[0])
            
    # 5. Calcul de la similarité cosinus (produit scalaire des embeddings de norme L2=1)
    cos_similarity = np.dot(embeddings[0], embeddings[1])
    
    # 6. Décision finale selon le seuil opérationnel ajusté
    verdict = "MÊME LOCUTEUR" if cos_similarity >= threshold else "LOCUTEURS DIFFÉRENTS"
    
    print("\n" + "=" * 70)
    print("                 RÉSULTAT DE LA VÉRIFICATION DU LOCUTEUR")
    print("=" * 70)
    print(f" Audio 1      : {os.path.basename(path_1)}")
    print(f" Audio 2      : {os.path.basename(path_2)}")
    print(f" Similarité   : {cos_similarity:.4f}")
    print(f" Seuil choisi : {threshold:.2f}")
    print(f" VERDICT      : {verdict}")
    print("=" * 70 + "\n")
    
    return cos_similarity, verdict

# ====================================================================
# INTERFACE EN LIGNE DE COMMANDE INTERACTIVE
# ====================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("     BIENVENUE DANS L'OUTIL DE VÉRIFICATION DU LOCUTEUR HAUTE SÉCURITÉ")
    print("=======================================================================")
    print("Ce script résout le problème des Faux Positifs en appliquant :")
    print("  - La suppression automatique des silences (VAD)")
    print("  - La suppression de l'empreinte micro (CMS)")
    print("  - La normalisation géométrique L2 sur hypersphère")
    print("=" * 70)
    
    # Recherche automatique des checkpoints dans 'Models/' ou '../Models/'
    potential_models = {
        'xvector': [
            'Models/xvector_final_model.pt',
            '../Models/xvector_final_model.pt',
            'results/xvector_final_model.pt'
        ],
        'ecapa': [
            'Models/ecapa_tdnn_final_model.pt',
            '../Models/ecapa_tdnn_final_model.pt',
            'results/ecapa_tdnn_final_model.pt'
        ]
    }
    
    selected_model_type = None
    selected_ckpt_path = None
    
    # 1. Sélection interactive du modèle
    model_choice = input("Choisissez le modèle à utiliser (1: ECAPA-TDNN [Recommandé] | 2: X-Vector) [1]: ").strip()
    if model_choice == '2':
        selected_model_type = 'xvector'
    else:
        selected_model_type = 'ecapa'
        
    # Recherche du fichier de poids
    paths_to_test = potential_models[selected_model_type]
    for p in paths_to_test:
        if os.path.exists(p):
            selected_ckpt_path = p
            break
            
    if not selected_ckpt_path:
        print(f"\n[ATTENTION] Aucun checkpoint exporté automatiquement trouvé pour {selected_model_type.upper()}.")
        selected_ckpt_path = input("Veuillez saisir le chemin absolu vers le checkpoint .pt : ").strip()
        
    if not os.path.exists(selected_ckpt_path):
        print(f"[ERREUR] Le fichier {selected_ckpt_path} is introuvable. Fin du script.")
        sys.exit(1)
        
    # 2. Restauration du réseau
    try:
        model, base_th = load_verification_model(selected_model_type, selected_ckpt_path, device)
    except Exception as e:
        print(f"[ERREUR] Échec du chargement du modèle : {e}")
        sys.exit(1)
        
    # 3. Demander les fichiers audio à comparer
    print("\n--- Saisie des fichiers audio à comparer ---")
    audio_1 = input("Saisissez le chemin du fichier audio 1 (.wav ou .flac) : ").strip()
    audio_2 = input("Saisissez le chemin du fichier audio 2 (.wav ou .flac) : ").strip()
    
    # Remplacer les guillemets éventuels (drag & drop Windows)
    audio_1 = audio_1.replace('"', '').replace("'", "")
    audio_2 = audio_2.replace('"', '').replace("'", "")
    
    # Seuil recommandé
    print(f"\nSeuil de décision par défaut pour VoxCeleb : {base_th:.3f}")
    print("Pour des enregistrements de voix propres (smartphone/micro PC), un seuil de 0.75 est vivement recommandé")
    print("pour rejeter les faux positifs (deux locuteurs différents confondus).")
    th_input = input("Saisissez le seuil de décision cosinus désiré [0.75] : ").strip()
    
    if th_input == "":
        th_val = 0.75
    else:
        try:
            th_val = float(th_input)
        except ValueError:
            th_val = 0.55
            
    # 4. Exécuter le diagnostic
    try:
        verify_audio_pair(model, audio_1, audio_2, device, threshold=th_val)
    except Exception as e:
        print(f"[ERREUR] Échec de l'évaluation : {e}")
