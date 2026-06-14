"""FastAPI service for speaker verification.

Inlined and completely decoupled from `/src` for extreme robustness in Docker and local execution.
"""

import io
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Speaker Verification System",
    description="Production-ready speaker verification API",
    version="1.0.0"
)

# Global variables
model = None
models = {}
preprocessor = None
verifier = None
verifiers = {}
speaker_db = None
config = None
model_type = None
device = None

MODEL_CHECKPOINT_CANDIDATES = {
    "ecapa_tdnn": [
        "Models/ecapa_tdnn_final_model.pt",
        #"Models/best_model_xvector.pt",
        #"Models/ecapa_tdnn_final_model_finetune.pt",
    ],
    "xvector": [
        "Models/best_model_xvector.pt",
        #"Models/ecapa_tdnn_final_model.pt",
        #"Models/best_model_xvector_finetune.pt",
    ],
}


class VerifyRequest(BaseModel):
    """Request model for speaker verification."""
    threshold: Optional[float] = 0.75


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    speaker_id: str
    embedding: list
    success: bool


class VerifyResponse(BaseModel):
    """Response model for verification."""
    similarity_score: float
    is_same_speaker: bool
    threshold_used: float


class IdentifyResponse(BaseModel):
    """Response model for speaker identification."""
    top_matches: list
    success: bool


class ModelComparisonResponse(BaseModel):
    """Response model for ECAPA vs XVector comparison."""
    reference_file: str
    comparison_file: str
    best_model: str
    models: dict


# ====================================================================
# DEFINITIONS OF INLINED ARCHITECTURES & HELPERS
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
    def __init__(self, input_dim=80, num_classes=1211, embedding_dim=512):
        super().__init__()
        self.tdnn1=TDNNBlock(input_dim,512,5,1); self.tdnn2=TDNNBlock(512,512,3,2)
        self.tdnn3=TDNNBlock(512,512,3,3); self.tdnn4=TDNNBlock(512,512,1,1)
        self.tdnn5=TDNNBlock(512,1500,1,1); self.pool=StatsPooling()
        self.fc1=nn.Linear(3000,embedding_dim); self.bn1=nn.BatchNorm1d(embedding_dim)
        self.fc2=nn.Linear(embedding_dim,embedding_dim); self.bn2=nn.BatchNorm1d(embedding_dim)
        self.classifier=nn.Linear(embedding_dim,num_classes)
        self.input_type = "mel"
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
    def __init__(self,input_dim=80,num_classes=1211,embedding_dim=192):
        super().__init__()
        self.conv1=nn.Conv1d(input_dim,512,5,padding=2);self.bn1=nn.BatchNorm1d(512)
        self.l1=SERes2NetBlock(512,3,2);self.l2=SERes2NetBlock(512,3,3);self.l3=SERes2NetBlock(512,3,4)
        self.conv2=nn.Conv1d(2048,1536,1);self.bn2=nn.BatchNorm1d(1536)
        self.pool=AttentiveStatsPooling(1536)
        self.fc=nn.Linear(3072,embedding_dim);self.bnfc=nn.BatchNorm1d(embedding_dim)
        self.classifier=nn.Linear(embedding_dim,num_classes)
        self.input_type = "mel"
    def extract_embedding(self,x):
        x0=F.relu(self.bn1(self.conv1(x)));x1=self.l1(x0);x2=self.l2(x1);x3=self.l3(x2)
        return self.bnfc(self.fc(self.pool(F.relu(self.bn2(self.conv2(torch.cat([x0,x1,x2,x3],1)))))))
    def forward(self,x): return self.classifier(F.relu(self.extract_embedding(x)))

# --- AUDIO PREPROCESSING ---
class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        f_min: int = 50,
        f_max: int = 7600,
        normalization: str = "cmvn"
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalization = normalization
    
    def __call__(self, audio_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception:
            y = np.zeros(int(self.sample_rate * self.duration), dtype=np.float32)
            sr = self.sample_rate

        # 1. Voice Activity Detection (VAD) - Trim silences under 30 dB
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        # 2. Force fixed duration (3 seconds = 48000 samples)
        max_samples = int(self.sample_rate * self.duration)
        if len(y_trimmed) > max_samples:
            y_trimmed = y_trimmed[:max_samples]
        elif len(y_trimmed) < max_samples:
            y_trimmed = np.pad(y_trimmed, (0, max_samples - len(y_trimmed)), mode='constant')

        # 3. Compute Mel Spectrogram exactly like PyTorch T.MelSpectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y_trimmed,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=400,
            hop_length=160,
            win_length=400,
            fmin=0.0,
            fmax=None
        )

        # 4. Natural logarithm
        log_mel = np.log(mel_spectrogram + 1e-6)

        # 5. Cepstral Mean Subtraction (CMS)
        log_mel = log_mel - np.mean(log_mel)

        return log_mel


def load_audio(
    audio_path: str,
    sr: int = 16000,
    duration: float = 3.0,
    mono: bool = True
) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, mono=mono)
    if duration is not None:
        max_samples = int(sr * duration)
        if len(y) > max_samples:
            y = y[:max_samples]
        elif len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)), mode='constant')
    return y


def add_noise(
    y: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 10.0
) -> np.ndarray:
    if len(noise) > len(y):
        noise = noise[:len(y)]
    elif len(noise) < len(y):
        noise = np.tile(noise, (len(y) // len(noise) + 1))[:len(y)]
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
    noisy_y = y + noise_scale * noise
    return noisy_y


def load_and_preprocess_audio(audio_source, model_input_type="mel", sample_rate=16000, duration=3.0, n_mels=80) -> torch.Tensor:
    """Loads audio from path or bytes, applies VAD, standardizes duration, and outputs the appropriate tensor."""
    # Load audio
    if isinstance(audio_source, bytes):
        y, sr = librosa.load(io.BytesIO(audio_source), sr=sample_rate, mono=True)
    else:
        y, sr = librosa.load(audio_source, sr=sample_rate, mono=True)
        
    # 1. Voice Activity Detection (VAD) - Trim silences under 30 dB
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    # 2. Force fixed duration (3 seconds = 48000 samples)
    max_samples = int(sample_rate * duration)
    if len(y_trimmed) > max_samples:
        y_trimmed = y_trimmed[:max_samples]
    elif len(y_trimmed) < max_samples:
        y_trimmed = np.pad(y_trimmed, (0, max_samples - len(y_trimmed)), mode='constant')
        
    if model_input_type == "waveform":
        # Returns raw audio waveform as (1, time_steps)
        return torch.FloatTensor(y_trimmed).unsqueeze(0)
    else:
        # Returns log-mel spectrogram as (1, n_mels, time_steps)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y_trimmed,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160,
            win_length=400,
            fmin=0.0,
            fmax=None
        )
        log_mel = np.log(mel_spectrogram + 1e-6)
        log_mel = log_mel - np.mean(log_mel)
        return torch.FloatTensor(log_mel).unsqueeze(0)


# --- DATA AUGMENTATION ---
class DataAugmenter:
    def __init__(self, sample_rate: int = 16000, musan_path: Optional[str] = None):
        self.sample_rate = sample_rate
        self.musan_path = musan_path if musan_path else "data/musan"
        
    def _load_musan_files(self, category: str) -> list:
        musan_dir = Path(self.musan_path) / category
        if not musan_dir.exists():
            return []
        files = sorted(musan_dir.glob("**/*.wav"))
        return [str(f) for f in files]
        
    def add_noise_augmentation(self, y: np.ndarray, snr_db: float = 10.0, category: str = "noise") -> np.ndarray:
        if self.musan_path is None or not Path(self.musan_path).exists():
            return y
        noise_files = self._load_musan_files(category)
        if not noise_files:
            return y
        noise_file = np.random.choice(noise_files)
        try:
            noise = load_audio(noise_file, sr=self.sample_rate, duration=None, mono=True)
            return add_noise(y, noise, snr_db=snr_db)
        except Exception as e:
            logger.warning(f"Error loading noise file {noise_file}: {e}")
            return y


# --- SPEAKER VERIFICATION UTILITIES ---
class SpeakerDatabase:
    def __init__(self):
        self.speakers = {}
    def enroll(self, speaker_id: str, embedding: np.ndarray) -> None:
        self.speakers[speaker_id] = embedding
    def remove(self, speaker_id: str) -> None:
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
    def get(self, speaker_id: str) -> Optional[np.ndarray]:
        return self.speakers.get(speaker_id)
    def list_speakers(self) -> list:
        return list(self.speakers.keys())
    def save(self, path: str) -> None:
        import json
        data = {sid: emb.tolist() for sid, emb in self.speakers.items()}
        with open(path, 'w') as f:
            json.dump(data, f)
    def load(self, path: str) -> None:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.speakers = {sid: np.array(emb) for sid, emb in data.items()}


class SpeakerVerificationInference:
    def __init__(self, model: nn.Module, device: str = "cpu", threshold: float = 0.75):
        self.model = model
        self.device = device
        self.threshold = threshold

    def verify(self, feature1: np.ndarray, feature2: np.ndarray, threshold: Optional[float] = None) -> tuple:
        if threshold is None:
            threshold = self.threshold
        
        # Convert to tensors
        feat1_tensor = torch.FloatTensor(feature1).unsqueeze(0).to(self.device)
        feat2_tensor = torch.FloatTensor(feature2).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            emb1 = self.model.extract_embedding(feat1_tensor)
            emb2 = self.model.extract_embedding(feat2_tensor)
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            similarity = torch.sum(emb1 * emb2, dim=1).item()
            
        is_same = similarity >= threshold
        return float(similarity), bool(is_same)

    def identify_speaker(self, test_features: np.ndarray, speaker_embeddings: dict, top_k: int = 1) -> list:
        test_tensor = torch.FloatTensor(test_features).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            test_emb = self.model.extract_embedding(test_tensor)
            test_emb = F.normalize(test_emb, p=2, dim=1).cpu().numpy()[0]
            
        similarities = {}
        for speaker_id, speaker_emb in speaker_embeddings.items():
            speaker_emb_norm = speaker_emb / (np.linalg.norm(speaker_emb) + 1e-10)
            similarity = np.dot(test_emb, speaker_emb_norm)
            similarities[speaker_id] = float(similarity)
            
        sorted_speakers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_speakers[:top_k]


# --- DYNAMIC CHECKPOINT LOADER ---
def load_checkpoint_and_model(model_key: str, checkpoint_path: Path, device: str):
    logger.info(f"Loading {model_key} checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Detect if it's a SpeechBrain wrapper model by checking state dict keys
    is_speechbrain = any(k.startswith("embedding_model.") or k.startswith("feat_extractor.") for k in state_dict.keys())
    
    input_dim = checkpoint.get("input_dim", 80)
    embedding_dim = checkpoint.get("embedding_dim", 512 if model_key == "xvector" else 192)
    num_classes = checkpoint.get("num_classes", 1211)
    
    if is_speechbrain:
        logger.info(f"Detected SpeechBrain checkpoint for {model_key}")
        try:
            import speechbrain
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                f"Checkpoint {checkpoint_path} requires the 'speechbrain' package to be installed. "
                "Please install it or use the standard non-finetuned checkpoint instead."
            )
            
        source_repo = "speechbrain/spkrec-xvect-voxceleb" if model_key == "xvector" else "speechbrain/spkrec-ecapa-voxceleb"
        logger.info(f"Loading base SpeechBrain model structure from {source_repo}...")
        classifier = EncoderClassifier.from_hparams(source=source_repo, run_opts={"device": device})
        
        if model_key == "xvector":
            class PretrainedXVectorWrapper(nn.Module):
                def __init__(self, classifier, num_classes=1211, embedding_dim=512):
                    super().__init__()
                    self.feat_extractor = classifier.mods.compute_features
                    self.mean_var_norm = classifier.mods.mean_var_norm
                    self.embedding_model = classifier.mods.embedding_model
                    self.fc = nn.Linear(embedding_dim, num_classes)
                    self.input_type = "waveform"
                def extract_embedding(self, wavs):
                    if wavs.dim() == 3:
                        wavs = wavs.squeeze(1)
                    wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
                    feats = self.feat_extractor(wavs)
                    feats = self.mean_var_norm(feats, wav_lens)
                    embeddings = self.embedding_model(feats, wav_lens)
                    return embeddings.squeeze(1)
            model = PretrainedXVectorWrapper(classifier, num_classes, embedding_dim)
        else:
            class PretrainedECAPAWrapper(nn.Module):
                def __init__(self, classifier, num_classes=1211, embedding_dim=192):
                    super().__init__()
                    self.feat_extractor = classifier.mods.compute_features
                    self.mean_var_norm = classifier.mods.mean_var_norm
                    self.embedding_model = classifier.mods.embedding_model
                    self.input_type = "waveform"
                def extract_embedding(self, wavs):
                    if wavs.dim() == 3:
                        wavs = wavs.squeeze(1)
                    wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
                    feats = self.feat_extractor(wavs)
                    feats = self.mean_var_norm(feats, wav_lens)
                    embeddings = self.embedding_model(feats, wav_lens)
                    return embeddings.squeeze(1)
            model = PretrainedECAPAWrapper(classifier, num_classes, embedding_dim)
            
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.info(f"Detected Custom scratch-trained checkpoint for {model_key}")
        if model_key == "xvector":
            model = XVectorModel(input_dim=input_dim, num_classes=num_classes, embedding_dim=embedding_dim)
        else:
            model = ECAPATDNNModel(input_dim=input_dim, num_classes=num_classes, embedding_dim=embedding_dim)
        model.input_type = "mel"
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model, checkpoint.get("optimal_threshold", 0.75)


# ====================================================================
# FASTAPI LIFECYCLE AND ENDPOINTS
# ====================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model, models, preprocessor, verifier, verifiers, speaker_db, config, model_type, device
    
    logger.info("Loading configuration...")
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        f_min=config['data']['f_min'],
        f_max=config['data']['f_max']
    )
    
    model_type = config['api'].get('model_type', 'ecapa_tdnn')

    def _resolve_checkpoint(model_key: str):
        explicit_key = f"{model_key}_checkpoint_path"
        explicit_value = config['api'].get(explicit_key)
        if explicit_value and Path(explicit_value).exists():
            return Path(explicit_value)

        for candidate in MODEL_CHECKPOINT_CANDIDATES.get(model_key, []):
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        return None

    for key in ("ecapa_tdnn", "xvector"):
        checkpoint_path = _resolve_checkpoint(key)

        if checkpoint_path is not None:
            try:
                loaded_model, optimal_th = load_checkpoint_and_model(key, checkpoint_path, device)
                logger.info(f"Loaded {key} checkpoint successfully! Optimal threshold: {optimal_th:.4f}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path} for {key}: {e}. Initializing with random weights.")
                loaded_model = XVectorModel(num_classes=1211) if key == "xvector" else ECAPATDNNModel(num_classes=1211)
                loaded_model.input_type = "mel"
                loaded_model.to(device)
                loaded_model.eval()
                optimal_th = config['api'].get('verification_threshold', 0.75)
        else:
            logger.warning(f"Checkpoint not found for {key}; using random weights")
            loaded_model = XVectorModel(num_classes=1211) if key == "xvector" else ECAPATDNNModel(num_classes=1211)
            loaded_model.input_type = "mel"
            loaded_model.to(device)
            loaded_model.eval()
            optimal_th = config['api'].get('verification_threshold', 0.75)

        models[key] = loaded_model
        verifiers[key] = SpeakerVerificationInference(
            model=loaded_model,
            device=device,
            threshold=optimal_th
        )

    model = models[model_type]
    verifier = verifiers[model_type]
    
    # Initialize speaker database
    speaker_db = SpeakerDatabase()
    speaker_db_path = config['api'].get('speaker_db_path', 'speaker_db.json')
    try:
        if Path(speaker_db_path).exists():
            speaker_db.load(speaker_db_path)
            logger.info(f"Loaded speaker DB from {speaker_db_path}")
    except Exception as e:
        logger.warning(f"Could not load speaker DB: {e}")

    logger.info("Models loaded successfully!")


def _extract_audio_features(audio_bytes: bytes) -> np.ndarray:
    """Load audio bytes and extract normalized mel features."""
    import tempfile
    audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=config['data']['sample_rate'], mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        import soundfile as sf
        sf.write(tmp.name, audio_array, config['data']['sample_rate'])
        tmp_path = tmp.name
    try:
        return preprocessor(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def _model_similarity(model_key: str, y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute similarity for a specific loaded model using raw waveforms."""
    model_input_type = models[model_key].input_type
    
    if model_input_type == "waveform":
        t1 = torch.FloatTensor(y1).unsqueeze(0).to(device)
        t2 = torch.FloatTensor(y2).unsqueeze(0).to(device)
    else:
        # Log mel extraction + CMS
        def to_mel(y):
            mel = librosa.feature.melspectrogram(
                y=y, sr=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400, fmin=0.0, fmax=None
            )
            log_mel = np.log(mel + 1e-6)
            return log_mel - np.mean(log_mel)
        t1 = torch.FloatTensor(to_mel(y1)).unsqueeze(0).to(device)
        t2 = torch.FloatTensor(to_mel(y2)).unsqueeze(0).to(device)
        
    models[model_key].eval()
    with torch.no_grad():
        emb1 = models[model_key].extract_embedding(t1)
        emb2 = models[model_key].extract_embedding(t2)
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        sim = torch.sum(emb1 * emb2, dim=1).item()
        
    return float(sim)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "model_type": model_type,
        "optimal_threshold": verifier.threshold if verifier else 0.75
    }


@app.post("/embed")
async def extract_embedding(file: UploadFile = File(...)) -> EmbeddingResponse:
    """Extract speaker embedding from audio file."""
    try:
        audio_data = await file.read()
        t = load_and_preprocess_audio(audio_data, model_input_type=model.input_type).to(device)
        
        model.eval()
        with torch.no_grad():
            emb = model.extract_embedding(t)
            emb = F.normalize(emb, p=2, dim=1).cpu().numpy()[0]
            
        return EmbeddingResponse(
            speaker_id=file.filename,
            embedding=emb.tolist(),
            success=True
        )
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_speaker(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: Optional[float] = None,
    model_type: Optional[str] = None
) -> VerifyResponse:
    """Verify if two audio files are from the same speaker."""
    selected_model_type = model_type if model_type else config['api'].get('model_type', 'ecapa_tdnn')
    if selected_model_type not in models:
        raise HTTPException(status_code=400, detail=f"Model type {selected_model_type} not loaded")
        
    if threshold is None:
        threshold = verifiers[selected_model_type].threshold
        
    try:
        audio_data1 = await file1.read()
        audio_data2 = await file2.read()
        
        model_input_type = models[selected_model_type].input_type
        t1 = load_and_preprocess_audio(audio_data1, model_input_type=model_input_type).to(device)
        t2 = load_and_preprocess_audio(audio_data2, model_input_type=model_input_type).to(device)
        
        current_model = models[selected_model_type]
        current_model.eval()
        with torch.no_grad():
            emb1 = current_model.extract_embedding(t1)
            emb2 = current_model.extract_embedding(t2)
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            similarity = torch.sum(emb1 * emb2, dim=1).item()
            
        is_same = similarity >= threshold
        
        return VerifyResponse(
            similarity_score=float(similarity),
            is_same_speaker=bool(is_same),
            threshold_used=threshold
        )
    except Exception as e:
        logger.error(f"Error verifying speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/augment")
async def augment_audio(
    file: UploadFile = File(...),
    category: str = "noise",
    snr_db: float = 10.0
) -> StreamingResponse:
    """Return an augmented audio (WAV) using white noise or MUSAN augmentation."""
    try:
        audio_data = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=config['data']['sample_rate'], mono=True)

        augmenter = DataAugmenter(sample_rate=config['data']['sample_rate'], musan_path=config['augmentation']['musan_path'])
        augmented = augmenter.add_noise_augmentation(y, snr_db=snr_db, category=category)

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, augmented, config['data']['sample_rate'], format='WAV')
        buf.seek(0)
        return StreamingResponse(buf, media_type='audio/wav')
    except Exception as e:
        logger.error(f"Error augmenting audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/augmentation-categories")
async def augmentation_categories():
    """List available MUSAN categories and sample files."""
    try:
        musan_path = config['augmentation'].get('musan_path', 'data/musan')
        p = Path(musan_path)
        if not p.exists():
            return {"musan_path": str(p), "categories": []}

        cats = [d.name for d in p.iterdir() if d.is_dir()]
        samples = {}
        for c in cats:
            cpath = p / c
            files = sorted([str(f.relative_to(p)) for f in cpath.rglob("*.wav")])
            samples[c] = files[:10]

        return {"musan_path": str(p), "categories": cats, "samples": samples}
    except Exception as e:
        logger.error(f"Error listing augmentation categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/augment-with-file")
async def augment_with_file(
    file: UploadFile = File(...),
    noise_file: str = Form(...),
    snr_db: float = Form(10.0)
) -> StreamingResponse:
    """Augment uploaded audio using a specific MUSAN file (relative to musan_path)."""
    try:
        musan_path = Path(config['augmentation'].get('musan_path', 'data/musan'))
        candidate = musan_path.joinpath(noise_file).resolve()
        if not str(candidate).startswith(str(musan_path.resolve())) or not candidate.exists():
            raise HTTPException(status_code=400, detail="Invalid noise_file")

        audio_bytes = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=config['data']['sample_rate'], mono=True)
        noise = load_audio(str(candidate), sr=config['data']['sample_rate'], duration=None, mono=True)
        augmented = add_noise(y, noise, snr_db=float(snr_db))

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, augmented, config['data']['sample_rate'], format='WAV')
        buf.seek(0)
        return StreamingResponse(buf, media_type='audio/wav')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error augmenting with file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    categories: Optional[str] = None
) -> dict:
    """Generate a robustness report for the uploaded audio."""
    try:
        audio_data = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        
        max_samples = 48000
        if len(y_trimmed) > max_samples:
            y_trimmed = y_trimmed[:max_samples]
        elif len(y_trimmed) < max_samples:
            y_trimmed = np.pad(y_trimmed, (0, max_samples - len(y_trimmed)), mode='constant')
            
        model_input_type = model.input_type
        if model_input_type == "waveform":
            t_orig = torch.FloatTensor(y_trimmed).unsqueeze(0).to(device)
        else:
            def to_mel(y):
                mel = librosa.feature.melspectrogram(
                    y=y, sr=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400, fmin=0.0, fmax=None
                )
                log_mel = np.log(mel + 1e-6)
                return log_mel - np.mean(log_mel)
            t_orig = torch.FloatTensor(to_mel(y_trimmed)).unsqueeze(0).to(device)
            
        model.eval()
        with torch.no_grad():
            orig_emb = model.extract_embedding(t_orig)
            orig_emb = F.normalize(orig_emb, p=2, dim=1).cpu().numpy()[0]
            
        if categories:
            cats = [c.strip() for c in categories.split(',') if c.strip()]
        else:
            cats = ['noise', 'music', 'babble']
            
        snr_list = config['augmentation'].get('noise_snr', [20, 10, 5])
        augmenter = DataAugmenter(sample_rate=16000, musan_path=config['augmentation']['musan_path'])
        
        similarities = []
        generated = 0
        
        for cat in cats:
            for snr in snr_list:
                aug_y = augmenter.add_noise_augmentation(y_trimmed, snr_db=snr, category=cat)
                
                if len(aug_y) > max_samples:
                    aug_y = aug_y[:max_samples]
                elif len(aug_y) < max_samples:
                    aug_y = np.pad(aug_y, (0, max_samples - len(aug_y)), mode='constant')
                    
                if model_input_type == "waveform":
                    t_aug = torch.FloatTensor(aug_y).unsqueeze(0).to(device)
                else:
                    t_aug = torch.FloatTensor(to_mel(aug_y)).unsqueeze(0).to(device)
                    
                with torch.no_grad():
                    aug_emb = model.extract_embedding(t_aug)
                    aug_emb = F.normalize(aug_emb, p=2, dim=1).cpu().numpy()[0]
                    
                sim = float(np.dot(orig_emb, aug_emb))
                similarities.append({'category': cat, 'snr': snr, 'similarity': sim})
                generated += 1
                
        sims = [s['similarity'] for s in similarities]
        avg = float(np.mean(sims)) if sims else 0.0
        mx = float(np.max(sims)) if sims else 0.0
        mn = float(np.min(sims)) if sims else 0.0
        voice_quality_score = int(np.clip((avg + 1) / 2 * 100, 0, 100))
        
        return {
            'voice_quality_score': voice_quality_score,
            'best_model': config['api'].get('model_type', 'ecapa_tdnn'),
            'average_model_score': avg,
            'max_score': mx,
            'min_score': mn,
            'generated_samples': generated,
            'per_sample': similarities
        }
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-models", response_model=ModelComparisonResponse)
async def compare_models(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
) -> ModelComparisonResponse:
    """Compare ECAPA-TDNN and X-Vector using one uploaded pair and SNR sweeps."""
    try:
        audio_bytes1 = await file1.read()
        audio_bytes2 = await file2.read()
        
        y1, _ = librosa.load(io.BytesIO(audio_bytes1), sr=16000, mono=True)
        y1_trimmed, _ = librosa.effects.trim(y1, top_db=30)
        
        y2, _ = librosa.load(io.BytesIO(audio_bytes2), sr=16000, mono=True)
        y2_trimmed, _ = librosa.effects.trim(y2, top_db=30)
        
        max_samples = 48000
        if len(y1_trimmed) > max_samples:
            y1_trimmed = y1_trimmed[:max_samples]
        elif len(y1_trimmed) < max_samples:
            y1_trimmed = np.pad(y1_trimmed, (0, max_samples - len(y1_trimmed)), mode='constant')
            
        if len(y2_trimmed) > max_samples:
            y2_trimmed = y2_trimmed[:max_samples]
        elif len(y2_trimmed) < max_samples:
            y2_trimmed = np.pad(y2_trimmed, (0, max_samples - len(y2_trimmed)), mode='constant')
            
        snr_levels = config['augmentation'].get('noise_snr', [20, 15, 10, 5])
        augmenter = DataAugmenter(
            sample_rate=16000,
            musan_path=config['augmentation']['musan_path']
        )
        
        comparison_payload = {}
        original_scores = {}
        
        for key in ("ecapa_tdnn", "xvector"):
            if key not in models:
                continue
            original_scores[key] = _model_similarity(key, y1_trimmed, y2_trimmed)
            snr_curve = []
            
            for snr in snr_levels:
                noisy_y2 = augmenter.add_noise_augmentation(y2_trimmed, snr_db=float(snr), category="noise")
                if len(noisy_y2) > max_samples:
                    noisy_y2 = noisy_y2[:max_samples]
                elif len(noisy_y2) < max_samples:
                    noisy_y2 = np.pad(noisy_y2, (0, max_samples - len(noisy_y2)), mode='constant')
                    
                snr_curve.append({
                    "snr_db": float(snr),
                    "similarity": _model_similarity(key, y1_trimmed, noisy_y2)
                })
                
            comparison_payload[key] = {
                "original_similarity": original_scores[key],
                "optimal_threshold": verifiers[key].threshold,
                "snr_curve": snr_curve,
            }
            
        best_model = max(original_scores.items(), key=lambda item: item[1])[0] if original_scores else "none"
        
        return ModelComparisonResponse(
            reference_file=file1.filename or "reference.wav",
            comparison_file=file2.filename or "comparison.wav",
            best_model=best_model,
            models=comparison_payload,
        )
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll")
async def enroll_speaker(
    speaker_id: str,
    file: UploadFile = File(...)
) -> dict:
    """Enroll a new speaker."""
    try:
        audio_data = await file.read()
        t = load_and_preprocess_audio(audio_data, model_input_type=model.input_type).to(device)
        
        model.eval()
        with torch.no_grad():
            emb = model.extract_embedding(t)
            emb = F.normalize(emb, p=2, dim=1).cpu().numpy()[0]
            
        speaker_db.enroll(speaker_id, emb)
        
        return {
            "status": "success",
            "speaker_id": speaker_id,
            "message": f"Speaker {speaker_id} enrolled successfully"
        }
    except Exception as e:
        logger.error(f"Error enrolling speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify")
async def identify_speaker(
    file: UploadFile = File(...),
    top_k: int = 3
) -> IdentifyResponse:
    """Identify speaker from audio file."""
    try:
        if not speaker_db.list_speakers():
            raise HTTPException(
                status_code=400,
                detail="No speakers enrolled in database"
            )
        
        audio_data = await file.read()
        t = load_and_preprocess_audio(audio_data, model_input_type=model.input_type).to(device)
        
        model.eval()
        with torch.no_grad():
            test_emb = model.extract_embedding(t)
            test_emb = F.normalize(test_emb, p=2, dim=1).cpu().numpy()[0]
            
        similarities = {}
        for speaker_id in speaker_db.list_speakers():
            speaker_emb = speaker_db.get(speaker_id)
            speaker_emb_norm = speaker_emb / (np.linalg.norm(speaker_emb) + 1e-10)
            sim = np.dot(test_emb, speaker_emb_norm)
            similarities[speaker_id] = float(sim)
            
        sorted_speakers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        matches = sorted_speakers[:top_k]
        
        return IdentifyResponse(
            top_matches=[
                {"speaker_id": sid, "similarity": float(sim)}
                for sid, sim in matches
            ],
            success=True
        )
    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers")
async def list_speakers():
    """List all enrolled speakers."""
    return {"speakers": speaker_db.list_speakers()}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Delete speaker enrollment."""
    if speaker_id not in speaker_db.list_speakers():
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    speaker_db.remove(speaker_id)
    return {"status": "success", "message": f"Speaker {speaker_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    # Load settings directly to launch local process if executed as script
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    uvicorn.run(
        "api.app:app",
        host=cfg['api']['host'],
        port=cfg['api']['port'],
        workers=cfg['api']['workers']
    )
