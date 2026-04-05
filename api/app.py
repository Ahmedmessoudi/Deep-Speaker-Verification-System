"""FastAPI service for speaker verification."""

import io
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.data import AudioPreprocessor
from src.models import ECAPATDNN, XVector
from src.inference import SpeakerDatabase, SpeakerVerificationInference


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
preprocessor = None
verifier = None
speaker_db = None
config = None


class VerifyRequest(BaseModel):
    """Request model for speaker verification."""
    threshold: Optional[float] = 0.5


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


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model, preprocessor, verifier, speaker_db, config
    
    logger.info("Loading configuration...")
    
    # Load configuration
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing models...")
    
    # Get device
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
    
    # Load model
    model_type = config['api'].get('model_type', 'ecapa_tdnn')
    
    if model_type == "ecapa_tdnn":
        model = ECAPATDNN(
            input_dim=config['model']['ecapa_tdnn']['input_dim'],
            num_channels=config['model']['ecapa_tdnn']['num_channels'],
            num_speakers=config['model']['ecapa_tdnn']['num_speakers'],
            embeddings_dim=config['model']['ecapa_tdnn']['embeddings_dim']
        )
    else:
        model = XVector(
            input_dim=config['model']['xvector']['input_dim'],
            tdnn_dim=config['model']['xvector']['tdnn_dim'],
            num_speakers=config['model']['xvector']['num_speakers'],
            embeddings_dim=config['model']['xvector']['embeddings_dim']
        )
    
    model.to(device)
    model.eval()
    
    # Initialize verifier
    verifier = SpeakerVerificationInference(
        model=model,
        device=device,
        threshold=0.5
    )
    
    # Initialize speaker database
    speaker_db = SpeakerDatabase()
    
    logger.info("Models loaded successfully!")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/embed")
async def extract_embedding(file: UploadFile = File(...)) -> EmbeddingResponse:
    """
    Extract speaker embedding from audio file.
    
    Args:
        file: Audio file
        
    Returns:
        Embedding vector
    """
    try:
        # Read audio file
        audio_data = await file.read()
        audio_array, sr = librosa.load(
            io.BytesIO(audio_data),
            sr=config['data']['sample_rate'],
            mono=True
        )
        
        # Save temporarily to extract features
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, config['data']['sample_rate'])
            tmp_path = tmp.name
        
        # Extract features
        features = preprocessor(tmp_path)
        
        # Extract embedding
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        embedding = model.extract_embedding(features_tensor)
        embedding = embedding.detach().cpu().numpy()[0]
        
        # Clean up
        Path(tmp_path).unlink()
        
        return EmbeddingResponse(
            speaker_id=file.filename,
            embedding=embedding.tolist(),
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_speaker(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: Optional[float] = 0.5
) -> VerifyResponse:
    """
    Verify if two audio files are from the same speaker.
    
    Args:
        file1: First audio file
        file2: Second audio file
        threshold: Decision threshold
        
    Returns:
        Verification result
    """
    try:
        # Process first audio
        audio_data1 = await file1.read()
        audio_array1, sr1 = librosa.load(
            io.BytesIO(audio_data1),
            sr=config['data']['sample_rate'],
            mono=True
        )
        
        # Process second audio
        audio_data2 = await file2.read()
        audio_array2, sr2 = librosa.load(
            io.BytesIO(audio_data2),
            sr=config['data']['sample_rate'],
            mono=True
        )
        
        # Extract features using temporary files
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
            sf.write(tmp1.name, audio_array1, config['data']['sample_rate'])
            tmp_path1 = tmp1.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            sf.write(tmp2.name, audio_array2, config['data']['sample_rate'])
            tmp_path2 = tmp2.name
        
        features1 = preprocessor(tmp_path1)
        features2 = preprocessor(tmp_path2)
        
        # Verify
        feat1_tensor = torch.FloatTensor(features1).unsqueeze(0)
        feat2_tensor = torch.FloatTensor(features2).unsqueeze(0)
        
        similarity, is_same = verifier.verify(feat1_tensor, feat2_tensor, threshold)
        
        # Clean up
        Path(tmp_path1).unlink()
        Path(tmp_path2).unlink()
        
        return VerifyResponse(
            similarity_score=float(similarity),
            is_same_speaker=bool(is_same),
            threshold_used=threshold
        )
    
    except Exception as e:
        logger.error(f"Error verifying speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll")
async def enroll_speaker(
    speaker_id: str,
    file: UploadFile = File(...)
) -> dict:
    """
    Enroll a new speaker.
    
    Args:
        speaker_id: Speaker identifier
        file: Audio file
        
    Returns:
        Enrollment result
    """
    try:
        # Read audio
        audio_data = await file.read()
        audio_array, sr = librosa.load(
            io.BytesIO(audio_data),
            sr=config['data']['sample_rate'],
            mono=True
        )
        
        # Extract features
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, config['data']['sample_rate'])
            tmp_path = tmp.name
        
        features = preprocessor(tmp_path)
        
        # Extract embedding
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        embedding = model.extract_embedding(features_tensor)
        embedding = embedding.detach().cpu().numpy()[0]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        # Store in database
        speaker_db.enroll(speaker_id, embedding)
        
        # Clean up
        Path(tmp_path).unlink()
        
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
    """
    Identify speaker from audio file.
    
    Args:
        file: Audio file
        top_k: Number of top matches
        
    Returns:
        Identification results
    """
    try:
        if not speaker_db.list_speakers():
            raise HTTPException(
                status_code=400,
                detail="No speakers enrolled in database"
            )
        
        # Read audio
        audio_data = await file.read()
        audio_array, sr = librosa.load(
            io.BytesIO(audio_data),
            sr=config['data']['sample_rate'],
            mono=True
        )
        
        # Extract features
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, config['data']['sample_rate'])
            tmp_path = tmp.name
        
        features = preprocessor(tmp_path)
        
        # Get speaker embeddings
        speaker_embeddings = {sid: speaker_db.get(sid) for sid in speaker_db.list_speakers()}
        
        # Identify
        matches = verifier.identify_speaker(features, speaker_embeddings, top_k)
        
        # Clean up
        Path(tmp_path).unlink()
        
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
    """
    List all enrolled speakers.
    
    Returns:
        List of speaker IDs
    """
    return {"speakers": speaker_db.list_speakers()}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """
    Delete speaker enrollment.
    
    Args:
        speaker_id: Speaker identifier
        
    Returns:
        Deletion result
    """
    if speaker_id not in speaker_db.list_speakers():
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    speaker_db.remove(speaker_id)
    
    return {"status": "success", "message": f"Speaker {speaker_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers']
    )
