# Quick Start Guide - Speaker Verification System

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
cd speaker_verification
pip install -r requirements.txt
```

### 2. Download Sample Data

For testing purposes, you can use a subset of VoxCeleb or create dummy data.

### 3. Configure System

Edit `config/config.yaml` to match your setup:

```yaml
data:
  raw_path: "./data/raw"
  sample_rate: 16000
  
training:
  batch_size: 32
  learning_rate: 0.01
```

### 4. Start Training

```bash
cd scripts
python train.py --model ecapa_tdnn --epochs 10
```

### 5. Run Inference

```bash
python infer.py \
    --mode verify \
    --checkpoint ../checkpoints/ecapa_tdnn/best_model.pt \
    --audio1 ../data/raw/sample1.wav \
    --audio2 ../data/raw/sample2.wav
```

### 6. Start API

```bash
cd ..
python -m uvicorn api.app:app --reload
```

Open: http://localhost:8000/docs

---

## 📋 Checklist

- [ ] Installed Python 3.8+
- [ ] Installed dependencies from requirements.txt
- [ ] Downloaded VoxCeleb dataset (optional)
- [ ] Downloaded MUSAN dataset for augmentation (optional)
- [ ] Updated config.yaml with your paths
- [ ] Ran training script successfully
- [ ] API is responding at localhost:8000

## 🔗 Useful Links

- **VoxCeleb Dataset**: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **MUSAN Dataset**: https://www.openslr.org/17/
- **FastAPI Docs**: http://localhost:8000/docs

## 🆘 Troubleshooting

### CUDA not available?
```bash
# CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Out of memory?
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16
```

### Data not found?
```bash
# Check data path
ls -la data/raw/
```

## 📞 Support

For more help, see README.md in the project root.
