# Pipeline Guide

This guide explains the lightweight training pipeline, how to resume from checkpoints, and how to compare X-Vector vs ECAPA-TDNN.

## What was added

- A Windows pipeline script: run train, evaluate, and optional inference in one command.
- A Linux pipeline script: same flow as Windows.
- Resume-from-checkpoint support in the training script.
- A comparison script to evaluate both models on the same test pairs.

## Data location

- The pipeline uses ./data by default. Keep your dataset under ./data/raw and test pairs under ./data/test_pairs.txt.

## Windows pipeline

Run training + evaluation for one model:

```powershell
.\run_pipeline.ps1 -Model ecapa_tdnn -Epochs 3 -BatchSize 4
```

Resume from the latest/best checkpoint:

```powershell
.\run_pipeline.ps1 -Model ecapa_tdnn -Resume
```

Resume from a specific checkpoint:

```powershell
.\run_pipeline.ps1 -Model ecapa_tdnn -Resume -ResumePath checkpoints\ecapa_tdnn\checkpoint_epoch_3.pt
```

Force CPU:

```powershell
.\run_pipeline.ps1 -Model xvector -CpuOnly
```

## Linux pipeline

Run training + evaluation for one model:

```bash
./run_pipeline.sh
```

Resume from the latest/best checkpoint:

```bash
RESUME=1 ./run_pipeline.sh
```

Resume from a specific checkpoint:

```bash
RESUME=1 RESUME_PATH=checkpoints/ecapa_tdnn/checkpoint_epoch_3.pt ./run_pipeline.sh
```

Force CPU:

```bash
CPU_ONLY=1 ./run_pipeline.sh
```

## Compare the two models

This uses existing checkpoints and the same test pairs file.

```bash
python scripts/compare_models.py --test_data data/test_pairs.txt
```

Optional: train both models first (light settings):

```bash
python scripts/compare_models.py --auto_train --epochs 3 --batch_size 4
```

## Notes for low pressure on your machine

- Defaults are set to small batch sizes and few epochs.
- Use X-Vector for faster experiments; ECAPA-TDNN is heavier.
- If you hit GPU memory errors, reduce batch size first.
