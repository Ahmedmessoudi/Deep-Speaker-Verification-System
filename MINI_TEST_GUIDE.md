# Testing the Speaker Verification System Locally (Mini Mode)

This guide explains how to test the Speaker Verification System on your local machine using a small subset of the data (100 voice elements), without extracting the massive ~22GB zip files or modifying the production configuration.

## Prerequisites

Make sure you have your Python environment activated and dependencies installed:

```bash
# If using a virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

> [!IMPORTANT]
> To enable data augmentation during preparation, you will need `librosa` and `soundfile`. If you saw a warning about this in the output, install them:
> ```bash
> pip install librosa soundfile
> ```

---

## Step 1: Prepare the Mini Dataset

We have created a dedicated script that peers inside `Voxceleb.zip` and `Musan.zip`, extracts exactly 100 audio files from 10 speakers, extracts a few noise files, and creates noise-augmented copies—all while leaving the original zips untouched.

Run this command from the project root:

```bash
python scripts/prepare_mini_dataset.py
```

**Expected Result:**
- A new folder `data/raw_mini/` is created containing the 100 extracted VoxCeleb `.wav` files split into `train`, `val`, and `test`.
- A new folder `data/musan_mini/` is created containing a small subset of noise files.
- A new folder `data/augmented_mini/` is created containing noise-augmented versions of the training data (if `librosa` is installed).
- Total disk space used is very small (~50 MB).

---

## Step 2: Train on the Mini Dataset

We have created a dedicated training script (`train_mini.py`) and a corresponding config (`config_mini.yaml`) specifically for this test mode. It automatically points to the `raw_mini` folders, uses a smaller batch size, defaults to CPU (if CUDA isn't available), and adjusts the model architecture to match the 10-speaker test set.

Run the mini training script:

```bash
python scripts/train_mini.py --model ecapa_tdnn --epochs 5
```

**What this does:**
1. Loads settings from `config/config_mini.yaml`.
2. Reads the 100 audio files from `data/raw_mini`.
3. Auto-detects that there are only 10 speakers and adjusts the model's output layer accordingly.
4. Trains the `ecapa_tdnn` model for 5 epochs.
5. Saves the training history and checkpoints to a new folder called `checkpoints_mini/`.

> [!TIP]
> You can easily switch the model to the baseline X-Vector by changing the `--model` argument:
> ```bash
> python scripts/train_mini.py --model xvector --epochs 5
> ```

---

## Step 3: Run Inference (Optional)

Once the mini training finishes, you can test the trained model by verifying if two audio files belong to the same speaker.

```bash
python scripts/infer.py \
    --mode verify \
    --checkpoint checkpoints_mini/ecapa_tdnn/best_model.pt \
    --audio1 data/raw_mini/test/wav/id10271/session0/00001.wav \
    --audio2 data/raw_mini/test/wav/id10271/session0/00002.wav
```
*(Note: Replace the `id10271` paths with actual file paths printed out during the `prepare_mini_dataset.py` step, as the 10 speakers are chosen randomly).*

---

## Moving to Production

When you move to the dedicated machine and want to run the **full dataset** (~6,000 speakers, 1.2 million files):

1. Fully extract `Voxceleb.zip` into `data/raw/`.
2. Fully extract `Musan.zip` into `data/musan/`.
3. Run the original, untouched scripts:
   ```bash
   python scripts/train.py --model ecapa_tdnn --epochs 100
   ```

Because we separated the "mini" mode into its own script and config, **your production workflow remains completely pristine and untouched!**
