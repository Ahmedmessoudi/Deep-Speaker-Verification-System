param(
    [string]$Model = "ecapa_tdnn",
    [string]$Config = "config/config.yaml",
    [string]$DataRoot = "data/raw",
    [string]$CheckpointDir = "checkpoints",
    [int]$Epochs = 3,
    [int]$BatchSize = 4,
    [double]$LearningRate = 0.001,
    [switch]$SkipInstall,
    [switch]$CpuOnly,
    [switch]$Resume,
    [string]$ResumePath = "",
    [string]$TestPairs = "data/test_pairs.txt",
    [string]$Audio1 = "",
    [string]$Audio2 = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    python -m venv venv
}

. .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

if (-not $SkipInstall) {
    pip install -r requirements.txt
    if (-not $CpuOnly) {
        pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade
    }
}

if ($CpuOnly) {
    $env:CUDA_VISIBLE_DEVICES = ""
}

if (-not (Test-Path $DataRoot)) {
    Write-Error "Data root not found: $DataRoot"
    exit 1
}

$resumeArg = @()
if ($Resume) {
    if ($ResumePath -and (Test-Path $ResumePath)) {
        $resumeArg = @("--resume", $ResumePath)
    } else {
        $modelDir = Join-Path $CheckpointDir $Model
        if (Test-Path $modelDir) {
            $latest = Get-ChildItem $modelDir -Filter "checkpoint_epoch_*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $best = Join-Path $modelDir "best_model.pt"
            if (Test-Path $best) {
                $resumeArg = @("--resume", $best)
            } elseif ($latest) {
                $resumeArg = @("--resume", $latest.FullName)
            }
        }
    }
}

python scripts\train.py --config $Config --model $Model --data_root $DataRoot --output_dir $CheckpointDir --epochs $Epochs --batch_size $BatchSize --learning_rate $LearningRate @resumeArg

$checkpointPath = Join-Path $CheckpointDir "$Model\best_model.pt"

if (Test-Path $checkpointPath) {
    if (Test-Path $TestPairs) {
        python scripts\evaluate.py --config $Config --model $Model --checkpoint $checkpointPath --test_data $TestPairs
    } else {
        Write-Warning "Test pairs file not found: $TestPairs"
    }
} else {
    Write-Warning "Checkpoint not found: $checkpointPath"
}

if ($Audio1 -and $Audio2 -and (Test-Path $Audio1) -and (Test-Path $Audio2)) {
    python scripts\infer.py --config $Config --model $Model --checkpoint $checkpointPath --mode verify --audio1 $Audio1 --audio2 $Audio2
}
