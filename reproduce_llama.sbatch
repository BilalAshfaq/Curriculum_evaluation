#!/bin/bash
# Submit from the repository root:
#   mkdir -p slurm-logs
#   sbatch reproduce_llama.sbatch
# Optional overrides, for example:
#   sbatch --gres=gpu:2 --export=ALL,NUM_GPUS=2 reproduce_llama.sbatch
# To use an existing ordered dataset, export DATASET_PATH before submission.
#SBATCH --job-name=curriculum-llama-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:10
#SBATCH --account=eduard
#SBATCH --qos=normal
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-logs/curriculum-llama-repro-%j.out
#SBATCH --error=slurm-logs/curriculum-llama-repro-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
CONDA_ENV="${CONDA_ENV:-curr-eval-env}"
CONDA_SH="${CONDA_SH:-/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-llama-repro-seed${SEED}-${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/results/reproduction/$RUN_NAME}"
# Paper Table 6 settings; the original per-run GPU count remains unconfirmed.
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_GPUS="${NUM_GPUS:-1}"
case "$NUM_GPUS" in 1|2) ;; *) echo "NUM_GPUS must be 1 or 2"; exit 1 ;; esac
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-$((8 / NUM_GPUS))}"
VALID_NUM="${VALID_NUM:-500}"
# Retain the training script's sequence-length default pending confirmation.
MAX_LENGTH="${MAX_LENGTH:-512}"

cd "$REPO_DIR"
# Conda compiler hooks may read unset variables while switching environments.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/shared/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
# Dataset preparation needs a writable cache separate from the shared model cache.
export HF_DATASETS_CACHE="$REPO_DIR/.cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"
# Avoid an interactive login on the compute node. Override with WANDB_MODE=online.
export WANDB_MODE="${WANDB_MODE:-offline}"

# A fresh directory prevents accidentally mixing checkpoints from different jobs.
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
export WANDB_DIR="$RUN_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_DIR"
echo "Run directory: $RUN_DIR"
echo "Started: $(date --iso-8601=seconds)"
echo "Conda environment: $CONDA_ENV"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUMULATION))"
echo "Scheduler: Trainer cosine decay with 10% warmup (paper Table 6)."
echo "DeepSpeed scheduler override removed; verify the corrected trial before full training."

python - <<'PY'
import os
import torch
num_gpus = int(os.environ["NUM_GPUS"])
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA devices:", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() != num_gpus:
    raise RuntimeError(f"This job requires exactly {num_gpus} visible CUDA GPU(s)")
for index in range(num_gpus):
    with torch.cuda.device(index):
        print(index, torch.cuda.get_device_name(index),
              "memory GiB:", torch.cuda.get_device_properties(index).total_memory / 2**30,
              "bfloat16:", torch.cuda.is_bf16_supported())
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The training script requires bfloat16 support")
        # Exercise a CUDA operation as well as checking device visibility.
        x = torch.ones((32, 32), device=f"cuda:{index}", dtype=torch.bfloat16)
        (x @ x).sum().item()
PY

python -m pip freeze > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
git status --short > "$RUN_DIR/git-status.txt"
git diff > "$RUN_DIR/source-changes.patch"
cp "$REPO_DIR/reproduce_llama.sbatch" "$RUN_DIR/submitted-script.sbatch"

# Preserve the authors' unseeded baseline-generation behavior and retain its output.
# Reuse DATASET_PATH for subsequent jobs that must use the exact same ordering.
if [[ -z "${DATASET_PATH:-}" ]]; then
    python metrics/SpilitByMetric/random_baseline.py \
        --input_file "$REPO_DIR/dataset/MetaMathQA-20K_train.jsonl" \
        --output_dir "$RUN_DIR/data"
    DATASET_PATH="$RUN_DIR/data/MetaMathQA-20K_train_baseline_full_shuffled.jsonl"
fi
[[ -f "$DATASET_PATH" ]] || { echo "ERROR: Dataset missing: $DATASET_PATH"; exit 1; }
DATASET_PATH="$(realpath "$DATASET_PATH")"
sha256sum "$DATASET_PATH" > "$RUN_DIR/dataset.sha256"

# Copy the launch configuration and DeepSpeed JSON for this job. Only the machine
# path and process count change here. The source JSON now delegates scheduling
# to the Trainer so cosine decay and 10% warmup match the paper.
python - "$REPO_DIR" "$RUN_DIR" <<'PY'
import os
import pathlib
import shutil
import sys
import yaml

repo, run = map(pathlib.Path, sys.argv[1:])
source = repo / "train_base" / "accelerate_config"
config = yaml.safe_load((source / "train_config.yaml").read_text())
shutil.copyfile(source / "ds_config.json", run / "ds_config.json")
config["deepspeed_config"]["deepspeed_config_file"] = str(run / "ds_config.json")
config["num_processes"] = int(os.environ["NUM_GPUS"])
(run / "accelerate.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY

COMMAND=(
    accelerate launch --config_file "$RUN_DIR/accelerate.yaml"
    "$REPO_DIR/train_base/train_llama3_8b_lora.py"
    --model_name "$MODEL_NAME"
    --dataset_path "$DATASET_PATH"
    --output_dir "$RUN_DIR/adapter"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --batch_size "$BATCH_SIZE"
    --grad_accumulation "$GRAD_ACCUMULATION"
    --valid_num "$VALID_NUM"
    --max_length "$MAX_LENGTH"
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05
    --gradient_checkpointing
    --wandb_project curriculum-reproduction
    --wandb_run_name "$RUN_NAME"
)
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/train.log"

echo "Finished: $(date --iso-8601=seconds)"
echo "Adapter: $RUN_DIR/adapter"
echo "Run configuration, generated data, and training log: $RUN_DIR"
echo "Benchmark evaluation is a separate step."
