#!/bin/bash
# From the repository root: mkdir -p slurm-logs && sbatch reproduction_trial.sbatch
# Uses the existing curr-eval-env Conda environment; does not install dependencies.
# This diagnostic run is not a reproduction result: 50 updates, 256 input
# examples, 16 validation examples, and 8 evaluation examples per benchmark.
#SBATCH --job-name=curriculum-trial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --account=eduard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-logs/curriculum-trial-%j.out
#SBATCH --error=slurm-logs/curriculum-trial-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONDA_SH="${CONDA_SH:-/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-curr-eval-env}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/results/reproduction/trial-${SLURM_JOB_ID}}"
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
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
export WANDB_DIR="$RUN_DIR"
echo "Job: $SLURM_JOB_ID; node: $(hostname); started: $(date --iso-8601=seconds)"
echo "Trial artifacts: $RUN_DIR"
python -m pip freeze > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
git status --short > "$RUN_DIR/git-status.txt"
git diff > "$RUN_DIR/source-changes.patch"
cp "$REPO_DIR/reproduction_trial.sbatch" "$RUN_DIR/submitted-script.sbatch"
cp "$REPO_DIR/scripts/reproduction_trial.py" "$RUN_DIR/trial-runner.py"

# Check actual CUDA operations and imports before fetching model weights.
python - <<'PY'
import importlib
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("Exactly two allocated CUDA GPUs are required")
for i in range(2):
    with torch.cuda.device(i):
        print("GPU", i, torch.cuda.get_device_name(i),
              "GiB:", torch.cuda.get_device_properties(i).total_memory / 2**30)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required")
        x = torch.ones((32, 32), device=f"cuda:{i}", dtype=torch.bfloat16)
        print("CUDA matmul:", (x @ x).sum().item())
for name in ("transformers", "accelerate", "datasets", "peft", "deepspeed", "flash_attn", "wandb", "yaml"):
    package = importlib.import_module(name)
    print(name, getattr(package, "__version__", "import OK"))
PY

python - "$REPO_DIR" "$RUN_DIR" "$MODEL_NAME" "$MODEL_REVISION" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
import yaml

repo, run = map(Path, sys.argv[1:3])
name, revision = sys.argv[3:5]
if Path(name).is_dir():
    snapshot = Path(name).resolve()
    resolved_revision = None
else:
    snapshot = Path(snapshot_download(
        name, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.py"],
    ))
    resolved_revision = snapshot.name
config = AutoConfig.from_pretrained(snapshot)
tokenizer = AutoTokenizer.from_pretrained(snapshot)
metadata = {
    "model_name": name, "requested_revision": revision,
    "resolved_revision": resolved_revision, "snapshot_path": str(snapshot),
    "model_type": config.model_type, "tokenizer_class": type(tokenizer).__name__,
    "local_model_note": "For local directories, use recorded file hashes as provenance.",
}
(run / "model_revision.json").write_text(json.dumps(metadata, indent=2))
(run / "model_path.txt").write_text(str(snapshot))
# Hash local snapshot contents, including weight files, to record the exact model.
with (run / "model_files.sha256").open("w") as output:
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            output.write(f"{digest.hexdigest()}  {path.relative_to(snapshot)}\n")
source = repo / "dataset/MetaMathQA-20K_train.jsonl"
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) < 256:
    raise RuntimeError("Trial requires at least 256 input examples")
# Retain original file order for this diagnostic subset. No curriculum claim.
(run / "trial_data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:256]))
launch_source = repo / "train_base/accelerate_config"
launch = yaml.safe_load((launch_source / "train_config.yaml").read_text())
shutil.copyfile(launch_source / "ds_config.json", run / "ds_config.json")
launch["deepspeed_config"]["deepspeed_config_file"] = str(run / "ds_config.json")
launch["num_processes"] = 2
(run / "accelerate.yaml").write_text(yaml.safe_dump(launch, sort_keys=False))
print("Resolved model:", json.dumps(metadata), flush=True)
PY

MODEL_PATH="$(cat "$RUN_DIR/model_path.txt")"
sha256sum "$RUN_DIR/trial_data.jsonl" > "$RUN_DIR/dataset.sha256"
TRAIN_COMMAND=(accelerate launch --config_file "$RUN_DIR/accelerate.yaml"
    "$REPO_DIR/scripts/reproduction_trial.py"
    --model_name "$MODEL_PATH" --dataset_path "$RUN_DIR/trial_data.jsonl"
    --output_dir "$RUN_DIR/adapter" --max_steps 50)
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train-command.sh"
printf '\n' >> "$RUN_DIR/train-command.sh"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/train.log"

[[ -f "$RUN_DIR/adapter/adapter_config.json" ]] || { echo "Missing saved adapter"; exit 1; }
[[ -f "$RUN_DIR/adapter/trial_report.json" ]] || { echo "Missing trial report"; exit 1; }

# One evaluation process can use both allocated GPUs via the original device_map.
# Running from RUN_DIR keeps the original evaluator's relative output paths here.
cd "$RUN_DIR"
for benchmark in MetaMathQA ASDiv gsm8k mathbench math; do
    DATASET="$REPO_DIR/test_base/DataForTest/${benchmark}_test_512.jsonl"
    sha256sum "$DATASET" >> "$RUN_DIR/benchmarks.sha256"
    EVAL_COMMAND=(python "$REPO_DIR/test_base/Llama8b/train/test_train.py"
        --base_model_path "$MODEL_PATH" --lora_adapter_path "$RUN_DIR/adapter"
        --dataset_path "$DATASET" --limit 8 --batch_size 4
        --max_new_tokens 1024 --print_first_n 1)
    printf '%q ' "${EVAL_COMMAND[@]}" >> "$RUN_DIR/eval-commands.sh"
    printf '\n' >> "$RUN_DIR/eval-commands.sh"
    "${EVAL_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/eval-${benchmark}.log"
    PREDICTIONS="$RUN_DIR/answer_results/default/${benchmark}_test_512/adapter.jsonl"
    python - "$PREDICTIONS" <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(rows) != 8 or not all("is_correct" in row for row in rows):
    raise RuntimeError("Expected eight scored predictions")
print("Verified scored predictions:", len(rows))
PY
done
echo "Trial completed: $(date --iso-8601=seconds)"
echo "Review $RUN_DIR/adapter/trial_report.json and eval-*.log before full training."
echo "These eight-example benchmark scores are diagnostic only, not reproduction results."
