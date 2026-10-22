#!/bin/bash
# Submit from the repository root: sbatch evaluate_llama.sbatch
# Defaults to the adapter trained by job 57609. To evaluate another adapter:
#   sbatch --export=ALL,ADAPTER_PATH=/absolute/path/to/adapter evaluate_llama.sbatch
# EVAL_BATCH_SIZE and MAX_NEW_TOKENS may also be overridden.
#SBATCH --job-name=curriculum-llama-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:10
#SBATCH --account=eduard
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-logs/curriculum-llama-eval-%j.out
#SBATCH --error=slurm-logs/curriculum-llama-eval-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
CONDA_ENV="${CONDA_ENV:-curr-eval-env}"
CONDA_SH="${CONDA_SH:-/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-$REPO_DIR/results/reproduction/llama-repro-seed42-57609/adapter}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
EVAL_DIR="${EVAL_DIR:-$REPO_DIR/results/evaluation/llama-eval-${SLURM_JOB_ID}}"
BENCHMARKS=(MetaMathQA ASDiv gsm8k mathbench math)

cd "$REPO_DIR"
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/shared/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="$REPO_DIR/.cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

ADAPTER_PATH="$(realpath "$ADAPTER_PATH")"
[[ -f "$ADAPTER_PATH/adapter_config.json" ]] || { echo "Missing adapter configuration"; exit 1; }
[[ -f "$ADAPTER_PATH/adapter_model.safetensors" || -f "$ADAPTER_PATH/adapter_model.bin" ]] || { echo "Missing adapter weights"; exit 1; }
for benchmark in "${BENCHMARKS[@]}"; do
    [[ -f "$REPO_DIR/test_base/DataForTest/${benchmark}_test_512.jsonl" ]] || { echo "Missing benchmark: $benchmark"; exit 1; }
done
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Evaluation requires exactly one visible CUDA GPU")
with torch.cuda.device(0):
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Evaluation requires BF16 support")
    x = torch.ones((32, 32), device="cuda:0", dtype=torch.bfloat16)
    (x @ x).sum().item()
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)
PY

mkdir -p "$(dirname "$EVAL_DIR")"
mkdir "$EVAL_DIR"
EVAL_DIR="$(realpath "$EVAL_DIR")"
echo "Job: $SLURM_JOB_ID; node: $(hostname); started: $(date --iso-8601=seconds)"
echo "Base model: $MODEL_NAME"
echo "Adapter: $ADAPTER_PATH"
echo "Results: $EVAL_DIR"
python -m pip freeze > "$EVAL_DIR/packages.txt"
git rev-parse HEAD > "$EVAL_DIR/commit.txt"
git diff > "$EVAL_DIR/source-changes.patch"
cp "$REPO_DIR/evaluate_llama.sbatch" "$EVAL_DIR/submitted-script.sbatch"
cp "$REPO_DIR/test_base/Llama8b/train/test_train.py" "$EVAL_DIR/evaluator.py"
sha256sum "$ADAPTER_PATH"/adapter_config.json "$ADAPTER_PATH"/adapter_model.* > "$EVAL_DIR/adapter.sha256"

# The evaluator writes relative result paths; isolate them for this job.
cd "$EVAL_DIR"
for benchmark in "${BENCHMARKS[@]}"; do
    DATASET="$REPO_DIR/test_base/DataForTest/${benchmark}_test_512.jsonl"
    sha256sum "$DATASET" >> "$EVAL_DIR/benchmarks.sha256"
    COMMAND=(python "$EVAL_DIR/evaluator.py"
        --base_model_path "$MODEL_NAME" --lora_adapter_path "$ADAPTER_PATH"
        --dataset_path "$DATASET" --limit -1
        --batch_size "$EVAL_BATCH_SIZE" --max_new_tokens "$MAX_NEW_TOKENS"
        --print_first_n 1)
    printf '%q ' "${COMMAND[@]}" >> "$EVAL_DIR/commands.sh"
    printf '\n' >> "$EVAL_DIR/commands.sh"
    "${COMMAND[@]}" 2>&1 | tee "$EVAL_DIR/eval-${benchmark}.log"
    python - "$DATASET" "$ADAPTER_PATH" "$benchmark" <<'PY'
import csv
import json
import sys
from pathlib import Path

dataset, adapter, benchmark = sys.argv[1:]
expected = sum(bool(line.strip()) for line in Path(dataset).read_text().splitlines())
matches = list(Path("answer_results").rglob(f"{Path(dataset).stem}/{Path(adapter).name}.jsonl"))
if len(matches) != 1:
    raise RuntimeError(f"Expected one predictions file, found {len(matches)}")
rows = [json.loads(line) for line in matches[0].read_text().splitlines() if line.strip()]
if not expected or len(rows) != expected or not all(isinstance(row.get("is_correct"), bool) for row in rows):
    raise RuntimeError(f"Expected {expected} scored predictions, found {len(rows)}")
correct = sum(row["is_correct"] for row in rows)
summary = Path("summary.csv")
new_file = not summary.exists()
with summary.open("a", newline="") as stream:
    writer = csv.writer(stream)
    if new_file:
        writer.writerow(["benchmark", "samples", "correct", "accuracy_percent", "avg_output_tokens"])
    writer.writerow([benchmark, expected, correct, 100 * correct / expected,
                     sum(row["output_token_length"] for row in rows) / expected])
print(f"Verified {expected} predictions; {benchmark}: {100 * correct / expected:.2f}%")
PY
done
echo "Evaluation finished: $(date --iso-8601=seconds)"
echo "Combined benchmark scores: $EVAL_DIR/summary.csv"
cat "$EVAL_DIR/summary.csv"
