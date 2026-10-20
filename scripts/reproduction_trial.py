"""Run the original Llama trainer with a trial-only step limit and diagnostics.

The production trainer is imported without editing it. The only training-setting
override here is max_steps; evaluation/save intervals and scheduler stay intact.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()
    if args.max_steps < 50:
        parser.error("Use at least 50 steps to exercise the original save interval")

    source = Path(__file__).resolve().parents[1] / "train_base/train_llama3_8b_lora.py"
    spec = importlib.util.spec_from_file_location("original_llama_training", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from transformers import TrainerCallback

    original_arguments = module.TrainingArguments

    def trial_arguments(*positional, **kwargs):
        kwargs["max_steps"] = args.max_steps
        return original_arguments(*positional, **kwargs)

    module.TrainingArguments = trial_arguments

    class Diagnostics(TrainerCallback):
        def __init__(self, report):
            self.report = report

        def on_train_begin(self, training_args, state, control, **kwargs):
            if state.is_world_process_zero:
                scheduler = kwargs.get("lr_scheduler")
                chain = []
                seen = set()
                while scheduler is not None and id(scheduler) not in seen:
                    seen.add(id(scheduler))
                    chain.append(f"{type(scheduler).__module__}.{type(scheduler).__name__}")
                    scheduler = getattr(scheduler, "lr_scheduler", None)
                self.report["scheduler_classes"] = chain
                print("TRIAL scheduler classes:", chain, flush=True)

    original_trainer = module.SequentialTrainer

    class TrialTrainer(original_trainer):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            lengths = [len(row["input_ids"]) for row in self.train_dataset]
            sample = [self.train_dataset[i] for i in range(min(8, len(lengths)))]
            batch = self.data_collator(sample)
            valid_labels = (batch["labels"] != -100).sum(dim=1).tolist()
            self.trial_report = {
                "trial_only": True,
                "max_steps": args.max_steps,
                "training_examples": len(lengths),
                "validation_examples": len(self.eval_dataset),
                "max_tokenized_training_length": max(lengths),
                "training_examples_over_512_tokens": sum(n > 512 for n in lengths),
                "valid_response_labels_first_8": valid_labels,
                "world_size": self.args.world_size,
                "effective_batch_size": self.args.world_size
                * self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps,
                "sample_order_note": "Original SequentialSampler retained; distributed order is not independently verified.",
            }
            if self.is_world_process_zero():
                print("TRIAL preprocessing:", json.dumps(self.trial_report), flush=True)
            if not all(valid_labels):
                raise RuntimeError("Trial sample contains an example with no response labels")
            self.add_callback(Diagnostics(self.trial_report))

        def train(self, *positional, **kwargs):
            result = super().train(*positional, **kwargs)
            if self.is_world_process_zero():
                self.trial_report.update({
                    "global_step": self.state.global_step,
                    "best_model_checkpoint": self.state.best_model_checkpoint,
                    "best_metric": self.state.best_metric,
                    "load_best_model_at_end": self.args.load_best_model_at_end,
                    "log_history": self.state.log_history,
                })
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "trial_report.json").write_text(
                    json.dumps(self.trial_report, indent=2), encoding="utf-8"
                )
                if not self.state.best_model_checkpoint:
                    raise RuntimeError("No best checkpoint was recorded during the trial")
                if not any("eval_loss" in row for row in self.state.log_history):
                    raise RuntimeError("No validation loss was recorded during the trial")
            return result

    module.SequentialTrainer = TrialTrainer
    # Keep these settings aligned with reproduce_llama.sbatch. Only the trial's
    # step limit and small input/validation sets differ from the full run.
    settings = argparse.Namespace(
        model_name=args.model_name, dataset_path=args.dataset_path,
        output_dir=args.output_dir, seed=42, epochs=3, batch_size=4,
        grad_accumulation=4, learning_rate=5e-5, valid_num=16,
        max_length=512, lora_r=64, lora_alpha=128, lora_dropout=0.05,
        gradient_checkpointing=True, wandb_project="curriculum-reproduction-trial",
        wandb_run_name=f"trial-{os.environ.get('SLURM_JOB_ID', 'local')}",
    )
    module.main(settings)


if __name__ == "__main__":
    main()
