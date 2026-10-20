"""CPU-only regression checks for the audit and trial diagnostics."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("length_audit", ROOT / "scripts/audit_sequence_lengths.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CharacterTokenizer:
    eos_token = "~"
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [0 if char == "~" else ord(char) for char in text]


class ReproductionChecks(unittest.TestCase):
    def tokenize(self, text, limit):
        _, function = audit.load_preprocessing(
            ROOT / "train_base/train_llama3_8b_lora.py", CharacterTokenizer(), limit
        )
        return function({"text": [text]})["input_ids"][0]

    def test_audit_exercises_actual_zero_budget_bug(self):
        template = "\n\n---\n\nResponse:\n"
        text = "long prompt" + template + "response~"
        # With zero prompt budget, the original [-0:] retains the whole prompt.
        limit = len(template) + len("response~") + 1
        self.assertGreater(len(self.tokenize(text, limit)), limit)

    def test_audit_exercises_actual_negative_budget_bug(self):
        text = "long prompt" + "\n\n---\n\nResponse:\n" + "r" * 100 + "~"
        self.assertGreater(len(self.tokenize(text, 40)), 40)

    def test_short_examples_unchanged(self):
        text = "prompt\n\n---\n\nResponse:\nresponse~"
        self.assertEqual(self.tokenize(text, 512), CharacterTokenizer().encode(text))

    def test_trial_distinguishes_cosine_from_constant_after_warmup(self):
        tree = ast.parse((ROOT / "scripts/reproduction_trial.py").read_text())
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "Diagnostics")
        namespace = {"TrainerCallback": object, "math": math}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "trial-callback", "exec"), namespace)
        settings = SimpleNamespace(
            lr_scheduler_type=SimpleNamespace(value="cosine"),
            get_warmup_steps=lambda steps: 5, learning_rate=5e-5,
        )
        state = SimpleNamespace(is_world_process_zero=True, max_steps=50, global_step=25)
        scheduler = SimpleNamespace(get_last_lr=lambda: [2.934120444167326e-5])
        report = {}
        diagnostics = namespace["Diagnostics"](report)
        diagnostics.on_train_begin(settings, state, None, lr_scheduler=scheduler)
        diagnostics.on_step_end(settings, state, None, lr_scheduler=scheduler)
        self.assertTrue(report["learning_rate_trace"][-1]["matches"])
        scheduler.get_last_lr = lambda: [5e-5]
        diagnostics.on_step_end(settings, state, None, lr_scheduler=scheduler)
        self.assertFalse(report["learning_rate_trace"][-1]["matches"])


if __name__ == "__main__":
    unittest.main()
