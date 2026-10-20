"""Audit all input records with the Llama trainer's exact formatting/tokenization.

Does not load model weights, alter examples, or change truncation policy. The
nested tokenizer is extracted from the checked-out training source with AST so
the audit checks the real implementation without duplicating its algorithm.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace


def load_preprocessing(source, tokenizer, max_length):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = []
    patterns = {"THINK_PATTERN", "REPEATED_ANSWER_PATTERN", "MULTIPLE_BOXED_PATTERN"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in patterns for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"clean_solution_text", "format_prompt"}:
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            selected.extend(
                child for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "tokenize_function"
            )
    namespace = {"re": re, "tokenizer": tokenizer, "args": SimpleNamespace(max_length=max_length)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["format_prompt"], namespace["tokenize_function"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", default="dataset/MetaMathQA-20K_train.jsonl")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("max_length must be at least 2")
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer needs an EOS token")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    source = Path(__file__).resolve().parents[1] / "train_base/train_llama3_8b_lora.py"
    format_prompt, tokenize = load_preprocessing(source, tokenizer, args.max_length)
    dataset = Path(args.dataset_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lengths = []
    overflow = []
    total_full_over = 0
    nonpositive_budgets = 0
    duplicate_eos = 0
    template = "\n\n---\n\nResponse:\n"
    template_length = len(tokenizer.encode(template, add_special_tokens=False))
    with dataset.open(encoding="utf-8") as stream, (output / "lengths.jsonl").open("w", encoding="utf-8") as records:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = format_prompt(row)["text"]
            full = tokenizer.encode(text, add_special_tokens=False)
            encoded = tokenize({"text": [text]})["input_ids"][0]
            length = len(encoded)
            lengths.append(length)
            total_full_over += len(full) > args.max_length
            duplicate_eos += len(encoded) >= 2 and encoded[-2:] == [tokenizer.eos_token_id] * 2
            response = text.split(template, 1)[1]
            response_length = len(tokenizer.encode(response, add_special_tokens=False))
            prompt_budget = args.max_length - response_length - template_length - 1
            nonpositive_budgets += len(full) > args.max_length and prompt_budget <= 0
            record = {
                "line": line_number, "full_length": len(full), "tokenized_length": length,
                "response_length": response_length, "prompt_budget": prompt_budget,
            }
            records.write(json.dumps(record) + "\n")
            if length > args.max_length:
                overflow.append(record)
            if len(lengths) % 1000 == 0:
                print(f"Audited {len(lengths)} records; over limit: {len(overflow)}", flush=True)
    if not lengths:
        raise ValueError("No examples found")
    ordered = sorted(lengths)
    context = getattr(config, "max_position_embeddings", None)
    report = {
        "dataset": str(dataset.resolve()), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "training_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_name": args.model_name, "resolved_config_revision": getattr(config, "_commit_hash", None),
        "examples": len(lengths), "configured_max_length": args.max_length,
        "max_tokenized_length": max(lengths), "examples_over_limit": len(overflow),
        "formatted_examples_over_limit_before_truncation": total_full_over,
        "long_examples_with_nonpositive_prompt_budget": nonpositive_budgets,
        "examples_ending_with_duplicate_eos": duplicate_eos,
        "model_context_length": context,
        "examples_over_model_context": sum(n > context for n in lengths) if context else None,
        "length_percentiles": {str(p): ordered[round((len(ordered) - 1) * p / 100)] for p in (50, 90, 95, 99)},
        "largest_overflows": sorted(overflow, key=lambda row: row["tokenized_length"], reverse=True)[:20],
        "policy": "Original preprocessing unchanged. 512 is a code default, not a verified paper setting.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
