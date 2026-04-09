import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np

from common import continuation_token_logprobs_raven, ensure_dir, load_model_and_tokenizer


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Shape:
    def __init__(self, shape_dict):
        self.type = int(shape_dict["Type"]) + 2
        self.size = int(shape_dict["Size"]) + 1
        self.color = int(shape_dict["Color"])

    def __str__(self):
        return f"({self.type},{self.size/10},{self.color*10})"


def build_center_single_context_and_choices(sample: Dict, n: int):
    items = [Shape(sample["rpm"][j][0]) for j in range(16)]
    item_strings = [str(x) for x in items]
    choices = item_strings[8:]
    if n == 1:
        context = "{}, {}, ".format(*item_strings[6:8])
    elif n == 2:
        context = "row 1: {}, {}, {}; row 2: {}, {}, ".format(*item_strings[3:8])
    else:
        context = "row 1: {}, {}, {}; row 2: {}, {}, {}; row 3: {}, {}, ".format(*item_strings[:8])
    return context, choices


def mean_or_nan(values: List[float]) -> float:
    return float(sum(values) / len(values)) if len(values) > 0 else float("nan")


def parse_subset_ids(path: str) -> List[str]:
    raw = read_json(path)
    return [str(x) for x in raw]


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(context: str, choice: str, n: int) -> str:
    prompt = context + choice
    if n != 1:
        prompt += ";"
    return prompt


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    default_center_single = os.path.join(repo_root, "Code/effects/center_single.json")
    default_subset = os.path.join(repo_root, "Code/effects/subset.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--center_single_json", default=default_center_single)
    parser.add_argument("--subset_json", default=default_subset)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    ensure_dir(args.output_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device)
    samples = read_json(args.center_single_json)
    subset_ids = parse_subset_ids(args.subset_json)

    correct = 0
    predictions = []

    for sample_id in subset_ids:
        sample = samples[sample_id]
        context, choices = build_center_single_context_and_choices(sample, args.n)
        context = args.prefix + context

        scores = []
        for choice in choices:
            text = build_prompt(context, choice, args.n)
            token_logprobs = continuation_token_logprobs_raven(model, tokenizer, text, context, args.device)
            scores.append(mean_or_nan(token_logprobs))

        pred_idx = int(np.nanargmax(scores)) if not all(math.isnan(x) for x in scores) else -1
        answer_idx = 0
        is_correct = pred_idx == 0
        correct += int(is_correct)
        predictions.append(
            {
                "sample_id": sample_id,
                "pred_idx": pred_idx,
                "answer_idx": answer_idx,
                "correct": is_correct,
                "scores": scores,
            }
        )

    accuracy = correct / len(subset_ids) if subset_ids else float("nan")
    metrics = {
        "task": "fluid_reasoning_rpm_center_single",
        "center_single_json": args.center_single_json,
        "subset_json": args.subset_json,
        "num_examples": len(subset_ids),
        "n": args.n,
        "accuracy": accuracy,
    }

    with open(f"{args.output_path}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{args.output_path}/predictions.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
