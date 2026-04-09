import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

from common import continuation_avg_prob_legacy, ensure_dir, load_model_and_tokenizer, token_representation_by_layer


def average_ranks(values: List[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = average_ranks(x)
    ry = average_ranks(y)
    x_centered = rx - rx.mean()
    y_centered = ry - ry.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def read_dataset(path: str) -> Dict[str, List[Tuple[str, float]]]:
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Category", "Response", "Typicality_Scores"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Dataset CSV must include columns: Category, Response, Typicality_Scores")
        for row in reader:
            category = row["Category"].strip()
            member = row["Response"].strip()
            score = row["Typicality_Scores"].strip()
            if score == "":
                continue
            typicality = float(score)
            grouped[category].append((member, typicality))
    return grouped


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    default_csv = os.path.join(repo_root, "Code/files/castro_et_al_typicality.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_csv", default=default_csv)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ensure_dir(args.output_path)
    grouped = read_dataset(args.dataset_csv)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device)

    context_zero_shot = " "
    context_one_shot = "a cat is a mammal\n "
    context_two_shot = "a cat is a mammal\n a square is shape\n "
    context_three_shot = "a cat is a mammal\n a square is shape\n a tomato is a vegetable\n"

    cache_vectors: Dict[str, np.ndarray] = {}

    def get_text_layers(text: str) -> np.ndarray:
        if text not in cache_vectors:
            cache_vectors[text] = token_representation_by_layer(model, tokenizer, text, args.device)
        return cache_vectors[text]

    per_category = {}
    all_rep = []
    all_zero = []
    all_one = []
    all_two = []
    all_three = []

    for category, rows in grouped.items():
        rep_scores = []
        zero_scores = []
        one_scores = []
        two_scores = []
        three_scores = []
        human_scores = []

        category_layers = get_text_layers(category)
        for member, typicality in rows:
            member_layers = get_text_layers(member)
            cos_vals = []
            for layer_idx in range(category_layers.shape[0]):
                cat_vec = category_layers[layer_idx]
                mem_vec = member_layers[layer_idx]
                denom = np.linalg.norm(cat_vec) * np.linalg.norm(mem_vec)
                cos_vals.append(float(np.dot(cat_vec, mem_vec) / denom) if denom != 0 else 0.0)

            rep_scores.append(max(cos_vals))

            prompt = "a " + member + " is " + category
            zero_scores.append(
                continuation_avg_prob_legacy(
                    model,
                    tokenizer,
                    context_zero_shot + prompt,
                    context_zero_shot,
                    args.device,
                    lowercase_prompt=True,
                )
            )
            one_scores.append(
                continuation_avg_prob_legacy(
                    model,
                    tokenizer,
                    context_one_shot + prompt,
                    context_one_shot,
                    args.device,
                    lowercase_prompt=True,
                )
            )
            two_scores.append(
                continuation_avg_prob_legacy(
                    model,
                    tokenizer,
                    context_two_shot + prompt,
                    context_two_shot,
                    args.device,
                    lowercase_prompt=True,
                )
            )
            three_scores.append(
                continuation_avg_prob_legacy(
                    model,
                    tokenizer,
                    context_three_shot + prompt,
                    context_three_shot,
                    args.device,
                    lowercase_prompt=True,
                )
            )

            human_scores.append(typicality)

        rep_corr = float(spearmanr(human_scores, rep_scores, nan_policy="omit").correlation)
        zero_corr = float(spearmanr(human_scores, zero_scores, nan_policy="omit").correlation)
        one_corr = float(spearmanr(human_scores, one_scores, nan_policy="omit").correlation)
        two_corr = float(spearmanr(human_scores, two_scores, nan_policy="omit").correlation)
        three_corr = float(spearmanr(human_scores, three_scores, nan_policy="omit").correlation)

        all_rep.append(rep_corr)
        all_zero.append(zero_corr)
        all_one.append(one_corr)
        all_two.append(two_corr)
        all_three.append(three_corr)

        per_category[category] = {
            "num_members": len(rows),
            "representation_spearman": rep_corr,
            "zero_shot_spearman": zero_corr,
            "one_shot_spearman": one_corr,
            "two_shot_spearman": two_corr,
            "three_shot_spearman": three_corr,
        }

    metrics = {
        "task": "concept_typicality",
        "dataset_csv": args.dataset_csv,
        "average_representation_spearman": float(np.nanmean(all_rep)),
        "average_zero_shot_spearman": float(np.nanmean(all_zero)),
        "average_one_shot_spearman": float(np.nanmean(all_one)),
        "average_two_shot_spearman": float(np.nanmean(all_two)),
        "average_three_shot_spearman": float(np.nanmean(all_three)),
        "num_categories": len(per_category),
        "per_category": per_category,
    }

    with open(f"{args.output_path}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
