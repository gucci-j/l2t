import argparse
import json
import math
from typing import Dict, List

import numpy as np
from scipy import optimize
from sklearn.manifold import MDS

from common import cosine_matrix, ensure_dir, load_model_and_tokenizer, token_representation_by_layer


NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def best_fit_line(x_values: List[float], y_values: List[float]):
    xbar = sum(x_values) / len(x_values)
    ybar = sum(y_values) / len(y_values)
    n = len(x_values)
    numer = sum([xi * yi for xi, yi in zip(x_values, y_values)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in x_values]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    return a, b


def normalize_size(size_rows: List[List[float]], size_avg: List[float]):
    max1 = max(max(row) for row in size_rows)
    min1 = min(min(row) for row in size_rows)
    diff = max1 - min1
    if diff == 0:
        return size_rows, size_avg

    norm_rows = [[(v - min1) / diff for v in row] for row in size_rows]
    norm_avg = [(v - min1) / diff for v in size_avg]
    return norm_rows, norm_avg


def grouped_pairs(sim_matrix: np.ndarray, numbers: List[int]):
    grouped = {}
    all_pairs = []
    for i in range(len(numbers)):
        for j in range(i, len(numbers)):
            diff = j - i
            n1 = i + 1
            n2 = j + 1
            sim = float(sim_matrix[i, j])
            grouped.setdefault(diff, []).append([n1, n2, sim])
            all_pairs.append([n1, n2, sim])
    return grouped, all_pairs


def number_strings(numbers: List[int], mode: str) -> List[str]:
    if mode == "lower":
        return [NUMBER_WORDS[n] for n in numbers]
    if mode == "upper":
        return [NUMBER_WORDS[n].capitalize() for n in numbers]
    if mode == "digits":
        return [str(n) for n in numbers]
    raise ValueError(f"Unsupported mode: {mode}")


def distance_effect_metrics(sim_matrix: np.ndarray, numbers: List[int]):
    grouped, _ = grouped_pairs(sim_matrix, numbers)
    averages = []
    for diff in range(len(numbers)):
        items = grouped[diff]
        averages.append(float(sum(v[2] for v in items) / len(items)))
    distance_curve = averages[1:]
    x_vals = list(range(1, len(distance_curve) + 1))
    a, b = best_fit_line(x_vals, distance_curve)
    residuals = np.var([(b * xx + a - yy) for xx, yy in zip(x_vals, distance_curve)])
    variance = np.var(distance_curve)
    rsqr = float("nan") if variance == 0 else float(1 - residuals / variance)
    return rsqr, float(max(distance_curve) - min(distance_curve)), float(max(distance_curve))


def size_effect_r2(sim_matrix: np.ndarray, numbers: List[int]):
    grouped, _ = grouped_pairs(sim_matrix, numbers)
    size_rows = []
    for diff in range(len(numbers)):
        size_rows.append([v[2] for v in grouped[diff]])
    size_rows = size_rows[1:]

    max_len = max(len(row) for row in size_rows)
    padded = np.full((len(size_rows), max_len), np.nan, dtype=np.float64)
    for idx, row in enumerate(size_rows):
        padded[idx, : len(row)] = row
    size_avg = np.nanmean(padded.T, axis=1).tolist()
    _, size_avg = normalize_size(size_rows, size_avg)

    x_vals = list(range(1, len(size_avg) + 1))
    a, b = best_fit_line(x_vals, size_avg)
    residuals = np.var([(b * xx + a - yy) for xx, yy in zip(x_vals, size_avg)])
    variance = np.var(size_avg)
    return float("nan") if variance == 0 else float(1 - residuals / variance)


def ratio_effect_r2(sim_matrix: np.ndarray, numbers: List[int]):
    _, all_pairs = grouped_pairs(sim_matrix, numbers)
    sorted_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)
    y_vals = [p[2] for p in sorted_pairs if p[2] != 1]
    x_vals = [p[1] / p[0] for p in sorted_pairs if p[2] != 1]

    xs = np.array(x_vals, dtype=np.float64)
    ys = np.array(y_vals, dtype=np.float64)
    if len(xs) < 3:
        return float("nan")

    try:
        params, _ = optimize.curve_fit(
            lambda t, a, b, c: a * np.exp(-b * t) + c,
            xs,
            ys,
            maxfev=1000000,
        )
        a, b, c = params
        squared_diffs = np.square(ys - (a * np.exp(-b * xs) + c))
        squared_diffs_from_mean = np.square(ys - np.mean(ys))
        denom = np.sum(squared_diffs_from_mean)
        if denom == 0:
            return float("nan")
        return float(1 - np.sum(squared_diffs) / denom)
    except Exception:
        return float("nan")


def mds_metrics(sim_matrix: np.ndarray):
    dissim = 1.0 - sim_matrix
    mds = MDS(
        random_state=0,
        n_components=1,
        metric=False,
        dissimilarity="precomputed",
        normalized_stress=True,
    )
    transformed = mds.fit_transform(dissim).flatten().tolist()
    if transformed[0] > 0:
        transformed = [-1 * x for x in transformed]

    min_val = min(transformed)
    max_val = max(transformed)
    if math.isclose(min_val, max_val):
        norm = [0.0 for _ in transformed]
    else:
        norm = [(x - min_val) / (max_val - min_val) for x in transformed]

    target = [math.log10(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]]
    corr = float(np.corrcoef(norm, target)[0][1])
    return float(mds.stress_), corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_number", type=int, default=1)
    parser.add_argument("--max_number", type=int, default=8)
    parser.add_argument("--layer_mode", default="all", choices=["all", "first"])
    args = parser.parse_args()

    numbers = list(range(args.min_number, args.max_number + 1))
    if min(numbers) < 1 or max(numbers) > 8:
        raise ValueError("This implementation supports numbers in [1, 8] to match the released protocol.")

    ensure_dir(args.output_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device)

    formats = ["lower", "upper", "digits"]
    format_results: Dict[str, Dict] = {}
    all_distance, all_ratio = [], []
    all_size, all_stress, all_mds_corr = [], [], []

    for mode in formats:
        strings = number_strings(numbers, mode)
        reps = [token_representation_by_layer(model, tokenizer, text, args.device) for text in strings]
        num_layers = reps[0].shape[0]
        layer_indices = [0] if args.layer_mode == "first" else list(range(num_layers))

        layer_distance_r2, layer_ratio_r2 = [], []
        layer_size_r2, layer_stress, layer_mds_corr = [], [], []
        layer_distance_range, layer_distance_max = [], []
        for layer_idx in layer_indices:
            vectors = [rep[layer_idx] for rep in reps]
            sim_matrix = cosine_matrix(vectors)

            d_r2, d_range, d_max = distance_effect_metrics(sim_matrix, numbers)
            s_r2 = size_effect_r2(sim_matrix, numbers)
            r_r2 = ratio_effect_r2(sim_matrix, numbers)
            stress, mds_corr = mds_metrics(sim_matrix)

            layer_distance_r2.append(d_r2)
            layer_distance_range.append(d_range)
            layer_distance_max.append(d_max)
            layer_size_r2.append(s_r2)
            layer_ratio_r2.append(r_r2)
            layer_stress.append(stress)
            layer_mds_corr.append(mds_corr)

        mode_distance_mean = float(np.nanmean(layer_distance_r2))
        mode_ratio_mean = float(np.nanmean(layer_ratio_r2))
        mode_size_mean = float(np.nanmean(layer_size_r2))
        mode_stress_mean = float(np.nanmean(layer_stress))
        mode_mds_corr_mean = float(np.nanmean(layer_mds_corr))
        all_distance.append(mode_distance_mean)
        all_ratio.append(mode_ratio_mean)
        all_size.append(mode_size_mean)
        all_stress.append(mode_stress_mean)
        all_mds_corr.append(mode_mds_corr_mean)

        format_results[mode] = {
            "distance_r2_mean_over_layers": mode_distance_mean,
            "distance_range_mean_over_layers": float(np.nanmean(layer_distance_range)),
            "distance_max_mean_over_layers": float(np.nanmean(layer_distance_max)),
            "size_r2_mean_over_layers": mode_size_mean,
            "ratio_r2_mean_over_layers": mode_ratio_mean,
            "mds_stress_mean_over_layers": mode_stress_mean,
            "mds_correlation_mean_over_layers": mode_mds_corr_mean,
            "distance_r2_by_layer": layer_distance_r2,
            "size_r2_by_layer": layer_size_r2,
            "ratio_r2_by_layer": layer_ratio_r2,
            "mds_stress_by_layer": layer_stress,
            "mds_correlation_by_layer": layer_mds_corr,
        }

    metrics = {
        "task": "numeric_magnitude_comparison",
        "numbers": numbers,
        "formats": formats,
        "layer_mode": args.layer_mode,
        "distance_effect_r2": float(np.nanmean(all_distance)),
        "size_effect_r2": float(np.nanmean(all_size)),
        "ratio_effect_r2": float(np.nanmean(all_ratio)),
        "mds_stress": float(np.nanmean(all_stress)),
        "mds_correlation": float(np.nanmean(all_mds_corr)),
        "per_format": format_results,
    }

    with open(f"{args.output_path}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
