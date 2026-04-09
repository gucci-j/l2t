Psychometric evaluations
===

This directory adds three evaluation pipelines aligned with ["Development of Cognitive Intelligence in Pre-trained Language Models"](https://aclanthology.org/2024.emnlp-main.539/):

- Numeric abilities: magnitude comparison effects (distance and ratio)
- Concept understanding: typicality via surprisal
- Fluid reasoning: Raven-style matrix completion via surprisal


## 1. Numeric abilities

The numeric benchmark follows the paper protocol:

- numbers `1..8`
- three formats: lower-case words, upper-case words, digits
- cosine similarities from model hidden states
- `R^2` for distance, size, and ratio effects
- MDS stress and MDS correlation (appendix metrics)

Command:

```bash
python evaluation/src/psychometric/numeric_magnitude.py \
  --model_path /path/to/model \
  --output_path /path/to/output
```

## 2. Concept understanding (typicality)

To evaluate concept understanding, we use the `castro_et_al_typicality.csv` dataset by default. The CSV file can be downloaded from the original paper's released assets: https://aclanthology.org/2024.emnlp-main.539/ (Select "Software" button to download the file; the file is located in the `Code/files/` directory of the released assets).

Expected CSV columns:

- `Category`
- `Response`
- `Typicality_Scores`

Reported metrics:

- representation-based Spearman
- zero/one/two/three-shot surprisal-based Spearman

Command:

```bash
python evaluation/src/psychometric/concept_typicality.py \
  --model_path /path/to/model \
  --dataset_csv /path/to/typicality.csv \
  --output_path /path/to/output
```

## 3. Fluid reasoning (RPM / I-RAVEN text conversion)

Uses the released files by default:

- `Code/effects/center_single.json`
- `Code/effects/subset.json`

Note that these files can be downloaded from the original paper's released assets: https://aclanthology.org/2024.emnlp-main.539/ (Select "Software" button to download the files; the files are located in the `Code/effects/` directory of the released assets).

The evaluator mirrors the released protocol where candidate index `0` is the gold answer for each sampled item.

Command:

```bash
python evaluation/src/psychometric/fluid_rpm.py \
  --model_path /path/to/model \
  --center_single_json /path/to/center_single.json \
  --subset_json /path/to/subset.json \
  --output_path /path/to/output
```

## Output

Each script writes:

- `metrics.json`
- optional per-example outputs (`predictions.jsonl` where applicable)
