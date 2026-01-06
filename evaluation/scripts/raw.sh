#!/bin/bash

source $SCRATCH/envs/arr_2026_jan_eval/bin/activate

rocminfo
rocm-smi

# Set configurations
MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH argument is required."
    exit 1
fi

MODEL_BASENAME=$(basename ${MODEL_PATH})
log_base_dir=/path/to/l2t/repository/evaluation/logs/raw
mkdir -p ${log_base_dir}
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=$SCRATCH/cache/
export HF_HUB_CACHE=$SCRATCH/cache/
export HF_DATASETS_CACHE=$SCRATCH/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Run the script

# Zero-shot
tasks=(
    "piqa"
    "hellaswag"
    "sciq"
    "copa"
    "race"
    "record"
    "lambada"
    "blimp"
)
for task in "${tasks[@]}"; do
    lm_eval --model hf \
        --model_args pretrained=${MODEL_PATH},dtype=bfloat16  \
        --tasks ${task} \
        --batch_size 1 \
        --output_path ${log_base_dir}/${task}/${MODEL_BASENAME}
done

# 5-shot
tasks=(
    "logiqa"
    "arc_easy"
    "social_iqa"
    "openbookqa"
    "hellaswag"
)
for task in "${tasks[@]}"; do
    lm_eval --model hf \
        --model_args pretrained=${MODEL_PATH},dtype=bfloat16  \
        --tasks ${task} \
        --batch_size 1 \
        --output_path ${log_base_dir}/${task}/${MODEL_BASENAME} \
        --num_fewshot 5    
done
