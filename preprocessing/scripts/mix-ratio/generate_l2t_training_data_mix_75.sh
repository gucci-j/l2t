#!/bin/bash

source $SCRATCH/envs/arr_2026_jan/bin/activate

# Set configurations
tokenizer_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
ntp_tokenizer_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
cache_dir="$SCRATCH/cache"
output_dir="$SCRATCH/data/l2t_training_data_mix_75"
seq_len=2048
num_proc=$(( $(nproc) - 1 ))
shard_index=$1

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=$SCRATCH/cache/
export HF_HUB_CACHE=$SCRATCH/cache/
export HF_DATASETS_CACHE=$SCRATCH/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Run the script
cd /path/to/l2t/repository/preprocessing/src

mkdir -p $output_dir

python generate_l2t_training_data.py \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --shard_index $shard_index \
    --output_dir "${output_dir}" \
    --cache_dir "${cache_dir}" \
    --num_workers "${num_proc}" \
    --max_length "${seq_len}" \
    --generate_samples_on_the_fly \
    --batched \
    --min_num_words 20 \
    --mix_ntp_samples \
    --mix_ntp_ratio 0.75 \
    --ntp_tokenizer_name_or_path "${ntp_tokenizer_name_or_path}" \
    --loose_chunking \
    --loose_chunking_token_count 512

# Deactivate the virtual environment
deactivate
