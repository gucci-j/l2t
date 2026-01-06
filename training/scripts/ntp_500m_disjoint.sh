#!/bin/bash

source $SCRATCH/envs/arr_2026_jan/bin/activate

# Set configurations
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
tokenizer_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
cache_dir="$SCRATCH/cache"
log_dir=/path/to/l2t/repository/training/logs/raw-500m-disjoint
output_dir="$SCRATCH/models/raw-500m-disjoint"
dataset_dir="$SCRATCH/data/l2t_ntp_training_data"
num_proc=$(( $(nproc) / 2 ))

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=$SCRATCH/cache/
export HF_HUB_CACHE=$SCRATCH/cache/
export HF_DATASETS_CACHE=$SCRATCH/cache/
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TORCH_HOME="$SCRATCH/cache"
export NCCL_DEBUG=INFO
rocm-smi --showtopo
rocm-smi
rocm-smi --showhw

# Run the script
cd /path/to/l2t/repository/training/src/

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=12410 \
main_ntp.py \
    --train_dataset_path "${dataset_dir}/*/train/*.arrow.zst" \
    --dev_dataset_path "${dataset_dir}/*/dev/*.arrow.zst" \
    --output_dir "${output_dir}" \
    --logging_dir "${log_dir}" \
    --model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --optim adamw_apex_fused \
    --seed 42 \
    --eval_strategy no \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_steps 200000 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns True \
    --save_strategy steps \
    --save_steps 0.05 \
    --bf16 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --model_size 500m \
    --use_streaming
