#!/bin/bash -l



# Set configurations
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
tokenizer_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
cache_dir="$SCRATCH/cache"
log_dir=/path/to/l2t/repository/training/logs/l2t-500m-fill_middle
output_dir="$SCRATCH/models/l2t-500m-fill_middle"
dataset_dir="$SCRATCH/data/l2t_fill_middle_training_data"


export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="$SCRATCH/cache"
export HF_HUB_CACHE="$SCRATCH/cache"
export HF_DATASETS_CACHE="$SCRATCH/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TORCH_HOME="$SCRATCH/cache"
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat/lib.real:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp
export NCCL_DEBUG=INFO

# Run the script
cd /path/to/l2t/repository/training/src



python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=12390 \
main_ntp.py \
    --train_dataset_path "${dataset_dir}/*/train/*.arrow.zst" \
    --dev_dataset_path "${dataset_dir}/*/dev/*.arrow.zst" \
    --output_dir "${output_dir}" \
    --logging_dir "${log_dir}" \
    --model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --optim adamw_torch \
    --seed 42 \
    --eval_strategy no \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_steps 200000 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns True \
    --save_strategy steps \
    --save_steps 5000 \
    --bf16 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --model_size 500m \
    --use_streaming \
    --stop_on_checkpoint \
    --stop_after_n_checkpoints 10
