#!/bin/bash
#SBATCH --job-name=generate_typo_training_data
#SBATCH --partition=your_partition_name
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH --time=6:00:00

shard_index="$1"
export shard_index

apptainer exec --fakeroot --bind $SCRATCH:$SCRATCH \
    --rocm $SCRATCH/containers/pytorch_rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.3.0.sif \
    bash /path/to/l2t/repository/preprocessing/scripts/single-task/generate_typo_training_data.sh $shard_index
