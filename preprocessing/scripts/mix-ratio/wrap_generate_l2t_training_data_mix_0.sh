#!/bin/bash
#SBATCH --job-name=generate_l2t_training_data_mix_0
#SBATCH --partition=your_partition_name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH --time=3:00:00

shard_index="$1"
export shard_index

apptainer exec --fakeroot --bind $SCRATCH:$SCRATCH \
    --rocm $SCRATCH/containers/pytorch_rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.3.0.sif \
    bash /path/to/l2t/repository/preprocessing/scripts/mix-ratio/generate_l2t_training_data_mix_0.sh $shard_index
