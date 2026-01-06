#!/bin/bash
#SBATCH --job-name=train_l2t_500m_mix_75
#SBATCH --partition=your_partition_name
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=20-00:00:00

apptainer exec --fakeroot --bind $SCRATCH:$SCRATCH \
    --rocm $SCRATCH/containers/pytorch_rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.3.0.sif \
    bash /path/to/l2t/repository/training/scripts/mix-ratio/l2t_500m_mix_75.sh
