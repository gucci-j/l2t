#!/bin/bash
#SBATCH --job-name=train_raw_500m_shared
#SBATCH --partition=your_partition_name
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=20-00:00:00

# Create a symbolic link from $SCRATCH/data/l2t_ntp_training_data/0~7 to $SCRATCH/data/l2t_ntp_training_data_shared/0~7
mkdir -p $SCRATCH/data/l2t_ntp_training_data_shared
ln -s $SCRATCH/data/l2t_ntp_training_data/0 $SCRATCH/data/l2t_ntp_training_data_shared/0
ln -s $SCRATCH/data/l2t_ntp_training_data/1 $SCRATCH/data/l2t_ntp_training_data_shared/1
ln -s $SCRATCH/data/l2t_ntp_training_data/2 $SCRATCH/data/l2t_ntp_training_data_shared/2
ln -s $SCRATCH/data/l2t_ntp_training_data/3 $SCRATCH/data/l2t_ntp_training_data_shared/3
ln -s $SCRATCH/data/l2t_ntp_training_data/4 $SCRATCH/data/l2t_ntp_training_data_shared/4
ln -s $SCRATCH/data/l2t_ntp_training_data/5 $SCRATCH/data/l2t_ntp_training_data_shared/5
ln -s $SCRATCH/data/l2t_ntp_training_data/6 $SCRATCH/data/l2t_ntp_training_data_shared/6
ln -s $SCRATCH/data/l2t_ntp_training_data/7 $SCRATCH/data/l2t_ntp_training_data_shared/7

apptainer exec --fakeroot --bind $SCRATCH:$SCRATCH \
    --rocm $SCRATCH/containers/pytorch_rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.3.0.sif \
    bash /path/to/l2t/repository/training/scripts/ntp_500m_shared.sh
