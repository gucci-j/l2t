#!/bin/bash -l
#SBATCH --job-name=train_raw_1b_shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=15-00:00:00

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

chmod +x /path/to/l2t/repository/training/scripts/ntp_1b_shared.sh
apptainer exec \
    --bind $SCRATCH:$SCRATCH \
    --bind $HOME:$HOME \
    --nv $SCRATCH/containers/pytorch_25.04-py3.sif \
    /path/to/l2t/repository/training/scripts/ntp_1b_shared.sh
