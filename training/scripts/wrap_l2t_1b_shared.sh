#!/bin/bash -l
#SBATCH --job-name=train_l2t_1b_shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=15-00:00:00

chmod +x /path/to/l2t/repository/training/scripts/l2t_1b_shared.sh
apptainer exec \
    --bind $SCRATCH:$SCRATCH \
    --bind $HOME:$HOME \
    --nv $SCRATCH/containers/pytorch_25.04-py3.sif \
    /path/to/l2t/repository/training/scripts/l2t_1b_shared.sh
