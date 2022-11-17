#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:2

#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH --mem=128g

# Load a CUDA module
module load cuda
module load miniconda/4.12.0
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda init bash

conda activate SlotAttention
conda list tensorflow
which python

# Run program
cd /users/bjoo2/data/bjoo2/SlotAttention/SlotAttention
python3 main.py