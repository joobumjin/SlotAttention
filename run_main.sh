#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:2

#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH --mem=128g

# Load a CUDA module
module load cuda
module load miniconda/4.12.0
conda activate SlotAttention
conda list tensorflow

# Run program
cd /users/bjoo2/data/bjoo2/SlotAttention/SlotAttention
python main.py