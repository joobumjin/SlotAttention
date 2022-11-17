#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:2

# Request 1 CPU core
#SBATCH -n 4
#SBATCH -t 04:00:00

# Load a CUDA module
module load cuda
module load miniconda/4.12.0
conda activate SlotAttention

# Run program
cd data/bjoo2/SlotAttention/SlotAttention
./main.py