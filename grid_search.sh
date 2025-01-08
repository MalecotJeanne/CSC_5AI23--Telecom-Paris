#!/bin/bash


#SBATCH --output=logs_%j.out
#SBATCH --error=errs_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1


set -x
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jupyterenv
srun python3 ./grid_search.py
