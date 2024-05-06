#!/bin/bash

#SBATCH --job-name=test-nvidia
#SBATCH --output=test-nvidia.out
#SBATCH --error=test-nvidia.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A40:1

# Load necessary modules
module load singularity

# Start an interactive shell in the nvidia.sif container
srun --pty singularity shell --nv nvidia.sif
