#!/bin/bash

#SBATCH --job-name=eu-ai-act
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --qos=batch-long
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:a100:2  # Request two A100 GPUs
#SBATCH --time=10000

# Ensure Singularity container is built only once
if [ ! -f ai_act.sif ]; then
    echo "Building Singularity container hnsw_experiment.sif..."
    singularity build --fakeroot ai_act.sif Singularity
else
    echo "Singularity container already exists."
fi

singularity exec ai_act.sif python3 main_with_rag.py --dataset="$DATASET"