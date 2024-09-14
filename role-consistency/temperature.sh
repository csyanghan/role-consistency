#!/bin/bash
#SBATCH --job-name=temperature-infer
#SBATCH --partition=gpu1_10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=slurms/infer_temp.out
#SBATCH --error=slurms/infer_temp.err

CUDA_VISIBLE_DEVICES="0" python role-consistency/infer.py \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --tp 1 \
        --decoding temperature \
        --split test \
        --dataset math_oai

# generate training data for role retrieval
CUDA_VISIBLE_DEVICES="0" python role-consistency/infer.py \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --tp 1 \
        --decoding temperature \
        --split train \
        --dataset math_oai