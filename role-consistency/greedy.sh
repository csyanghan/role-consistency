#!/bin/bash
#SBATCH --job-name=llama3-8b-greedy-math-oai
#SBATCH --partition=gpu11_12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=slurms/infer_3_gsm8k.out
#SBATCH --error=slurms/infer_3_gsm8k.err


CUDA_VISIBLE_DEVICES="0" python role-consistency/infer.py \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --tp 1 \
        --decoding greedy \
        --dataset math_oai \
        --split test


# generate training data for role retrieval
CUDA_VISIBLE_DEVICES="0" python role-consistency/infer.py \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --tp 1 \
        --decoding greedy \
        --dataset math_oai \
        --split train
