#!/bin/bash
#SBATCH --job-name=eval_8b_temperature_math_oai
#SBATCH --partition=dcu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=400G
#SBATCH --output=slurms/eval_8b.out
#SBATCH --error=slurms/eval_8b.err


python ./eval.py \
    --file_path results/math_oai/Meta-Llama-3-8B-Instruct-temperature_test.jsonl \
    --data_name math_oai
