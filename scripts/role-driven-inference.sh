#SBATCH --job-name=role-driven-inferences
#SBATCH --partition=gpu1_10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=slurms/role-driven-inferences.out
#SBATCH --error=slurms/role-driven-inferences.err

# process raw train data
python role-driven-inference/process_data.py \
    --file_path results/math_oai/Meta-Llama-3-8B-Instruct-greedy_train.jsonl \
    --data_name math_oai

# build BM25/Instructor/GTR index and retrieve
bash ./retrieval.sh

# role-driven inference
CUDA_VISIBLE_DEVICES="0" python role-driven-inference/infer_by_retrieval.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --infer_data_path results/retrieval/top3_roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl.problem.instructor_math_oai_test.jsonl \
    --tp 1 \
    --decoding greedy \
    --role_vote normal \
    --dataset math_oai \
    --split test

# eval
python role-driven-inference/eval_direct.py \
    --file_path results/math_oai/top3_roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl.problem.instructor_math_oai_test.jsonl-greedy_test_normal.jsonl \
    --data_name math_oai
