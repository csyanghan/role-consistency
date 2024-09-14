### BM25
### BM25

python retrieval/build_index.py \
    --index_type bm25 \
    --key problem \
    --dataset_path role-driven-inference/data/math_oai/roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl \
    --index_root_dir retrieval_indices

python retrieval/evaluate_index.py \
    --index_name roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl.problem.bm25 \
    --top_k 3 \
    --retrieval_results_root_dir results/retrieval \
    --index_root_dir retrieval_indices \
    --dataset_path data/math_oai_test.jsonl \
    --key problem

### GTR
### GTR
python retrieval/build_index.py \
    --index_type gtr \
    --key problem \
    --dataset_path role-driven-inference/data/math_oai/roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl \
    --index_root_dir retrieval_indices

python retrieval/evaluate_index.py \
    --index_name roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl.problem.gtr \
    --top_k 3 \
    --retrieval_results_root_dir results/retrieval \
    --index_root_dir retrieval_indices \
    --dataset_path data/math_oai_test.jsonl \
    --key problem

## Instructor 
## Instructor 

python retrieval/build_index.py \
    --index_type instructor \
    --key problem \
    --dataset_path role-driven-inference/data/math_oai/roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl \
    --index_root_dir retrieval_indices

python retrieval/evaluate_index.py \
    --index_name roles_Meta-Llama-3-8B-Instruct-greedy_train.jsonl.problem.instructor \
    --top_k 3 \
    --retrieval_results_root_dir results/retrieval \
    --index_root_dir retrieval_indices \
    --dataset_path data/math_oai_test.jsonl \
    --key problem
