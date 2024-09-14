import os
import sys
sys.path.append(os.getcwd())
import argparse
import datasets
from tqdm import tqdm
from retrieval.kv_store import KVStore
from utils.data_process import read_jsonl, save_arr

def load_index(index_path: str) -> KVStore:
    index_type = os.path.basename(index_path).split(".")[-1]
    if index_type == "bm25":
        from retrieval.bm25 import BM25
        index = BM25(None).load(index_path)
    elif index_type == "instructor":
        from retrieval.instructor import Instructor
        index = Instructor(None, None, None).load(index_path)
    elif index_type == "e5":
        from retrieval.e5 import E5
        index = E5(None).load(index_path)
    elif index_type == "gtr":
        from retrieval.gtr import GTR
        index = GTR(None).load(index_path)
    elif index_type == "grit":
        from retrieval.grit import GRIT
        index = GRIT(None, None).load(index_path)
    else:
        raise ValueError("Invalid index type")
    return index

parser = argparse.ArgumentParser()
parser.add_argument("--index_name", type=str, required=True)

parser.add_argument("--top_k", type=int, required=False, default=10)
parser.add_argument("--retrieval_results_root_dir", type=str, required=False, default="results/retrieval")
parser.add_argument("--index_root_dir", type=str, required=False, default="retrieval_indices")
parser.add_argument("--dataset_path", required=False, default="data/math_oai_test.jsonl")
parser.add_argument("--key", required=True), # question, problem
args = parser.parse_args()

index = load_index(os.path.join(args.index_root_dir, args.index_name))
query_set = read_jsonl(args.dataset_path)
for query in tqdm(query_set):
    query_text = query[args.key]
    top_k = index.query(query_text, args.top_k, return_keys=True)
    query["retrieved"] = top_k

os.makedirs(args.retrieval_results_root_dir, exist_ok=True)
data_name = "top" + str(args.top_k) + "_" + args.index_name + "_" + args.dataset_path.split("/")[-1].split(".")[0]
output_path = os.path.join(args.retrieval_results_root_dir, f"{data_name}.jsonl")
save_arr(query_set, output_path)
print("save to: ", output_path)
