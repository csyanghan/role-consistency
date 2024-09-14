import os
import sys
sys.path.append(os.getcwd())
import argparse
import datasets
from typing import List
from retrieval.kv_store import KVStore
from utils.data_process import read_jsonl

def get_index_name(args: argparse.Namespace) -> str:
    return os.path.basename(args.dataset_path) + "." + args.key

def create_index(args: argparse.Namespace) -> KVStore:
    index_name = get_index_name(args)

    if args.index_type == "bm25":
        from retrieval.bm25 import BM25
        index = BM25(index_name)
    elif args.index_type == "instructor":
        from retrieval.instructor import Instructor
        query_instruction = "Represent the {} retrieving relevant {}:".format(args.key, args.key)
        key_instruction = "Represent the {} for retrieval:".format(args.key)
        index = Instructor(index_name, key_instruction, query_instruction)
    elif args.index_type == "gtr":
        from retrieval.gtr import GTR
        index = GTR(index_name)
    else:
        raise ValueError("Invalid index type")
    return index

def create_kv_pairs(data: List[dict], key: str) -> dict:
    return {record[key]: record["role"] for record in data}

parser = argparse.ArgumentParser()
parser.add_argument("--index_type", required=True) # bm25, instructor, e5, gtr, grit
parser.add_argument("--key", required=True), # question, problem

parser.add_argument("--dataset_path", required=False, default="results/math_oai/Meta-Llama-3-8B-Instruct-greedy_train.jsonl")
parser.add_argument("--index_root_dir", required=False, default="retrieval_indices")
args = parser.parse_args()

corpus_data = read_jsonl(args.dataset_path)

index = create_index(args)
kv_pairs = create_kv_pairs(corpus_data, args.key)
index.create_index(kv_pairs)

index_name = get_index_name(args)
index.save(args.index_root_dir)
