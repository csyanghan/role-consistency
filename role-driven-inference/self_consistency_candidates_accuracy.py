import os 
import sys
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter
sys.path.append(os.getcwd())
from utils.math_parser import extract_answer
from utils.math_grade import math_equal
from utils.data_process import read_jsonl
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from typing import List
import nltk


# https://github.com/cshaib/diversity/blob/main/diversity/ngram_diversity.py

def ngram_diversity_score(
        data: List[str],
        num_n: int = 4, 
) -> float:
    """ Calculates corpus-level ngram diversity based on unique ngrams 
       (e.g., https://arxiv.org/pdf/2202.00666.pdf).

    Args:
        data (List[str]): List of documents. 
        num_n (int): Max ngrams to test up to. Defaults to 5. 

    Returns:
        float: ngram diveristy score.
    """
    score = 0 
    data = ' '.join(data).split(' ') # format to list of words

    for i in range(1, num_n + 1): 
        ngrams = list(nltk.ngrams(data, i))
        # num unique ngrams / all ngrams for each size n 
        score += len(set(ngrams)) / len(ngrams) 

    return round(score, 3)


def calculate_answers_count_given_arr(arr):
    count = {}
    for ans in arr:
        keys = list(count.keys())
        hit = False
        for count_key in keys:
            if math_equal(ans, count_key, match_type=match_type):
                count[count_key] += 1
                hit = True
        if not hit: count[ans] = 1
    cnt = len(count.keys())
    return cnt


def process_sample(sample, data_name):
    sample_answers = sample["generated_text"]
    if data_name == "math_oai": gt_answer = extract_answer(sample["solution"], data_name)
    elif data_name in ["gsm8k"]: gt_answer = sample["answer"]

    if isinstance(sample_answers, list):
        # 为了公平比较，只采样9个答案
        sample_answers = sample_answers[:9]
        inner_pred_answer = [extract_answer(sa, data_name) for sa in sample_answers]
        rst = []
        for ans in inner_pred_answer:
            if not ans:
                rst.append(0)
                continue
            if math_equal(gt_answer, ans, match_type=match_type):
                rst.append(1)
        return {
            "solution": gt_answer,
            "correct_arr": rst,
        }
    else:
        print("sample_answers is not list", sample_answers, flush=True)
        exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args parse')

    parser.add_argument('--file_path', default="./results/math_oai/Meta-Llama-3-8B-Instruct-temperature.jsonl", required=True)
    parser.add_argument("--data_name", choices=["gsm8k", "math_oai"], required=True)

    args = parser.parse_args()

    match_type = "math"

    print("Eval with: ", args, flush=True)
    
    target_dataset_path = "./data/{}_test.jsonl".format(args.data_name)
    target_dataset = read_jsonl(target_dataset_path)
    math_len = len(target_dataset)

    file_path = args.file_path
    actor_roles = ("NoRole","Mathematician", "Chemist", "Physicist", "Economist", "Computer scientist", "Anthropologist", "Doctor", "Linguist")

    generated_texts = read_jsonl(file_path)[:math_len*len(actor_roles)]
    print("generate results: ", len(generated_texts), "roles: ", len(actor_roles), flush=True)
    assert len(generated_texts) % len(actor_roles) == 0, "generate error"
    
    # https://github.com/ZubinGou/math-evaluation-harness/blob/db77bc6bb2a17c90ed2d63e88ebd8c40a82858c5/evaluate.py#L54
    extract_pred_answer = []
    with ProcessPool() as pool:
        def ps(x): return process_sample(x, args.data_name)
        extract_pred_answer_future = pool.map(ps, generated_texts, timeout=120)
        extract_pred_answer_iterator = extract_pred_answer_future.result()
        with tqdm(total=len(generated_texts), desc="Extract answer") as progress_bar:
            while True:
                try:
                    result = next(extract_pred_answer_iterator)
                    extract_pred_answer.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error, flush=True)
                except Exception as error:
                    print(error.traceback, flush=True)
                    exit()
                progress_bar.update(1) 
    print("extract answer", flush=True)

    # 初始化 final_result 字典
    final_result = {role: {"correct_arr": []} for role in actor_roles}
    # 计算每个角色的准确率
    for role_idx in range(len(actor_roles)):
        cur_role_idxs = range(role_idx * math_len, (role_idx + 1) * math_len)
        role_generate_texts = [extract_pred_answer[idx] for idx in cur_role_idxs]
        for ans in role_generate_texts:
            if ans:
                final_result[actor_roles[role_idx]]["sc_count_arr"].append(ans["self_consistency_count"])
                final_result[actor_roles[role_idx]]["sc_diversity_arr"].append(ans["self_consistency_diversity"])

    role_pred_answer = []
    for i in range(math_len):
        role_cur_idxs = [idx * math_len + i for idx in range(len(actor_roles))]
        # 变成一个二维数组， 同一个角色的答案放在第二维度数组里面， 第一维度是题目个数
        role_pred_answer.append([extract_pred_answer[idx] for idx in role_cur_idxs])
    
    def calculate_role_vote_accuracy(role_pred_answer):
        role_ans = []
        role_raw_ans = []
        for ans in role_pred_answer:
            if ans:
                role_ans.append(ans["role_answer"])
                role_raw_ans.append(ans["role_raw_answers"])
        return {
            "rc_answer": calculate_answers_count_given_arr(role_ans),
            "rc_diversity": ngram_diversity_score(role_raw_ans)
        }

    role_correct_arrs = []
    with ProcessPool() as pool:
        role_correct_arrs_future = pool.map(calculate_role_vote_accuracy, role_pred_answer, timeout=120)
        role_correct_arrs_iterator = role_correct_arrs_future.result()
        with tqdm(total=len(role_pred_answer), desc="Calculate Role Answer Diversity: ") as progress_bar:
            while True:
                try:
                    result = next(role_correct_arrs_iterator)
                    role_correct_arrs.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    # role_correct_arrs.append(False)
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    print("role consistency answer count:", np.mean([r["rc_answer"] for r in role_correct_arrs]), flush=True)
    print("role consistency reason path diversity:", np.mean([r["rc_diversity"] for r in role_correct_arrs]), flush=True)

