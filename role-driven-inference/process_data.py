import os 
import sys
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter
sys.path.append(os.getcwd())

from utils.math_parser import extract_answer
from utils.math_grade import math_equal
from utils.data_process import read_jsonl, save_arr
from pebble import ProcessPool
from concurrent.futures import TimeoutError

def find_mode(answers):
    # match_type: math/char
    # print("find_mode", match_type, flush=True)
    count = {}
    for ans in answers:
        keys = list(count.keys())
        hit = False
        for count_key in keys:
            if math_equal(ans, count_key, match_type=match_type):
                count[count_key] += 1
                hit = True
        if not hit: count[ans] = 1
    max_count = max(count.values())
    mode = [k for k, v in count.items() if v == max_count]
    return mode[0]

def process_sample(samples, data_name):
    rst_role_data = []
    selected_roles = []
    selected_sample = None
    for sample, role in zip(samples, actor_roles):

        if isinstance(sample["answer"], bool): break

        sample_answers = sample["generated_text"]
        if data_name == "math_oai": gt_answer = extract_answer(sample["solution"], data_name)
        elif data_name in ["gsm8k"]: gt_answer = sample["answer"]

        if isinstance(sample_answers, list):
            # temperature
            inner_pred_answer = [extract_answer(sa, data_name) for sa in sample_answers]
            # 筛选一下， 太长的回答直接过滤掉
            inner_pred_answer = list(filter(lambda x: x, inner_pred_answer))
            ans = find_mode(inner_pred_answer)
            if math_equal(gt_answer, ans):
                sample["role"] = role
                rst_role_data.append(sample)
                break
        else:
            ans = extract_answer(sample_answers, data_name)
            if math_equal(gt_answer, ans):
                # sample["role"] = role
                selected_roles.append(role)
                selected_sample = sample
                # rst_role_data.append(sample)
                # break
    if len(selected_roles) > 0:
        selected_sample["role"] = selected_roles
        rst_role_data.append(selected_sample)
    return rst_role_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args parse')

    parser.add_argument('--file_path', default="./results/math_oai/Meta-Llama-3-8B-Instruct-greedy.jsonl", required=True)
    parser.add_argument("--data_name", choices=["gsm8k", "math_oai"], required=True)

    args = parser.parse_args()

    match_type = "math"

    print("Eval with: ", args, flush=True)
    
    target_dataset = read_jsonl(args.file_path)
    actor_roles = ("NoRole","Mathematician", "Chemist", "Physicist", "Economist", "Computer scientist", "Anthropologist", "Doctor", "Linguist")

    assert len(target_dataset) % len(actor_roles) == 0, "generate error"
    math_len = int(len(target_dataset) / len(actor_roles))

    role_pred_answer = []
    for i in range(math_len):
        role_cur_idxs = [idx * math_len + i for idx in range(len(actor_roles))]
        # 变成一个二维数组， 不同角色的答案放在第二维度数组里面， 第一维度是题目个数
        role_pred_answer.append([target_dataset[idx] for idx in role_cur_idxs])

    # https://github.com/ZubinGou/math-evaluation-harness/blob/db77bc6bb2a17c90ed2d63e88ebd8c40a82858c5/evaluate.py#L54
    role_correct_data = []
    correct_cnt = 0
    with ProcessPool() as pool:
        def ps(x): return process_sample(x, args.data_name)
        extract_pred_answer_future = pool.map(ps, role_pred_answer, timeout=120)
        extract_pred_answer_iterator = extract_pred_answer_future.result()
        with tqdm(total=len(role_pred_answer), desc="Extract answer") as progress_bar:
            while True:
                try:
                    result = next(extract_pred_answer_iterator)
                    role_correct_data += result
                    if len(result) > 0:
                        correct_cnt += 1
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error, flush=True)
                except Exception as error:
                    print(error.traceback, flush=True)
                    exit()
                progress_bar.update(1) 
    
    save_name = "roles_" + args.file_path.split("/")[-1]
    save_arr(role_correct_data, "./role-driven-inference/data/{}/{}".format(args.data_name, save_name))

    print("role save success, correct rate: {}".format(correct_cnt / len(role_pred_answer)), flush=True)

    