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

def process_sample(sample, data_name):
    sample_answers = sample["generated_text"]
    if data_name == "math_oai": gt_answer = extract_answer(sample["solution"], data_name)
    elif data_name in ["gsm8k"]: gt_answer = sample["answer"]

    if isinstance(sample_answers, list):
        # 为了公平比较，只采样9个答案
        sample_answers = sample_answers[:9]
        inner_pred_answer = [extract_answer(sa, data_name) for sa in sample_answers]
        # 筛选一下， 太长的回答直接过滤掉
        inner_pred_answer = list(filter(lambda x: x, inner_pred_answer))
        # print(gt_answer, inner_pred_answer)
        if len(inner_pred_answer) == 0:
            return {
                "solution": gt_answer,
                "greedy_answer": "-1",
                "temperature_vote_answer": "-1"
            }
        return {
            "solution": gt_answer,
            "greedy_answer": inner_pred_answer[0],
            "temperature_vote_answer": find_mode(inner_pred_answer)
        }
    else:
        ans = extract_answer(sample_answers, data_name)
        return {
            "solution": gt_answer,
            "greedy_answer": ans,
            "temperature_vote_answer": ans
        }

# 定义计算角色准确率的函数
def calculate_role_accuracy(cur_idx):
    if not cur_idx: return False
    greedy_correct = 0
    temperature_vote_correct = 0
    if math_equal(cur_idx["solution"], cur_idx["greedy_answer"], match_type=match_type):
        greedy_correct += 1
    if math_equal(cur_idx["solution"], cur_idx["temperature_vote_answer"], match_type=match_type):
        temperature_vote_correct += 1
    return {
        "greedy_correct": greedy_correct,
        "temperature_vote_correct": temperature_vote_correct
    }

def calculate_role_vote_accuracy(role_pred_answer_sample):
    role_pred_answer_sample = list(filter(lambda x: x, role_pred_answer_sample))
    role_greedy_answer = find_mode([role_sample["greedy_answer"] for role_sample in role_pred_answer_sample])
    role_temperature_vote_answer = find_mode([role_sample["temperature_vote_answer"] for role_sample in role_pred_answer_sample])
    role_greedy_correct = 0
    role_temperature_vote_correct = 0
    if math_equal(role_pred_answer_sample[0]["solution"] ,role_greedy_answer, match_type=match_type):
        role_greedy_correct = 1
        # print(role_pred_answer_sample, flush=True)
    if math_equal(role_pred_answer_sample[0]["solution"] , role_temperature_vote_answer, match_type=match_type):
        role_temperature_vote_correct = 1


    #  各角色之间不是投票，而是只要角色之间有对的即算答对
    one_correct_greedy = 0
    one_correct_temperature = 0
    for ans in role_pred_answer_sample:
        if math_equal(role_pred_answer_sample[0]["solution"] , ans["greedy_answer"], match_type=match_type):
            one_correct_greedy = 1
            break
    for ans in role_pred_answer_sample:
        if math_equal(role_pred_answer_sample[0]["solution"] , ans["temperature_vote_answer"], match_type=match_type):
            one_correct_temperature = 1
            break

    return {
        "role_greedy_correct": role_greedy_correct,
        "role_temperature_vote_correct": role_temperature_vote_correct,
        "one_correct_greedy": one_correct_greedy,
        "one_correct_temperature": one_correct_temperature
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args parse')

    parser.add_argument('--file_path', default="./results/math_oai/Meta-Llama-3-8B-Instruct-greedy.jsonl", required=True)
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
                    extract_pred_answer.append(False)
                except Exception as error:
                    print(error.traceback, flush=True)
                    exit()
                progress_bar.update(1) 
    print("extract answer", flush=True)

    # 初始化 final_result 字典
    final_result = {role: {"greedy_correct_rate": 0, "temperature_vote_correct_rate": 0} for role in actor_roles}
    # 计算每个角色的准确率
    for role_idx in range(len(actor_roles)):
        cur_role_idxs = range(role_idx * math_len, (role_idx + 1) * math_len)
        role_generate_texts = [extract_pred_answer[idx] for idx in cur_role_idxs]
        role_accuracy_arr = []
        with ProcessPool() as pool:
            # 将 find_mode 函数传递给进程池
            role_accuracy_arr_future = pool.map(calculate_role_accuracy, role_generate_texts, timeout=120)
            role_accuracy_arr_iterator = role_accuracy_arr_future.result()
            with tqdm(total=len(role_generate_texts), desc="Calculate "+ actor_roles[role_idx]) as progress_bar:
                while True:
                    try:
                        result = next(role_accuracy_arr_iterator)
                        role_accuracy_arr.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        print(error)
                        role_accuracy_arr.append(False)
                    except Exception as error:
                        print(error.traceback)
                        exit()
                    progress_bar.update(1) 
        for acc in role_accuracy_arr:
            if not acc: continue
            final_result[actor_roles[role_idx]]["greedy_correct_rate"] += acc["greedy_correct"]
            final_result[actor_roles[role_idx]]["temperature_vote_correct_rate"] += acc["temperature_vote_correct"]
        print("calculate ", actor_roles[role_idx], " success", final_result[actor_roles[role_idx]], flush=True)

    final_result["role_greedy_correct_rate"] = 0
    final_result["role_temperature_vote_correct_rate"] = 0
    final_result["one_correct_greedy_rate"] = 0
    final_result["one_correct_temperature_rate"] = 0
    role_pred_answer = []
    for i in range(math_len):
        role_cur_idxs = [idx * math_len + i for idx in range(len(actor_roles))]
        # 变成一个二维数组， 同一个角色的答案放在第二维度数组里面， 第一维度是题目个数
        role_pred_answer.append([extract_pred_answer[idx] for idx in role_cur_idxs])
    
    role_correct_arrs = []
    with ProcessPool() as pool:
        role_correct_arrs_future = pool.map(calculate_role_vote_accuracy, role_pred_answer, timeout=120)
        role_correct_arrs_iterator = role_correct_arrs_future.result()
        with tqdm(total=len(role_pred_answer), desc="Calculate Vote: ") as progress_bar:
            while True:
                try:
                    result = next(role_correct_arrs_iterator)
                    role_correct_arrs.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    role_correct_arrs.append(False)
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 
    for rc in role_correct_arrs:
        if not rc: continue
        final_result["role_greedy_correct_rate"] += rc["role_greedy_correct"]
        final_result["role_temperature_vote_correct_rate"] += rc["role_temperature_vote_correct"]
        final_result["one_correct_greedy_rate"] += rc["one_correct_greedy"]
        final_result["one_correct_temperature_rate"] += rc["one_correct_temperature"]
    
    for k in final_result.keys():
        if isinstance(final_result[k], dict):
            for k1 in final_result[k].keys():
                final_result[k][k1] = (final_result[k][k1] / math_len) *100
        else:
            final_result[k] = (final_result[k] / math_len) *100

    greedy_result = []
    temperature_result = []
    for r in actor_roles:
        greedy_result.append(final_result[r]["greedy_correct_rate"])
        temperature_result.append(final_result[r]["temperature_vote_correct_rate"])
    final_result["greedy_correct_rate_mean"] = np.mean(greedy_result)
    final_result["greedy_correct_rate_std"] = np.std(greedy_result)
    final_result["temperature_vote_correct_rate_mean"] = np.mean(temperature_result)
    final_result["temperature_vote_correct_rate_std"] = np.std(temperature_result)

    print(final_result, flush=True)

