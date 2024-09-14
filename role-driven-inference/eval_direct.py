import os 
import sys
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter
sys.path.append(os.getcwd())
from utils.math_parser import read_jsonl, extract_answer
from utils.math_grade import math_equal
from multiprocessing import cpu_count
from pebble import ProcessPool
from concurrent.futures import TimeoutError

def find_mode(answers):
    count = {}
    for ans in answers:
        keys = list(count.keys())
        hit = False
        for count_key in keys:
            if math_equal(ans, count_key):
                count[count_key] += 1
                hit = True
        if not hit: count[ans] = 1
    max_count = max(count.values())
    mode = [k for k, v in count.items() if v == max_count]
    return mode[0]

def process_sample(sample, data_name):
    gt_answer = extract_answer(sample["solution"], data_name)
    decoding_texts = sample["generated_text"]

    generated_answer = extract_answer(decoding_texts, data_name)
    if math_equal(gt_answer, generated_answer):
        return 1
    else:
        return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args parse')

    parser.add_argument('--file_path', default="./results/math_oai/Meta-Llama-3-8B-Instruct-greedy.jsonl", required=True)
    parser.add_argument("--data_name", choices=["gsm8k", "math_oai"], required=True)

    args = parser.parse_args()
    print("Eval with: ", args, flush=True)

    generated_texts = read_jsonl(args.file_path)

    print("generate results: ", len(generated_texts))

    result_arr = []

    def process_sample_with_name(sample):
        return process_sample(sample, args.data_name)
    
    with ProcessPool() as pool:
        result_arr_future = pool.map(process_sample_with_name, generated_texts, timeout=120)
        result_arr_iterator = result_arr_future.result()
        with tqdm(total=len(generated_texts), desc="Calculate token vote: ") as progress_bar:
            while True:
                try:
                    result = next(result_arr_iterator)
                    result_arr.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error, flush=True)
                    result_arr.append(0)
                except Exception as error:
                    print(error, flush=True)
                    exit()
                progress_bar.update(1) 

    print("correct number: ", sum(result_arr), flush=True)
    print("total number: ", len(result_arr), flush=True)
    print("correct rate: ", (sum(result_arr) / len(result_arr)) * 100, flush=True)
