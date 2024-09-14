import os 
import sys
import argparse
sys.path.append(os.getcwd())

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils.data_process import read_jsonl, save_arr
from utils.prompts import GSM8K_examples, MATH_examples
from copy import deepcopy
import math

parser = argparse.ArgumentParser(description='args parse')

parser.add_argument('--model_name_or_path', default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('--lora_path', default=None)
parser.add_argument('--tp', type=int, default=1)
parser.add_argument('--nshots', type=int, default=0)
parser.add_argument('--slice', type=int, default=0)
parser.add_argument('--slice_total', type=int, default=0) # 大于0表示分为几段，0表示不分段
parser.add_argument("--decoding", choices=["greedy", "temperature"], required=True)
parser.add_argument("--split", choices=["test", "train"], required=True)
parser.add_argument("--dataset", choices=["gsm8k", "math_oai"], required=True)

args = parser.parse_args()


few_shots_prompt = ""
if args.nshots > 0:
    if args.dataset == "gsm8k":
        few_shots_prompt = "\n\n".join(GSM8K_examples[:args.nshots])
    elif args.dataset == "math_oai":
        few_shots_prompt =  " ".join(MATH_examples[:args.nshots])

# https://stackoverflow.com/questions/72464756/ray-on-slurm-problems-with-initialization
if args.tp >= 2:
    import ray
    ray.init(ignore_reinit_error=True, num_cpus=4)

print("Initialize LLM with parameters: ", args, flush=True)

use_lora = True if args.lora_path else False

if use_lora:
    # https://docs.vllm.ai/en/latest/models/lora.html
    # https://docs.vllm.ai/en/latest/offline_inference/llm.html
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp, enable_lora=True)
else:
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp, swap_space=8)

tokenizer = llm.get_tokenizer()

if args.decoding == "greedy":
    # https://docs.vllm.ai/en/stable/dev/sampling_params.html
    sampling_params = SamplingParams(
        temperature=0, # Zero means greedy sampling.
        max_tokens=512,
        stop=["</s>", "<|eot_id|>"] # https://github.com/vllm-project/vllm/issues/4180
    )
elif args.decoding == "temperature":
    sampling_params = SamplingParams(
        temperature=1,
        max_tokens=512,
        n=20,
        stop=["</s>", "<|eot_id|>"] # https://github.com/vllm-project/vllm/issues/4180
    )

actor_roles = ("NoRole","Mathematician", "Chemist", "Physicist", "Economist", "Computer scientist", "Anthropologist", "Doctor", "Linguist")

target_dataset_path = "./data/{}_{}.jsonl".format(args.dataset, args.split)
target_dataset = read_jsonl(target_dataset_path)
if args.slice_total > 0:
    # slice_total 大于0表示分段
    target_dataset_len = len(target_dataset)
    slice_len = math.ceil(target_dataset_len / args.slice_total)
    start_idx = args.slice * slice_len
    end_idx = (args.slice+1)*  slice_len if (args.slice+1)* slice_len < target_dataset_len else target_dataset_len
    target_dataset = target_dataset[start_idx:end_idx]


def preprocess_samples_gsm8k(samples):
    prompts_in_chat_template = []
    for role in actor_roles:
        system_prompt = "You are a " + role +  ". Please think step by step. Finally summarize your reason steps and give your answer with format: 'The answer is: {number}'"
        if role == "NoRole":
            system_prompt = "Please think step by step. Finally summarize your reason steps and give your answer with format: 'The answer is: {number}'"
        for sample in samples:
            message = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": few_shots_prompt + "Problem:\n{}\nSolution:\n".format(sample['question'])}
                    ]
            prompts_in_chat_template.append(llm.llm_engine.tokenizer.tokenizer.apply_chat_template(message, tokenize=False, add_generation_template=False))
    return prompts_in_chat_template


def preprocess_samples_math(samples):
    prompts_in_chat_template = []
    for role in actor_roles:
        system_prompt = "You are a " + role +  ". Please reason step by step, and put your final answer within \\boxed{{}}"
        if role == "NoRole":
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"
        for sample in samples:
            message = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": few_shots_prompt + "Problem:\n{}\nSolution:\n".format(sample['problem'])}
                    ]
            prompts_in_chat_template.append(llm.llm_engine.tokenizer.tokenizer.apply_chat_template(message, tokenize=False, add_generation_template=False))
    return prompts_in_chat_template

if args.dataset == "gsm8k":
    prompts_in_chat_template = preprocess_samples_gsm8k(target_dataset)
elif args.dataset == "math_oai":
    prompts_in_chat_template = preprocess_samples_math(target_dataset)


if use_lora:
    outputs = llm.generate(
        prompts_in_chat_template,
        sampling_params,
        lora_request=LoRARequest("lora", 1, args.lora_path)
    )
else:
    outputs = llm.generate(
        prompts_in_chat_template,
        sampling_params
    )

generated_samples = []
for output, sample in zip(outputs, target_dataset * len(actor_roles)):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    if args.decoding == "temperature":
        generated_text = list(map(lambda x: x.text, output.outputs))

    new_sample = deepcopy(sample)
    new_sample["prompt"] = prompt
    new_sample["generated_text"] = generated_text
    generated_samples.append(new_sample)

save_name = args.model_name_or_path.split("/")[-1]
if args.nshots > 0:
    save_name = args.model_name_or_path.split("/")[-1] + "_{}shots".format(args.nshots)

if args.slice_total > 0:
    save_arr(generated_samples,  "./results/{}/{}-{}_{}_{}.jsonl".format(args.dataset, save_name, args.decoding, args.split, args.slice))
else:
    save_arr(generated_samples,  "./results/{}/{}-{}_{}.jsonl".format(args.dataset,  save_name, args.decoding, args.split))
print(args.model_name_or_path, flush=True)

# print(total_result, flush=True)
