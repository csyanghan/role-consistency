# https://github.com/ZubinGou/math-evaluation-harness/blob/main/parser.py

import re
import regex
import sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n
import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any
import jsonlines

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def read_jsonl(data_path):
    contents = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            contents.append(obj)
    return contents

def save_arr(arr, save_path):
    with jsonlines.open(save_path, "w") as writer:
        for d in arr:
            writer.write(d)

def lower_keys(example):  
    new_example = {}  
    for key, value in example.items():  
        if key != key.lower():  
            new_key = key.lower()  
            new_example[new_key] = value  
        else:  
            new_example[key] = value  
    return new_example 


def load_prompt(data_name, prompt_type):
    if data_name in ['gsm_hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']:
        data_name = "gsm8k"
    if data_name in ['math_oai', "hungarian_exam"]:
        data_name = "math"
    if data_name in ['sat_math']:
        data_name = "mmlu_stem"
    if prompt_type in ['platypus_fs']:
        prompt_type = "cot"
    if prompt_type in ['tool-integrated']:
        prompt_type = "tora"

    if prompt_type in ['cot', 'pal', 'tora']:
        prompt_path = "./prompts/{}/{}.md".format(prompt_type, data_name)
        if not os.path.exists(prompt_path):
            prompt_path = "./prompts/{}.md".format(prompt_type)
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                prompt = fp.read().strip() + "\n\n\n"
        else:
            print(f"Error: prompt file {prompt_path} not found")
            prompt = ""
    else:
        prompt = ""
    return prompt

def construct_prompt(example, data_name, args):
    # Base models
    if args.prompt_type in ["direct", "cot", "pal", "tool-integrated"]:
        demo_prompt = load_prompt(data_name, args.prompt_type)
        if args.prompt_type in ["direct", "cot"]:
            if data_name in ["minerva_math", "math", "math_oai", "mmlu_stem", "sat_math", "mathqa", "hungarian_exam"]:
                context = f"Problem:\n{example['question']}\nSolution:"
            else:
                context = f"Question: {example['question']}\nAnswer:"
            full_prompt = demo_prompt + context
        elif args.prompt_type == "pal":
            context = f"Question: {example['question']}"
            full_prompt = demo_prompt + context
        elif args.prompt_type in ['tool-integreted']:
            context = f"Question: {example['question']}\n\nSolution:"
            full_prompt = demo_prompt + context

    # SFT models
    elif args.prompt_type in ['self-instruct', 'tora']:
        full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
    elif args.prompt_type in ['self-instruct-boxed']:
        full_prompt = f"<|user|>\n{example['question']}\nEnclose the final answer using \\boxed{{}}.\n<|assistant|>\n"
    elif args.prompt_type == "wizard_zs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "deepseek-math":
        full_prompt = (
            "User: {instruction}\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}.\n\nAssistant:"
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "kpmath":
        full_prompt = (
            'User: Please reason step by step and put your final answer at the end '
            'with "The answer is: ".\n\n{instruction}\n\nAssistant:'
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    else:
        raise NotImplementedError(args.prompt_type)
    return full_prompt

key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}

def show_sample(sample, print_all_preds=False):
    print("=="*20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample['question']))
    if 'code' in sample:
        if print_all_preds:
            for code in sample['code']:
                print('-'*20)
                print("code:", code)
            print("Execution:", sample['report'])
        else:
            print("Solution:\n", sample['code'][0])
            print("Execution:", sample['report'][0])
    if 'pred' in sample:
        print("Prediction:", repr(sample['pred'][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key  = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text:str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

# units mainly from MathQA
unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m", "deg", "mile",
    "q .", "monkey", "prime", "ratio", "profit of rs",  "rd", "o", "gm",
    "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
    "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h",
    "no change", "men", "soldier", "pie", "bc", "excess", "st",
    "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise",
    "a . m", "th", "π r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "° f", "profit", "minw", "yr", "women",
    "feet", "am", "pm", "hr", "cu cm", "square", "v â € ™", "are",
    "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number",
    "kmph", "day", "hour", "minute", "min", "second", "man", "woman", 
    "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³", "hectare", "more",
    "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month",
    "km", "m", "cm", "mm", "apple", "liter", "loss", "yard",
    "pure", "year", "increase", "decrease", "d", "less", "Surface",
    "litre", "pi sq m", "s .", "metre", "meter", "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r'\\begin\{array\}\{.*?\}', r'\\begin{pmatrix}', string)  
    string = re.sub(r'\\end\{array\}', r'\\end{pmatrix}', string)  
    string = string.replace("bmatrix", "pmatrix")


    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string
    
    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ['x=', 'y=', 'z=', 'x\\in', 'y\\in', 'z\\in', 'x\\to', 'y\\to', 'z\\to']:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if string.startswith("{") and string.endswith("}") and string.isalnum() or \
        string.startswith("(") and string.endswith(")") and string.isalnum() or \
        string.startswith("[") and string.endswith("]") and string.isalnum():
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_multi_choice_answer(pred_str):
    # TODO: SFT models
    if 'Problem:' in pred_str:
        pred_str = pred_str.split("Problem:", 1)[0]
    pred_str = pred_str.replace("choice is", "answer is")
    patt = regex.search(r"answer is \(?(?P<ans>[abcde])\)?", pred_str.lower())
    if patt is not None:
        return patt.group('ans').upper()
    return 'placeholder'

def extract_numbers(text):
    # 使用正则表达式匹配数字
    pattern = re.compile(r'The answer is:.+-?\d+(?:\.\d+)?', re.DOTALL | re.I)
    numbers = pattern.findall(text)
    if len(numbers) == 0: return []
    format_answer = numbers[0]
    format_answer = format_answer.replace(",", "")
    numbers = re.findall(r'-?\d+(?:\.\d+)?', format_answer)
    return numbers

def extract_answer(pred_str, data_name="math_oai"):
    if data_name in ["mmlu_stem", "sat_math", "mathqa"]:
        return extract_multi_choice_answer(pred_str)
        
    if data_name == "gsm8k":
        pred_answer = extract_numbers(pred_str)
        if len(pred_answer) == 0: return False
        else: return pred_answer[0]

    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        # minerva_math
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = tmp.split('$. I hope', 1)[0].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    elif ('he answer is' in pred_str):
        pred = pred_str.split('he answer is')[-1].strip()
    elif ('final answer is' in pred_str):
        pred = pred_str.split('final answer is')[-1].strip()
    # elif extract_program_output(pred_str) != "":
        # fall back to program
        # pred = extract_program_output(pred_str)
    else: # use the last number
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if(len(pred) >= 1):
            pred = pred[-1]
        else: pred = ''

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    if len(pred) > 32 or len(pred) == 0:
        return False
    return pred


def parse_ground_truth(example: Dict[str, Any], data_name):
    if 'gt_cot' in example and 'gt' in example:
        if data_name in ["math", "math_oai", "ocw", "amps", "hungarian_exam"]:
            gt_ans = extract_answer(example['gt_cot'], data_name)
        else:
            gt_ans = strip_string(example['gt'])
        return example['gt_cot'], gt_ans

    # parse ground truth
    if data_name in ["math", "math_oai", "minerva_math", "ocw", "amps", "hungarian_exam"]:
        gt_cot = example['solution']
        gt_ans = extract_answer(gt_cot, data_name)
    
    # post process
    gt_cot = str(gt_cot).strip()
    gt_ans = strip_string(gt_ans)
    return gt_cot, gt_ans


def parse_question(example, data_name):
    question = ""
    if data_name == "asdiv":
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":
        body = example["Body"].strip()
        if not body.endswith("."):
            body = body + "."
        question = f'{body} {example["Question"].strip()}'
    elif data_name == "tabmwp":
        title_str = f'regarding "{example["table_title"]}" ' if example['table_title'] else ""
        question = f'Read the following table {title_str}and answer a question:\n'
        question += f'{example["table"]}\n{example["question"]}'
        if example['choices']:
            question += f' Please select from the following options: {example["choices"]}'
    elif data_name == "theorem_qa":
        question = f"{example['question'].strip()}\nTheorem: {example['theorem_def'].strip()}"
    elif data_name == "mmlu_stem":
        options = example['choices']
        assert len(options) == 4
        for i, (label, option) in enumerate(zip('ABCD', options)):
            options[i] = f"({label}) {str(option).strip()}"
        options = ", ".join(options)
        question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
    elif data_name == "sat_math":
        options = example['options'].strip()
        assert 'A' == options[0]
        options = '(' + options
        for ch in 'BCD':
            if f' {ch}) ' in options:
                options = regex.sub(f' {ch}\) ', f" ({ch}) ", options)
        question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
    elif data_name == "mathqa":
        example['problem'] = example['problem'][0].upper() + example['problem'][1:]
        options = example['options'].strip()
        if options[0] == '[':
            options = eval(options)
            options = ", ".join(options)
        assert 'a' == options[0], options
        for ch in 'abcde':
            if f'{ch} ) ' in options:
                options = regex.sub(f'{ch} \) {ch} \) ', f'{ch} ) ', options)
                options = regex.sub(f'{ch} \) ', f"({ch.upper()}) ", options)
        options = options.replace(' , ', ', ')
        question = f"{example['problem'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
    else:
        for key in ['question', 'problem', 'Question', 'input']:
            if key in example:
                question = example[key]
                break
    assert question != ""
    # Yes or No question
    _, gt_ans = parse_ground_truth(example, data_name)
    gt_lower = gt_ans.lower()
    if gt_lower in ["true", "false"]:
        question += " (True or False)"
    if gt_lower in ["yes", "no"]:
        question += " (Yes or No)"
    return question.strip()


def _test_extract_answer():
    text= """
    The answer is $\\boxed{\left(                                                                                                                      
\\begin{array}{ccc}                                                                                                                                          
 -13 & 4 & -2 \\\\
 7 & 8 & -3 \\\\
 0 & 18 & -7 \\\\
 6 & 12 & 5 \\\\
\\end{array}
\\right)}$.
"""
    print(extract_answer(text, "math"))
    # should output a dict


if __name__ == "__main__":
    _test_extract_answer()
    