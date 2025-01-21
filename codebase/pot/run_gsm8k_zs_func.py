import openai
from time import sleep
from tool import synthesize_program, safe_execute, floatify_ans, simplify_ans, finqa_equal
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import os
import json
import argparse

client = openai.Client(base_url='http://127.0.0.1:37000/v1', api_key="None")


def zs_pot(question):
    model_name = "/media/zilun/wd-161/hf_download/Qwen2-3B-Instruct"
    full_prompt = f"""
import math
import numpy as np

# Question: {question}
# Answer this question by implementing a solver() function.
def solver():
    # Let's write a Python program step by step, and then return the answer
    # Firstly, we need define the following variable:
"""
    print(full_prompt)
    print('=======================')

    # greedy decoding
    got_result = False
    while not got_result:
        try:
            params = dict(
                max_tokens=360,
                top_p=1,
                timeout=60,
                temperature=0.0,
            )

            msg = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=msg,
                **params
            )

            model_output = response.choices[0].message.content.strip()
            print(model_output)
            got_result = True

        except Exception as e:
            print(e)
            sleep(3)

    program = synthesize_program(model_output, full_prompt)
    ans = safe_execute(program)
    prediction = floatify_ans(simplify_ans(ans, False))
    return prediction, full_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default='OPENAI_KEY', type=str)
    parser.add_argument("--dry_run", default=False, action='store_true')
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--model_name", default="/media/zilun/wd-161/hf_download/Qwen2-3B-Instruct", type=str)

    args = parser.parse_args()

    with open('data/gsm8K.json') as f:
        gsm_test = json.load(f)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    gsm_test = gsm_test[args.start:args.end]

    filename = f'outputs/gsm8K_zs_s{args.start}_e{args.end}_{dt_string}.jsonl'
    print(filename)

    writer = open(filename, 'w')
    for example in tqdm(gsm_test):
        # greedy decoding
        prediction, full_prompt = zs_pot(example)
        program = synthesize_program(prediction, full_prompt)
        ans = safe_execute(program)
        prediction = floatify_ans(simplify_ans(ans, False))
        gt_ans = example['answer']
        if finqa_equal(prediction, gt_ans):
            correct += 1
        else:
            wrong += 1
        print(prediction, '$', gt_ans, '$', correct / (correct + wrong))

        try:
            tmp = {'question': example['question'], 'answer': gt_ans, 'executed': prediction, 'generated': program}
            writer.write(json.dumps(tmp) + '\n')
        except Exception:
            continue
            
    writer.close()
    print()
    print(correct / (correct + wrong))


if __name__ == "__main__":
    main()