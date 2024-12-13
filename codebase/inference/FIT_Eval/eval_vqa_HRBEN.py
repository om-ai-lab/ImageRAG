import argparse
import torch
import os
import json
from tqdm import tqdm

def Combine_VQA_GT_Category(answers_file, output_file, questions_file, answers_gt_file):
    with open(answers_file, 'r') as f1:
        answers_data_lines = f1.readlines()

    with open(questions_file, 'r') as f2:
        questions_data = json.load(f2)['questions']
    with open(answers_gt_file, 'r') as f3:
        answers_gt_data = json.load(f3)['answers']

    answers_gt_data_dict = {a['id']: a for a in answers_gt_data}
    questions_data_dict = {q['id']: q for q in questions_data}

    with open(output_file, 'w') as f:
        for line in tqdm(answers_data_lines, desc="Processing", unit="line"):
            data = json.loads(line)
            question_id = data['question_id']
            answer = answers_gt_data_dict.get(question_id)
            if answer is not None:
                data['ground_truth'] = answer['answer']
            else:
                data['ground_truth'] = ''
                print(f"No {question_id} answer!")
            question = questions_data_dict.get(question_id)
            if question is not None:
                data['category'] = question['type']
            else:
                data['category'] = ''
                print(f"No {question_id} type!")
        
            f.write(json.dumps(data) + '\n')

    print('done!')

def evaluation_metrics_HRBEN(data_path):
    base = [json.loads(q) for q in open(data_path, "r")]
    category_correct = {"presence": 0, "comp": 0}
    category_incorrect = {"presence": 0, "comp": 0}
    correct = 0
    incorrect = 0
    for answers in tqdm(base):
        gt = answers['ground_truth'].lower()
        answer = answers['answer'].lower()
        category = answers['category'].lower()
        if gt == answer:
            correct += 1
            category_correct[category] += 1
        else:
            incorrect += 1
            category_incorrect[category] += 1
            
    print('correct:', correct)
    print('incorrect:', incorrect)
    print('Total:', correct + incorrect)

    over_acc = 0

    print("Category-wise accuracies:")
    for cat, cat_corr in category_correct.items():
        cat_total_count = cat_corr + category_incorrect[cat]
        cat_acc = cat_corr / cat_total_count
        
        print(f"{cat}: {cat_corr}/{cat_total_count} ({cat_acc*100:.2f}%)")
        over_acc += cat_acc

    print('Average Acc:', over_acc / len(category_correct))

    overall_acc = correct / (correct + incorrect)
    print('Overall Acc:', overall_acc)
    print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine VQA ground truth with model answers and evaluate metrics.")
    parser.add_argument("--answer-file", type=str, default="HRBEN_answers_Geochat-7B.jsonl")
    parser.add_argument("--output-file", type=str, default="HRBEN_answers_Geochat-7B_combined.jsonl")
    parser.add_argument("--questions-file", type=str, default="HRBEN/USGS_split_test_phili_questions.json")
    parser.add_argument("--answers-gt-file", type=str, default="HRBEN/USGS_split_test_phili_answers.json")

    args = parser.parse_args()

    Combine_VQA_GT_Category(args.answer_file, args.output_file, args.questions_file, args.answers_gt_file)

    evaluation_metrics_HRBEN(args.output_file)
    
