from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import argparse
import json
import os

class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score



def main(root_path, model_answers_file_list):
    for model_answer_gt in model_answers_file_list:
        golden_reference = []
        candidate_reference = []
        print(f'\n########### {model_answer_gt.split("/")[-1].split(".")[0]} ##########')

        with open(os.path.join(root_path, model_answer_gt), 'r') as file: 
            for line in file:
                data = json.loads(line)
                golden_reference.append(data['ground_truth'])
                candidate_reference.append(data['answer'])

        golden_reference = {k: [{'caption': v}] for k, v in enumerate(golden_reference)}
        candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}

        evaluator = Evaluator()
        evaluator.do_the_thing(golden_reference, candidate_reference)
        print(evaluator.evaluation_report)
        print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models based on their answers.")
    parser.add_argument("--root_path", type=str, required=True, help="Root path where model answer files are located.")
    parser.add_argument("--model_answers_file_list", nargs='+', type=str, default=[
        "geochat-7B/FITRS_image_caption_answer_geochat-7B.jsonl", 
        "geochat-7B/FITRS_region_caption_answer_geochat-7B.jsonl"],
         help="List of model answer file paths relative to root_path.")
    
    args = parser.parse_args()
    main(args.root_path, args.model_answers_file_list)