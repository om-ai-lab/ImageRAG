import json
import os
import pandas as pd
import re
from glob import glob

class JSONComparer:
    def __init__(self, jsonl_dir: str, baseline_file: str):
        self.jsonl_dir = jsonl_dir
        self.baseline_file = baseline_file
        self.results = {}

    def parse_option(self, output: str):
        match = re.match(r"\((.*?)\)", output)
        return match.group(1) if match else output

    def load_jsonl(self, file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question_id = item["Question_id"]

                # 处理 GT
                ground_truth = item["Ground truth"]
                if isinstance(ground_truth, list):
                    ground_truth = str(ground_truth[0]).strip()
                else:
                    ground_truth = str(ground_truth).strip()

                output = self.parse_option(item["output"].strip())

                # 解析路径信息
                summary = item.get("imagerag_summary", [])
                path_T = item.get("path_T", 0)
                lrsd_T = item.get("lrsd_T", 0)
                crsd_T = item.get("crsd_T", 0)

                data[question_id] = {
                    "Ground truth": ground_truth,
                    file_path: output,  # 只保留 output
                    "imagerag_summary": summary,
                    "path_T": path_T,
                    "lrsd_T": lrsd_T,
                    "crsd_T": crsd_T,
                }
        return data

    def compare_results(self):
        all_files = glob(os.path.join(self.jsonl_dir, "*.jsonl"))
        if self.baseline_file not in all_files:
            raise FileNotFoundError(f"Baseline file {self.baseline_file} not found in {self.jsonl_dir}")

        # 先加载 baseline 作为基准
        self.results = self.load_jsonl(self.baseline_file)
        baseline_name = "baseline"

        # 重命名 baseline 文件名
        for qid in self.results:
            self.results[qid][baseline_name] = self.results[qid].pop(self.baseline_file)

        # 加载其他文件并合并数据
        for file in all_files:
            if file == self.baseline_file:
                continue
            new_data = self.load_jsonl(file)
            for qid, values in new_data.items():
                if qid in self.results:
                    self.results[qid].update(values)
                else:
                    self.results[qid] = values

        # 计算 baseline_correct 和 comparison_correct
        for qid, data in self.results.items():
            gt = data["Ground truth"]
            baseline_output = data.get(baseline_name, "")

            # baseline 是否正确
            data["baseline_correct"] = baseline_output == gt

            # 遍历对比文件，检查是否正确
            comparison_correct = False
            correction_summary = None

            for file_name, output in data.items():
                if file_name in ["Ground truth", baseline_name, "imagerag_summary", "path_T", "lrsd_T", "crsd_T", "baseline_correct"]:
                    continue

                if output == gt:
                    comparison_correct = True

                    # 仅当 baseline 错误但当前文件正确时，记录 summary
                    if not data["baseline_correct"]:
                        correction_summary = data.get("imagerag_summary")

            data["comparison_correct"] = comparison_correct
            data["correction_summary"] = correction_summary if correction_summary else []

    def save_results(self, output_csv="comparison_results.csv"):
        df = pd.DataFrame.from_dict(self.results, orient='index').reset_index().rename(columns={'index': 'Question_id'})
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def display_results(self, num_rows=10):
        df = pd.DataFrame.from_dict(self.results, orient='index').reset_index().rename(columns={'index': 'Question_id'})
        print(df.head(num_rows))

# 示例调用
if __name__ == "__main__":
    # json_dir = "/data1/zilun/ImageRAG0226/data/eval"
    # baseline = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_baseline.jsonl"

    json_dir = "/data1/zilun/ImageRAG0226/data/eval"
    baseline = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_baseline.jsonl"
    comparer = JSONComparer(json_dir, baseline)
    comparer.compare_results()
    comparer.display_results()
    comparer.save_results()
