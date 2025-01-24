import json
from tqdm import tqdm

# def process_jsonl(input_file_path, output_file_path):
#     count = 0
#     # 打开输入和输出文件
#     with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
#         for line in tqdm(input_file):
#             # 解析每一行 JSON 数据
#             data = json.loads(line)
#             if data['train_task_type'] in ['multiturn', 'image_classification', 'image_caption', 'region_caption', 'complex_compare']:
#                 continue
#             else:
#                 output_file.write(json.dumps(data) + '\n')
#                 count += 1
#     print(count)
# 
# # 示例使用
# input_file_path = '/data1/zilun/grsm/ImageRAG_git/data/train/AGMLLLM_final_5para_star_obb2_0-1000_nocoordquestion_0112.jsonl'  # 替换为你的输入 JSON Lines 文件路径
# output_file_path = '/data1/zilun/grsm/ImageRAG_git/data/train/AGMLLLM_final_5para_star_obb2_0-1000_nocoordquestion_0112_vqa.jsonl'  # 替换为你的输出 JSON Lines 文件路径
# process_jsonl(input_file_path, output_file_path)


def process_jsonl(input_file_path, output_file_path):
    import random
    # 打开输入和输出文件
    save_list = []
    negation_list = []
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in tqdm(input_file):
            # 解析每一行 JSON 数据
            data = json.loads(line)
            negation_flag = False
            for conversation in data['conversations']:
                if "I'm sorry" in conversation['value']:
                    negation_list.append(data)
                    negation_flag = True
                    break
            if not negation_flag:
                save_list.append(data)
                # output_file.write(json.dumps(data) + '\n')
        print(len(save_list), len(negation_list))
        negation_quota = 1000
        random.shuffle(save_list)
        random.shuffle(negation_list)
        final_list = save_list + negation_list[:negation_quota]
        print(len(final_list))
        for data in tqdm(final_list):
            output_file.write(json.dumps(data) + '\n')

    
# 示例使用
input_file_path = '/data1/zilun/grsm/ImageRAG_git/data/train/AGMLLLM_final_5para_star_obb2_0-1000_nocoordquestion_0112_cc_vqa_regionalcap.jsonl'  # 替换为你的输入 JSON Lines 文件路径
output_file_path = '/data1/zilun/grsm/ImageRAG_git/data/train/AGMLLLM_final_5para_star_obb2_0-1000_nocoordquestion_0112_cc_vqa_regionalcap_filternegation.jsonl'  # 替换为你的输出 JSON Lines 文件路径
process_jsonl(input_file_path, output_file_path)