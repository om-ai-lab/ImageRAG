import json
import jsonlines
from tqdm import tqdm


# 从 JSON 文件读取数据
with open('/mnt/cfs/zilun/GRSM/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS-train-1415k_8para.json', 'r') as f:
    json_data = json.load(f)

# 将 JSON 数据转换为 JSON Lines 格式并写入文件
with jsonlines.open('/mnt/cfs/zilun/GRSM/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS-train-1415k_8para.jsonl', 'w') as writer:
    for obj in tqdm(json_data):
        # 将每个 JSON 对象转换为字符串并写入文件，每个对象占一行
        writer.write(obj)