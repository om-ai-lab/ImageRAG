import json
from tqdm import tqdm
from collections import Counter
import os
from PIL import Image

mme_realworld_lite_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/codebase/inference/MME-RealWorld-RS/MME-RealWorld-Lite.json"
mme_realworld_lite_bbox_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/codebase/inference/MME-RealWorld-RS/MME-RealWorld-Lite-BBox.json"
mme_realworld_litebbox_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/codebase/inference/MME-RealWorld-RS/MME-RealWorld-Lite-toi.json"
mme_realworld_lite_qsid_bbox_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/codebase/inference/MME-RealWorld-RS/MME-RealWorld-Lite-BBox-addon.json"
img_dir = "/media/zilun/wd-161/datasets/MME-RealWorld-Lite/data/imgs"
with open(mme_realworld_lite_path, 'r') as file:
    questions = json.load(file)
questions = [question for question in questions if question["Subtask"] == "Remote Sensing"]
with open(mme_realworld_lite_bbox_path, 'r') as file:
    bboxes = json.load(file)
with open(mme_realworld_lite_qsid_bbox_path, 'r') as file:
    bboxes_addon = json.load(file)
deal_list = ['dota_v2_dota_v2_dota_v2_P4155.png', 'dota_v2_dota_v2_dota_v2_P8717.png', 'dota_v2_dota_v2_dota_v2_P5240.png', 'dota_v2_dota_v2_dota_v2_P8510.png', 'dota_v2_dota_v2_dota_v2_P6744.png', 'dota_v2_dota_v2_dota_v2_P9271.png', 'dota_v2_dota_v2_dota_v2_P10015.png', 'dota_v2_dota_v2_dota_v2_P3145.png', 'dota_v2_dota_v2_dota_v2_P8406.png', 'dota_v2_dota_v2_dota_v2_P6423.png', 'dota_v2_dota_v2_dota_v2_P6423.png', 'dota_v2_dota_v2_dota_v2_P6138.png']

print(len(questions))
print(len(bboxes))
print(len(deal_list))

ans_file = open(mme_realworld_litebbox_path, "w")
processed_list = []
for q in tqdm(questions):
    qs = q["Text"]
    qs_id = q["Question_id"]
    img_name = q["Image"]
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)
    w, h = img.size
    if img_name not in deal_list:
        bbox = bboxes[img_name][0]
    else:
        bbox = bboxes_addon[qs_id][0]
    bbox["x"] = max(0, bbox["x"])
    bbox["y"] = max(0, bbox["y"])
    bbox["width"] = max(0, bbox["width"])
    bbox["height"] = max(0, bbox["height"])
    new_box = [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
    q["gt_toi"] = new_box
    q["img_size"] = (w, h)
    processed_list.append(q)
    print(img_name, bbox)

print(len(processed_list))


with open(mme_realworld_litebbox_path, 'w') as f:
    json.dump(processed_list, f, indent=4)

ans_file.close()

