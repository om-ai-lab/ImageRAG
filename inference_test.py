import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import re
import ast
from function import *
from tqdm import tqdm
import warnings
# 忽略所有 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

image_dir = '/data9/zilun/grsm/dataset/georsmllm_dataset/SegDataset/test/images'
output_path = '/data9/zilun/grsm/dataset/georsmllm_dataset/SegDataset/test/test_miou.json'
ground_truth_mask_dir = '/data9/zilun/grsm/dataset/georsmllm_dataset/SegDataset/test/label'
QA_path = '/data9/zilun/grsm/dataset/georsmllm_dataset/SegDataset/test/QA_new.json'



import sys
sys.path.insert(0, "/data9/zilun/segment-anything-2") #优先级最高，第一个从这个目录下找需要导入的模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_checkpoint = "/data9/zilun/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "../sam2_configs/sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)



miou = {}
total_iou = 0
count = 0
QAS = json.load(open(QA_path,'r'))
for qa in tqdm(QAS):
    image_path = qa['image'].split('/')
    image = image_path[-2]+'/'+image_path[-1]
    answer = qa['ground_truth']
    image = Image.open(os.path.join(image_dir,image))
    width,height = qa['image_resolution']
    question = qa['question']
    pattern = r'<region>(.+?)</region>'
    match_box = re.findall(pattern, question)[0]
    pattern = r'\[(.+?)\]'
    num_str = re.findall(pattern, match_box)[0].split(',')
    q_box = denormlization([float(x) for x in num_str],width,height)
    q_box = [int(x) for x in q_box]
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)

    pattern = r'<box>(.+?)</box>'
    matches_box = re.findall(pattern, answer)
    pattern = r'<keypoint>(.+?)</keypoint>'
    matches_keypoint = re.findall(pattern, answer)

    boxs = []
    keypoints = []
    if len(matches_box)==0:
        continue
    count += 1
    for i in range(len(matches_box)):
        pattern = r'\[(.+?)\]'
        num_str = re.findall(pattern, matches_box[i])[0].split(',')
        box = denormlization([float(x) for x in num_str],width,height)
        boxs.append(box)
        pattern = r'\[(.+)\]'
        keypoint = list(ast.literal_eval(re.findall(pattern, matches_keypoint[i])[0]))
        keypoint = denormlization(keypoint,width,height)
        keypoints.append(keypoint)

    boxs = np.array(boxs)
    keypoints = np.array(keypoints)
    input_label = np.array([[1, 1, 1, 1, 1]] * len(keypoints))

    masks, scores, _ = predictor.predict(
        point_coords=keypoints,
        point_labels=input_label,
        multimask_output=False,
        box=boxs
        
    )

    ground_truth_mask = np.array(Image.open(f"{ground_truth_mask_dir}/{qa['question_id']}.png").convert('L'))//255
    pred_mask = np.zeros(ground_truth_mask.shape)
    if len(masks.shape)==4:
        assert masks.shape[1]==1
        masks = masks.squeeze(1)
    temp = np.ceil(np.mean(masks,axis=0)-1e-6)
    h,w = temp.shape
    q_box[2] = min(q_box[2],w)
    q_box[3] = min(q_box[3],h)
    pred_mask[q_box[1]:q_box[3],q_box[0]:q_box[2]] = np.array(temp,dtype=np.uint8)[q_box[1]:q_box[3],q_box[0]:q_box[2]]
    iou = calculate_miou(pred_mask,ground_truth_mask,2)
    miou[qa['question_id']] = iou
    total_iou += iou
miou['miou'] = total_iou/count

with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(miou, file, ensure_ascii=False, indent=4)