import os
import pdb

import open_clip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ColorJitter
from PIL import Image
from collections import Counter
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import yaml
import pickle as pkl
import math
import regex as re
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from codebase.llm_template import clip_text_template, georsclip_text_template
from sklearn.cluster import DBSCAN


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def enlarge_roibox(roi_box, enlarge_factor, img_size):
    print("enlarge_factor: {}".format(enlarge_factor))
    if enlarge_factor == 1:
        return roi_box
    else: 
        w, h = img_size
        gt_bbox = roi_box

        x1, y1, x2, y2 = gt_bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width_bbox = x2 - x1
        height_bbox = y2 - y1

        new_width = width_bbox * enlarge_factor
        new_height = height_bbox * enlarge_factor

        new_x1 = center_x - new_width / 2.0
        new_y1 = center_y - new_height / 2.0
        new_x2 = center_x + new_width / 2.0
        new_y2 = center_y + new_height / 2.0
        
        # 确保缩放后的边界框不会超出图像的边界
        new_x1 = max(new_x1, 0)
        new_y1 = max(new_y1, 0)
        new_x2 = min(new_x2, w)
        new_y2 = min(new_y2, h)

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
        

def build_generative_vlm_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[0]
    y = rbboxes[1]
    w = rbboxes[2]
    h = rbboxes[3]
    a = rbboxes[4]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])
    polys = np.expand_dims(polys, axis=0)
    return polys


def convert_obb_to_region_str(rbox_np):
    angle = rbox_np[-1]
    polys = obb2poly_np_oc(rbox_np)
    x_left = np.clip(np.min(polys[:, [0, 2, 4, 6]], axis=1), 0, None)
    y_top = np.clip(np.min(polys[:, [1, 3, 5, 7]], axis=1), 0, None)
    x_right = np.max(polys[:, [0, 2, 4, 6]], axis=1)
    y_bottom = np.max(polys[:, [1, 3, 5, 7]], axis=1)
    region_str = f"<{int(x_left[0])}><{int(y_top[0])}><{int(x_right[0])}><{int(y_bottom[0])}>|<{int(angle)}>"
    return region_str


def load_yaml(config_filepath):
    # Load the YAML file
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def collect_fn(batch):
    batch_img_list = []
    bbox_coordinate_list = []
    for data in batch:
        batch_img, bbox_coordinate = data
        batch_img_list.append(batch_img.unsqueeze(0))
        bbox_coordinate_list.append(bbox_coordinate)
    batch_img_list = torch.cat(batch_img_list)
    return batch_img_list, bbox_coordinate_list


def extract_vlm_img_text_feat(query, key_text, coordinate_patchname_dict, patch_saving_dir, img_preprocess, text_tokenizer, fast_path_vlm, img_batch_size, feat_saving_dir, fastvlm_encoder_name):
    device = fast_path_vlm.logit_scale.device
    visfeat_saving_path = os.path.join(feat_saving_dir, "{}_vis_feat.pkl".format(fastvlm_encoder_name))
    print("Check visual feature saving path: {}".format(visfeat_saving_path))
    if os.path.exists(visfeat_saving_path):
        save_dict = pkl.load(open(visfeat_saving_path, "rb"))
        print("Cache found: {}".format(visfeat_saving_path))
        image_features, bbox_coordinate_list = save_dict["image_features"], save_dict["bbox_coordinate_list"]
    else:
        print("Does not contain {}, begin extracting features".format(visfeat_saving_path))
        with torch.no_grad(), torch.cuda.amp.autocast():
            patch_dataset = CCDataset(coordinate_patchname_dict, patch_saving_dir, img_preprocess)
            patch_dataloader = DataLoader(patch_dataset, pin_memory=True, batch_size=img_batch_size, num_workers=os.cpu_count() // 2, shuffle=False, collate_fn=collect_fn)
            image_feature_list = []
            bbox_coordinate_list = []
            for batch_img, bbox_coordinate in tqdm(patch_dataloader):
                batch_img = batch_img.to(device)
                image_features = fast_path_vlm.encode_image(batch_img)
                image_feature_list.append(image_features)
                bbox_coordinate_list.extend(bbox_coordinate)
            image_features = torch.cat(image_feature_list)
            save_dict = {
                "image_features": image_features,
                "bbox_coordinate_list": bbox_coordinate_list
            }
            pkl.dump(
                save_dict,
                open(visfeat_saving_path, "wb")
            )

    # prompt = "a photo contains "
    
    # if len(key_text) == 1:
    #     prompt += "{}".format(key_text[0])
    #     text_content = [text_tokenizer(prompt)]

    # elif len(key_text) > 1:
    #     for c in key_text:
    #         # c = c.replace("the ", "")
    #         if c != key_text[-1]:
    #             prompt += "{}, ".format(c)
    #         else:
    #             prompt += "and {}.".format(c)
    #     text_content = [text_tokenizer(prompt)] + [text_tokenizer(f"a photo of the {c}") for c in key_text]
    # else:
    #     print("No keyword detected, exiting...")
    #     exit()
    # print(prompt)
    
    
    if fastvlm_encoder_name == "clip":
        templates = clip_text_template
    elif fastvlm_encoder_name == "remoteclip" or fastvlm_encoder_name == "georsclip":
        templates = georsclip_text_template
        
    if len(key_text) == 1:
        all_keyphrase_text_without_prompt = [key_text[0]]
        texts = [template.replace('{}', key_text[0]) for template in templates]
        text_content = [text_tokenizer(texts, context_length=77).to(device)]
    
    elif len(key_text) > 1:
        all_keyphrase_text = []
        all_keyphrase_text_without_prompt = []
        temp_prompt = ""
        for i, c in enumerate(key_text):
            texts = [template.replace('{}', key_text[i]) for template in templates]
            all_keyphrase_text.append(texts)
            if i != len(key_text) - 1:
                if len(key_text) == 2:
                    temp_prompt += "{} ".format(c)
                else:
                    temp_prompt += "{}, ".format(c)
            else:
                temp_prompt += "and {}".format(c)
        # pdb.set_trace()
        all_keyphrase_text_without_prompt = key_text + [temp_prompt]
        summary_texts = [template.replace('{}', temp_prompt) for template in templates]
        all_keyphrase_text.append(summary_texts)
        text_content = [text_tokenizer(keyphrase_text, context_length=77).to(device) for keyphrase_text in all_keyphrase_text]

    else:
        print("No keyword detected, exiting...")
        exit()
    
    text_features_list = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for text_input in text_content:
            text_feature = fast_path_vlm.encode_text(text_input)
            text_feature = text_feature.mean(dim=0)
            # text_features = F.normalize(text_features, dim=-1)
            # text_features /= text_features.norm()
            text_features_list.append(text_feature.unsqueeze(0))
    text_features = torch.cat(text_features_list)
    keyword_feat_map = dict(zip(all_keyphrase_text_without_prompt, text_features_list))
    return image_features, text_features, keyword_feat_map, bbox_coordinate_list


def set_up_paraphrase_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def paraphrase_model_inference(model, tokenizer, query_text):
    prompt = query_text
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def setup_vlm_model(model_path, fast_vlm_model_name, device):
    if fast_vlm_model_name == "georsclip":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14-336-quickgelu',
            pretrained='openai',
            precision="fp16",
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14-336-quickgelu')
        checkpoint = torch.load(model_path, map_location=device)
        msg = model.load_state_dict(checkpoint)
        model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        print("Load GeoRSCLIP")

    elif fast_vlm_model_name == "clip":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14-336-quickgelu',
            pretrained='openai',
            precision="fp16",
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14-336-quickgelu')
        model = model.to(device).eval()
        print("Load CLIP")


    elif fast_vlm_model_name == "remoteclip":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14-quickgelu',
            pretrained='openai',
            precision="fp16"
        )
        tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14')
        checkpoint = torch.load(model_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint)
        print(msg)
        model = model.to(device).eval()
        print("Load RemoteCLIP")

    return model, img_preprocess, tokenizer


def setup_slow_text_encoder_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def setup_vqallm(llmvqa_model_path, llmvqa_model_name, generation_config, input_size, device, load_in_8bit=False):
    if "llava-onevision-qwen2-0.5b-ov" in llmvqa_model_name.lower():
        from codebase.vqa_llm.vaq_llm import VQA_LLM
        import warnings
        warnings.filterwarnings("ignore")
        vqa_llm = VQA_LLM(
            model_path=llmvqa_model_path,
            device=device
        )
        return vqa_llm

    elif "internvl" in llmvqa_model_name.lower():
        # init model
        model_path = llmvqa_model_path

        # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
        # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        # generation_config = dict(
        #     max_new_tokens=1024,
        #     do_sample=False
        # )
        generative_vlm_transform = build_generative_vlm_transform(input_size)
        return model, tokenizer, generation_config, dynamic_preprocess, generative_vlm_transform


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def obb2minhbb(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x_ctr,y_ctr,w,h]
    """
    x = rbboxes[0]
    y = rbboxes[1]
    w = rbboxes[2]
    h = rbboxes[3]
    a = np.radians(rbboxes[4])
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    # polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])
    # polys = np.expand_dims(polys, axis=0)
    vertices = [(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)]

    # 找到最小和最大的 x、y 坐标
    min_x = min(vertex[0] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    max_y = max(vertex[1] for vertex in vertices)

    # 计算 AABB 的中心点
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    # 计算 AABB 的宽度和高度
    w = max_x - min_x
    h = max_y - min_y

    return cx, cy, w, h


def convert_bboxes(obb1_bboxes):
    obb2_bboxes = []
    for obb1_bbox in obb1_bboxes:
        cx, cy, w, h = obb1_bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        obb2_bboxes.append((x1, y1, x2, y2))
    return obb2_bboxes


def sole_visualcue2mergedvisualcue(obb2_bboxes):

    # 初始化最小和最大坐标
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    # obb2_bboxes = convert_bboxes(obb1_bboxes)

    # 遍历所有边界框
    for bbox in obb2_bboxes:
        x1, y1, x2, y2 = bbox
        # 更新最小和最大坐标
        min_x = min(min_x, x1)
        min_y = min(min_x, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    # 计算大边界框的中心点坐标
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    # 计算大边界框的宽度和高度
    w = max_x - min_x
    h = max_y - min_y

    # 返回包含所有边界框的大边界框
    return [min_x, min_y, max_x, max_y]


def get_patch_scale_bbox(bbox, patch_scale, lower, upper):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    new_w = w * patch_scale
    new_h = h * patch_scale

    x1 = cx - new_w / 2
    y1 = cy - new_h / 2
    x2 = cx + new_w / 2
    y2 = cy + new_h / 2

    x1 = max(x1, lower)
    y1 = max(y1, lower)
    x2 = min(x2, upper)
    y2 = min(y2, upper)
    return [x1, y1, x2, y2]


# TODO: hard coding to 0-100
def visualcue2imagepatch(visual_cues, image, question, transform, input_size, max_num, use_dynamic, patch_scale, lower, upper):
    def get_bbox(bbox, w, h, normalize_max):
        x1, y1, x2, y2 = bbox
        return [int(x1 / normalize_max * w), int(y1 / normalize_max * h), int(x2 / normalize_max * w), int(y2 / normalize_max * h)]

    visual_cues_obb2 = convert_bboxes(visual_cues)
    bboxes = []
    for bbox in visual_cues_obb2:
        real_bbox = get_patch_scale_bbox(bbox, patch_scale=patch_scale, lower=lower, upper=upper)
        bboxes.append(real_bbox)
    question = question.split('<image>\n')[-1]
    final_instruction = "<image>\n"
    final_instruction += "Additional information:\n"
    for i, bbox in enumerate(bboxes):
        # The `final_instruction` variable is being constructed by appending additional information
        # about sub-patches to the original instruction. It includes details about each sub-patch's
        # location and bounding box coordinates. The final instruction is then updated in the data
        # dictionary before being written to the output file in JSON format.
        final_instruction += "Sub-patch {} at location <box>[{:.2f}, {:.2f}, {:.2f}, {:.2f}]</box>: <image>\n".format(i + 1, *bbox)
    final_instruction += question

    images, num_tiles = [], []
    images.append(image)
    num_tiles.append(1)
    w, h = image.size

    for bbox in bboxes:
        real_bbox = get_bbox(bbox, w, h, normalize_max=upper)
        crop = image.crop(real_bbox)
        images.append(crop)
        num_tiles.append(1)
    pixel_values = [transform(image, input_size=input_size, max_num=max_num, use_dynamic=use_dynamic) for image in images]
    pixel_values = torch.stack(pixel_values)
    return final_instruction, pixel_values, num_tiles


def calculate_similarity_matrix(img_feats, text_feats, logit_scale_exp, need_feat_normalize=True):
    img_feats = img_feats.detach().cpu().type(torch.float32)
    text_feats = text_feats.detach().cpu().type(torch.float32)
    logit_scale_exp = logit_scale_exp.detach().cpu()
    with torch.no_grad(), torch.cuda.amp.autocast():
        if need_feat_normalize:
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
        text2patch_similarity = (logit_scale_exp * img_feats @ text_feats.t()).t().softmax(dim=-1)
    return text2patch_similarity


def ranking_patch_t2p(bbox_coordinate_list, t2p_similarity, top_k):
    """
    If t2p_similarity is a 3 * 600 matrix. There are 3 key phrases and 600 image patches
    
    Step 1. Select most frequenced appeared image patches (top k)
    """
    values, index = t2p_similarity.topk(top_k)

    # should be 5 * 3 = 15 candidates if 5 keywords and top_k=3
    # top1patch_per_keyphrase = values[:, :1].flatten().tolist()
    # all key words together, most frequent appeared patches (topk)
    flat_values = values.flatten().tolist()
    flat_index = index.flatten().tolist()
    counter = Counter(flat_index)
    top_k_element_index = counter.most_common(top_k)
    top_k_element_index = [element for element, count in top_k_element_index]

    select_index_similarity_dict = dict()
    for i, i_element in enumerate(top_k_element_index):
        for j, j_element in enumerate(flat_index):
            if i_element == j_element:
                if i_element not in select_index_similarity_dict:
                    select_index_similarity_dict[i_element] = flat_values[j]
                else:
                    if select_index_similarity_dict[i_element] < flat_values[j]:
                        select_index_similarity_dict[i_element] = flat_values[j]

    index_select = list(select_index_similarity_dict.keys())
    similarity_select = list(select_index_similarity_dict.values())
    print("Frequency selection: {}".format(similarity_select))

    # top1 patch for each text keywords
    top1patch_value_per_keyphrase = values[:, :1].flatten().tolist()
    top1patch_index_per_keyphrase = index[:, :1].flatten().tolist()
    assert set(top_k_element_index) == set(index_select)

    select_top1_value_list = []
    select_top1_index_list = []
    for i, top1_index in enumerate(top1patch_index_per_keyphrase):
        if top1_index not in top_k_element_index:
            select_top1_value_list.append(top1patch_value_per_keyphrase[i])
            select_top1_index_list.append(top1_index)
    
    print("Rank1 selection: {}".format(select_top1_value_list))
    candidate_index = index_select + select_top1_index_list
    candidate_similarity = similarity_select + select_top1_value_list

    selected_bbox_coordinate_list = np.array(bbox_coordinate_list)[candidate_index]

    return selected_bbox_coordinate_list, candidate_similarity


def text_expand_model_inference(model, tokenizer, query_text):
    response_list = []
    for text in query_text:
        prompt =  "Could you write a few sentences (output in string) to explain the phrase I give you? Provide a detailed explanation including synonyms (if they exist) of the phrase. The explanation should be comprehensive enough to match the phrase with relevant class names based on sentence embedding similarity. Below are a few examples of phrases and their corresponding explanations: \n User: Airport \n Assistant: An airport is a facility where aircraft, such as airplanes and helicopters, take off, land, and are serviced. It typically consists of runways, taxiways, terminals for passenger and cargo handling, control towers for air traffic management, and hangars for aircraft maintenance. Airports serve as hubs for commercial aviation, connecting travelers and goods between local and international destinations. Synonyms for 'airport' include airfield, aerodrome, airstrip, and terminal (in certain contexts, referring specifically to the passenger facilities. \n Follow this structure for any phrase provided. User: {}".format(text)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response_list.append(response)
    return response_list


def setup_logger(log_path):
    logging.basicConfig(level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('my_logger')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def meta_df2clsimg_dict(meta_vector_database_df_selected, vectordatabase_dir):
    """
    Input: meta df
    Output: {
                cls1: [img_path1, img_path2, ... ],
                cls2: [img_path1, img_path2, ... ],
                ......
            }
    """
    meta_vector_database_df_selected['img_path_list'] = meta_vector_database_df_selected.apply(
        lambda row: os.path.join(vectordatabase_dir, row['dataset'], row['img_name_list']), axis=1
    )
    cls_img_dict = meta_vector_database_df_selected.groupby('cls_list')['img_path_list'].apply(list).to_dict()
    return cls_img_dict

def img_reduce(cls_img_dict, vlm, img_preprocess, reduce_fn="mean"):
    device = vlm.logit_scale.device

    def batch_inference_clip_vision_encoder(cls_img_dataloader):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_feature_list = []
            for batch_img in tqdm(cls_img_dataloader):
                batch_img = batch_img.to(device)
                image_features = vlm.encode_image(batch_img)
                image_feature_list.append(image_features)
            image_features = torch.cat(image_feature_list)
        return image_features

    cls_feat_dict = dict()
    for class_label in tqdm(cls_img_dict):
        img_list =  cls_img_dict[class_label]
        cls_dataset = VanillaDataset(img_list, img_preprocess)
        cls_dataloader = DataLoader(cls_dataset, pin_memory=True, batch_size=20, num_workers=8, shuffle=False)
        cls_image_features = batch_inference_clip_vision_encoder(cls_dataloader)
        cls_feat_dict[class_label] = cls_image_features
    pkl.dump(cls_feat_dict, open("cls_img_dict.pkl", "wb"))

    if reduce_fn == "mean":
        for class_label in tqdm(cls_feat_dict):
            cls_feat_dict[class_label] = cls_feat_dict[class_label].mean(-1)
    elif reduce_fn == "cluster":
        pass
    elif reduce_fn == "rerank_topn":
        pass

    return cls_feat_dict

def ranking_patch_visualcue2patch(bbox_coordinate_list, visualcue2patch_similarity, top_k=2):
    values, index = visualcue2patch_similarity.topk(top_k)
    # should be 5 * 3 = 15 candidates
    # top1patch_per_keyphrase = values[:, :1].flatten().tolist()
    flat_values = values.flatten().tolist()
    flat_index = index.flatten().tolist()
    counter = Counter(flat_index)
    top_k_element_index = counter.most_common(top_k)
    top_k_element_index = [element for element, count in top_k_element_index]

    select_index_similarity_dict = dict()
    for i, i_element in enumerate(top_k_element_index):
        for j, j_element in enumerate(flat_index):
            if i_element == j_element:
                if i_element not in select_index_similarity_dict:
                    select_index_similarity_dict[i_element] = flat_values[j]
                else:
                    if select_index_similarity_dict[i_element] < flat_values[j]:
                        select_index_similarity_dict[i_element] = flat_values[j]

    index_select = list(select_index_similarity_dict.keys())
    similarity_select = list(select_index_similarity_dict.values())
    print("Frequency selection: {}".format(similarity_select))

    top1patch_value_per_keyphrase = values[:, :1].flatten().tolist()
    top1patch_index_per_keyphrase = index[:, :1].flatten().tolist()
    assert set(top_k_element_index) == set(index_select)

    select_top1_value_list = []
    select_top1_index_list = []
    for i, top1_index in enumerate(top1patch_index_per_keyphrase):
        if top1_index not in top_k_element_index:
            select_top1_value_list.append(top1patch_value_per_keyphrase[i])
            select_top1_index_list.append(top1_index)
    print("Rank1 selection: {}".format(select_top1_value_list))

    candidate_index = index_select + select_top1_index_list
    candidate_similarity = similarity_select + select_top1_value_list

    selected_bbox_coordinate_list = np.array(bbox_coordinate_list)[candidate_index]
    return selected_bbox_coordinate_list, candidate_similarity


def reduce_visual_cue_per_cls(visual_cue_candidates_dict, keyword_feat_map, fast_path_vlm, reverse_map, reduce_fn, need_feat_normalize):
    reduced_visual_cue_candidates_dict = dict()
    if reduce_fn == "mean":
        for class_label in tqdm(visual_cue_candidates_dict):
            vsd_cues_feats = visual_cue_candidates_dict[class_label]
            if need_feat_normalize:
                vsd_cues_feats /= vsd_cues_feats.norm(dim=-1, keepdim=True)
            reduced_visual_cue_candidates_dict[class_label] = vsd_cues_feats.mean(0)
            
            
    elif reduce_fn == "cluster":
        for class_label in tqdm(visual_cue_candidates_dict):
            vsd_cues_feats = visual_cue_candidates_dict[class_label]
            if need_feat_normalize:
                vsd_cues_feats = vsd_cues_feats / vsd_cues_feats.norm(dim=-1, keepdim=True)
                
            if len(vsd_cues_feats) > 1:
                # pdb.set_trace()
                feats_np = vsd_cues_feats.cpu().numpy()  # 形状: (N, d)
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(feats_np)
                labels = clustering.labels_
                # 排除-1噪声
                valid_indices = labels != -1
                if valid_indices.sum() == 0:
                    reduced_visual_cue_candidates_dict[class_label] = vsd_cues_feats.mean(0)
                else:
                    # 选择最大簇
                    valid_labels = labels[valid_indices]
                    unique_labels, counts = np.unique(valid_labels, return_counts=True)
                    largest_cluster = unique_labels[np.argmax(counts)]
                    indices = np.where(labels == largest_cluster)[0]
                    # 计算簇中心向量均值
                    cluster_center = vsd_cues_feats[indices].mean(0)
                    # cluster_center = torch.from_numpy(cluster_center)
                    reduced_visual_cue_candidates_dict[class_label] = cluster_center
            else:
                reduced_visual_cue_candidates_dict[class_label] = vsd_cues_feats.mean(0)
                
    elif reduce_fn == "rerank":
        for class_label in tqdm(visual_cue_candidates_dict):
            original_keyphrase = reverse_map[class_label]
            text_feat = keyword_feat_map[original_keyphrase]
            vsd_cues_feats = visual_cue_candidates_dict[class_label]
            if len(vsd_cues_feats) > 1:
                # pdb.set_trace()
                t2p_similarity = calculate_similarity_matrix(vsd_cues_feats, text_feat, fast_path_vlm.logit_scale.exp())
                topn = min(3, t2p_similarity.shape[1])
                values, top_indices = t2p_similarity.topk(topn)
                # top_indices = torch.topk(t2p_similarity, topk).indices
                select_feature = vsd_cues_feats[top_indices].squeeze(0)
                aggregated_feature = select_feature.mean(dim=0)
                reduced_visual_cue_candidates_dict[class_label] = aggregated_feature
            else:
                reduced_visual_cue_candidates_dict[class_label] = vsd_cues_feats.mean(0)
            
    return reduced_visual_cue_candidates_dict


def select_visual_cue(vlm_image_feats, bbox_coordinate_list, visual_cue_candidates_dict, logit_scale_exp, need_feat_normalize):
    # (166, 512)
    """
    vlm_image_feats: patch feats for single image
    bbox_coordinate_list: corresponding coord for each feat
    visual_cue_candidates_dict: related feats in vsd
    """
    if need_feat_normalize:
        vlm_image_feats /= vlm_image_feats.norm(dim=-1, keepdim=True)
    patch_feats = vlm_image_feats.detach().cpu().type(torch.float32)
    # (3, 512)
    logit_scale_exp = logit_scale_exp.detach().cpu()
    visual_cue_candidates_stacked = []
    for visual_cue_candidates in visual_cue_candidates_dict:
        visual_cue_candidates_stacked.append(visual_cue_candidates_dict[visual_cue_candidates].detach().cpu().type(torch.float32).unsqueeze(0))
    visual_cue_candidates_stacked = torch.cat(visual_cue_candidates_stacked)
    visual_cue_feats = visual_cue_candidates_stacked
    visualcue2patch_similarity = (logit_scale_exp * patch_feats @ visual_cue_feats.t()).t().softmax(dim=-1)
    # pdb.set_trace()
    visual_cues,  visual_cues_similarity = ranking_patch_visualcue2patch(bbox_coordinate_list, visualcue2patch_similarity, top_k=2)
    return visual_cues, visual_cues_similarity


def setup_lrsd_vsd(config):

    # Create Chroma vector store
    slow_text_emb_model_path = config["text_embed_model"]["model_path"]
    text_embeddings = HuggingFaceEmbeddings(model_name=slow_text_emb_model_path)

    vsd_wd_flag = False
    vs_work_dir = os.path.join(config["work_dir"], config["vector_database"]["lrsd_vector_database_dir"])
    if os.path.exists(vs_work_dir):
        vsd_wd_flag = True

    meta_pkl_path = config["vector_database"]["lrsd_meta_pkl_path"]
    # 'img_name_list' 'label_list' 'feat'
    vector_database_content = pkl.load(open(meta_pkl_path, "rb"))
    assert len(vector_database_content["img_name_list"]) == len(vector_database_content["label_list"]) == len(
        vector_database_content["feat"])
    # imgname2label_dict = dict()
    label2imgname_dict = dict()
    imgname2feat_dict = dict()

    for i in tqdm(range(len(vector_database_content["img_name_list"]))):
        img_name = vector_database_content["img_name_list"][i]
        label = vector_database_content["label_list"][i]
        feat = vector_database_content["feat"][i]
        # imgname2label_dict[img_name] = label
        if label not in label2imgname_dict:
            label2imgname_dict[label] = [img_name]
        else:
            label2imgname_dict[label].append(img_name)
        imgname2feat_dict[img_name] = feat

    vectorstore = Chroma(
        collection_name="lrsd_vector_store4keyphrase_label_matching",
        embedding_function=text_embeddings,
        persist_directory=vs_work_dir,
        # Where to save data locally, remove if not necessary
        collection_metadata = {"hnsw:space": "l2"},
    )

    if not vsd_wd_flag:
        labels_in_database = list(label2imgname_dict.keys())
        meta = [{'type': 'text'}] * len(labels_in_database)
        vectorstore.add_texts(texts=labels_in_database, metadatas=meta)

    return vectorstore, label2imgname_dict, imgname2feat_dict


def setup_pub11_vsd(config):
    # Create Chroma vector store
    slow_text_emb_model_path = config["text_embed_model"]["model_path"]
    text_embeddings = HuggingFaceEmbeddings(model_name=slow_text_emb_model_path)

    vsd_wd_flag = False
    vs_work_dir = os.path.join(config["work_dir"], config["vector_database"]["crsd_vector_database_dir"])
    if os.path.exists(vs_work_dir):
        vsd_wd_flag = True

    meta_pkl_path = config["vector_database"]["crsd_meta_pkl_path"]
    # 'img_name_list' 'label_list' 'feat'
    vector_database_content = pkl.load(open(meta_pkl_path, "rb"))
    assert len(vector_database_content["img_name_list"]) == len(vector_database_content["label_list"]) == len(
        vector_database_content["feat"])
    # imgname2label_dict = dict()
    label2imgname_dict = dict()
    imgname2feat_dict = dict()

    for i in tqdm(range(len(vector_database_content["img_name_list"]))):
        img_name = vector_database_content["img_name_list"][i]
        label = vector_database_content["label_list"][i]
        feat = vector_database_content["feat"][i]
        # imgname2label_dict[img_name] = label
        if label not in label2imgname_dict:
            label2imgname_dict[label] = [img_name]
        else:
            label2imgname_dict[label].append(img_name)
        imgname2feat_dict[img_name] = feat

    vectorstore = Chroma(
        collection_name="crsd_vector_store4keyphrase_label_matching",
        embedding_function=text_embeddings,
        persist_directory=vs_work_dir,
        # Where to save data locally, remove if not necessary
        collection_metadata = {"hnsw:space": "l2"},
    )

    if not vsd_wd_flag:
        labels_in_database = list(label2imgname_dict.keys())
        meta = [{'type': 'text'}] * len(labels_in_database)
        batch_size = 10000
        for i in tqdm(range(0, len(labels_in_database), batch_size)):
            batch_texts = labels_in_database[i:i + batch_size]
            batch_meta = meta[i:i + batch_size]
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta)

    return vectorstore, label2imgname_dict, imgname2feat_dict


def filter_visual_cue_basedon_T(visual_cues, visual_cues_similarity, T):
    selected_visual_cue = []
    selected_visual_cue_confidence = []
    for cue, conf in zip(visual_cues, visual_cues_similarity):
        if conf > T:
            selected_visual_cue.append(cue)
            selected_visual_cue_confidence.append(conf)
    
    selected_with_confidence = list(zip(selected_visual_cue, selected_visual_cue_confidence))
    selected_with_confidence.sort(key=lambda x: -x[1])  # 按 confidence 从高到低排序

    # 分离出 visual_cue 和 confidence
    sorted_visual_cue = [cue for cue, conf in selected_with_confidence]
    sorted_confidence = [conf for cue, conf in selected_with_confidence]
    return sorted_visual_cue, sorted_confidence


class VanillaDataset(Dataset):
    def __init__(self, img_path_list, transform):
        self.img_path_list = img_path_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path_list)


class CCDataset(Dataset):
    def __init__(self, coordinate_patchname_dict, img_dir, transform):
        self.img_dir = img_dir
        self.coordinate_patchname_dict = coordinate_patchname_dict
        self.data = list(coordinate_patchname_dict.keys())
        self.transform = transform

    def __getitem__(self, index):
        bbox_coordinate = self.data[index]
        sample_name = self.coordinate_patchname_dict[bbox_coordinate]
        img_path = os.path.join(self.img_dir, sample_name)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, bbox_coordinate

    def __len__(self):
        return len(self.data)


def bbox_location(image_width, image_height, bbox):
    """
    :param image_width:
    :param image_height:
    :param bbox: old: (up left x, up left y, w, h), new: (x1, y1, x2, y2)
    :return:
    """
    # Define the 3x3 grid dimensions
    grid_width = image_width / 3
    grid_height = image_height / 3

    # Extract bbox details
    x1, y1, x2, y2 = bbox
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    # x, y, w, h = bbox

    # Define the boundaries for each of the 9 regions
    regions = {
        "Top-left":      (grid_width * 0, grid_height * 0, grid_width, grid_height),
        "Top-center":    (grid_width * 1, grid_height * 0, grid_width, grid_height),
        "Top-right":     (grid_width * 2, grid_height * 0, grid_width, grid_height),
        "Center-left":   (grid_width * 0, grid_height * 1, grid_width, grid_height),
        "Center":        (grid_width * 1, grid_height * 1, grid_width, grid_height),
        "Center-right":  (grid_width * 2, grid_height * 1, grid_width, grid_height),
        "Bottom-left":   (grid_width * 0, grid_height * 2, grid_width, grid_height),
        "Bottom-center": (grid_width * 1, grid_height * 2, grid_width, grid_height),
        "Bottom-right":  (grid_width * 2, grid_height * 2, grid_width, grid_height)
    }

    def intersection_area(target_bbox, region_bbox):
        """

        :param target_bbox: x1, y1, w1, h1
        :param region_bbox: x2, y2, w2, h2
        :return:
        """

        x1, y1, w1, h1 = target_bbox
        x2, y2, w2, h2 = region_bbox

        # Calculate the overlap boundaries
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        intersection_area = max(0, xB - xA) * max(0, yB - yA)

        return intersection_area

    # Calculate intersection area for each region
    overlaps = {
        region: intersection_area([x, y, w, h], [rx, ry, rw, rh])
        for region, (rx, ry, rw, rh) in regions.items()
    }

    # Return the region with the maximum overlap
    first_return = max(overlaps, key=overlaps.get)
    # del overlaps[max(overlaps, key=overlaps.get)]
    # second_return = max(overlaps, key=overlaps.get)
    # return "{} and {} blocks".format(first_return, second_return)
    return first_return