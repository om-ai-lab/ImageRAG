import pdb
import shutil
from pygments.lexer import default
from tqdm import tqdm
import os
import uuid
import numpy as np
import pickle as pkl
from PIL import Image
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import argparse
import logging
import openai
import pytorch_lightning as pl
import json
import math
import regex as re
import shortuuid

from codebase.utils import load_yaml



Image.MAX_IMAGE_PIXELS = None

# export PYTHONPATH=$PYTHONPATH:/data1/zilun/grsm/ImageRAG_git
# export PYTHONPATH=$PYTHONPATH:/media/zilun/fanxiang4t/GRSM/ImageRAG_git

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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def fituhr_inference_internvl(config):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
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

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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

    def load_image(image_file, input_size=448, max_num=12, use_dynamic=True):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        if use_dynamic:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            print("Use Dynamic")

        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    # init model
    model_path = config['fast_vlm_model']['model_path']

    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=False
    )

    # setup inference data
    questions = [json.loads(q) for q in open(config['question_file_path'], "r")]
    questions = get_chunk(questions, config['fast_vlm_model']['num_chunks'], config['fast_vlm_model']['chunk_idx'])
    answers_file = os.path.expanduser(config['answers_file_path'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for i in tqdm(range(0, len(questions), config['batch_size'])):
        count=i
        num_patches_list = []
        image_folder = []
        question_list = []
        category_list = []
        batch_end = min(i + config['batch_size'], len(questions))
        item_list = []
        for j in range(i, batch_end):
            item_list.append(questions[j])
            image_file = questions[j]['image']
            if "fit" in config['input_image_dir'].lower():
                image_path = os.path.join(config['input_image_dir'], image_file)
            else:
                image_path = os.path.join(config['input_image_dir'], image_file+".png")
            # 判断问题类别,进而确定模板
            category = questions[j]['category']
            qs = questions[j]['question']
            # 需要的区域任务要针对geochat进行OBB的格式转换
            if category in ["task4", "task5", "task6"]:
                # if 'fgrs' not in answers_file.split("/")[-1] and 'geochat' in answers_file.split("/")[-1]:
                if 'geochat' in answers_file.split("/")[-1]:
                    pattern = r'\{(.+?)\}'
                    matches = re.findall(pattern, qs)
                    for match in matches:
                        numbers_str = match
                        pattern = r'<(.+?)>'
                        numbers = re.findall(pattern, numbers_str)
                        rbox_np = np.array(numbers, dtype=float)
                        region_str = convert_obb_to_region_str(rbox_np)
                        qs = qs.replace(numbers_str, region_str)

            # batch inference, single image per sample (单图批处理)
            pixel_values = load_image(image_path, input_size=config["model_input_image_size"], max_num=6, use_dynamic=config["use_dynamic"]).to(torch.bfloat16).cuda()
            image_folder.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
            question_list.append(qs)
            category_list.append(category)

        image_folder = torch.cat(image_folder, dim=0)

        with torch.inference_mode():
            responses = model.batch_chat(
                tokenizer,
                image_folder,
                num_patches_list=num_patches_list,
                questions=question_list,
                generation_config=generation_config
            )
        for item, response, category in zip(item_list, responses, category_list):
            print(f'Category: {category}\nUser: {item}\nAssistant: {response}')

            ans_file.write(json.dumps({
                "question_id": item["question_id"],
                "image_id": item["image"],
                "category": category,
                "ground_truth": item["ground_truth"],
                "answer": response,
            }) + "\n")
            count = count + 1
            ans_file.flush()
    return answers_file


def inference():
    pl.seed_everything(2024)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='/media/zilun/fanxiang4t/GRSM/ImageRAG_git/config/config_fituhr_inference_internvl.yaml', help='Path to the configuration file.')
    parser.add_argument('--query_text', type=str, default='Suppose the top of this image represents north. How many aircraft are heading northeast? What is the color of the building rooftop to their southeast?', help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to the log file.')
    parser.add_argument('--base_url', type=str, default='http://127.0.0.1:34000/v1', help='base url')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    config = load_yaml(args.cfg_path)

    patch_saving_dir = config['patch_saving_dir']
    os.makedirs(patch_saving_dir, exist_ok=True)

    llmvqa_model_path = config['llmvqa_model']['model_path']

    if "InternVL" in llmvqa_model_path:
        answers_file = fituhr_inference_internvl(config)
    print(answers_file)


def eval_ComplexCompre(answer_file, param=None, group=None):
    if param==8 and group=="double":
        from codebase.inference.FIT_Eval.eval_complex_comprehension_8para_double import evaluation_metrics_ComplexCompre
        evaluation_metrics_ComplexCompre(answer_file, param=param, group=group)
    elif param==8 and group=="single":
        from codebase.inference.FIT_Eval.eval_complex_comprehension_8para_single import evaluation_metrics_ComplexCompre
        evaluation_metrics_ComplexCompre(answer_file, param=param, group=group)
    elif param==5 and group=="obb1":
        from codebase.inference.FIT_Eval.eval_complex_comprehension_5para_obb1 import evaluation_metrics_ComplexCompre
        evaluation_metrics_ComplexCompre(answer_file, param=param, group=group)
    elif param==5 and group=="obb2":
        from codebase.inference.FIT_Eval.eval_complex_comprehension_5para_obb2 import evaluation_metrics_ComplexCompre
        evaluation_metrics_ComplexCompre(answer_file, param=param, group=group)


if __name__ == "__main__":
    # inference()
    
    # answer_file = "/data1/zilun/grsm/ImageRAG_git/data/eval/eval_FITRS_complex_comprehension_eval_5para_complete_obb1_100_512_star_inference_obb1_100_512_nondynamic.jsonl"
    # eval_ComplexCompre(answer_file, param=5, group="obb1")

    # answer_file = "/data1/zilun/grsm/ImageRAG_git/data/eval/test_FITRS_complex_comprehension_eval_5para_complete_fit_01000_obb2_eval_5param_nodynamic448_0-1000_obb2.jsonl"
    # eval_ComplexCompre(answer_file, param=5, group="obb2")
    
    # answer_file = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/codebase/data/eval/eval_FITRS_complex_comprehension_eval_8para_complete_double_100_448_fit_inference_double_100_448_dynamic.jsonl"
    # eval_ComplexCompre(answer_file, param=8, group="double")
    
    answer_file = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/eval_FITRS_complex_comprehension_eval_8para_complete_single_100_448_star_inference_single_100_448_dynamic.jsonl"
    eval_ComplexCompre(answer_file, param=8, group="single")
    

    