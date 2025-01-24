import argparse
import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import jsonlines
import PIL.Image as Image
import pickle as pkl
import cv2

Image.MAX_IMAGE_PIXELS = None
import openai
import shutil
import random


def load_json_jsonl(file_path):
    # Check the file extension and process accordingly
    if file_path.endswith(".jsonl"):
        # For JSONL files (line-delimited JSON)
        data = [json.loads(line) for line in open(file_path, "r")]
    elif file_path.endswith(".json"):
        # For standard JSON files
        with open(file_path, "r") as f:
            data = json.load(f)  # Load the entire JSON file as a single object
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return data


def dump_jsonline(content_lines, save_path):
    with jsonlines.open(save_path, 'w') as writer:
        for content_line in tqdm(content_lines):
            writer.write(content_line)


def make_segmentation_dataset(raw_json_path, save_jsonline_path, split):
    dumped_data_list = []
    raw_data = load_json_jsonl(raw_json_path)
    for i, raw_line_data in enumerate(tqdm(raw_data)):
        single_data = dict()
        user_input = raw_line_data["question"]
        gpt_input = raw_line_data["ground_truth"]
        conv = []
        user_conv = {"from": "human", "value": "<image>\n" + user_input}
        gpt_conv = {"from": "gpt", "value": gpt_input}
        width, height = raw_line_data["image_resolution"]
        conv.append(user_conv)
        conv.append(gpt_conv)
        single_data["id"] = raw_line_data["question_id"]
        single_data["image"] = raw_line_data["image"]
        single_data["width"] = width
        single_data["height"] = height
        single_data["conversations"] = conv
        single_data["object"] = raw_line_data["object"]
        single_data["task_category"] = raw_line_data["category"]
        single_data["mask"] = raw_line_data["ground_truth_mask"]
        dumped_data_list.append(single_data)
    dump_jsonline(dumped_data_list, save_jsonline_path)
    print("{} split done".format(split))


def make_geolocalization_dataset(raw_json_path, save_jsonline_path, split, image_root):
    dumped_data_list = []
    raw_data = load_json_jsonl(raw_json_path)
    for i, raw_line_data in enumerate(tqdm(raw_data)):
        single_data = dict()
        user_input = raw_line_data["question"]
        gpt_input = raw_line_data["ground_truth"]
        conv = []
        user_conv = {"from": "human", "value": "<image>\n" + user_input}
        gpt_conv = {"from": "gpt", "value": gpt_input}
        img_name = raw_line_data["image"]
        img_path = os.path.join(image_root, img_name)
        img = Image.open(img_path)
        width, height = img.size
        conv.append(user_conv)
        conv.append(gpt_conv)
        single_data["id"] = raw_line_data["question_id"]
        single_data["image"] = raw_line_data["image"]
        single_data["width"] = width
        single_data["height"] = height
        single_data["conversations"] = conv
        single_data["object"] = raw_line_data["object"]
        single_data["task_category"] = raw_line_data["category"]
        dumped_data_list.append(single_data)
    dump_jsonline(dumped_data_list, save_jsonline_path)
    print("{} split done".format(split))


def make_geolocalization_dataset(raw_json_path, save_jsonline_path, split, image_root):
    #     {
    #     "question_id": 100002,
    #     "category": "task9",
    #     "image": "lake_or_pond_0_3_rgb.jpg",
    #     "question": "[geolocalization]Can you find the geographic position and description for this image?",
    #     "ground_truth": "This is a lake or pond in Russian Federation, geolocated at UTM Grid Zones 37V, close to Yaroslavl, with precise GPS coordinates <gps>[57.738179, 39.751588]</gps>."
    # },
    # /data1/zilun/fmow/train/place_of_worship/place_of_worship_2493/place_of_worship_2493_0_rgb.jpg
    dumped_data_list = []
    raw_data = load_json_jsonl(raw_json_path)
    for i, raw_line_data in enumerate(tqdm(raw_data)):
        single_data = dict()
        user_input = raw_line_data["question"]
        gpt_input = raw_line_data["ground_truth"]
        conv = []
        user_conv = {"from": "human", "value": "<image>\n" + user_input}
        gpt_conv = {"from": "gpt", "value": gpt_input}
        img_name = raw_line_data["image"]

        parts = img_name.rsplit('_', 3)
        category = parts[0]
        id = parts[1]
        index = parts[2]
        extension = parts[3]
        rel_img_name = f"fmow/train/{category}/{category}_{id}/{img_name}"

        img_path = os.path.join(image_root, rel_img_name)
        img = Image.open(img_path)
        width, height = img.size
        conv.append(user_conv)
        conv.append(gpt_conv)
        single_data["id"] = raw_line_data["question_id"]
        single_data["image"] = rel_img_name
        single_data["width"] = width
        single_data["height"] = height
        single_data["conversations"] = conv
        single_data["object"] = category
        single_data["task_category"] = raw_line_data["category"]
        dumped_data_list.append(single_data)
    dump_jsonline(dumped_data_list, save_jsonline_path)
    print("{} split done".format(split))


def make_changedetection_dataset(raw_json_path, save_jsonline_path, split, image_root):
    dumped_data_list = []
    raw_data = load_json_jsonl(raw_json_path)
    all_data = []
    for raw_line_data in tqdm(raw_data):
        all_data += raw_line_data
    for i, raw_line_data in enumerate(tqdm(all_data)):
        single_data = dict()
        # raw_line_data = raw_line_data[0]
        user_input = raw_line_data["question"]
        gpt_input = raw_line_data["ground_truth"]
        conv = []
        user_conv = {"from": "human", "value": "<image>\n<image>\n" + user_input}
        gpt_conv = {"from": "gpt", "value": gpt_input}
        img_names = raw_line_data["image"]
        # test_000004_A.png
        # /data1/zilun/grsm/dataset/georsmllm_dataset/LEVIR-MCI-dataset/images/train/A
        img_name_A_list = img_names[0].split("_")
        img_name_B_list = img_names[1].split("_")
        img_A_rel_path = os.path.join(
            "LEVIR-MCI-dataset/images",
            img_name_A_list[0],
            "A",
            img_name_A_list[0] + "_" + img_name_A_list[1] + ".png"
        )
        img_B_rel_path = os.path.join(
            "LEVIR-MCI-dataset/images",
            img_name_B_list[0],
            "B",
            img_name_B_list[0] + "_" + img_name_B_list[1] + ".png"
        )
        img_path_A = os.path.join(image_root, img_A_rel_path)
        img_path_B = os.path.join(image_root, img_B_rel_path)
        img_A = Image.open(img_path_A)
        width_A, height_A = img_A.size
        img_B = Image.open(img_path_B)
        width_B, height_B = img_B.size
        conv.append(user_conv)
        conv.append(gpt_conv)

        single_data["id"] = raw_line_data["question_id"]
        single_data["image"] = [img_A_rel_path, img_B_rel_path]
        single_data["width"] = [width_A, width_B]
        single_data["height"] = [height_A, height_B]
        single_data["conversations"] = conv
        single_data["task_category"] = raw_line_data["category"]
        dumped_data_list.append(single_data)

    dump_jsonline(dumped_data_list, save_jsonline_path)
    print("{} split done".format(split))


def select_high_quality_data_train(raw_jsonl_path, save_jsonl_path, base_url="http://127.0.0.1:30000/v1"):
    import openai
    generation_param_dict = {
        "max_tokens": 128,
        "temperature": 0.5,
        "top_p": 0.99,
        "num_beams": 1,
        "timeout": 60
    }

    client = openai.Client(base_url=base_url, api_key="None")

    q_prompt = """
    Task: Parse if the given question consist of multiple conditions. Respond with only a python list of strings.

    Guidelines: 
    - If so, break down the instruction into a list of sub-questions, each containing one condition.
    - Otherwise return a list that contains the original question.

    Your task is to apply the same process to the question provided:

    Input question: {}
    Output: 
    """

    params = dict(
        temperature=generation_param_dict['temperature'],
        max_tokens=generation_param_dict['max_tokens'],
        top_p=generation_param_dict['top_p'],
        timeout=generation_param_dict['timeout'],
        # do_sample=True
    )
    dumped_data_list = []
    data_lines = load_json_jsonl(raw_jsonl_path)
    index = 0
    for data_line in tqdm(data_lines):
        conversations = data_line["conversations"]
        if len(conversations) == 2:
            question = conversations[0]["value"]
            question.replace("<image>\n", "").replace("[seg]", "").replace("[geolocation]", "").replace("[change]",
                                                                                                        "").replace(
                "[grounding]", "").replace("[detection]", "").replace("[reasoning]", "")
            msg = [
                {
                    "role": "user",
                    "content": q_prompt.format(question)
                }
            ]

            response = client.chat.completions.create(
                model="Qwen25-72B",
                messages=msg,
                **params
            )
            response_str = response.choices[0].message.content.strip()
            response_list = eval(response_str)
            if len(response_list) > 1:
                data_line["question_conditions"] = response_list
            else:
                data_line["question_conditions"] = []

        else:
            data_line["question_conditions"] = []
        dumped_data_list.append(data_line)

        if index % 100:
            dump_jsonline(dumped_data_list, save_jsonl_path)
            # print("save to {}".format(raw_jsonl_path))
        index += 1
    dump_jsonline(dumped_data_list, save_jsonl_path)


def select_high_quality_data_eval(raw_jsonl_path, save_jsonl_path, base_url="http://127.0.0.1:30000/v1"):
    generation_param_dict = {
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.99,
        # "num_beams": 1,
        "timeout": 60
    }

    client = openai.Client(base_url=base_url, api_key="None")

    # Example:
    # Input question: "Detect the location of all planes on the east bank of the river"
    # Output: ["Detect the location of all planes", "Select the one on the east bank of the river"]

    q_prompt = """
    Task: Parse if the given question consist of multiple conditions. Respond with only a python list of strings.

    Guidelines: 
    - If so, break down the instruction into a list of sub-questions, each containing one condition.
    - Otherwise return a list that contains the original question.

    Your task is to apply the same process to the question provided:

    Input question: {}
    Output: 
    """

    params = dict(
        temperature=generation_param_dict['temperature'],
        max_tokens=generation_param_dict['max_tokens'],
        top_p=generation_param_dict['top_p'],
        timeout=generation_param_dict['timeout'],
        # do_sample=True
    )
    dumped_data_list = []
    data_lines = load_json_jsonl(raw_jsonl_path)
    index = 0
    for data_line in tqdm(data_lines):
        question = data_line["question"]
        question.replace("<image>\n", "").replace("[seg]", "").replace("[geolocation]", "").replace("[change]",
                                                                                                    "").replace(
            "[grounding]", "").replace("[detection]", "").replace("[reasoning]", "")
        msg = [
            {
                "role": "user",
                "content": q_prompt.format(question)
            }
        ]

        response = client.chat.completions.create(
            model="Qwen25-72B",
            messages=msg,
            **params
        )
        response_str = response.choices[0].message.content.strip()
        response_list = eval(response_str)
        if len(response_list) > 1:
            data_line["question_conditions"] = response_list
            print("Breakdown to {}".format(response_list))
        else:
            data_line["question_conditions"] = []

        dumped_data_list.append(data_line)

        if index % 1000 == 0:
            dump_jsonline(dumped_data_list, save_jsonl_path)
            # print("save to {}".format(raw_jsonl_path))
        index += 1
    dump_jsonline(dumped_data_list, save_jsonl_path)


def copy_jpg_files(src_dir, dst_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Walk through the source directory
    for root, dirs, files in tqdm(os.walk(src_dir)):
        for file in files:
            # Check if the file ends with .jpg
            if file.lower().endswith('_rgb.jpg'):
                # Construct the full path to the source file
                src_file_path = os.path.join(root, file)
                # Construct the full path to the destination file
                dst_file_path = os.path.join(dst_dir, file)

                # Copy the file
                shutil.copy2(src_file_path, dst_file_path)
                # print(f"Copied: {src_file_path} -> {dst_file_path}")


def merge_train(input_file_path_list, output_file_path):
    # def jsons2jsonl(input_file_path_list, output_file_path):
    merged_data = []
    cutoff = 20000
    for file_path in tqdm(input_file_path_list):
        data = load_json_jsonl(file_path)
        random.shuffle(data)
        print(len(data))
        # if "geolocalization" in file_path:
        #     data = data[:cutoff]
        merged_data.extend(data)
    with jsonlines.open(output_file_path, 'w') as writer:
        for obj in tqdm(merged_data):
            writer.write(obj)


def main():
    # raw_jsonl_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/test_FITRS_complex_comprehension_eval_5para_complete_fit_1000_obb2.jsonl"
    # breakdown_jsonl_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/test_FITRS_complex_comprehension_eval_5para_complete_fit_1000_obb2_breakdown.jsonl"

    # # /media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task
    # select_high_quality_data_eval(raw_jsonl_path, breakdown_jsonl_path, base_url="http://127.0.0.1:30000/v1")

    # source_directory = '/data1/zilun/fmow/train'
    # destination_directory = '/data1/zilun/grsm/tmp_georsmllm_data/fmow/train'
    # copy_jpg_files(source_directory, destination_directory)

    json_root = "/data1/zilun/grsm/ImageRAG_0111/data/georsmllm_task_data"

    seg_train_raw_json_path = os.path.join(json_root, "seg_train.json")
    seg_train_processed_json_path = os.path.join(json_root, "seg_train_final.jsonl")
    make_segmentation_dataset(seg_train_raw_json_path, seg_train_processed_json_path, "train")

    # seg_test_raw_json_path = os.path.join(json_root, "seg_test.json")
    # seg_test_processed_json_path = os.path.join(json_root, "seg_test_final.jsonl")
    # make_segmentation_dataset(seg_test_raw_json_path, seg_test_processed_json_path, "test")

    change_train_raw_json_path = os.path.join(json_root, "change_train.json")
    change_train_processed_json_path = os.path.join(json_root, "change_train_final.jsonl")
    make_changedetection_dataset(change_train_raw_json_path, change_train_processed_json_path, "train", "/data1/zilun/grsm/dataset/georsmllm_dataset")

    # change_test_raw_json_path = os.path.join(json_root, "changedetection_test.json")
    # change_test_processed_json_path = os.path.join(json_root, "changedetection_test_final.jsonl")
    # make_changedetection_dataset(change_test_raw_json_path, change_test_processed_json_path, "test", "/data1/zilun/grsm/tmp_georsmllm_data")

    geoloc_train_raw_json_path = os.path.join(json_root, "geoloc_train.json")
    geoloc_train_processed_json_path = os.path.join(json_root, "geoloc_train_final.jsonl")
    make_geolocalization_dataset(geoloc_train_raw_json_path, geoloc_train_processed_json_path, "train", "/data1/zilun")

    output_file_path = os.path.join(json_root, "georsmllm_train.jsonl")
    merge_train(
        [seg_train_processed_json_path, change_train_processed_json_path, geoloc_train_processed_json_path],
        output_file_path
    )


if __name__ == "__main__":
    main()
    # tmp = "/data1/zilun/grsm/ImageRAG_0111/data/georsmllm_task_data/change_train.json"
    # raw_data = load_json_jsonl(tmp)
    # print(len(raw_data))
    # all_data = []
    # for raw_line_data in tqdm(raw_data):
    #     all_data += raw_line_data
    # for data in all_data:
    #     print(data)
    # print(len(all_data))