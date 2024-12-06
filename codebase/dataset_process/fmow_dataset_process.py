import json
import pdb
import shutil
from datetime import datetime

from tqdm import tqdm
import os
import uuid
import numpy as np
import pickle as pkl
from PIL import Image
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import glob
import pandas as pd
import re
from IRAG.imageRAG.codebase.statistics import visualize_hbbox
from torch.utils.data import Dataset, DataLoader


Image.MAX_IMAGE_PIXELS = None


def parse_fmow_coord(str_coord):
    """

    :param coord_text: "POLYGON ((73.9045104783130000 18.5879324392585232, 73.9333264511109292 18.5879324392585232,
    73.9333264511109292 18.5728217447537176, 73.9045104783130000 18.5728217447537176, 73.9045104783130000 18.5879324392585232))"

    :return:
    """
    # latitude = 18.5879324392585232
    # longitude = 73.9045104783130000

    # longitude, latitude = coord_text.split(",")[0].split("((")[1].split(" ")
    coord_text = str_coord.replace("POLYGON ((", "").replace("))", "").split(",")
    coord_text = [text.split(" ") for text in coord_text]
    coord_list = []
    for text in coord_text:
        coord = []
        for tmp in text:
            if len(tmp) == 0:
                pass
            else:
                coord.append(float(tmp))
        coord_list.append(coord)

    vertex_list = []
    for coordinate in coord_list:
        longitude, latitude = coordinate
        vertex_list.append([latitude, longitude])

    return vertex_list



def calculate_centroid(vertices):
    # 初始化坐标总和
    x_sum = 0
    y_sum = 0

    # 遍历顶点列表，累加所有x和y坐标
    for x, y in vertices:
        x_sum += x
        y_sum += y

    # 计算平均值
    centroid_x = x_sum / len(vertices)
    centroid_y = y_sum / len(vertices)

    return centroid_x, centroid_y


# def process_fmow(gt_dir, img_dir, out_dir, split):
#     img_name_list = []
#     bbox_list = []
#     cls_list = []
#     utm_list = []
#     image_original_size_list = []
#     raw_location_list = []
#     country_code_list = []
#
#     gt_dir = gt_dir + "/" + split
#     # search_pattern = os.path.join(gt_dir, '**', '*_rgb.json')
#     # rgb_json_files = glob.glob(search_pattern, recursive=True)
#     os.makedirs(os.path.join(out_dir, "fmow"), exist_ok=True)
#
#     all_subdir = os.listdir(gt_dir)
#     all_subdir = [subdir_name for subdir_name in all_subdir if os.path.isdir(os.path.join(gt_dir, subdir_name))]
#     for classname in tqdm(all_subdir):
#         all_subdir_name = os.listdir(os.path.join(gt_dir, classname))
#         for part in all_subdir_name:
#             all_filename = os.listdir(os.path.join(gt_dir, classname, part))
#             for img_name in all_filename:
#                 if img_name.endswith("_rgb.json"):
#                     rgb_json_file_path = os.path.join(gt_dir, classname, part, img_name)
#                     with open(rgb_json_file_path, "r") as f:
#                         json_f = json.load(f)
#                         bboxs = json_f["bounding_boxes"]
#                         img_name = json_f["img_filename"]
#                         img_width = json_f["img_width"]
#                         img_height = json_f["img_height"]
#                         country_code = json_f["country_code"]
#                         utm = json_f["utm"]
#                         raw_location = calculate_centroid(parse_fmow_coord(json_f["raw_location"]))
#                         for bbox in bboxs:
#                             cls = bbox["category"]
#                             id = bbox["ID"]
#                             if cls != "false_detection" and id !=-1:
#                                 bbox_coord = bbox["box"]
#                                 img_name_list.append(img_name)
#                                 bbox_list.append(bbox_coord)
#                                 cls_list.append(cls)
#                                 utm_list.append(utm)
#                                 country_code_list.append(country_code)
#                                 image_original_size_list.append((img_width, img_height))
#                                 raw_location_list.append(raw_location)
#                                 img_path = os.path.join(img_dir, split, classname, part, img_name)
#                                 img = Image.open(img_path)
#                                 crop_box = (bbox_coord[0], bbox_coord[1], bbox_coord[0] + bbox_coord[2], bbox_coord[1] + bbox_coord[3])
#                                 # visualize_hbbox(img_path, (bbox_coord[0], bbox_coord[1], bbox_coord[2], bbox_coord[3]), color=(0, 0, 255))
#                                 patch_img = img.crop(crop_box)
#                                 save_name = os.path.join(out_dir, "fmow", "{}_{}-{}-{}-{}.jpg".format(img_name.split(".")[0], bbox_coord[0], bbox_coord[1], bbox_coord[2], bbox_coord[3]))
#                                 patch_img.save(save_name)
#                                 del img
#     df = pd.DataFrame(
#         {
#             "img_name_list": img_name_list,
#             "cls_list": img_name_list,
#             "raw_location_list": raw_location_list,
#             "utm_list": utm_list,
#             "country_code_list": country_code_list,
#             "image_original_size_list": image_original_size_list,
#             "bbox_list": bbox_list,
#         }
#     )
#     pkl.dump(df, open(os.path.join(out_dir, "fmow", "info_{}.pkl".format(split)), "wb"))
#     return img_name_list, bbox_list, cls_list


def process_fmow_with_dataloader(gt_dir, img_dir, out_dir, split):

    class FmowDataset(Dataset):
        def __init__(self, json_path_list, classname_list, part_list, img_dir, split):
            self.img_dir = img_dir
            self.json_path_list = json_path_list
            self.split = split
            self.classname_list = classname_list
            self.part_list = part_list
            os.makedirs(os.path.join(out_dir, "fmow"), exist_ok=True)

        def __getitem__(self, index):
            rgb_json_file_path = self.json_path_list[index]
            cls_name = self.classname_list[index]
            part = self.part_list[index]

            with open(rgb_json_file_path, "r") as f:
                json_f = json.load(f)
                bboxs = json_f["bounding_boxes"]
                img_name = json_f["img_filename"]
                img_width = json_f["img_width"]
                img_height = json_f["img_height"]
                country_code = json_f["country_code"]
                utm = json_f["utm"]
                raw_location = calculate_centroid(parse_fmow_coord(json_f["raw_location"]))
                for bbox in bboxs:
                    cls = bbox["category"]
                    id = bbox["ID"]
                    if cls != "false_detection" and id != -1:
                        bbox_coord = bbox["box"]
                        img_path = os.path.join(self.img_dir, self.split, cls_name, part, img_name)
                        img = Image.open(img_path)
                        crop_box = (
                        bbox_coord[0], bbox_coord[1], bbox_coord[0] + bbox_coord[2], bbox_coord[1] + bbox_coord[3])
                        patch_img = img.crop(crop_box)
                        save_name = os.path.join(
                            out_dir,
                            "fmow",
                            "fmow_{}_{}-{}-{}-{}.jpg".format(
                                img_name.split(".")[0].split("/")[-1],
                                bbox_coord[0], bbox_coord[1], bbox_coord[2], bbox_coord[3]
                            )
                        )
                        patch_img.save(save_name)
                        del img
                        return img_name, bbox_coord, cls, utm, country_code, (img_width, img_height), raw_location


        def __len__(self):
            return len(self.json_path_list)


    def collect_fn(batch):
        img_name_list = []
        bbox_coord_list = []
        cls_list = []
        utm_list = []
        country_code_list = []
        img_size_list = []
        raw_location_list = []

        for data in batch:
            img_name, bbox_coord, cls, utm, country_code, (img_width, img_height), raw_location = data
            img_name_list.append(img_name)
            bbox_coord_list.append(bbox_coord)
            cls_list.append(cls)
            utm_list.append(utm)
            country_code_list.append(country_code)
            img_size_list.append((img_width, img_height))
            raw_location_list.append(raw_location)

        return img_name_list, bbox_coord_list, cls_list, utm_list, country_code_list, img_size_list, raw_location_list


    rgb_json_file_path_list = []
    classname_list = []
    part_list = []

    img_name_list = []
    bbox_list = []
    cls_list = []
    utm_list = []
    image_original_size_list = []
    raw_location_list = []
    country_code_list = []

    gt_dir = os.path.join(gt_dir, split)
    os.makedirs(os.path.join(out_dir, "fmow"), exist_ok=True)

    all_subdir = os.listdir(gt_dir)
    all_subdir = [subdir_name for subdir_name in all_subdir if os.path.isdir(os.path.join(gt_dir, subdir_name))]
    for classname in tqdm(all_subdir):
        all_subdir_name = os.listdir(os.path.join(gt_dir, classname))
        for part in all_subdir_name:
            all_filename = os.listdir(os.path.join(gt_dir, classname, part))
            for img_name in all_filename:
                if img_name.endswith("_rgb.json"):
                    rgb_json_file_path = os.path.join(gt_dir, classname, part, img_name)
                    rgb_json_file_path_list.append(os.path.join(classname, part, rgb_json_file_path))
                    classname_list.append(classname)
                    part_list.append(part)
    print("total {} images".format(len(rgb_json_file_path_list)))
    fmow_dataset = FmowDataset(rgb_json_file_path_list, classname_list, part_list, img_dir, split)
    fmow_dataloader = DataLoader(
        fmow_dataset,
        batch_size=64,
        pin_memory=True,
        num_workers=32,
        shuffle=False,
        collate_fn=collect_fn
    )

    for i, batch in tqdm(enumerate(fmow_dataloader)):
        img_name, bbox_coord, cls, utm, country_code, img_size, raw_location = batch
        img_name_list.extend(img_name)
        bbox_list.extend(bbox_coord)
        cls_list.extend(cls)
        utm_list.extend(utm)
        country_code_list.extend(country_code)
        image_original_size_list.extend(img_size)
        raw_location_list.extend(raw_location)

    df = pd.DataFrame(
        {
            "img_name_list": img_name_list,
            "cls_list": cls_list,
            "raw_location_list": raw_location_list,
            "utm_list": utm_list,
            "country_code_list": country_code_list,
            "image_original_size_list": image_original_size_list,
            "bbox_list": bbox_list,
        }
    )
    pkl.dump(df, open(os.path.join(out_dir, "fmow", "info_{}.pkl".format(split)), "wb"))
    return img_name_list, bbox_list, cls_list


def main():
    start = datetime.now()
    fmow_gt_dir = "/media/zilun/wd-161/datasets/fmow/groundtruth/fmow"
    fmow_img_dir = "/media/zilun/wd-161/datasets/fmow"
    img_output_root = "/media/zilun/mx500/ImageRAG_database/cropped_img"
    fmow_img_path_list, fmow_bbox_list, fmow_cls_list = process_fmow_with_dataloader(fmow_gt_dir, fmow_img_dir, img_output_root, "train")
    print(fmow_img_path_list[:10], fmow_cls_list[:10])
    print(datetime.now() - start)


if __name__ == "__main__":
    main()