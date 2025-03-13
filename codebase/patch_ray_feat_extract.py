import random
import ray
import os
import numpy as np
from PIL import Image
import torch
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pdb
import pandas as pd
import open_clip
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything


import pickle as pkl
import math
import json
from codebase.patchify import cc_patchify, vit_patchify, grid_patchify
from codebase.utils import extract_vlm_img_text_feat


Image.MAX_IMAGE_PIXELS = None


def get_all_img_path(dataset_dir):
    """
    recursively get all image path inside the dataset directory
    """

    all_img_path_list = []
    filenames = glob.glob(os.path.join(dataset_dir, '**', '*.*'), recursive=True)
    for file_path in filenames:
        if file_path.lower().endswith("jpg") or file_path.lower().endswith("png"):
            all_img_path_list.append(file_path)

    return all_img_path_list


def assign_img_per_gpu(num_runner, total_img_number):
    """
    Assign images to gpus
    :param num_runner:
    :param total_img_number:
    :return:
    """
    base_img_per_runner = total_img_number // num_runner
    resource_assignment = [base_img_per_runner] * num_runner
    residual = total_img_number - base_img_per_runner * num_runner
    if residual == 0:
        return resource_assignment
    else:
        i = 0
        while i < residual:
            resource_assignment[i] += 1
            i += 1
        return resource_assignment


def get_img_path_assignment(all_imgs_paths, resource_assignment):
    """
    all_imgs_paths: list of str, contains abs path of imgs
    resource_assignment: list of int, contains image assignment per runner
    return: list of list, replace int in resource_assignment to actual img path
    """
    assert sum(resource_assignment) == len(all_imgs_paths)
    rl = []
    i = 0
    for img_per_runner in resource_assignment:
        temp_path_list = all_imgs_paths[i:i + img_per_runner]
        rl.append(temp_path_list)
        i += img_per_runner
    return rl


class ImageFeatureExtractionDataset(Dataset):
    """
    Grid Feature Extractor Dataset
    """
    def __init__(self, img_path_list, img_preprocess):
        self.img_path_list = img_path_list
        self.transform = img_preprocess

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = Image.open(img_path)
        image_after_transform = self.transform(image)
        return image_after_transform, img_path


@ray.remote
def spilt_singl_image_list(task_id, img_path_list, save_dir, patch_method):
    seed_everything(2024)
    os.makedirs(save_dir, exist_ok=True)
    patch_saving_dir = os.path.join(save_dir, patch_method)
    for image_path in tqdm(img_path_list):
        print(task_id, image_path)
        if  patch_method == "vit":
            img_resize, original_image, coordinate_patchname_dict, image_save_dir = vit_patchify(image_path, patch_saving_dir, patch_size=448)
        elif patch_method == "cc":
            img_resize, original_image, coordinate_patchname_dict, image_save_dir = cc_patchify(image_path, patch_saving_dir, c_denom=20)
        elif patch_method == "grid":
            img_resize, original_image, coordinate_patchname_dict, image_save_dir = grid_patchify(image_path, patch_saving_dir, max_grid=10)
        print(
            "resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))
    
def collect_fn(batch):
    batch_img_list = []
    bbox_coordinate_list = []
    for data in batch:
        batch_img, bbox_coordinate = data
        batch_img_list.append(batch_img.unsqueeze(0))
        bbox_coordinate_list.append(bbox_coordinate)
    batch_img_list = torch.cat(batch_img_list)
    return batch_img_list, bbox_coordinate_list


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
    

@ray.remote
def extract_per_image_feat(task_id, img_dir_list, model_path, clip_encoder_name):
    print(clip_encoder_name)
    seed_everything(2024)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda'
    print("device: {}".format(device))
    print("Loading given ckpt")

    if clip_encoder_name=="RemoteCLIP":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            # pretrained='openai',
            # precision="fp16"
        )
        checkpoint = torch.load(model_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=True)
        print(msg)
        model = model.to(device).eval()
        print("Load RemoteCLIP")
        
    elif clip_encoder_name=="GeoRSCLIP":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14-336-quickgelu',
            pretrained='openai',
            precision="fp16",
        )
        checkpoint = torch.load(model_path, map_location=device)
        msg = model.load_state_dict(checkpoint, strict=True)
        print(msg)
        model=model.to(device).eval()
        print("Load GeoRSCLIP")

        
    elif clip_encoder_name == "CLIP":
        print("Load CLIP")
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14-336-quickgelu',
            pretrained='openai',
            precision="fp16",
        )
        model = model.to(device).eval()
    
    elif clip_encoder_name == "MCIPCLIP":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14-336-quickgelu',
            pretrained='openai',
            precision="fp16",
        )
        checkpoint = torch.load(model_path, map_location=device)
        msg = model.load_state_dict(checkpoint, strict=True)
        print(msg)
        model=model.to(device).eval()
        print("Load MCIPCLIP")
    
    fast_path_vlm = model
                    
    for i, img_dir in enumerate(img_dir_list):
        coordinate_patchname_dict = dict()
        all_file_name = os.listdir(img_dir)
        for fname in all_file_name:
            if fname.endswith("png"):
                # /data1/zilun/ImageRAG0226/cache/patch/mmerealworld/old_cc/03553_Toronto/03553_Toronto_0-0-11500-7500.png
                lastname = fname.split("_")[-1].split(".")[0]
                coord = lastname.split("-")
                coord = tuple([int(c) for c in coord])
                coordinate_patchname_dict[coord] = fname
        
        visfeat_saving_path = os.path.join(img_dir, "{}_vis_feat.pkl".format(clip_encoder_name.lower()))
        print("Check visual feature saving path: {}".format(visfeat_saving_path))
        if os.path.exists(visfeat_saving_path):
            save_dict = pkl.load(open(visfeat_saving_path, "rb"))
            print("Cache found: {}".format(visfeat_saving_path))
            image_features, bbox_coordinate_list = save_dict["image_features"], save_dict["bbox_coordinate_list"]
        else:
            print("Does not contain {}, begin extracting features".format(visfeat_saving_path))
            with torch.no_grad(), torch.cuda.amp.autocast():
                patch_dataset = CCDataset(coordinate_patchname_dict, img_dir, img_preprocess)
                patch_dataloader = DataLoader(patch_dataset, pin_memory=True, batch_size=50, num_workers=os.cpu_count() // 16, shuffle=False, collate_fn=collect_fn)
                image_feature_list = []
                bbox_coordinate_list = []
                for batch_img, bbox_coordinate in tqdm(patch_dataloader):
                    # print(task_id, i, batch_img.shape)
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
                
                
def extract_img(args):
    from datetime import datetime
    toc = datetime.now()
    pl.seed_everything(2024)

    if args.ray_mode == "debug_cpu":
        ray.init(local_mode=True)
        args.num_cpu = 0
        args.num_gpu = 0
    elif args.ray_mode == "debug_gpu":
        args.num_cpu = 0
        args.num_gpu = 1
        ray.init(local_mode=True)
    elif args.ray_mode == "local_run":
        args.num_cpu = 16
        args.num_gpu = 1
        ray.init(local_mode=True)
    else:
        ray.init("auto")

    print('runner, cpu, gpu: {}, {}, {}'.format(args.num_runner, args.num_cpu, args.num_gpu))
    ray_resources = ray.available_resources()
    print('available devices: {}'.format(ray_resources))

    all_dirs_paths = []
    for img_dir in tqdm(os.listdir(args.dataset_img_dir)):
        dir_path = os.path.join(args.dataset_img_dir, img_dir)
        if os.path.isdir(dir_path):
            all_dirs_paths.append(dir_path)
    
    # all_imgs_paths = get_all_img_path(os.path.join(args.dataset_img_dir))
    print("total number of image dirs in directory {}: {}".format(args.dataset_img_dir, len(all_dirs_paths)))
    print("non repeat img dirss: {}".format(len(set(all_dirs_paths))))
    resource_assignment = assign_img_per_gpu(args.num_runner, len(all_dirs_paths))
    print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
    img_path_assignments = get_img_path_assignment(all_dirs_paths, resource_assignment)

    result_status = []
    for task_id, img_dir_list in enumerate(img_path_assignments):
        status = extract_per_image_feat.options(num_cpus=4, num_gpus=1).remote(
            task_id, img_dir_list, args.model_path, args.encoder_name
        )
        result_status.append(status)
        print("runner: {}".format(task_id))
    ray.get(result_status)

    tic = datetime.now()
    print(tic - toc)
        

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def split_image(args):

    from datetime import datetime
    toc = datetime.now()
    seed_everything(2024)

    if args.ray_mode == "debug_cpu":
        ray.init(local_mode=True)
        args.num_cpu = 0
        args.num_gpu = 0
    elif args.ray_mode == "debug_gpu":
        args.num_cpu = 0
        args.num_gpu = 1
        ray.init(local_mode=True)
    elif args.ray_mode == "local_run":
        args.num_cpu = 16
        args.num_gpu = 1
        ray.init(local_mode=True)
    else:
        ray.init("auto")

    print('runner, cpu, gpu: {}, {}, {}'.format(args.num_runner, args.num_cpu, args.num_gpu))
    ray_resources = ray.available_resources()
    print('available devices: {}'.format(ray_resources))


    with open(args.question_fpath, 'r') as file:
        questions = json.load(file)
    questions = [question for question in questions if question["Subtask"] == "Remote Sensing"]
    questions = get_chunk(questions, 1, 0)
    
    patch_saving_dir = os.path.join(args.save_dir, args.patch_method)
    
    img_list = []
    for line in tqdm(questions):
        img_name = line["Image"]
        img_path = os.path.join(args.dataset_img_dir, img_name)
        img_list.append(img_path)


    print("total number of line in question file {}: {}".format(args.question_fpath, len(questions)))
    print("non repeat imgs: {}".format(len(set(img_list))))
    resource_assignment = assign_img_per_gpu(args.num_runner, len(img_list))
    print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
    img_path_assignments = get_img_path_assignment(img_list, resource_assignment)

    os.makedirs(args.save_dir, exist_ok=True)
    print("image feature save dir: {}".format(args.save_dir))

    result_status = []
    for task_id, img_path_list in enumerate(img_path_assignments):
        # task_id, img_path_list, save_dir, patch_method
        status = spilt_singl_image_list.options(num_cpus=1, num_gpus=0).remote(
            task_id, img_path_list, args.save_dir, args.patch_method
        )
        result_status.append(status)
        print("runner: {}".format(task_id))
    ray.get(result_status)

    tic = datetime.now()
    print(tic - toc)
        

def main_split_image():
    # ray start --head --port=6379
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runner', type=int, default=75,
                        help='number of runner')
    # #gpu per runner
    parser.add_argument('--num_gpu', type=int, default=0,
                        help='number of gpu per trail')
    # #cpu per runner
    parser.add_argument('--num_cpu', type=int, default=50,
                        help='number of cpu per trail')

    parser.add_argument('--ray_mode', type=str, default="auto", help='local mode or auto mode')

    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    parser.add_argument('--dataset_img_dir', type=str,
                        default="/data9/shz/dataset/MME-RealWorld/remote_sensing",
                        help='dir of images needed to be extracted')

    parser.add_argument('--save_dir', type=str,
                        default="/data1/zilun/ImageRAG0226/cache/patch/mmerealworld",
                        help='dir of images needed to be extracted')
                        
    parser.add_argument('--patch_method', type=str,
                        default="vit",
                        help='cc vit or grid')

    parser.add_argument('--question_fpath', type=str,
                        default="/data1/zilun/ImageRAG0226/codebase/inference/MME-RealWorld-RS/MME_RealWorld.json",
                        help='question fpath')
                        
    args = parser.parse_args()

    split_image(args)
    

def main_extract_image():
    # ray start --head --port=6379
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runner', type=int, default=50,
                        help='number of runner')
    # #gpu per runner
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    # #cpu per runner
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='number of cpu per trail')

    parser.add_argument('--ray_mode', type=str, default="local", help='local mode or auto mode')

    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    parser.add_argument('--dataset_img_dir', type=str,
                        default="/data1/zilun/ImageRAG0226/cache/patch/mmerealworld/vit",
                        help='dir of images needed to be extracted')
    
    #  args.model_path, args.encoder_name
    parser.add_argument('--encoder_name', type=str, 
                        # default="CLIP",
                        # default="GeoRSCLIP", 
                        # default="RemoteCLIP", 
                        default="MCIPCLIP", 
                        help='local mode or auto mode')

    parser.add_argument('--model_path', type=str,
                        # default="/data1/zilun/ImageRAG0226/checkpoint/RS5M_ViT-L-14-336.pt",
                        # default="/data1/zilun/ImageRAG0226/checkpoint/RemoteCLIP/RemoteCLIP-ViT-L-14.pt",
                        default="/data1/zilun/ImageRAG0226/checkpoint/MCIP-ViT-L-14-336.pth",
                        help='model_path')
                        
    args = parser.parse_args()

    extract_img(args)
    



if __name__ == "__main__":
    # main_split_image()
    main_extract_image()