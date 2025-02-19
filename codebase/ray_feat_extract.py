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
import json
import pickle as pkl
import pandas as pd
import open_clip
import pytorch_lightning as pl


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
def extract_image_feature(task_id, model_path, img_path_list, batch_size, num_gpu, save_dir, clip_encoder_name="GeoRSCLIP"):
    pl.seed_everything(2024)
    os.makedirs(save_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if num_gpu > 0 else 'cpu'
    print("device: {}".format(device))

    model, _, img_preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-L-14-336-quickgelu',
        pretrained='openai',
        precision="fp16",
    )
    model = model.to(device)
    if clip_encoder_name == "GeoRSCLIP":
        checkpoint = torch.load(model_path, map_location=device)
        msg = model.load_state_dict(checkpoint, strict=False)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

    dataset = ImageFeatureExtractionDataset(img_path_list, img_preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)


    with torch.no_grad(), torch.cuda.amp.autocast():
        for index, batch in tqdm(enumerate(dataloader)):
            x, img_path_list = batch
            x = x.to(device)
            clip_feat = model.encode_image(x).detach().cpu().numpy()
            # clip_feat = model.encode_image(x).detach().cpu()

            feat_list = []
            img_name_list = []
            for i, img_path in enumerate(img_path_list):
                img_name = img_path.split("/")[-1]
                # img_name = img_path.split("/")[-1].split(".")[0]
                # grid_image_feature_path = os.path.join(save_dir, img_name)
                feat = clip_feat[i].reshape(1, -1)
                feat_list.append(feat)
                img_name_list.append(img_name)
                # np.save(grid_image_feature_path, feat)

            dump_name = "img_name_feat_{}-{}.pkl".format(task_id, index)
            dump_path = os.path.join(save_dir, dump_name)
            result_dict = dict()
            result_dict["img_name_list"] = img_name_list
            result_dict["feat_list"] = feat_list
            with open(dump_path, "wb") as f:
                pkl.dump(result_dict, f)


def extract(args, modality="image"):

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

    if modality == "image":
        all_imgs_paths = get_all_img_path(os.path.join(args.dataset_img_dir))
        print("total number of images in directory {}: {}".format(args.dataset_img_dir, len(all_imgs_paths)))
        print("non repeat imgs: {}".format(len(set(all_imgs_paths))))
        resource_assignment = assign_img_per_gpu(args.num_runner, len(all_imgs_paths))
        print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
        img_path_assignments = get_img_path_assignment(all_imgs_paths, resource_assignment)

        os.makedirs(args.save_dir, exist_ok=True)
        print("image feature save dir: {}".format(args.save_dir))

        result_status = []
        for task_id, img_path_list in enumerate(img_path_assignments):
            status = extract_image_feature.options(num_cpus=4, num_gpus=args.num_gpu).remote(
                # model_path, img_path_list, batch_size, num_gpu, clip_encoder_name="GeoRSCLIP"
                task_id, args.model_path, img_path_list, args.batch_size, args.num_gpu, args.save_dir
            )
            result_status.append(status)
            print("runner: {}".format(task_id))
        ray.get(result_status)

        tic = datetime.now()
        print(tic - toc)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runner', type=int, default=1,
                        help='number of gpu per trail')
    # #gpu per runner
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    # #cpu per runner
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='number of cpu per trail')

    parser.add_argument('--ray_mode', type=str, default="local_run", help='local mode or auto mode')

    parser.add_argument('--batch_size', type=int, default=500, help='batch size')

    parser.add_argument('--dataset_img_dir', type=str,
                        default="/media/zilun/mx500/ImageRAG_database/cropped_img",
                        help='dir of images needed to be extracted')

    parser.add_argument('--save_dir', type=str,
                        default="/media/zilun/mx500/ImageRAG_database/cropped_img/img_feat",
                        help='dir of images needed to be extracted')

    parser.add_argument('--model_path', type=str,
                        default="/media/zilun/wd-161/RS5M/RS5M_codebase/ckpt/RS5M_ViT-L-14-336.pt",
                        help='model_path')

    args = parser.parse_args()

    extract(args)


def deduplicate_result_dict(result_dict_path, new_result_dict_path, sentence_bert_path="/media/zilun/wd-161/hf_download/all-MiniLM-L6-v2"):

    from sentence_transformers import SentenceTransformer, SimilarityFunction
    model = SentenceTransformer(sentence_bert_path)
    vector_database_content = pkl.load(open(result_dict_path, "rb"))
    assert len(vector_database_content["img_name_list"]) == len(vector_database_content["label_list"]) == len(vector_database_content["feat"])
    T = 1.0
    # # 示例用法：定义一个简单的语义去重函数
    def cal_semantically_sim(label1, label2):
        # # 假设完全相同的标签视为语义重复
        # embeddings1 = model.encode(label1)
        # embeddings2 = model.encode(label2)
        # # Compute cosine similarities
        # similarities = model.similarity(embeddings1, embeddings2)
        tmp_label1 = label1.lower().replace("-", " ").replace("_", " ")
        tmp_label2 = label2.lower().replace("-", " ").replace("_", " ")
        if tmp_label1 == tmp_label2:
            similarities = 1.0
        else:
            similarities = 0.0
        return similarities

    label_list = vector_database_content["label_list"]
    unique_labels = []
    label_map = dict()
    label_set = list(set(label_list))
    seen_labels = []  # 已处理的标签列表

    for label in tqdm(label_set):
        is_duplicate = False
        label_map[label] = label.lower().replace("-", " ").replace("_", " ")
        for seen_label in seen_labels:
            similarities = cal_semantically_sim(label, seen_label)
            if similarities >= T:
                print(label, seen_label, similarities)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_labels.append(label)
            seen_labels.append(label)
    print(unique_labels)
    print(len(unique_labels))
    print(label_map)
    print(len(label_map))

    new_label_list = []
    for label in label_list:
        label_simplified = label_map[label]
        new_label_list.append(label_simplified)
    vector_database_content["label_list"] = new_label_list
    print(set(new_label_list))
    pkl.dump(vector_database_content, open(new_result_dict_path, "wb"))
    return vector_database_content



if __name__ == "__main__":
    # main()
    new_vector_database_content = deduplicate_result_dict(
        "/media/zilun/fanxiang4t/GRSM/ImageRAG0214/data/georsclip_feat_label_all_server.pkl",
        "/media/zilun/fanxiang4t/GRSM/ImageRAG0214/data/georsclip_feat_label_1M.pkl"
    )