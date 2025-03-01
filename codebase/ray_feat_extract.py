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
import pickle as pkl


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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if num_gpu > 0 else 'cpu'
    print("device: {}".format(device))
    print("Loading given ckpt")

    print(clip_encoder_name)
    
    
    if clip_encoder_name=="RemoteCLIP":
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            # pretrained='openai',
            # precision="fp16"
        )
        checkpoint = torch.load(model_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint)
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
        msg = model.load_state_dict(checkpoint)
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

    if modality == "lrsd":
        
        pkl_file_path = args.merged_pkl_path

        base_dir = args.dataset_img_dir

        with open(pkl_file_path, "rb") as f:
            df = pkl.load(f)

        all_imgs_paths = []
        for rel_path in tqdm(df["relative_storage_path"]):
            file_path = os.path.join(base_dir, rel_path) if base_dir else rel_path
            all_imgs_paths.append(file_path)
        
        # all_imgs_paths = get_all_img_path(os.path.join(args.dataset_img_dir))
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
                task_id, args.model_path, img_path_list, args.batch_size, args.num_gpu, args.save_dir, clip_encoder_name=args.encoder_name
            )
            result_status.append(status)
            print("runner: {}".format(task_id))
        ray.get(result_status)

        tic = datetime.now()
        print(tic - toc)


    if modality == "crsd":
        # all_imgs_paths = get_all_img_path(os.path.join(args.dataset_img_dir))
        pub11_train_pd = pd.read_csv(args.pub11_csv_train_path)
        pub11_val_pd = pd.read_csv(args.pub11_csv_val_path)
        pub11_pd = pd.concat([pub11_train_pd, pub11_val_pd])
        all_img_names = pub11_pd["file_name"].tolist()
        all_imgs_paths = [os.path.join(args.dataset_img_dir, img_name) for img_name in all_img_names]
        print(all_imgs_paths[:5])
        # all_texts = pub11_pd["text"].tolist()
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
                task_id, args.model_path, img_path_list, args.batch_size, args.num_gpu, args.save_dir, clip_encoder_name=args.encoder_name
            )
            result_status.append(status)
            print("runner: {}".format(task_id))
        ray.get(result_status)

        tic = datetime.now()
        print(tic - toc)


def main_crsd():
    # ray start --head --port=6379
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runner', type=int, default=1,
                        help='number of runner')
    # #gpu per runner
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    # #cpu per runner
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='number of cpu per trail')

    parser.add_argument('--ray_mode', type=str, default="local_run", help='local mode or auto mode')

    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    parser.add_argument('--encoder_name', type=str, default="CLIP", help='local mode or auto mode')

    parser.add_argument('--dataset_img_dir', type=str,
                        default="/data9/zilun/grsm/dataset",
                        help='dir of images needed to be extracted')

    parser.add_argument('--save_dir', type=str,
                        default="/data1/zilun/dataset/pub11/img_feat/clip",
                        help='dir of images needed to be extracted')

    parser.add_argument('--model_path', type=str,
                        default="/data1/zilun/ImageRAG0226/checkpoint/RemoteCLIP/RemoteCLIP-ViT-L-14.pt",
                        help='model_path')

    parser.add_argument('--pub11_csv_train_path', type=str,
                        default="/data9/zilun/dataset/RS5M/pub11_train_metadata.csv",
                        help='pub11_csv_train_path')

    parser.add_argument('--pub11_csv_val_path', type=str,
                        default="/data9/zilun/dataset/RS5M/pub11_validation_metadata.csv",
                        help='pub11_csv_val_path')

    args = parser.parse_args()

    extract(args, modality="lrsd")
    
    

def main_lrsd():
    # /data9/zilun/grsm/dataset/dataset/new_merged_updated_millionaid.pkl

    # ray start --head --port=6379
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runner', type=int, default=1,
                        help='number of runner')
    # #gpu per runner
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    # #cpu per runner
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='number of cpu per trail')

    parser.add_argument('--ray_mode', type=str, default="local_run", help='local mode or auto mode')

    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    parser.add_argument('--encoder_name', type=str, default="CLIP", help='local mode or auto mode')
    
    
    parser.add_argument('--merged_pkl_path', type=str,
                        default="/data9/zilun/grsm/dataset/dataset/new_merged_updated_millionaid.pkl",
                        help='dir of images needed to be extracted')

    parser.add_argument('--dataset_img_dir', type=str,
                        default="/data9/zilun/grsm/dataset",
                        help='dir of images needed to be extracted')

    parser.add_argument('--save_dir', type=str,
                        default="/data1/zilun/dataset/lrsd/img_feat/clip",
                        help='dir of images needed to be extracted')

    parser.add_argument('--model_path', type=str,
                        # default="/data1/zilun/ImageRAG0226/checkpoint/RemoteCLIP/RemoteCLIP-ViT-L-14.pt",
                        default="/data1/zilun/ImageRAG0226/checkpoint/RS5M_ViT-L-14-336.pt",
                        help='model_path')
    
    
    
    args = parser.parse_args()

    extract(args, modality="lrsd")
    


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
    print(label_map)
    print(len(label_map))
    
    print(unique_labels)
    print(len(unique_labels))


    new_label_list = []
    for label in label_list:
        label_simplified = label_map[label]
        new_label_list.append(label_simplified)
    vector_database_content["label_list"] = new_label_list
    print(set(new_label_list))
    pkl.dump(vector_database_content, open(new_result_dict_path, "wb"))
    return vector_database_content


def merge_ray_feat_crsd(pub11_csv_train_path, pub11_csv_val_path, feat_dir, save_path):
    pub11_train_pd = pd.read_csv(pub11_csv_train_path)
    pub11_val_pd = pd.read_csv(pub11_csv_val_path)
    pub11_pd = pd.concat([pub11_train_pd, pub11_val_pd])
    pd_img_names = pub11_pd["file_name"].tolist()
    pd_texts = pub11_pd["text"].tolist()
    img_text_dict = dict(zip(pd_img_names, pd_texts))
    
    def list_imgname2text(img_names):
        text_list = []
        for img_name in img_names:
            text = img_text_dict[img_name]
            text_list.append(text)
        return text_list

    all_image_names = []
    all_features = []
    all_text = []

    # 遍历目录中的所有 pkl 文件
    pkl_files = glob.glob(os.path.join(feat_dir, "*.pkl"))

    for file_path in tqdm(pkl_files):
        # 加载 pkl 文件
        with open(file_path, 'rb') as f:
            data = pkl.load(f)  # 假设数据是一个字典，包含 image_name 和 feat
            
        # 提取 image_name 和 feat 数据
        image_names = data['img_name_list']
        feats = data['feat_list']
        texts = list_imgname2text(image_names)

        all_image_names.extend(image_names)
        all_features.extend(feats)
        all_text.extend(texts)

    # assert len(vector_database_content["img_name_list"]) == len(vector_database_content["label_list"]) == len(vector_database_content["feat"])

    result = {
        "img_name_list": all_image_names,
        "label_list": all_text,
        "feat": all_features
    }
    
    for i in range(5):
        print(all_image_names[i], all_text[i], all_features[i].shape)
        
        
    pkl.dump(result, open(save_path, "wb"))
    
    return result



def merge_ray_feat_lrsd(merged_pkl_path, feat_dir, save_path):
    
    pkl_file_path = merged_pkl_path

    with open(pkl_file_path, "rb") as f:
        df = pkl.load(f)
    
    all_imgs_paths = []
    for rel_path in tqdm(df["relative_storage_path"]):
        file_path = rel_path
        all_imgs_paths.append(file_path)
    print(len(all_imgs_paths))

    
    pd_img_names = df["relative_storage_path"].tolist()
    pd_img_names = [img_name.split("/")[-1] for img_name in pd_img_names]
    pd_texts = df["label"].tolist()
    img_text_dict = dict(zip(pd_img_names, pd_texts))
    
    def list_imgname2text(img_names):
        text_list = []
        for img_name in img_names:
            text = img_text_dict[img_name]
            text_list.append(text)
        return text_list

    all_image_names = []
    all_features = []
    all_text = []

    # 遍历目录中的所有 pkl 文件
    pkl_files = glob.glob(os.path.join(feat_dir, "*.pkl"))

    for file_path in tqdm(pkl_files):
        # 加载 pkl 文件
        with open(file_path, 'rb') as f:
            data = pkl.load(f)  # 假设数据是一个字典，包含 image_name 和 feat
            
        # 提取 image_name 和 feat 数据
        image_names = data['img_name_list']
        feats = data['feat_list']
        
        # pdb.set_trace()

        texts = list_imgname2text(image_names)

        all_image_names.extend(image_names)
        all_features.extend(feats)
        all_text.extend(texts)

    # assert len(vector_database_content["img_name_list"]) == len(vector_database_content["label_list"]) == len(vector_database_content["feat"])

    result = {
        "img_name_list": all_image_names,
        "label_list": all_text,
        "feat": all_features
    }
    
    for i in range(5):
        print(all_image_names[i], all_text[i], all_features[i].shape)
        
        
    pkl.dump(result, open(save_path, "wb"))
    print(save_path)
    return result
    



if __name__ == "__main__":
    # main_lrsd()
    
    new_vector_database_content = deduplicate_result_dict(
        "/data1/zilun/ImageRAG0226/data/lrsd_georsclip_1M.pkl",
        "/data1/zilun/ImageRAG0226/data/lrsd_georsclip_1M_dedup.pkl",
        sentence_bert_path="/data1/zilun/ImageRAG0226/checkpoint/all-MiniLM-L6-v2"
    )
    
    # merge_ray_feat(
    #     "/data9/zilun/dataset/RS5M/pub11_train_metadata.csv",
    #     "/data9/zilun/dataset/RS5M/pub11_validation_metadata.csv",
    #     "/data1/zilun/dataset/pub11/img_feat/remoteclip",
    #     "/data1/zilun/ImageRAG0226/data/remoteclip_pub11feat_label_3M.pkl"
    # )
    
    # merge_ray_feat_lrsd(
    #     "/data1/zilun/ImageRAG0226/data/new_merged_updated_millionaid.pkl",
    #     "/data1/zilun/dataset/lrsd/img_feat/clip",
    #     "/data1/zilun/ImageRAG0226/data/lrsd_clip_3M.pkl"
    # )