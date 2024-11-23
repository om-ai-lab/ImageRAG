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


def process_millionaid_with_dataloader(img_dir, out_dir, split):

    class MillionAIDDataset(Dataset):
        def __init__(self, img_dir, out_dir, split):
            self.img_dir = os.path.join(img_dir, split)
            self.out_dir = out_dir
            self.image_paths, self.superclass_level1_list, self.superclass_level2_list, self.class_list = self.load_imgs()
            os.makedirs(os.path.join(self.out_dir, "millionaid"), exist_ok=True)

        def load_imgs(self):
            image_paths = []
            superclass_level1_list = []
            superclass_level2_list = []
            class_list = []
            for superclass_level1 in os.listdir(self.img_dir):
                superclass_level1_path = os.path.join(self.img_dir, superclass_level1)
                if os.path.isdir(superclass_level1_path):
                    for superclass_level2 in os.listdir(superclass_level1_path):
                        superclass_level2_path = os.path.join(superclass_level1_path, superclass_level2)
                        if os.path.isdir(superclass_level2_path):
                            for class_dir in os.listdir(superclass_level2_path):
                                class_path = os.path.join(superclass_level2_path, class_dir)
                                if os.path.isdir(class_path):
                                    for image_file in os.listdir(class_path):
                                        if image_file.lower().endswith(
                                                ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                                            # image_path = os.path.join(class_path, image_file)
                                            image_paths.append(image_file)
                                            superclass_level1_list.append(superclass_level1)
                                            superclass_level2_list.append(superclass_level2)
                                            class_list.append(class_dir)

            return image_paths, superclass_level1_list, superclass_level2_list, class_list

        def __getitem__(self, index):
            image_name = self.image_paths[index]
            superclass_level1 = self.superclass_level1_list[index]
            superclass_level2 = self.superclass_level2_list[index]
            cls_name = self.class_list[index]

            img_path = os.path.join(self.img_dir, superclass_level1, superclass_level2, cls_name, image_name)
            img = Image.open(img_path)
            save_name = "millionaid_{}_{}_{}_{}.jpg".format(
                    superclass_level1.replace("_", "-"),
                    superclass_level2.replace("_", "-"),
                    cls_name.replace("_", "-"),
                    image_name.split(".")[0]
                )
            save_path = os.path.join(
                self.out_dir,
                "millionaid",
                save_name
            )
            img.save(save_path)
            return save_name, superclass_level1, superclass_level2, cls_name


        def __len__(self):
            return len(self.image_paths)


    def collect_fn(batch):
        image_paths = []
        superclass_level1_list = []
        superclass_level2_list = []
        class_list = []
        for data in batch:
            image_path, superclass_level1, superclass_level2, cls_name = data
            image_paths.append(image_path)
            superclass_level1_list.append(superclass_level1)
            superclass_level2_list.append(superclass_level2)
            class_list.append(cls_name)
        return image_paths, superclass_level1_list, superclass_level2_list, class_list


    millionaid_dataset = MillionAIDDataset(img_dir, out_dir, split)
    print("total {} images".format(len(millionaid_dataset)))

    millionaid_dataload = DataLoader(
        millionaid_dataset,
        batch_size=64,
        pin_memory=True,
        num_workers=32,
        shuffle=False,
        collate_fn=collect_fn
    )

    all_image_path, all_superclass_level1, all_superclass_level2, all_cls_name = [], [], [], []
    for i, batch in tqdm(enumerate(millionaid_dataload)):
        image_paths, superclass_level1_list, superclass_level2_list, class_list = batch
        all_image_path.extend(image_paths)
        all_superclass_level1.extend(superclass_level1_list)
        all_superclass_level2.extend(superclass_level2_list)
        all_cls_name.extend(class_list)

    df = pd.DataFrame(
        {
            "img_name_list": all_image_path,
            "level1_class": all_superclass_level1,
            "level2_class": all_superclass_level2,
            "level3_class": all_cls_name,
        }
    )
    pkl.dump(df, open(os.path.join(out_dir, "millionaid", "info_{}.pkl".format(split)), "wb"))
    return all_image_path, all_cls_name


def main():
    start = datetime.now()
    millionaid_dir = "/media/zilun/wd-161/ImageRAG_database/MillionAID"
    img_output_root = "/media/zilun/mx500/ImageRAG_database/cropped_img"
    all_image_path, all_cls_name = process_millionaid_with_dataloader(millionaid_dir, img_output_root, "train")
    print(all_image_path[:10], all_cls_name[:10])
    print(datetime.now() - start)


if __name__ == "__main__":
    main()