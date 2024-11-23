import pdb
import shutil
from tqdm import tqdm
import os
import uuid
import numpy as np
import pickle as pkl
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2


Image.MAX_IMAGE_PIXELS = None


def visualize_hbbox(image_path, hbbox, color=(0, 0, 255)):
    """
    denormalized hbbox
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print("Hbbox Vertices: {}".format(hbbox))
    bbox_left_top_x, bbox_left_top_y, w, h = hbbox
    bbox_left_top = [bbox_left_top_x, bbox_left_top_y]
    bbox_right_bottom = [bbox_left_top_x + w, bbox_left_top_y + h]
    cv2.rectangle(image, bbox_left_top, bbox_right_bottom, thickness=10, color=color)
    # cv2.imshow('Image: {}'.format(image_path), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("tmp.jpg", image)

def image_size_stats(img_dir):

    widths = []
    heights = []
    size_counts = {size: 0 for size in range(1000, 35000, 1000)}

    for filename in tqdm(os.listdir(img_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(img_dir, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)

                for size in size_counts.keys():
                    if width >= size or height >= size:
                        size_counts[size] += 1

    # 打印统计结果
    print("图像数量统计（宽度和高度均大于等于指定值）:")
    for size, count in size_counts.items():
        print(f"大于等于 {size}x{size}: {count} 张图像")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='blue', alpha=0.7)
    plt.title('Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='green', alpha=0.7)
    plt.title('Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def main():
    # STAR_trainset_dir = "/media/zilun/mx500/STAR/train/img"
    # STAR_valset_dir = "/media/zilun/mx500/STAR/val/img"
    # STAR_testset_dir = "/media/zilun/mx500/STAR/test/img264"
    # image_size_stats(STAR_testset_dir)

    visualize_hbbox("/media/zilun/wd-161/datasets/fmow/val/nuclear_powerplant/nuclear_powerplant_0/nuclear_powerplant_0_0_rgb.jpg", [2049, 1033, 5540, 3232], color=None)



if __name__ == "__main__":
    main()