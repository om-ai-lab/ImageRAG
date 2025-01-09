import torch.nn.functional as F
from datetime import datetime

from attr.validators import instance_of
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ColorJitter
import random
import os
import numpy as np
from PIL import Image
import torch
from math import ceil, floor
from einops import rearrange, reduce, repeat
from torchvision.utils import save_image
from tqdm import tqdm


# def vis_patches(save_dir, patch_coordinates, img_resize):
#     patch_container = []
#     for level in patch_coordinates:
#         level_container = []
#         for patch_coordinate in level:
#             h_range, w_range = patch_coordinate
#             # patch_img = img_resize[:, : , h_range[0]:h_range[1],w_range[0]:w_range[1]]
#             crop_box = (w_range[0], h_range[0], w_range[1], h_range[1])
#             patch_img = img_resize.crop(crop_box)
#             level_container.append(patch_img)
#         patch_container.append(level_container)
#
#     patch_coordinates_dict = dict()
#     assert isinstance(patch_coordinates, list)
#     for i, patch_level in enumerate(patch_container):
#         for k, batch_sample in enumerate(patch_level):
#             # for _, patch in enumerate(batch_sample):
#             # save_image(batch_sample, os.path.join(save_dir, "patchlevel-{}_patch-{}.png".format(i, k)))
#             patch_save_name = "patchlevel-{}_patch-{}.png".format(i, k)
#             batch_sample.save(os.path.join(save_dir, patch_save_name))
#
#     return patch_coordinates


def whrange2bbox(w_range, h_range):
    """
    w and h range to bbox (topleft_x, topleft_y, w, h)
    """
    return w_range[0], h_range[0], w_range[1] - w_range[0], h_range[1] - h_range[0]


def vis_patches(save_dir, patch_coordinates, img_resize):
    assert isinstance(patch_coordinates, list)
    coordinate_patchname_dict = dict()
    for level_index, level_content in enumerate(tqdm(patch_coordinates)):
        for patch_index, patch_coordinate in enumerate(level_content):
            h_range, w_range = patch_coordinate
            crop_box = (w_range[0], h_range[0], w_range[1], h_range[1])
            patch_img = img_resize.crop(crop_box)
            patch_save_name = "patchlevel-{}_patch-{}.png".format(level_index, patch_index)
            patch_img.save(os.path.join(save_dir, patch_save_name))
            # patch_bbox_coordinate = whrange2bbox(w_range, h_range)
            patch_bbox_coordinate = crop_box
            coordinate_patchname_dict[patch_bbox_coordinate] = patch_save_name
    return coordinate_patchname_dict


def img_2patch(img, c_denom=10, dump_imgs=False, patch_saving_dir=None):
    """
    Get image patches with patch-cc scheme
    (bs, 3, h, w) -> (bs, p, 3, h_p, w_p)
    :param img: img torch tensor
    :return: list of bbox (tl_x, tl_r, w, h), each represents a patch (cover)
    """
    if dump_imgs:
        assert patch_saving_dir is not None

    w, h = img.size
    if h % c_denom != 0:
        resize_h = (h // c_denom + 1) * c_denom
    else:
        resize_h = h

    if w % c_denom != 0:
        resize_w = (w // c_denom + 1) * c_denom
    else:
        resize_w = w

    img_resize = Image.new('RGB', (resize_w, resize_h), 0)

    left = (resize_w - w) // 2
    top = (resize_h - h) // 2

    img_resize.paste(img, (left, top))
    # img_resize.show()
    patch_container = []
    patch_container.append([([0, resize_h], [0, resize_w])])
    n = 2
    a_h = resize_h
    a_w = resize_w
    c_h = a_h // c_denom
    c_w = a_w // c_denom
    kernel_h = a_h
    kernel_w = a_w
    patch_size_list = [1]

    while kernel_h - 2 * c_h >= 0 and kernel_w - 2 * c_w + 1 >= 0:
        kernel_h = (a_h - (n - 1) * c_h)
        kernel_w = (a_w - (n - 1) * c_w)
        stride_h = floor(kernel_h - (kernel_h - c_h) / c_h)
        stride_w = floor(kernel_w - (kernel_w - c_w) / c_w)
        img_patches = return_sliding_windows(img_resize, kernel_h, kernel_w, stride_h, stride_w)
        patch_container.append(img_patches)
        print("level: {}, kernel h: {}, kernel w: {}, stride h: {}, stride w: {}".format(n, kernel_h, kernel_w, stride_h, stride_w))
        patch_size_list.append(len(img_patches))
        n += 1
    index_array, vl = index_of_last_apperance(patch_size_list)

    patch_container_deduplicate = [patch_container[i] for i in index_array]

    if dump_imgs:
        patch_container_deduplicate = vis_patches(save_dir=patch_saving_dir, patch_coordinates=patch_container_deduplicate, img_resize=img_resize)

    return img_resize, patch_container_deduplicate


def index_of_last_apperance(patch_size_list):
    rd = dict()
    for i, ele in enumerate(patch_size_list):
        if ele not in rd:
            rd[ele] = [i]
        else:
            rd[ele].append(i)
    rl = []
    vl = []
    for key in rd:
        rl.append(max(rd[key]))
        vl.append(key)
    return rl, vl


def return_sliding_windows(img, kernel_h, kernel_w, stride_h, stride_w):
    result = []
    w, h = img.size
    for i in range(0, h, stride_h):
        for j in range(0, w, stride_w):
            if i + kernel_h < h:
                if j+kernel_w < w:
                    result.append(([i, i+kernel_h], [j, j+kernel_w]))
                else:
                    result.append(([i, i+kernel_h], [w-kernel_w, w]))
                    break
            else:
                if j+kernel_w < w:
                    result.append(([h-kernel_h, h], [j,j+kernel_w]))
                else:
                    result.append(([h-kernel_h, h], [w-kernel_w, w]))
                    return result





if __name__ == "__main__":
    pass