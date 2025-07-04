import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from math import floor


def vit_patchify(image_path, patch_save_root, patch_size=448):
    """
    将图像分割为多个 patch，并保存到指定目录。
    如果目录已存在且非空，则直接读取已有的 patch 文件并整理成输出字典。

    参数:
        image_path (str): 输入图像的路径。
        patch_size (int): 每个 patch 的大小。
        patch_save_root (str): 保存 patch 的根目录。

    返回:
        original_image (PIL.Image): 原始图像。
        patch_dict (dict): key 是 patch 的坐标 (row, col)，value 是 patch 的相对路径。
    """
    # patch_save_root = os.path.join(patch_save_root, "vit")
    os.makedirs(patch_save_root, exist_ok=True)

    # 提取图像文件名（不包含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 打开原始图像
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size

    # 创建保存 patch 的目录
    image_save_dir = os.path.join(patch_save_root, image_name)
    os.makedirs(image_save_dir, exist_ok=True)

    # 如果图像尺寸不是 patch_size 的整数倍，需要对图像进行 resize
    new_width = int(np.ceil(original_width / patch_size)) * patch_size
    new_height = int(np.ceil(original_height / patch_size)) * patch_size

    if new_width != original_width or new_height != original_height:
        print(f"image resize to {new_width}x{new_height}")
        resized_image = original_image.resize((new_width, new_height))
    else:
        resized_image = original_image

    # 检查目录是否为空
    if os.listdir(image_save_dir):
        print(f"dir {image_save_dir} non-empty，load existed patch file")
        return resized_image, original_image, _load_existing_patches(image_save_dir, image_name), image_save_dir

    # 初始化 patch_dict
    patch_dict = {}

    # 遍历每个 patch
    for row in tqdm(range(new_height // patch_size)):
        for col in range(new_width // patch_size):
            # 计算 patch 在 resize 后图像中的边界
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            # 裁剪 patch
            patch = resized_image.crop((x1, y1, x2, y2))

            # 将 patch 的坐标映射回原始图像
            original_x1 = int(x1 * original_width / new_width)
            original_y1 = int(y1 * original_height / new_height)
            original_x2 = int(x2 * original_width / new_width)
            original_y2 = int(y2 * original_height / new_height)

            # 生成 patch 的文件名
            patch_filename = f"{image_name}_{original_x1}-{original_y1}-{original_x2}-{original_y2}.png"
            patch_path = os.path.join(image_save_dir, patch_filename)
            patch.save(patch_path)

            # 将 patch 的相对路径保存到 patch_dict
            relative_path = os.path.join(image_name, patch_filename)
            patch_dict[(original_x1, original_y1, original_x2, original_y2)] = relative_path

    return resized_image, original_image, patch_dict, image_save_dir


def cc_patchify(image_path, patch_save_root, c_denom):
    """
    Get image patches with patch-cc scheme
    (bs, 3, h, w) -> (bs, p, 3, h_p, w_p)
    :param img: img torch tensor
    :return: list of bbox (tl_x, tl_r, w, h), each represents a patch (cover)
    # img, img_name, c_denom=10, dump_imgs=False, patch_saving_dir=None
    """
    # patch_save_root = os.path.join(patch_save_root, "cc")
    os.makedirs(patch_save_root, exist_ok=True)

    # 提取图像文件名（不包含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 打开原始图像
    original_image = Image.open(image_path)
    # original_width, original_height = original_image.size

    # 创建保存 patch 的目录
    image_save_dir = os.path.join(patch_save_root, image_name)
    print("Make sure cc is in the patch saving dir")
    assert "cc" in image_save_dir
    os.makedirs(image_save_dir, exist_ok=True)

    # 如果图像尺寸不是 patch_size 的整数倍，需要对图像进行 resize
    w, h = original_image.size
    if h % c_denom != 0:
        resize_h = (h // c_denom + 1) * c_denom
    else:
        resize_h = h

    if w % c_denom != 0:
        resize_w = (w // c_denom + 1) * c_denom
    else:
        resize_w = w

    if resize_w != w or resize_h != h:
        print(f"image resize to {resize_w}x{resize_h}")
        img_resize = original_image.resize((resize_w, resize_h))
    else:
        img_resize = original_image
        
    # 检查目录是否为空
    if os.listdir(image_save_dir):
        print(f"dir {image_save_dir} non-empty，load existed patch file")
        return img_resize, original_image, _load_existing_patches(image_save_dir, image_name), image_save_dir

    # img_resize = Image.new('RGB', (resize_w, resize_h), 0)
    # left = (resize_w - w) // 2
    # top = (resize_h - h) // 2
    # img_resize.paste(original_image, (left, top))

    def vis_cc_patches(patch_coordinates, img_resize, img_name):
        assert isinstance(patch_coordinates, list)
        patch_dict = {}
        for level_index, level_content in enumerate(tqdm(patch_coordinates)):
            for patch_index, patch_coordinate in enumerate(level_content):
                h_range, w_range = patch_coordinate
                x1 = w_range[0]
                y1 = h_range[0]
                x2 = w_range[1]
                y2 = h_range[1]

                crop_box = (x1, y1, x2, y2)
                patch = img_resize.crop(crop_box)

                # 将 patch 的坐标映射回原始图像
                original_x1 = int(x1 * w / resize_w)
                original_y1 = int(y1 * h / resize_h)
                original_x2 = int(x2 * w / resize_w)
                original_y2 = int(y2 * h / resize_h)

                # 生成 patch 的文件名
                patch_filename = f"{image_name}_{original_x1}-{original_y1}-{original_x2}-{original_y2}.png"
                patch_path = os.path.join(image_save_dir, patch_filename)
                patch.save(patch_path)

                # 将 patch 的相对路径保存到 patch_dict
                relative_path = os.path.join(image_name, patch_filename)
                patch_dict[(original_x1, original_y1, original_x2, original_y2)] = relative_path

        return patch_dict

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
                    if j + kernel_w < w:
                        result.append(([i, i + kernel_h], [j, j + kernel_w]))
                    else:
                        result.append(([i, i + kernel_h], [w - kernel_w, w]))
                        break
                else:
                    if j + kernel_w < w:
                        result.append(([h - kernel_h, h], [j, j + kernel_w]))
                    else:
                        result.append(([h - kernel_h, h], [w - kernel_w, w]))
                        return result

    patch_container = [[([0, resize_h], [0, resize_w])]]

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
        print("level: {}, count: {}, kernel h: {}, kernel w: {}, stride h: {}, stride w: {}".format(n, len(img_patches), kernel_h, kernel_w, stride_h, stride_w))
        patch_size_list.append(len(img_patches))
        n += 1
    index_array, vl = index_of_last_apperance(patch_size_list)
    patch_container_deduplicate = [patch_container[i] for i in index_array]
    patch_dict = vis_cc_patches(patch_coordinates=patch_container_deduplicate, img_resize=img_resize, img_name=image_name)

    return img_resize, original_image, patch_dict, image_save_dir


def _load_existing_patches(image_save_dir, image_name):
    """
    从已有的目录中加载 patch 文件并整理成输出字典。

    参数:
        image_save_dir (str): 保存 patch 的目录。
        image_name (str): 图像文件名（不包含扩展名）。

    返回:
        original_image (PIL.Image): 原始图像（None，因为无法直接从 patch 恢复原始图像）。
        patch_dict (dict): key 是 patch 的坐标 (row, col)，value 是 patch 的相对路径。
    """
    patch_dict = {}
    for filename in os.listdir(image_save_dir):
        if filename.startswith(image_name) and filename.endswith(".png"):
            # 解析文件名中的坐标
            coords = filename[len(image_name) + 1:-4].split("-")
            x1, y1, x2, y2 = map(int, coords)

            # 保存到 patch_dict
            relative_path = os.path.join(image_name, filename)
            patch_dict[(x1, y1, x2, y2)] = relative_path

    return patch_dict



def grid_patchify(image_path, patch_save_root, max_grid=10, enlarge_factor=1.0):
    """
    将图像按 1x1, 2x2, …, max_gridxmax_grid 网格划分为多个 patch, 并保存到指定目录。
    若图像尺寸不能被当前网格数整除，则对图像进行 resize(扩展到最近能整除的尺寸)。
    最终输出的 patch 坐标会映射回原始图像的坐标，并可按 enlarge_factor 放大。
    
    参数:
      image_path (str): 输入图像路径。
      patch_save_root (str): 保存所有 patch 的根目录。
      max_grid (int): 最大网格数（默认 10, 即划分到 10x10)。
      enlarge_factor (float): 放大倍数（默认 1.0, 即不放大）。
    
    返回:
      resized_image (PIL.Image): 用于 patch 裁剪的图像（可能经过 resize)。
      original_image (PIL.Image): 原始图像。
      patch_dict (dict): 字典，其 key 为 grid 数(n)，value 为该 grid 下的 patch 字典，
                         每个 patch 字典的 key 是 (x1, y1, x2, y2)（原始图像坐标，经过 enlarge_factor 调整），
                         value 是对应 patch 的相对保存路径。
      image_save_dir (str): 保存该图像所有 patch 的顶级目录。
    """
    from math import ceil
    os.makedirs(patch_save_root, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 打开原始图像
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size

    # 创建以图像名为子目录的保存目录
    image_save_dir = os.path.join(patch_save_root, image_name)
    os.makedirs(image_save_dir, exist_ok=True)
    
    
    # # 如果图像尺寸不是 patch_size 的整数倍，需要对图像进行 resize
    # new_width = int(np.ceil(original_width / patch_size)) * patch_size
    # new_height = int(np.ceil(original_height / patch_size)) * patch_size

    # if new_width != original_width or new_height != original_height:
    #     print(f"image resize to {new_width}x{new_height}")
    #     resized_image = original_image.resize((new_width, new_height))
    # else:
    #     resized_image = original_image
    
    
    # 检查目录是否为空
    if os.listdir(image_save_dir):
        resized_image = original_image
        print(f"dir {image_save_dir} non-empty, load existed patch file")
        return resized_image, original_image, _load_existing_patches(image_save_dir, image_name), image_save_dir
    

    patch_dict = {}  # 用于存储各 grid 下的 patch 信息

    # 对于 1×1 到 max_grid×max_grid 的划分
    for n in range(1, max_grid + 1):
        # 为确保整除，将图像扩展到 n 的整数倍
        new_width = int(ceil(original_width / n)) * n
        new_height = int(ceil(original_height / n)) * n

        if new_width != original_width or new_height != original_height:
            print(f"[Grid {n}x{n}] Resizing image to {new_width}x{new_height}")
            resized = original_image.resize((new_width, new_height))
        else:
            resized = original_image

        cell_width = new_width // n
        cell_height = new_height // n

        grid_dict = {}  # 存储当前 grid 下所有 patch 信息

        for row in range(n):
            for col in range(n):
                # 在 resized 图像上计算 patch 边界
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height

                patch = resized.crop((x1, y1, x2, y2))
                
                # 将 resized 上的坐标映射回原始图像坐标
                orig_x1 = int(x1 * original_width / new_width)
                orig_y1 = int(y1 * original_height / new_height)
                orig_x2 = int(x2 * original_width / new_width)
                orig_y2 = int(y2 * original_height / new_height)

                # 若需要放大，则以 patch 中心为基准放大 bbox 尺寸
                if enlarge_factor != 1.0:
                    center_x = (orig_x1 + orig_x2) / 2.0
                    center_y = (orig_y1 + orig_y2) / 2.0
                    width_bbox = orig_x2 - orig_x1
                    height_bbox = orig_y2 - orig_y1
                    new_width_bbox = width_bbox * enlarge_factor
                    new_height_bbox = height_bbox * enlarge_factor
                    orig_x1 = int(center_x - new_width_bbox / 2.0)
                    orig_y1 = int(center_y - new_height_bbox / 2.0)
                    orig_x2 = int(center_x + new_width_bbox / 2.0)
                    orig_y2 = int(center_y + new_height_bbox / 2.0)
                    # 可添加边界检测，避免超出原图范围

                # 生成 patch 文件名，包含 grid 信息和映射后的原始坐标
                patch_filename = f"{image_name}_{orig_x1}-{orig_y1}-{orig_x2}-{orig_y2}.png"
                # 保存到 image_save_dir（与 cc_patchify 保持一致，不创建子文件夹）
                patch_path = os.path.join(image_save_dir, patch_filename)
                patch.save(patch_path)

                # 存储 patch 的相对路径，格式与 cc_patchify 相同
                relative_path = os.path.join(image_name, patch_filename)
                patch_dict[(orig_x1, orig_y1, orig_x2, orig_y2)] = relative_path
        # 以最后一次的 resized 图像为返回值
        resized_image = resized

    return resized_image, original_image, patch_dict, image_save_dir



# 示例用法
if __name__ == "__main__":
    # 输入图像路径
    image_path = "/data9/shz/dataset/MME-RealWorld/remote_sensing/remote_sensing/dota_v2_dota_v2_dota_v2_P6397.png"  # 替换为你的图像路径
    patch_size = 448  # 设置 patch 的大小
    patch_save_root = "/data9/zilun/ImageRAG0214/cache/patch/mmerealworldlite/cc"  # 设置保存 patch 的根目录

    # 调用函数
    resized_image, patch_dict, img_save_dir = cc_patchify(image_path, patch_save_root, 10)

    print("原始图像：", resized_image.size)
    print("Patch 字典：", patch_dict)