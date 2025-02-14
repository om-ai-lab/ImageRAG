import os
from PIL import Image
import numpy as np
from tqdm import tqdm


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

    # 提取图像文件名（不包含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 打开原始图像
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size

    # 创建保存 patch 的目录
    image_save_dir = os.path.join(patch_save_root, image_name)
    os.makedirs(image_save_dir, exist_ok=True)

    # 检查目录是否为空
    if os.listdir(image_save_dir):
        print(f"dir {image_save_dir} non-empty，load existed patch file")
        return original_image, _load_existing_patches(image_save_dir, image_name), image_save_dir


    # 如果图像尺寸不是 patch_size 的整数倍，需要对图像进行 resize
    new_width = int(np.ceil(original_width / patch_size)) * patch_size
    new_height = int(np.ceil(original_height / patch_size)) * patch_size

    if new_width != original_width or new_height != original_height:
        print(f"image resize to {new_width}x{new_height}")
        resized_image = original_image.resize((new_width, new_height))
    else:
        resized_image = original_image

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

    return original_image, patch_dict, image_save_dir


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


# 示例用法
if __name__ == "__main__":
    # 输入图像路径
    image_path = "/media/zilun/wd-161/datasets/MME-RealWorld-Lite/data/imgs/dota_v2_dota_v2_dota_v2_P6397.png"  # 替换为你的图像路径
    patch_size = 448  # 设置 patch 的大小
    patch_save_root = "/media/zilun/fanxiang4t/GRSM/ImageRAG0214/cache"  # 设置保存 patch 的根目录

    # 调用函数
    original_image, patch_dict = patchify_image(image_path, patch_size, patch_save_root)

    print("原始图像：", original_image.size)
    print("Patch 字典：", patch_dict)