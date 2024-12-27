import argparse
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import re

def obb2poly_np_oc_2rad(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[0]
    y = rbboxes[1]
    w = rbboxes[2]
    h = rbboxes[3]
    a = np.radians(rbboxes[4])
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])
    polys = np.expand_dims(polys, axis=0)
    return polys

def main():
    bench_json_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/FIT-RS-train-1415k_5para.json"
    output_file_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/FIT-RS-train-1415k_5para_0_1000.json"

    with open(bench_json_path, "r") as f:
        base = json.load(f)

    modified_data = []
    # 匹配 <rbox>
    for i, instruction in enumerate(tqdm(base)):
        conv = instruction['conversations']
        for sentence in conv:
            if '<rbox>' in sentence['value']:
                # 进行替换,5参数->8参数
                todo_str = sentence['value']
                # 使用正则表达式查找所有 <rbox> 标签中的内容
                pattern = r'\{(<.*?>)\}'
                # 使用正则表达式找到所有的矩形框
                matches = re.findall(pattern, todo_str)
                rboxes = []
                for match in matches:
                    # 在每个矩形框中，找到所有的数字
                    numbers_str = re.findall(r'<(.*?)>', match)
                    # 将数字转换为浮点数，并将角度转换为弧度
                    rbox = np.array(numbers_str, dtype=np.float32)

                    # polys = obb2poly_np_oc_2rad(rbox)[0]
                    # x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = polys
                    # rbox_str = "[(%.2f,%.2f),(%.2f,%.2f),(%.2f,%.2f),(%.2f,%.2f)]" % (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_)
                    # rbox_str = "{<%.2f,%.2f><%.2f,%.2f><%.2f,%.2f><%.2f,%.2f>}" % (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_)

                    rbox_0_1000 = rbox[:-1] / 100 * 512 / 512 * 1000
                    xc, yc, w, h = rbox_0_1000
                    theta_degree = rbox[-1]
                    rbox_str = "{<%.2f><%.2f><%.2f><%.2f>|<%d>}" % (xc, yc, w, h, theta_degree)

                    todo_str = todo_str.replace(f'{{{match}}}', rbox_str)
                sentence['value'] = todo_str

        # 将修改后的数据添加到新的变量中
        modified_data.append(instruction)

    with open(output_file_path, 'w') as outfile:
        json.dump(modified_data, outfile, indent=4)
    print('done!')


def jsonl2json():
    import json

    # 打开 JSONL 文件
    with open('/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/8-coordinate-FIT-RS-train-1415k.jsonl', 'r') as jsonl_file:
        # 打开新的 JSON 文件准备写入
        with open('/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/8-coordinate-FIT-RS-train-1415k_5para_0_1000.json', 'w') as json_file:
            # 初始化一个空数组
            json_array = []

            # 逐行读取 JSONL 文件
            for line in jsonl_file:
                # 将每行解析为 JSON 对象并添加到数组中
                json_array.append(json.loads(line))
            # 将数组转换为 JSON 格式并写入文件
            json.dump(json_array, json_file, indent=4)


if __name__ == "__main__":
    main()