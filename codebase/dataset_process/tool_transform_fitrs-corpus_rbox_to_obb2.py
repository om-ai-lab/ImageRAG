import argparse
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import re
import jsonlines


def obb12obb2(rbbox):
    cx, cy, w, h = rbbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


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
    bench_json_path = "/mnt/cfs/zilun/GRSM/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS-train-1415k.json"
    output_file_path = "/mnt/cfs/zilun/GRSM/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS-train-1415k_nodynamic_resize448_0-1000_obb2.json"

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
                    rbox_0_1000 = rbox[:-1] / 100 * 512 / 512 * 1000
                    xc, yc, w, h = rbox_0_1000
                    theta_degree = rbox[-1]
                    x1, y1, x2, y2 = obb12obb2((xc, yc, w, h))
                    rbox_str = "{<%.2f><%.2f><%.2f><%.2f>|<%d>}" % (x1, y1, x2, y2, theta_degree)
                    todo_str = todo_str.replace(f'{{{match}}}', rbox_str)
                sentence['value'] = todo_str

        # 将修改后的数据添加到新的变量中
        modified_data.append(instruction)

    with open(output_file_path, 'w') as outfile:
        json.dump(modified_data, outfile, indent=4)
    print('done!')

    # 从 JSON 文件读取数据
    with open(output_file_path, 'r') as f:
        json_data = json.load(f)

    # 将 JSON 数据转换为 JSON Lines 格式并写入文件
    with jsonlines.open(output_file_path + 'l', 'w') as writer:
        for obj in tqdm(json_data):
            # 将每个 JSON 对象转换为字符串并写入文件，每个对象占一行
            writer.write(obj)


if __name__ == "__main__":
    main()