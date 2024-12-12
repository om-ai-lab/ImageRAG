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
  
    bench_jsonl_path = "test_FITRS_complex_comprehension_eval.jsonl"
    # bench_jsonl_path = "test_FITRS_region_caption_eval.jsonl"
    base = [json.loads(q) for q in open(bench_jsonl_path, "r")]
    output_file_path = 'test_FITRS_complex_comprehension_eval_8para.jsonl'


    # 匹配 <rbox>
    for i, answers in enumerate(tqdm(base)):
        
        question = answers['question']
        gt = answers['ground_truth']
        process_str=[question, gt]
        
        # 3) 进行替换,5参数->8参数
        for j, todo_str in enumerate(process_str):
            # 使用正则表达式查找所有 <rbox> 标签中的内容
            # pattern = r'<rbox>\((.*?)\)</rbox>'
            pattern = r'\{(<.*?>)\}'
            # 使用正则表达式找到所有的矩形框
            matches = re.findall(pattern, todo_str)
            rboxes = []
            for match in matches:
                # 在每个矩形框中，找到所有的数字
                numbers_str = re.findall(r'<(.*?)>', match)
                # 将数字转换为浮点数，并将角度转换为弧度
                rbox = np.array(numbers_str, dtype=float)
                polys = obb2poly_np_oc_2rad(rbox)[0]
                x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = polys
                rbox_str = "[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]" % (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_)

                todo_str = todo_str.replace(f'{{{match}}}', rbox_str)
            process_str[j] = todo_str

        question, gt = process_str
        answers['question'] = question
        answers['ground_truth'] = gt

    with open(output_file_path, 'w') as outfile:
        for entry in base:
            json.dump(entry, outfile)
            outfile.write('\n')

    print('done!')