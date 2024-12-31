import argparse
import torch
import os
import json
from tqdm import tqdm
import re
from codebase.inference.FIT_Eval.sgg_eval.sgg_eval import Compute_Pred_Matches
from codebase.inference.FIT_Eval.sgg_eval.vg_eval import do_vg_evaluation
import numpy as np
from codebase.inference.FIT_Eval.eval_map_5para import eval_rbbox_map
import math
import cv2


# ## all categories
label_id = ['airplane', 'boat', 'taxiway', 'boarding_bridge', 'tank', 'ship', 'crane',
            'car', 'apron', 'dock', 'storehouse', 'goods_yard', 'truck', 'terminal',
            'runway', 'breakwater', 'car_parking', 'bridge', 'cooling_tower',
            'truck_parking', 'chimney', 'vapor', 'coal_yard', 'genset', 'smoke',
            'gas_station', 'lattice_tower', 'substation', 'containment_vessel', 'flood_dam', 'ship_lock', 'gravity_dam',
            'arch_dam', 'cement_concrete_pavement', 'toll_gate', 'tower_crane', 'engineering_vehicle',
            'unfinished_building', 'foundation_pit',
            'wind_mill', 'intersection', 'roundabout', 'ground_track_field', 'soccer_ball_field', 'basketball_court',
            'tennis_court', 'baseball_diamond', 'stadium', 'null']

# ## all relationships
relations = ['over', 'not co-storage with', 'connect', 'parallelly parked on', 'intersect', 'co-storage with',
             'converge', 'parallelly docked at', 'adjacent', 'within safe distance of', 'through', 'approach',
             'away from', 'randomly parked on', 'run along', 'isolatedly parked on', 'around', 'randomly docked at',
             'drive off',
             'drive toward', 'within danger distance of', 'supply to', 'isolatedly docked at', 'pass across',
             'not run along', 'slightly emit', 'exhaust to', 'violently emit',
             'incorrectly parked on', 'pass under', 'directly transmit electricity to',
             'indirectly transmit electricity to', 'pass through', 'within same line of', 'within different line of',
             'directly connected to', 'indirectly connected to', 'driving in the same direction with',
             'driving in the opposite direction with', 'driving alongside with', 'driving in the same lane with',
             'driving in the different lane with', 'working on', 'not working on', 'parked alongside with',
             'not parked alongside with',
             'in the same parking with', 'in the different parking with', 'parking in the same apron with',
             'parking in the different apron with', 'running along the same taxiway with',
             'running along the different taxiway with',
             'running along the different runway with', 'docking at the same breakwater with',
             'docking at the same dock with', 'docking at the different dock with', 'docked alongside with',
             'not docked alongside with']

label_id_to_index = {label: index for index, label in enumerate(label_id)}
relation_to_index = {relation: index for index, relation in enumerate(relations)}


# target_id, target_cat, relation, obj_id_count, obj_cat
def convert_to_numpy_triplet(sub_id, sub_cat, rel, obj_id, obj_cat):
    sub_cat_index = label_id_to_index.get(sub_cat, -1)
    rel_index = relation_to_index.get(rel, -1)
    obj_cat_index = label_id_to_index.get(obj_cat, -1)
    return (sub_id, sub_cat_index, rel_index, obj_id, obj_cat_index)


def obb2poly_np_oc(rbboxes):
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
    a = rbboxes[4]
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


# 过滤过小box,否则后续计算会出错
def filter_rbox(rbox):
    if len(rbox) == 5:
        _, _, w, h, _ = rbox
    elif len(rbox) == 6:
        _, _, w, h, _, _ = rbox
    else:  # 长度不对
        return False
    if w < 2 or h < 2:
        return False
    # elif w < 10 or h <10:
    #     rbox[2] = rbox[2]*10
    #     rbox[3] = rbox[3]*10 #放大
    else:
        return True


def convert_obb_to_region_str(rbox_np):
    angle = rbox_np[-1]
    polys = obb2poly_np_oc(rbox_np)
    x_left = np.clip(np.min(polys[:, [0, 2, 4, 6]], axis=1), 0, None)
    y_top = np.clip(np.min(polys[:, [1, 3, 5, 7]], axis=1), 0, None)
    x_right = np.max(polys[:, [0, 2, 4, 6]], axis=1)
    y_bottom = np.max(polys[:, [1, 3, 5, 7]], axis=1)
    region_str = f"<{int(x_left[0])}><{int(y_top[0])}><{int(x_right[0])}><{int(y_bottom[0])}>|<{int(angle)}>"
    return region_str


def extract_rbox_from_str(match,
                          pattern=r'<(.*?)>'):
    '''
    input: <cx><cy><w><h>|<angle> (under 'oc' definition, angle is degree), str '<cx><cy><w><h>|<angle>'
    output: (cx, cy, w, h, angle) (angle is rad)
    '''
    numbers_str = re.findall(pattern, match)
    try:
        rbox = np.array(numbers_str, dtype=float)
    except ValueError:
        default_rbox = np.array([0., 0., 0., 0., 0], dtype=float)
        rbox = default_rbox
    if len(rbox) == 0:  # 没提取到
        return np.array([0., 0., 0., 0., 0], dtype=float)
    rbox[-1] = np.deg2rad(rbox[-1])
    return rbox


def extract_multi_rboxes_from_str(input_str):
    # 定义正则表达式模式，用于匹配每个矩形框
    pattern = r'\{(<.*?>)\}'
    # 使用正则表达式找到所有的矩形框
    matches = re.findall(pattern, input_str)
    rboxes = []
    # default_rbox = '({<-3><-3><3><3>|<0>})'
    default_rbox = np.array([0., 0., 0., 0., 0], dtype=float)
    for match in matches:
        # 在每个矩形框中，找到所有的数字
        numbers_str = re.findall(r'<(.*?)>', match)
        # 将数字转换为浮点数，并将角度转换为弧度
        try:
            rbox = np.array(numbers_str, dtype=float)
        except ValueError:
            # 如果转换失败，返回默认的数组
            rbox = default_rbox
        rbox[-1] = np.deg2rad(rbox[-1])
        # if filter_rbox(rbox):
        rboxes.append(rbox)
    # 将所有的矩形框参数合并成一个 numpy 数组
    return np.array(rboxes)


### for list convert to numpy for calculate mAP
def convert_list_to_rboxeval(det_result_input, annotation_input):
    det_results = [[] for _ in range(len(det_result_input))]
    num_classes = len(label_id)
    annotations = []
    # 遍历每个图像的检测结果
    for i, image_results in enumerate(det_result_input):
        ## 1) 处理annotation_input为要求格式
        image_annotations = annotation_input[i]
        bboxes = []
        labels = []
        # 遍历这个图像的每个注释
        for annotation in image_annotations:
            # 将这个注释的bbox和label添加到结果列表中
            bboxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])
        if not bboxes:
            continue
        bboxes = np.vstack(bboxes)
        labels = np.array(labels)
        # 将这个图像的bbox和label结果添加到总结果列表中
        annotations.append({'bboxes': bboxes, 'labels': labels})
        ## 2) 处理det_result_input为要求格式
        # 初始化一个列表来保存每个类别的检测结果
        per_class_results = [np.zeros((0, 6)) for _ in range(num_classes)]
        per_class_tmp_list = [[] for _ in range(num_classes)]
        # 遍历这个图像的每个检测结果
        for result in image_results:
            # 将这个检测结果添加到对应类别的结果列表中
            category_id = result['category_id']
            per_class_tmp_list[category_id].append(result['bbox'])
        # 将每个类别的结果合并为一个 (n, 6) 的数组，并添加到总结果列表中
        for j in range(num_classes):
            if per_class_tmp_list[j]:
                per_class_results[j] = np.vstack(per_class_tmp_list[j])
        det_results[i] = per_class_results

    det_results = [x for x in det_results if x != []]
    return det_results, annotations


### for task2
def calculate_relationships_acc(gt_relationships, pred_relationships):
    gt_rels = set(gt_relationships)
    pred_rels = set(pred_relationships)
    # Calculate the number of true positives (tp), false positives (fp), and false negatives (fn)
    tp = len(gt_rels & pred_rels)
    fp = len(pred_rels - gt_rels)
    fn = len(gt_rels - pred_rels)
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def calculate_relationships_tpfp(gt_relationships, pred_relationships):
    gt_rels = set(gt_relationships)
    pred_rels = set(pred_relationships)
    # Calculate the number of true positives (tp), false positives (fp), and false negatives (fn)
    tp = len(gt_rels & pred_rels)
    fp = len(pred_rels - gt_rels)
    fn = len(gt_rels - pred_rels)
    return tp, fp, fn


def calculate_relationships_PRF1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def parse_single_triplet(triplet_str):
    # 使用正则表达式找到三元组的各个部分
    region1 = re.findall(r'subject: (.+?),', triplet_str)
    region2 = re.findall(r'object: (.+?),', triplet_str)
    # 这里是单类别1对1, 还未考虑1对多匹配
    relationship = re.findall(r'<rel>(.*?)</rel>', triplet_str)
    # 如果任何一个部分的格式不正确，返回 None
    if len(region1) == 0 or len(region2) == 0 or len(relationship) == 0:
        return [], [], []

    return region1[0], region2[0], relationship


def parse_multi_catgory_rbox(input_string, add_score=False):
    # 提取所有的目标类别和对应的rbox
    # pattern = r'<ref>(.*?)</ref><rbox>\((.*?)\)</rbox>'
    if add_score:
        pattern = r'\b(\w+)\s*<rbox>\((.*?)\)</rbox>'
    else:
        pattern = r'<ref>(.*?)</ref><rbox>\((.*?)\)</rbox>'
    matches = re.findall(pattern, input_string)
    categories = []
    rboxes = []
    for match in matches:
        # 提取类别，并转换为对应的label_id
        category = match[0]
        if category.endswith('s'):
            category = category[:-1]
        category_id = label_id_to_index.get(category, -1)
        categories.append(category_id)
        # 提取rbox，并转换为numpy数组
        rbox_strs = match[1]
        tmp_rboxes = extract_multi_rboxes_from_str(rbox_strs)
        num_obj = tmp_rboxes.shape[0]
        for i in range(num_obj):
            rbox = tmp_rboxes[i]
            if add_score:
                rbox = np.append(rbox, 1.0)
            if filter_rbox(rbox):
                rboxes.append(rbox)

    if len(rboxes) > 0:
        rboxes_categories = list(zip(map(tuple, rboxes), categories))
        rboxes_categories = list(dict.fromkeys(rboxes_categories))
        rboxes, categories = zip(*rboxes_categories)
        rboxes = [np.array(rbox) for rbox in rboxes]

    det_result_per_image = [{'bbox': rbox, 'category_id': category_id} for rbox, category_id in zip(rboxes, categories)]

    return det_result_per_image


def parse_multi_rbox_nocatgory(input_string, add_score=False):
    pattern = r'(\{.*?\})'
    matches = re.findall(pattern, input_string)
    categories = []
    rboxes = []
    for match in matches:
        # 提取目标类别，并转换为对应的label_id
        category_id = -1  # 默认值
        categories.append(category_id)
        # 提取rbox，并转换为numpy数组
        rbox = extract_rbox_from_str(match)
        if add_score:
            rbox = np.append(rbox, 1.0)
        if filter_rbox(rbox):
            rboxes.append(rbox)
    if len(rboxes) > 0:
        # 将rboxes和categories合并为一个列表，每个元素是一个元组(rbox, category_id)
        rboxes_categories = list(zip(map(tuple, rboxes), categories))
        # 使用dict来删除重复的元素并保持原始顺序
        rboxes_categories = list(dict.fromkeys(rboxes_categories))
        # 分离rboxes和categories
        rboxes, categories = zip(*rboxes_categories)
        # 将rboxes转换回numpy.ndarray
        rboxes = [np.array(rbox) for rbox in rboxes]
    ##
    det_result_per_image = [{'bbox': rbox, 'category_id': category_id} for rbox, category_id in zip(rboxes, categories)]
    return det_result_per_image


size = ["small", "medium", "large", "giant"]

RBOX_START = '<rbox>'
RBOX_END = '</rbox>'
REF_START = '<ref>'
REF_END = '</ref>'
REL_START = '<rel>'
REL_END = '</rel>'


#### for Task5
def extract_triplets_from_str(str, if_gt=True):
    # 提取指示目标(区域)类别
    target_cat = ''
    target = ''
    match = re.search(r'(.*) on the .* part of the image', str.split('.')[0])
    if match is not None:
        target = match.group(1)
        for s in size:
            if s in target:
                match = re.search(s + r' (.*)', target)
                if match is None:
                    target = ''
                else:
                    target = match.group(1)
                # target_cat = re.search(s + r' (.*)', target).group(1)
                break
    elif target == '' and if_gt != True:  # 对于answer,如果回答中第一句格式不标准,无类别则用gt的类别来代替
        print('first sentence:', str.split('.')[0])
        target_cat = if_gt

    # 提取关系和其他对象
    # relations = re.findall(r'(\d+)? (.*?) \((.*?)\).*?<(.*)>', str)
    # 根据句号"."进行断句, 逐句提取三元组
    sentences = str.replace('\n', ' ').split('. ')[1:]
    triplets = []
    bboxes = []
    gt_bboxes = np.array((50., 50., 20., 20., 0.))
    obj_id_count = 1
    target_id = 0

    default_rel = 'background'
    default_ref = 'background'
    default_rbox = '({<0.><0.><0.><0.>|<0>})'
    # 在每一句中寻找relation ("<>"内的短语)
    for sentence in sentences:
        if sentence == "":
            continue
        sentence = sentence.lower()
        relation = re.findall(r'<rel>(.*?)</rel>', sentence)
        obj_cat = re.findall(r'<ref>(.*?)</ref>', sentence)
        unknow_boxes_str = re.findall(r'<rbox>(.*?)</rbox>', sentence)

        relation = next((item for item in re.findall(r'<rel>(.*?)</rel>', sentence)), default_rel)
        obj_cat = next((item for item in re.findall(r'<ref>(.*?)</ref>', sentence)), default_ref)
        unknow_boxes_str = next((item for item in re.findall(r'<rbox>(.*?)</rbox>', sentence)), default_rbox)

        rboxes_ = extract_multi_rboxes_from_str(unknow_boxes_str)
        count = int(rboxes_.shape[0])

        if "it is" in sentence:  # it-<rel>-obj
            # 考虑复数
            if count > 1 and obj_cat.endswith('s'):
                obj_cat = obj_cat[:-1]
            obj_rboxes = rboxes_
            for i in range(count):
                if filter_rbox(obj_rboxes[i]):
                    triplets.append(convert_to_numpy_triplet(target_id, target_cat, relation, obj_id_count, obj_cat))
                    bboxes.append((gt_bboxes, obj_rboxes[i]))
                    obj_id_count += 1
        elif "> it" in sentence:  # subj-<rel>-it
            if count > 1 and obj_cat.endswith('s'):
                obj_cat = obj_cat[:-1]
            obj_rboxes = rboxes_
            for i in range(count):
                if filter_rbox(obj_rboxes[i]):
                    triplets.append(convert_to_numpy_triplet(obj_id_count, obj_cat, relation, target_id, target_cat))
                    bboxes.append((obj_rboxes[i], gt_bboxes))
                    obj_id_count += 1

    if if_gt == True:
        return triplets, bboxes, target_cat
    else:
        return triplets, bboxes


#### for Task6
def extract_triplets_from_str_task6(str, add_score=False):
    sentences = str.replace('\n', ' ').split('. ')
    triplets = []
    bboxes = []
    # det_results_per_image = []
    rboxes_score = []
    categories = []
    id_count = 0

    for sentence in sentences:
        sentence = sentence.lower()
        if "sorry" in sentence and add_score == False:  # gt为负样本
            continue
        # Find all <rel> tags
        relation = re.findall(r'<rel>(.*?)</rel>', sentence)
        ## 1) SGG
        if relation:
            relation = relation[0]
            ref_values = re.findall(r'<ref>(.*?)</ref>', sentence)
            rbox_values = re.findall(r'<rbox>(.*?)</rbox>', sentence)
            default_ref = 'background'  # 考虑错误情况
            default_rbox = '({<0.><0.><0.><0.>|<0>})'  # 考虑错误情况
            while len(ref_values) < 2:
                ref_values.append(default_ref)
            subj_cat, obj_cat = ref_values
            while len(rbox_values) < 2:
                rbox_values.append(default_rbox)
            subj_boxes_str, obj_boxes_str = rbox_values

            # 考虑复数
            if subj_cat.endswith('s'):
                subj_cat = subj_cat[:-1]
            if obj_cat.endswith('s'):
                obj_cat = obj_cat[:-1]
            subj_rboxes = extract_multi_rboxes_from_str(subj_boxes_str)
            obj_rboxes = extract_multi_rboxes_from_str(obj_boxes_str)
            num_subj = subj_rboxes.shape[0]
            if obj_rboxes.shape[0] == 0:
                continue
            assert obj_rboxes.shape[0] <= 1
            obj_rboxes = obj_rboxes[0]
            if not filter_rbox(obj_rboxes):
                continue

            for i in range(num_subj):
                if filter_rbox(subj_rboxes[i]):
                    triplets.append(convert_to_numpy_triplet(id_count, subj_cat, relation, id_count + 1, obj_cat))
                    bboxes.append((subj_rboxes[i], obj_rboxes))  # 这里注意形状要是一维数组
                    id_count += 2

        ## 2) Object Detection
        elif not relation and RBOX_START in sentence:
            default_ref = 'background'
            default_rbox = '({<0.><0.><0.><0.>|<0>})'
            category = next((item for item in re.findall(r'<ref>(.*?)</ref>', sentence)), default_ref)
            rboxes_str = next((item for item in re.findall(r'<rbox>(.*?)</rbox>', sentence)), default_rbox)

            # 1) extract category
            if category.endswith('s'):
                category = category[:-1]
            # 2) extract rboxes in ground truth and answer
            rboxes = extract_multi_rboxes_from_str(rboxes_str)
            num_obj = rboxes.shape[0]
            for i in range(num_obj):
                rbox = rboxes[i]
                if add_score:
                    rbox = np.append(rbox, 1.0)
                if filter_rbox(rbox):
                    # 添加得分
                    rboxes_score.append(rbox)
                    # categories.append(label_id.index(category))
                    categories.append(label_id_to_index.get(category, -1))
            # det_result_per_image = [{'bbox': rbox, 'category_id': label_id.index(category)} for rbox in rboxes_score]

    det_results_per_image = [{'bbox': rbox, 'category_id': category_id} for rbox, category_id in
                             zip(rboxes_score, categories)]

    return triplets, bboxes, det_results_per_image


def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def replace_8para_groupdouble_to_5para(input_data):
    """
    8 param to 5 param
    """
    question = input_data['answer']
    gt = input_data['ground_truth']
    process_str = [question, gt]
    # 3) 进行替换,5参数->8参数
    for j, todo_str in enumerate(process_str):
        # 使用正则表达式查找所有 <rbox> 标签中的内容
        # pattern = r'\{(<[^>]+>[^}]+)\}'
        pattern = r'\{(<.*?>)\}'
        # pattern = r'<(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)>'
        # pattern = r'<rbox>\((.*?)\)</rbox>'
        # 使用正则表达式找到所有的矩形框
        matches = re.findall(pattern, todo_str)
        for match in matches:
            # 在每个矩形框中，找到所有的数字
            subpattern = r'<(.+?)>'
            numbers_str = re.findall(subpattern, match)
            print(todo_str)
            print(numbers_str)
            numbers_str_reorder = [numbers_str[0], numbers_str[1],numbers_str[6],numbers_str[7],numbers_str[4],numbers_str[5],numbers_str[2],numbers_str[3]]
            # 将数字转换为浮点数，并将角度转换为弧度
            rbox = np.array(numbers_str_reorder, dtype=np.float32).reshape(-1)
            polys = poly2obb_np_oc(rbox)
            cx_, cy_, w_, h_, a_rad = polys
            a_degrees = math.degrees(a_rad)
            rbox_str = "<%.2f><%.2f><%.2f><%.2f>|<%d>" % (cx_, cy_, w_, h_, a_degrees)
            todo_str = todo_str.replace(match, rbox_str)
            if " <ref> " in todo_str:
                todo_str = todo_str.replace(" <ref> ", "<ref>")
            if " </ref> " in todo_str:
                todo_str = todo_str.replace(" </ref> ", "</ref>")
        process_str[j] = todo_str
    question, gt = process_str
    input_data['answer'] = question
    input_data['ground_truth'] = gt

    return input_data


def evaluation_metrics_ComplexCompre(data_path, param, group=None):
    base = [json.loads(q) for q in open(data_path, "r")]
    ######## pre definition #########
    ## Task1 Object Detection
    det_results_task1 = [[] for _ in range(len(base))]
    gt_annotations_task1 = [[] for _ in range(len(base))]
    ## Task2 Relation Detection
    count_task2 = 0
    recall_task2 = 0
    precision_task2 = 0
    tp_task2 = 0
    fp_task2 = 0
    fn_task2 = 0
    ## Task3 Relation Reasoning
    recall_task3 = 0
    tp_task3 = 0
    fp_task3 = 0
    fn_task3 = 0
    ## Task4 Object Reasoning
    det_results_task4 = [[] for _ in range(len(base))]
    gt_annotations_task4 = [[] for _ in range(len(base))]
    ## Task5 Region Grounding
    gt_inputs_task5 = []
    predictions_task5 = []
    ## Task6 Image Grounding
    gt_inputs_task6 = []
    predictions_task6 = []
    det_results_task6 = [[] for _ in range(len(base))]
    gt_annotations_task6 = [[] for _ in range(len(base))]

    ################################
    # for answers in tqdm(base):
    for i, answers in enumerate(tqdm(base)):
        # if answers['answer'] == "A fully visible medium car on the top right part of the image.\n 1 car<rbox>({<91.75,16.78><93.59,23.99><89.91,24.98><88.07,17.77>)})</rbox> is <rel>parked alongside with</rel> it.":
        #     print()
        answers = replace_8para_groupdouble_to_5para(answers)
        gt = answers['ground_truth']
        answer = answers['answer']
        task_category = answers['category']
        if "due to the context length" in gt or "..." in gt:  # NOTE: too long to evaluate, "..."则是出现在grounding任务中
            continue
        pattern_loc = r'\{(.+?)\}'
        pattern_ = r'<(.+?)>'
        if task_category == "task1":  # Object Detection## Metrics: mAP for all, mean IoU
            # 1) extract category
            category_match = re.search(r'There (?:is|are) \d+ (.+?)s? in the image', gt)
            if category_match is None:  # 负样本
                continue
            category = category_match.group(1)
            category = category.rstrip('s')
            # 2) extract rboxes in ground truth and answer
            rbox_matches_gt = re.findall(pattern_loc, gt)
            rboxes_gt = []
            for match in rbox_matches_gt:
                rbox = extract_rbox_from_str(match)
                if filter_rbox(rbox):
                    rboxes_gt.append(rbox)
            rbox_matches_pre = re.findall(pattern_loc, answer)
            rboxes_pre = []
            for match in rbox_matches_pre:
                rbox = extract_rbox_from_str(match)
                if filter_rbox(rbox):
                    rbox = np.append(rbox, 1.0)
                    rboxes_pre.append(rbox)
            # 3) append to det_results and gt_annotations
            det_results_per_image1 = [{'bbox': rbox, 'category_id': label_id_to_index.get(category, -1)} for rbox in
                                      rboxes_pre]
            det_results_task1[i].extend(det_results_per_image1)
            gt_annotations_per_image1 = [{'bbox': rbox, 'category_id': label_id_to_index.get(category, -1)} for rbox in
                                         rboxes_gt]

            gt_annotations_task1[i].extend(gt_annotations_per_image1)
            continue

        elif task_category == "task2":  # Relationship Detection
            # "ground_truth": "There are 2 relationships between tank and tank: tank <not co-storage with> tank, tank <co-storage with> tank"
            # Metrics: Recall, Precision
            pattern_r = re.compile(r'<(.*?)>')
            rel_gt = re.findall(pattern_r, gt)
            rel_pre = re.findall(pattern_r, answer)
            tp, fp, fn = calculate_relationships_tpfp(rel_gt, rel_pre)
            tp_task2 += tp
            fp_task2 += fp
            fn_task2 += fn
            continue

        elif task_category == "task3":  # Referring Relationship Reasoning
            cat1_gt, cat2_gt, rel_gt = parse_single_triplet(gt)
            cat1_pre, cat2_pre, rel_pre = parse_single_triplet(answer)
            if not rel_gt:  # 负样本
                continue
            # calculate accuracy
            # acc为单标签分类,用于多标签时不会考虑顺序
            if cat1_gt == cat1_pre and cat2_gt == cat2_pre:
                tp, fp, fn = calculate_relationships_tpfp(rel_gt, rel_pre)
                tp_task3 += tp
                fp_task3 += fp
                fn_task3 += fn
            elif cat1_pre != [] and cat2_pre != []:  # 类别预测错误
                tp = 0
                fp = len(rel_pre)
                fn = len(rel_gt)
            else:  # 类别预测为空
                tp = 0
                fp = 0
                fn = len(rel_gt)
            continue

        elif task_category == "task4":  # Object Reasoning
            if 'categories' in gt:  # 类别+box
                det_results_per_image4 = parse_multi_catgory_rbox(answer, add_score=True)
                gt_annotations_per_image4 = parse_multi_catgory_rbox(gt)
            else:  # 仅box
                det_results_per_image4 = parse_multi_rbox_nocatgory(answer, add_score=True)
                gt_annotations_per_image4 = parse_multi_rbox_nocatgory(gt)
            det_results_task4[i].extend(det_results_per_image4)
            gt_annotations_task4[i].extend(gt_annotations_per_image4)
            continue

        elif task_category == "task5":  # Region Grounding
            obj_gt = re.findall(pattern_loc, gt)
            if not obj_gt:  # gt不含rbox tag, 无法计算三元组
                continue
            # obj_pre = re.findall(pattern_loc, answer)
            ## 1) 首先从gt和prediction分别提取三元组、关系
            # 提取目标对象并保存提及的三元组
            gt_triplets, gt_bboxes, target_cat = extract_triplets_from_str(gt, if_gt=True)
            pre_triplets, pre_bboxes = extract_triplets_from_str(answer, if_gt=target_cat)
            ## 2) 按照SGG中的eval方式来进行评估
            # Compute_Pred_Matches(gt_triplets, pre_triplets, gt_bboxes, pre_bboxes, iou_thres=0.5, phrdet=False)
            gt_input = {'gt_triplet': gt_triplets, 'gt_bboxes': gt_bboxes}
            prediction = {'pred_triplet': pre_triplets, 'pred_bboxes': pre_bboxes}
            gt_inputs_task5.append(gt_input)
            predictions_task5.append(prediction)
            continue

        elif task_category == "task6":  # Image Grounding
            obj_gt = re.findall(pattern_loc, gt)
            if not obj_gt:  # gt不含grounding标签, 无法计算三元组
                continue
            if 'sorry' in gt:  # negative sample
                continue
            gt_triplets_t6, gt_bboxes_t6, gt_annotations_per_image6 = extract_triplets_from_str_task6(gt)
            pre_triplets_t6, pre_bboxes_t6, det_results_per_image6 = extract_triplets_from_str_task6(answer,
                                                                                                     add_score=True)

            ## 2) 按照SGG中的eval方式来进行评估
            # Compute_Pred_Matches(gt_triplets, pre_triplets, gt_bboxes, pre_bboxes, iou_thres=0.5, phrdet=False)
            gt_input_t6 = {'gt_triplet': gt_triplets_t6, 'gt_bboxes': gt_bboxes_t6}
            prediction_t6 = {'pred_triplet': pre_triplets_t6, 'pred_bboxes': pre_bboxes_t6}
            gt_inputs_task6.append(gt_input_t6)
            predictions_task6.append(prediction_t6)

            ## 目标检测评估
            gt_annotations_task6[i].extend(gt_annotations_per_image6)
            det_results_task6[i].extend(det_results_per_image6)

    ######## Output Results #######
    iou_thr = 0.25
    print(f"=======iou thr: {iou_thr}========")
    ### Task1
    # convert format
    det_task_1, gt_task_1 = convert_list_to_rboxeval(det_results_task1, gt_annotations_task1)
    # eval map
    mean_ap_1, result_1 = eval_rbbox_map(det_task_1, gt_task_1, iou_thr=iou_thr)
    print(f"Task-Object Detection mean ap: {mean_ap_1}")
    ## Task 2
    # 新方式
    precision_task2, recall_task2, f1_task2 = calculate_relationships_PRF1(tp_task2, fp_task2, fn_task2)
    print(f'Task-Relation Detection Average Precision: {precision_task2:.4f}')
    print(f'Task-Relation Detection Average Recall: {recall_task2:.4f}')
    print(f'Task-Relation Detection F1 score: {f1_task2:.4f}')

    ### Task 3
    precision_task3, recall_task3, f1_task3 = calculate_relationships_PRF1(tp_task3, fp_task3, fn_task3)
    print(f'Task-Relation Reasoning Average Precision: {precision_task3:.4f}')
    print(f'Task-Relation Reasoning Average Recall: {recall_task3:.4f}')
    print(f'Task-Relation Reasoning F1 score: {f1_task3:.4f}')

    ### Task 4
    det_task_4, gt_task_4 = convert_list_to_rboxeval(det_results_task4, gt_annotations_task4)
    # eval map
    mean_ap_4, result_4 = eval_rbbox_map(det_task_4, gt_task_4, iou_thr=iou_thr)
    print(f"Task-Object Reasoning mean ap: {mean_ap_4}")
    ### Task 5
    print("Task-Region-level SGG result:")
    do_vg_evaluation(gt_inputs_task5, predictions_task5, iou_thres=[iou_thr])
    ## Task 6
    print("Task-Image-level SGG result:")
    do_vg_evaluation(gt_inputs_task6, predictions_task6, iou_thres=[iou_thr])
    det_task_6, gt_task_6 = convert_list_to_rboxeval(det_results_task6, gt_annotations_task6)
    mean_ap_6, _ = eval_rbbox_map(det_task_6, gt_task_6, iou_thr=iou_thr)
    print(f"Task-Image-level SGG mean ap: {mean_ap_6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file", type=str,
                        default="/project/luojunwei/VisionLanguage/Code/GeoChat/output_answers/geochat-7B/FITRS_complex_comprehension_eval_geochat-7B.jsonl")
    args = parser.parse_args()

    evaluation_metrics_ComplexCompre(args.answer_file)