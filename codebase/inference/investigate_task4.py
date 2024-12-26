import json
import jsonlines
import os
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import re
import cv2
from IPython.display import display
import PIL.Image as Image
import math
import matplotlib.pyplot as plt
from multiprocessing import get_context
from terminaltables import AsciiTable
import pdb
import helper

import matplotlib.pyplot as plt
import matplotlib.patches as patches


try:
    from mmcv.utils import print_log
except ImportError:
    from mmengine.logging import print_log


Image.MAX_IMAGE_PIXELS = None


def read_eval_jsonl(jsonl_path):
    base = [json.loads(q) for q in open(jsonl_path, "r")]
    return base


def fit_denormalization(rbbox, image_w=512, image_h=512):
    """obb2poly_np_oc_2rad
    rbbox: normalzied rbbox
    return: unnormalized rbbox (deg angle)
    """
    cx, cy, w, h, angle_deg = rbbox
    cx = cx / 100 * image_w
    cy = cy / 100 * image_h
    w = w / 100 * image_w
    h = h / 100 * image_h
    # angle_rad = math.radians(angle_deg)
    return cx, cy, w, h, angle_deg


def obb2poly_np_oc_2rad(rbbox):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbbox[0]
    y = rbbox[1]
    w = rbbox[2]
    h = rbbox[3]
    a = np.radians(rbbox[4])
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.array([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])
    return polys.tolist()


def obb2poly_np_oc_2rad_obb2(rbbox):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x1 = rbbox[0]
    y1 = rbbox[1]
    x2 = rbbox[2]
    y2 = rbbox[3]

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    a = np.radians(rbbox[4])
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.array([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])
    return polys.tolist()


def rbbox_to_corners(rbbox):
    # https://blog.csdn.net/W1995S/article/details/115583492
    cx, cy, w, h, angle_deg = rbbox
    angle_rad = math.radians(angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    c = np.array([cx, cy]).reshape(-1, 1)
    v = np.array(
        [
            [-w / 2, -h / 2],
            [-w / 2, h / 2],
            [w / 2, h / 2],
            [w / 2, -h / 2],
        ]
    ).T
    rotation_matrix = np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ]
    )
    corners = rotation_matrix @ v + c
    corners = corners.T.flatten().tolist()
    return corners


def visualize_rbbox(image, rbbox, color):
    """
    denormalized rbbox
    """
    print("Rbbox Vertices: {}".format(rbbox))
    vertices = np.array(
        [
            [int(rbbox[0]), int(rbbox[1])],
            [int(rbbox[2]), int(rbbox[3])],
            [int(rbbox[4]), int(rbbox[5])],
            [int(rbbox[6]), int(rbbox[7])],
        ]
    )
    print(vertices)
    cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=1)
    return image


def extract_coord(input_str):
    pattern = r'\{(<.*?>)\}'
    # 使用正则表达式找到所有的矩形框
    matches = re.findall(pattern, input_str)
    rbox_list = []
    for match in matches:
        numbers_str = re.findall(r'<(.*?)>', match)
        # 将数字转换为浮点数，并将角度转换为弧度
        rbox = np.array(numbers_str, dtype=np.float32)
        rbox_list.append(rbox)
    return np.array(rbox_list)


def visualize_fit(example, image_root):
    print("FIT Dataset Vis......")
    question_id = example["question_id"]
    image_id = example["image_id"]
    task_category = example["category"]
    gt = example["ground_truth"]
    gt_coords = extract_coord(gt)
    prediction = example["answer"]
    prediction_coords = extract_coord(prediction)

    image = cv2.imread(os.path.join(image_root, image_id), cv2.IMREAD_UNCHANGED)
    height, width = image.shape[0], image.shape[1]
    for gt_rbox in gt_coords:
        print("gt_rbox: {}".format(gt_rbox))
        rbbox_denorm_5param = fit_denormalization(gt_rbox, width, height)
        print("rbbox_denorm_5param: {}".format(rbbox_denorm_5param))
        rbbox_denorm_8param = obb2poly_np_oc_2rad(rbbox_denorm_5param)
        print("rbbox_denorm_8param: {}".format(rbbox_denorm_8param))
        image = visualize_rbbox(image, rbbox_denorm_8param, color=(0, 255, 0))
        print()
    print()
    for pred_rbox in prediction_coords:
        print("pred_rbox: {}".format(pred_rbox))
        rbbox_denorm_5param = fit_denormalization(gt_rbox, width, height)
        print("rbbox_denorm_5param: {}".format(rbbox_denorm_5param))
        rbbox_denorm_8param = obb2poly_np_oc_2rad(rbbox_denorm_5param)
        print("rbbox_denorm_8param: {}".format(rbbox_denorm_8param))
        image = visualize_rbbox(image, rbbox_denorm_8param, color=(0, 0, 255))
        print()
    return image


def show_vis_result(image, para5_star_example, para5_star_question_example):
    for k, v in para5_star_example.items():
        print("{}: {}".format(k, v))
    for k, v in para5_star_question_example.items():
        print("{}: {}".format(k, v))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    # plt.imshow(image)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    display(Image.fromarray(image))


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


def merge_q_gt(q_jsonl, a_jsonl):
    qagt_dict = dict()
    for q in tqdm(q_jsonl):
        q_id = q["question_id"]
        qagt_dict[q_id] = q
        for a in a_jsonl:
            if a["question_id"] == q_id:
                qagt_dict[q_id]["answer"] = a["answer"]
    return qagt_dict


def draw_rotated_box(ax, vertices, color='red', alpha=0.5):
    box = patches.Polygon(np.array(vertices).reshape(-1, 2), closed=True, edgecolor=color, facecolor='none',
                          linewidth=2, alpha=alpha)
    ax.add_patch(box)


def visualize_rotated_bboxes(image_path, gt_boxes, pred_bboxes):
    # 读取图像
    img = plt.imread(image_path)

    # 创建一个1行2列的子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 可视化真实旋转边界框
    axs[0].imshow(img)
    axs[0].set_title('Ground Truth Rotated BBoxes')
    for bbox in gt_boxes:
        draw_rotated_box(axs[0], bbox, color='red')
    axs[0].axis('off')  # 关闭坐标轴
    # 在子图下方显示图片路径
    axs[0].set_xlabel(image_path)

    # 可视化预测旋转边界框
    axs[1].imshow(img)
    axs[1].set_title('Prediction Rotated BBoxes')
    for bbox in pred_bboxes:
        draw_rotated_box(axs[1], bbox, color='blue')
    axs[1].axis('off')  # 关闭坐标轴
    # 在子图下方显示图片路径
    axs[1].set_xlabel(image_path)

    # 调整子图间距
    plt.tight_layout()

    # 显示图像
    plt.show()


def normalized_bbox2obb(normalized_bbox, image_w=512, image_h=512):
    denormed_bbox = fit_denormalization(normalized_bbox, image_w, image_h)
    poly = obb2poly_np_oc_2rad(denormed_bbox)
    return poly


# ## all categories
label_id = ['airplane', 'boat', 'taxiway', 'boarding_bridge', 'tank', 'ship', 'crane',
            'car', 'apron', 'dock', 'storehouse', 'goods_yard', 'truck', 'terminal',
            'runway', 'breakwater', 'car_parking', 'bridge', 'cooling_tower',
            'truck_parking', 'chimney', 'vapor', 'coal_yard', 'genset', 'smoke',
            'gas_station', 'lattice_tower', 'substation', 'containment_vessel', 'flood_dam', 'ship_lock', 'gravity_dam',
            'arch_dam', 'cement_concrete_pavement', 'toll_gate', 'tower_crane', 'engineering_vehicle',
            'unfinished_building', 'foundation_pit',
            'wind_mill', 'intersection', 'roundabout', 'ground_track_field', 'soccer_ball_field', 'basketball_court',
            'tennis_court', 'baseball_diamond', 'stadium']

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


def parse_multi_catgory_rbox(input_string, add_score=False):
    # 提取所有的目标类别和对应的rbox
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


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes

        tpfp = pool.starmap(
            helper.tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # tpfp = tpfp_default(cls_dets, cls_gts, cls_gts_ignore,  [iou_thr for _ in range(num_imgs)], [area_ranges for _ in range(num_imgs)] )
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        # try:
        #     cls_dets = np.vstack(cls_dets)
        # except:
        #     print()
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


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


# 过滤过小box,否则后续计算会出错
def filter_rbox(rbox):
    if len(rbox) == 5:
        _, _, w, h, _ = rbox
    elif len(rbox) == 6:
        _, _, w, h, _, _ = rbox
    else:  # 长度不对
        return False
    if w < 0 or h < 0:
        return False
    # elif w < 10 or h <10:
    #     rbox[2] = rbox[2]*10
    #     rbox[3] = rbox[3]*10 #放大
    else:
        return True


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


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
            # cls_gts_ignore.append(torch.zeros((0, 8), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


def parse_gt_answer_investigate(qagt_dict, image_root, task_id="task4", vis_num=10):
    size = ["small", "medium", "large", "giant"]

    RBOX_START = '<rbox>'
    RBOX_END = '</rbox>'
    REF_START = '<ref>'
    REF_END = '</ref>'
    REL_START = '<rel>'
    REL_END = '</rel>'

    iou_thr = 0.25
    det_results_task4 = [[] for _ in range(len(qagt_dict))]
    gt_annotations_task4 = [[] for _ in range(len(qagt_dict))]

    vis_flag = True
    count_vis = 1
    i = 0
    for question_id, example in tqdm(qagt_dict.items()):
        if count_vis > vis_num:
            vis_flag = False
        if example["category"] == task_id:
            question = example["question"]
            image_id = example["image"]
            task_category = example["category"]
            gt = example["ground_truth"]
            prediction = example["answer"]
            if vis_flag:
                print("Image ID: {}".format(image_id))
                print("Question: {}".format(question))
                print("Answer: {}".format(prediction))
                print("Ground Truth: {}".format(gt))
                gt_coords = extract_coord(gt)
                gt_coords = [normalized_bbox2obb(gt_coord) for gt_coord in gt_coords]
                prediction_coords = extract_coord(prediction)
                prediction_coords = [normalized_bbox2obb(prediction_coord) for prediction_coord in prediction_coords]
                print("{} gt boxes; {} prediction boxes".format(len(gt_coords), len(prediction_coords)))
                visualize_rotated_bboxes(os.path.join(image_root, image_id), gt_coords, prediction_coords)
                count_vis += 1
                print()
            if 'categories' in gt:  # 类别+box
                det_results_per_image4 = parse_multi_catgory_rbox(prediction, add_score=True)
                gt_annotations_per_image4 = parse_multi_catgory_rbox(gt)
            else:  # 仅box
                det_results_per_image4 = parse_multi_rbox_nocatgory(prediction, add_score=True)
                gt_annotations_per_image4 = parse_multi_rbox_nocatgory(gt)
            det_results_task4[i].extend(det_results_per_image4)
            gt_annotations_task4[i].extend(gt_annotations_per_image4)
        i += 1
    ### Task 4
    det_task_4, gt_task_4 = convert_list_to_rboxeval(det_results_task4, gt_annotations_task4)
    # eval map
    mean_ap_4, result_4 = eval_rbbox_map(det_task_4, gt_task_4, iou_thr=iou_thr)
    print(f"Task-Object Reasoning mean ap: {mean_ap_4}")


def main():
    root = os.getcwd()
    fit_image_root = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS_Img/imgv2_split_512_100_vaild"
    para5_fit_question_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/test_FITRS_complex_comprehension_eval_5para_complete_fit.jsonl"
    para5_fit_eval_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/test_FITRS_complex_comprehension_eval_5para_complete_fit_eval.jsonl"
    # para5_fit_eval_path = "/media/zilun/fanxiang4t/GRSM/ov/inference/rsvqa/result/skysensegpt-fullft-1e6/FITRS_complex_comprehension_eval_skysensegpt-fullft-1e6.jsonl"
    para5_fit_question = read_eval_jsonl(para5_fit_question_path)
    para5_fit_eval = read_eval_jsonl(para5_fit_eval_path)
    qagt_dict = merge_q_gt(para5_fit_question, para5_fit_eval)
    parse_gt_answer_investigate(qagt_dict, fit_image_root, task_id="task4", vis_num=0)


if __name__ == "__main__":
    main()