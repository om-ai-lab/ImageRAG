import argparse
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import re
import jsonlines
import PIL.Image as Image
import pickle as pkl
import cv2
Image.MAX_IMAGE_PIXELS = None


def obb12obb2(rbbox):
    cx, cy, w, h, theta = rbbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2, theta


def obb22obb1(rbbox):
    x1, y1, x2, y2, theta = rbbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h, theta


def obb12poly_np_oc_2rad(rbboxes):
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


def obb2_to_min_out_hbb(obb2):
    """Convert obb2 to minium out hbb.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    obb1 = obb22obb1(obb2)
    poly = obb12poly_np_oc_2rad(obb1)
    bboxps = np.array(poly).reshape((4, 2))

    HBB_x1 = np.min(bboxps[:,0])
    HBB_y1 = np.min(bboxps[:,1])
    HBB_x2 = np.max(bboxps[:,0])
    HBB_y2 = np.max(bboxps[:,1])

    return HBB_x1, HBB_y1, HBB_x2, HBB_y2


def dump_wh(star_dir_list, save_path):
    check_dict = dict()
    for star_dir in star_dir_list:
        for img_name in os.listdir(star_dir):
            if img_name.endswith("png"):
                img_path = os.path.join(star_dir, img_name)
                img = Image.open(img_path)
                w, h = img.size
                check_dict[img_name] = (w, h, star_dir)
    pkl.dump(check_dict, open(save_path, "wb"))


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


def star_denormalization(rbbox, image_w, image_h):
    """obb2poly_np_oc_2rad
    rbbox: normalzied rbbox
    return: unnormalized rbbox (deg angle)
    """
    x1, y1, x2, y2, angle_deg = rbbox
    x1 = x1 / 1000 * image_w
    y1 = y1 / 1000 * image_h
    x2 = x2 / 1000 * image_w
    y2 = y2 / 1000 * image_h
    # angle_rad = math.radians(angle_deg)
    return x1, y1, x2, y2, angle_deg


def convert_bboxes(obb1_bboxes):
    obb2_bboxes = []
    for obb1_bbox in obb1_bboxes:
        cx, cy, w, h = obb1_bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        obb2_bboxes.append((x1, y1, x2, y2))
    return obb2_bboxes


def sole_visualcue2mergedvisualcue(obb2_boxes):

    # 初始化最小和最大坐标
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    # 遍历所有边界框
    for bbox in obb2_boxes:
        x1, y1, x2, y2 = bbox
        # 更新最小和最大坐标
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    # 返回包含所有边界框的大边界框
    return min_x, min_y, max_x, max_y



def get_patch_scale_bbox(bbox, scale_factor, x_upper, y_upper):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    new_w = w * scale_factor
    new_h = h * scale_factor

    x1 = cx - new_w / 2
    y1 = cy - new_h / 2
    x2 = cx + new_w / 2
    y2 = cy + new_h / 2

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, x_upper)
    y2 = min(y2, y_upper)
    return x1, y1, x2, y2



def fit2star(fit_coord_denorm, star_upleft_coord_x, star_upleft_coord_y):
    """
    Denormed OBB2 FIT coord to STAR coord
    """

    # target obb2
    x1, y1, x2, y2, theta = fit_coord_denorm
    print("FIT Coord: {}".format(fit_coord_denorm))
    x1_star = star_upleft_coord_x + x1
    y1_star = star_upleft_coord_y + y1
    x2_star = star_upleft_coord_x + x2
    y2_star = star_upleft_coord_y + y2
    print("STAR Coord: {}".format((x1_star, y1_star, x2_star, y2_star, theta)))
    return x1_star, y1_star, x2_star, y2_star, theta


def vis_bbox(img_path, list_bbox, dataset_name):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def draw_rotated_box(ax, vertices, color='red', alpha=0.5):
        box = patches.Polygon(np.array(vertices).reshape(-1, 2), closed=True, edgecolor=color, facecolor='none',
                              linewidth=5, alpha=alpha)
        ax.add_patch(box)

    img = plt.imread(img_path)

    # 创建一个1行2列的子图
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # 可视化真实旋转边界框
    axs.imshow(img)
    axs.set_title('Ground Truth Rotated BBoxes')
    for obb2_bbox in list_bbox:
        obb1_bbox = obb22obb1(obb2_bbox)
        poly = obb12poly_np_oc_2rad(obb1_bbox)
        draw_rotated_box(axs, poly, color='red')
    axs.axis('off')  # 关闭坐标轴
    # 在子图下方显示图片路径
    axs.set_xlabel(img_path)

    # 调整子图间距
    plt.tight_layout()
    # 显示图像
    plt.show()


def bbox_location(image_width, image_height, bbox):
    """
    :param image_width:
    :param image_height:
    :param bbox: old: (up left x, up left y, w, h), new: (x1, y1, x2, y2)
    :return:
    """
    # Define the 3x3 grid dimensions
    grid_width = image_width / 3
    grid_height = image_height / 3

    # Extract bbox details
    x1, y1, x2, y2 = bbox
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    # x, y, w, h = bbox

    # Define the boundaries for each of the 9 regions
    regions = {
        "Top-left":      (grid_width * 0, grid_height * 0, grid_width, grid_height),
        "Top-center":    (grid_width * 1, grid_height * 0, grid_width, grid_height),
        "Top-right":     (grid_width * 2, grid_height * 0, grid_width, grid_height),
        "Center-left":   (grid_width * 0, grid_height * 1, grid_width, grid_height),
        "Center":        (grid_width * 1, grid_height * 1, grid_width, grid_height),
        "Center-right":  (grid_width * 2, grid_height * 1, grid_width, grid_height),
        "Bottom-left":   (grid_width * 0, grid_height * 2, grid_width, grid_height),
        "Bottom-center": (grid_width * 1, grid_height * 2, grid_width, grid_height),
        "Bottom-right":  (grid_width * 2, grid_height * 2, grid_width, grid_height)
    }

    def intersection_area(target_bbox, region_bbox):
        """

        :param target_bbox: x1, y1, w1, h1
        :param region_bbox: x2, y2, w2, h2
        :return:
        """

        x1, y1, w1, h1 = target_bbox
        x2, y2, w2, h2 = region_bbox

        # Calculate the overlap boundaries
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        intersection_area = max(0, xB - xA) * max(0, yB - yA)

        return intersection_area

    # Calculate intersection area for each region
    overlaps = {
        region: intersection_area([x, y, w, h], [rx, ry, rw, rh])
        for region, (rx, ry, rw, rh) in regions.items()
    }

    # Return the region with the maximum overlap
    first_return = max(overlaps, key=overlaps.get)
    # del overlaps[max(overlaps, key=overlaps.get)]
    # second_return = max(overlaps, key=overlaps.get)
    # return "{} and {} blocks".format(first_return, second_return)
    return first_return


def json2jsonl(input_file_path):
    with open(input_file_path, 'r') as f:
        json_data = json.load(f)

    with jsonlines.open(input_file_path + 'l', 'w') as writer:
        for obj in tqdm(json_data):
            writer.write(obj)


def jsons2jsonl(input_file_path_list, output_file_path):
    merged_data = []

    for file_path in tqdm(input_file_path_list):
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)

    with jsonlines.open(output_file_path, 'w') as writer:
        for obj in tqdm(merged_data):
            writer.write(obj)


def process_intermediate(json_path, star_stats, output_file_path, task_type):
    with open(json_path, "r") as f:
        base = json.load(f)
    modified_data = []
    # 匹配 <rbox>
    for i, instruction in enumerate(tqdm(base)):
        conv = instruction['conversations']
        # 0000__512__0___0.png
        img_name = instruction['image']
        star_img_name = img_name.split("__")[0] + ".png"
        instruction['star_image'] = star_img_name
        instruction['train_task_type'] = task_type

        # FIT/FIT-RS/FIT-RS_Instruction/FIT-RS_Img/imgv2_split_512_100_vaild/0000__512__0___824.png
        star_upleft_coord_x = int(img_name.split("___")[0].split("__")[-1])
        star_upleft_coord_y = int(img_name.split("___")[1].split(".")[0])
        w_fit, h_fit = int(img_name.split("___")[0].split("__")[-2]), int(img_name.split("___")[0].split("__")[-2])
        w_star, h_star, star_img_dir = star_stats[star_img_name]
        instruction['star_width'] = w_star
        instruction['star_height'] = h_star
        fit_relative_pos_star = bbox_location(w_star, h_star, [star_upleft_coord_x, star_upleft_coord_y, star_upleft_coord_x + w_fit, star_upleft_coord_y + h_fit])
        instruction['fit_relative_pos_star'] = fit_relative_pos_star
        instruction['additional_roi_coord'] = dict()
        instruction['additional_roi_coord']['fit'] = [[
            star_upleft_coord_x / w_star * 1000,
            star_upleft_coord_y / h_star * 1000,
            (star_upleft_coord_x + w_fit) / w_star * 1000,
            (star_upleft_coord_y + h_fit) / h_star * 1000
        ]]
        instruction['additional_roi_coord']['object'] = []
        star_img_path = os.path.join(star_img_dir, star_img_name)

        # TODO: convert fit coord to star
        for sentence in conv:
            if '<rbox>' in sentence['value']:
                # 进行替换,5参数->8参数
                todo_str = sentence['value']
                print(todo_str)
                # 使用正则表达式查找所有 <rbox> 标签中的内容
                pattern = r'\{(<.*?>)\}'
                # 使用正则表达式找到所有的矩形框
                matches = re.findall(pattern, todo_str)
                fit_bbox_list = []
                star_bbox_list = []
                for match in matches:
                    # 在每个矩形框中，找到所有的数字
                    numbers_str = re.findall(r'<(.*?)>', match)
                    # 将数字转换为浮点数，并将角度转换为弧度
                    rbox = np.array(numbers_str, dtype=np.float32)
                    # obb1: (cx, cy, w, h, theta_degree)
                    fit_denorm = fit_denormalization(rbox, image_w=w_fit, image_h=h_fit)
                    # obb2: (x1, y1, x2, y2, theta)
                    x1, y1, x2, y2, theta_degree = obb12obb2(fit_denorm)
                    # star obb2: (x1_star, y1_star, x2_star, y2_star, theta_degree)
                    x1_star, y1_star, x2_star, y2_star, theta_degree = fit2star((x1, y1, x2, y2, theta_degree), star_upleft_coord_x, star_upleft_coord_y)
                    fit_bbox_list.append([x1, y1, x2, y2, theta_degree])
                    star_bbox_list.append([x1_star, y1_star, x2_star, y2_star, theta_degree])
                    # normalize star coord to 0~1000
                    x1_star_norm, x2_star_norm = np.array([x1_star, x2_star]) / w_star * 1000
                    y1_star_norm, y2_star_norm = np.array([y1_star, y2_star]) / h_star * 1000
                    rbox_str = "{<%.2f><%.2f><%.2f><%.2f>|<%d>}" % (x1_star_norm, y1_star_norm, x2_star_norm, y2_star_norm, theta_degree)
                    todo_str = todo_str.replace(f'{{{match}}}', rbox_str)

                    # TODO: Separate QA's ROI
                    # out_hbb = obb2_to_min_out_hbb([x1_star_norm, y1_star_norm, x2_star_norm, y2_star_norm, theta_degree])
                    # out_hbb_scaled = get_patch_scale_bbox(out_hbb, scale_factor=1.2, x_upper=1000, y_upper=1000)

                    out_hbb = obb2_to_min_out_hbb([x1_star, y1_star, x2_star, y2_star, theta_degree])
                    out_hbb_scaled = get_patch_scale_bbox(out_hbb, scale_factor=1.3, x_upper=w_star, y_upper=h_star)
                    out_hbb_scaled_x = np.array([out_hbb_scaled[0], out_hbb_scaled[2]]) / w_star * 1000
                    out_hbb_scaled_y = np.array([out_hbb_scaled[1], out_hbb_scaled[3]]) / h_star * 1000
                    instruction['additional_roi_coord']['object'].append([out_hbb_scaled_x[0], out_hbb_scaled_y[0], out_hbb_scaled_x[1], out_hbb_scaled_y[1]])
                    # star_bbox_list.append(out_hbb)

                # vis_bbox(fit_img_path, fit_bbox_list, "fit")
                # vis_bbox(star_img_path, star_bbox_list, "star")
                merged_visual_cue = sole_visualcue2mergedvisualcue(instruction['additional_roi_coord']['object'])
                instruction['additional_roi_coord']['object'].append(merged_visual_cue)
                sentence['value'] = todo_str

        # 将修改后的数据添加到新的变量中
        modified_data.append(instruction)

    with open(output_file_path, 'w') as outfile:
        json.dump(modified_data, outfile, indent=4)
    print('done!')


def process_final_cc_vqa_multiturn_imageclassification(intermediate_path, final_path):
    # https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html
    def generate_cc_conv_single(old_value, additional_roi_coord, fit_relative_pos_star):
        final_instruction = "<image>\n"
        final_instruction += "Additional information:\n"
        fit_coord = additional_roi_coord["fit"][0]
        # TODO: 这么多<image>，怎么送patch进去？
        final_instruction += "Region of Interest: <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(*fit_coord)
        if len(additional_roi_coord["object"]) > 0:
            union_patch = additional_roi_coord["object"][-1]
            final_instruction += "Union patch of targets: <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(*union_patch)
            for i, bbox in enumerate(additional_roi_coord["object"][:-1]):
                # The `final_instruction` variable is being constructed by appending additional information
                # about sub-patches to the original instruction. It includes details about each sub-patch's
                # location and bounding box coordinates. The final instruction is then updated in the data
                # dictionary before being written to the output file in JSON format.
                # <box>[[243, 469, 558, 746]]</box>
                final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(i + 1, *bbox)
        final_instruction += "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        old_value = old_value.split('<image>\n')[-1]
        final_instruction += old_value

        return final_instruction

    def generate_cc_conv(old_conv, additional_roi_coord, fit_relative_pos_star):
        new_conv = []
        for sentence in old_conv:
            if sentence["from"] == "human":
                tmp_conv = dict()
                tmp_conv["from"] = sentence["from"]
                tmp_conv["value"] = generate_cc_conv_single(sentence["value"], additional_roi_coord, fit_relative_pos_star)
                new_conv.append(tmp_conv)
            else:
                new_conv.append(sentence.copy())
        return new_conv


    with open(intermediate_path, "r") as f:
        base = json.load(f)

    # TODO: Not right?
    # {
    #     "id": 0,
    #     "image": [
    #         "cimages/multimages/16/5pc.png",
    #         "cimages/multimages/16/5pd.png",
    #         "cimages/multimages/16/1602207874_p5b.png",
    #         "cimages/multimages/16/5pe.png",
    #         "cimages/multimages/16/1473016381_p5a.png"
    #     ],
    #     "height_list": [23, 22, 23, 41, 52],
    #     "width_list": [240, 240, 240, 240, 240],
    #     "conversations": [
    #         {
    #             "from": "human",
    #             "value": "Let F = {2, 5, 7, 9}\n\nLet G = {1, 4, 6, 8}\n\nWhich of the following is true?\nA. \n<image>\n\nB. /\n<image>\n\nC. /\n<image>\n\nD. /\n<image>\n\nE. /\n<image>\n\nAnswer with the option's letter from the given choices directly."
    #         },
    #         {
    #             "from": "gpt",
    #             "value": "A"
    #         }
    #     ]
    # }

    modified_data = []
    for i, instruction in enumerate(tqdm(base)):
        new_instruction = dict()
        new_instruction["id"] = instruction["id"]
        new_instruction["image"] = instruction["star_image"]
        new_instruction["height"] = instruction["star_height"]
        new_instruction["width"] = instruction["star_width"]
        new_instruction["train_task_type"] = instruction["train_task_type"]
        new_instruction["additional_roi_coord"] = instruction["additional_roi_coord"]
        new_conv = generate_cc_conv(instruction["conversations"], instruction["additional_roi_coord"], instruction["fit_relative_pos_star"])
        new_instruction["conversations"] = new_conv
        modified_data.append(new_instruction)

    with open(final_path, 'w') as outfile:
        json.dump(modified_data, outfile, indent=4)
    print('done!')


def process_final_imagecaption(intermediate_path, final_path):
    # https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html
    def generate_imagecaption_human(old_value, additional_roi_coord, fit_relative_pos_star):
        final_instruction = "<image>\n"
        final_instruction += "Additional information:\n"
        fit_coord = additional_roi_coord["fit"][0]
        # TODO: 这么多<image>，怎么送patch进去？
        final_instruction += "Region of Interest: <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(*fit_coord)
        final_instruction += "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        old_value = old_value.split('<image>\n')[-1]
        # final_instruction += "Here is a description of a region in the image: "
        final_instruction += old_value
        final_instruction += " Could you provide the bounding box coordinate of the region described above?"
        return final_instruction

    def generate_imagecaption_gpt(additional_roi_coord):
        fit_coord = additional_roi_coord["fit"][0]
        # <rbox>({<23.86><479.36><29.71><541.63>|<1>})</rbox>
        final_instruction = "The location of the described region is: <rbox>({<%.2f><%.2f><%.2f><%.2f>|<%d>})</rbox>" % (fit_coord[0], fit_coord[1], fit_coord[2], fit_coord[3], 0)
        return final_instruction


    def generate_imagecaption_conv(old_conv, additional_roi_coord, fit_relative_pos_star):
        assert len(old_conv) == 2
        new_conv = []
        human_conv = dict()
        human_conv["from"] = "human"
        human_conv["value"] = generate_imagecaption_human(old_conv[1]["value"], additional_roi_coord, fit_relative_pos_star)
        new_conv.append(human_conv)
        gpt_conv = dict()
        gpt_conv["from"] = "gpt"
        gpt_conv["value"] = generate_imagecaption_gpt(additional_roi_coord)
        new_conv.append(gpt_conv)
        return new_conv

    with open(intermediate_path, "r") as f:
        base = json.load(f)

    modified_data = []
    for i, instruction in enumerate(tqdm(base)):
        new_instruction = dict()
        new_instruction["id"] = instruction["id"]
        new_instruction["image"] = instruction["star_image"]
        new_instruction["height"] = instruction["star_height"]
        new_instruction["width"] = instruction["star_width"]
        # complex_compare task actually
        new_instruction["train_task_type"] = instruction["train_task_type"]
        new_instruction["additional_roi_coord"] = instruction["additional_roi_coord"]
        new_conv = generate_imagecaption_conv(instruction["conversations"], instruction["additional_roi_coord"], instruction["fit_relative_pos_star"])
        new_instruction["conversations"] = new_conv
        modified_data.append(new_instruction)

    with open(final_path, 'w') as outfile:
        json.dump(modified_data, outfile, indent=4)
    print('done!')


def process_final_regioncaption(intermediate_path, final_path):
    # https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html
    def generate_regioncaption_human(old_value, additional_roi_coord, fit_relative_pos_star):
        final_instruction = "<image>\n"
        final_instruction += "Additional information:\n"
        fit_coord = additional_roi_coord["fit"][0]
        # TODO: 这么多<image>，怎么送patch进去？
        final_instruction += "Region of Interest: <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(*fit_coord)
        final_instruction += "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        old_value = old_value.split('<image>\n')[-1]
        # final_instruction += "Here is a description of a region in the image: "
        final_instruction += old_value
        final_instruction += " Could you provide the bounding box coordinate of the region described above?"
        return final_instruction

    def generate_regioncaption_gpt(additional_roi_coord):
        ans_coord = additional_roi_coord["object"][0]
        # <rbox>({<23.86><479.36><29.71><541.63>|<1>})</rbox>
        final_instruction = "The location of the described region is: <rbox>({<%.2f><%.2f><%.2f><%.2f>|<%d>})</rbox>" % (ans_coord[0], ans_coord[1], ans_coord[2], ans_coord[3], 0)
        return final_instruction


    def generate_regioncaption_conv(old_conv, additional_roi_coord, fit_relative_pos_star):
        assert len(old_conv) == 2
        new_conv = []
        human_conv = dict()
        human_conv["from"] = "human"
        human_conv["value"] = generate_regioncaption_human(old_conv[1]["value"], additional_roi_coord, fit_relative_pos_star)
        new_conv.append(human_conv)
        gpt_conv = dict()
        gpt_conv["from"] = "gpt"
        gpt_conv["value"] = generate_regioncaption_gpt(additional_roi_coord)
        new_conv.append(gpt_conv)
        return new_conv

    with open(intermediate_path, "r") as f:
        base = json.load(f)

    modified_data = []
    for i, instruction in enumerate(tqdm(base)):
        new_instruction = dict()
        new_instruction["id"] = instruction["id"]
        new_instruction["image"] = instruction["star_image"]
        new_instruction["height"] = instruction["star_height"]
        new_instruction["width"] = instruction["star_width"]
        # complex_compare task actually
        new_instruction["train_task_type"] = instruction["train_task_type"]
        new_instruction["additional_roi_coord"] = instruction["additional_roi_coord"]
        new_conv = generate_regioncaption_conv(instruction["conversations"], instruction["additional_roi_coord"], instruction["fit_relative_pos_star"])
        new_instruction["conversations"] = new_conv
        modified_data.append(new_instruction)

    with open(final_path, 'w') as outfile:
        json.dump(modified_data, outfile, indent=4)
    print('done!')


def main():
    cc_input_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task/train_instruction_complexcompare_708k.json"
    cc_intermediate_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/cc_intermediate_5para_star_obb2_0-1000.json"
    cc_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/cc_final_5para_star_obb2_0-1000.json"

    vqa_input_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task/train_instruction_vqa_400k.json"
    vqa_intermediate_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/vqa_intermediate_5para_star_obb2_0-1000.json"
    vqa_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/vqa_final_5para_star_obb2_0-1000.json"

    imageclassification_input_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task/train_instruction_imageclassification_130k.json"
    imageclassification_intermediate_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/imageclassification_intermediate_5para_star_obb2_0-1000.json"
    imageclassification_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/imageclassification_final_5para_star_obb2_0-1000.json"

    multiturn_json_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task/train_instruction_multiturn_50k.json"
    multiturn_intermediate_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/multiturn_intermediate_5para_star_obb2_0-1000.json"
    multiturn_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/multiturn_final_5para_star_obb2_0-1000.json"

    imagecaption_input_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task/train_instruction_imagecaption_65k.json"
    imagecaption_intermediate_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/imagecaption_intermediate_5para_star_obb2_0-1000.json"
    imagecaption_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/imagecaption_final_5para_star_obb2_0-1000.json"

    regioncaption_input_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/train_data_of_each_individual_task/train_instruction_regioncaption_72k.json"
    regioncaption_intermediate_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/regioncaption_intermediate_5para_star_obb2_0-1000.json"
    regioncaption_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/regioncaption_final_5para_star_obb2_0-1000.json"

    star_stats_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/codebase/dataset_process/star_statistics.pkl"
    fit_img_dir = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS_Img/imgv2_split_512_100_vaild"
    star_stats = pkl.load(open(star_stats_path, "rb"))

    process_intermediate(cc_input_path, star_stats, cc_intermediate_path, task_type="complex_compare")
    process_intermediate(vqa_input_path, star_stats, vqa_intermediate_path, task_type="vqa")
    process_intermediate(imageclassification_input_path, star_stats, imageclassification_intermediate_path, task_type="image_classification")
    process_intermediate(multiturn_json_path, star_stats, multiturn_intermediate_path, task_type="multiturn")
    process_intermediate(imagecaption_input_path, star_stats, imagecaption_intermediate_path, task_type="image_caption")
    process_intermediate(regioncaption_input_path, star_stats, regioncaption_intermediate_path, task_type="region_caption")

    process_final_cc_vqa_multiturn_imageclassification(cc_intermediate_path, cc_final_path)
    process_final_cc_vqa_multiturn_imageclassification(vqa_intermediate_path, vqa_final_path)
    process_final_cc_vqa_multiturn_imageclassification(imageclassification_intermediate_path, imageclassification_final_path)
    process_final_cc_vqa_multiturn_imageclassification(multiturn_intermediate_path, multiturn_final_path)
    process_final_imagecaption(imagecaption_intermediate_path, imagecaption_final_path)
    process_final_regioncaption(regioncaption_intermediate_path, regioncaption_final_path)

    summary_final_path = "/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/train/AGMLLLM_final_5para_star_obb2_0-1000.jsonl"
    jsons2jsonl([cc_final_path, vqa_final_path, imageclassification_final_path, multiturn_final_path, imagecaption_final_path, regioncaption_final_path], summary_final_path)


if __name__ == "__main__":
    main()