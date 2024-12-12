import json
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import math
import os
from PIL import Image
import re
import BboxToolkit as bt



def transform_predformat_obb(x1, y1, x2, y2, angle_deg):
    # 计算中心点坐标

    a = math.radians(angle_deg)
    x_ctr = (x1 + x2) / 2
    y_ctr = (y1 + y2) / 2

    R = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    w = R * np.cos(a)
    h = R * np.sin(a)
    coords = np.array([[x_ctr, y_ctr, w, h, a]])
    return coords


def evaluation_vg(data_path, images_dir):
    def bbox_and_angle_to_polygon(x1, y1, x2, y2, a):
        # 计算中心点坐标
        x_ctr = (x1 + x2) / 2
        y_ctr = (y1 + y2) / 2

        # 计算宽度和高度
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        # 计算角度（弧度）
        angle_rad = math.radians(a)

        # 计算旋转后的四个角点坐标
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        x1_rot = cos_a * (-w / 2) - sin_a * (-h / 2) + x_ctr
        y1_rot = sin_a * (-w / 2) + cos_a * (-h / 2) + y_ctr

        x2_rot = cos_a * (w / 2) - sin_a * (-h / 2) + x_ctr
        y2_rot = sin_a * (w / 2) + cos_a * (-h / 2) + y_ctr

        x3_rot = cos_a * (w / 2) - sin_a * (h / 2) + x_ctr
        y3_rot = sin_a * (w / 2) + cos_a * (h / 2) + y_ctr

        x4_rot = cos_a * (-w / 2) - sin_a * (h / 2) + x_ctr
        y4_rot = sin_a * (-w / 2) + cos_a * (h / 2) + y_ctr

        # 返回多边形坐标
        polygon_coords = np.array((x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot))

        return polygon_coords

    def extract_bboxes(output):
        """
        Extract bounding box coordinates from the given string using regular expressions.
        :param output: String containing bounding box coordinates in the format {<bx_left><by_top><bx_right><by_bottom>|θ}
        :return: List of bounding boxes, each in the format [bx_left, by_top, bx_right, by_bottom, θ]
        """
        # 修改正则表达式，确保最后一个数字和管道符号能够正确匹配
        pattern = r'{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(\d+)>}'
        # pattern = r'{<(\d+)><(\d+)><(\d+)><(\d+)>}(?:\|<(\d+)>)?'
        matches = re.findall(pattern, output)
        bboxes = []
        for match in matches:
            # 将所有匹配的坐标转换为浮点数，并添加到 bboxes 列表中
            bbox = [int(coord) for coord in match]  # 用int而不是float, 坐标是整数
            bboxes.append(bbox)
        return bboxes


    # read the answer file output by `GeoChat/geochat/eval/batch_geochat_referring.py`, and save as a list `geochat_predict`.
    geochat_predict = [json.loads(q) for q in open(data_path, "r")]
    correct = 0
    total_cnt = len(geochat_predict)
    scale = 1
    for _, predict in tqdm(enumerate(geochat_predict)):
        answer = predict['answer']
        answer = answer.replace("<unk>", "").replace(" ", "").strip()
        image_path = os.path.join(images_dir, predict['image_id'] + '.png')
        image = Image.open(image_path)
        gt_bboxes = predict['ground_truth']  # list
        predict_boxes = extract_bboxes(answer)  # list
        for i in range(len(gt_bboxes)):
            # convert coordinates to float
            poly = np.array(gt_bboxes[i]).astype(np.float32).reshape(-1)  # [4,2]
            gt_obb = bt.poly2obb(poly).reshape(1, 5)  # convert to [cx, cy, w, h, theta]
            for j, pred_bbox in enumerate(predict_boxes):
                pred_obb = transform_predformat_obb(*pred_bbox)
                iou_score = bt.geometry.bbox_overlaps(pred_obb, gt_obb)[0][0]  # calcualte obb Iou by BboxToolkit.
                if iou_score > 0:
                    print(iou_score)
                if iou_score >= 0:
                    correct += 1
                # except:
                #     continue

    dataset = 'GeoChat Bench referring'
    print(f"Evaluating {dataset} ...")
    print(f'Precision @ 0.5: {correct / total_cnt} \n')


def evaluation_regioncaptioning(data_path):
    def blue_score(infer, gt):
        gt = gt.lower().strip().split(" ")
        infer = infer.lower().strip().split(" ")
        scores = sentence_bleu(
            gt, infer,
        )
        return scores

    def get_rougeL_score(infer, gt):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(gt, infer)
        scores = scores["rougeL"].fmeasure
        return scores

    def get_rouge1_score(infer, gt):
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(gt, infer)
        scores = scores["rouge1"].fmeasure
        return scores

    def get_meteor_score(infer, gt):
        gt = gt.lower().strip().split(" ")
        infer = infer.lower().strip().split(" ")

        meteor_scores = single_meteor_score(
            gt, infer
        )
        return meteor_scores

    base = [json.loads(q) for q in open(data_path, "r")]
    rougel_list = []
    rouge1_list = []
    meteor_list = []

    for answers in tqdm(base):
        gt = answers['ground_truth'].lower()
        answer = answers['answer'].lower().replace('.', '').replace("<p>", "").replace("</p>", "")
        rougel = get_rougeL_score(answer, gt)
        rouge1 = get_rouge1_score(answer, gt)
        meteor = get_meteor_score(answer, gt)
        rougel_list.append(rougel)
        rouge1_list.append(rouge1)
        meteor_list.append(meteor)

    print("Rouge1: ", np.array(rouge1_list).mean())
    print("RougeL: ", np.array(rougel_list).mean())
    print("METEOR: ", np.array(meteor_list).mean())


def evaluation_scene(data_path):
    base = [json.loads(q) for q in open(data_path, "r")]
    correct = 0
    incorrect = 0
    for answers in tqdm(base):
        gt = answers['question_id'].split('/')[0].lower()
        answer = answers['answer'].replace(' ', '').lower().replace('.', '')
        if gt == answer:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        # else:
        #     continue
    print('correct:', correct)
    print('incorrect:', incorrect)
    print('Total:', correct + incorrect)
    print('Acc:', (correct / (correct + incorrect)))


def evaluation_vqa(data_path):
    base = []
    with open(data_path, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        base.append(json.loads(line))

    correct = 0
    incorrect = 0
    comp_correct = 0
    comp_incorrect = 0
    pre_correct = 0
    pre_incorrect = 0
    ru_correct = 0
    ru_incorrect = 0
    for answers in tqdm(base):
        gt = answers["gt"].lower()
        type_ = answers["type"]
        answer = answers["answer"].replace(" ", "").lower().replace(".", "")
        if gt == answer:
            correct = correct + 1
            if type_ == "comp":
                comp_correct = comp_correct + 1
            if type_ == "presence":
                pre_correct = pre_correct + 1
            if type_ == "rural_urban":
                ru_correct = ru_correct + 1
        else:
            incorrect = incorrect + 1
            if type_ == "comp":
                comp_incorrect = comp_incorrect + 1
            if type_ == "presence":
                pre_incorrect = pre_incorrect + 1
            if type_ == "rural_urban":
                ru_incorrect = ru_incorrect + 1

    print("presence_correct:", pre_correct)
    print("presence_incorrect:", pre_incorrect)
    print("presence_Total:", pre_correct + pre_incorrect)
    print("presence_Acc:", (pre_correct / (pre_correct + pre_incorrect)))
    print("-" * 100)
    print("comparison_correct:", comp_correct)
    print("comparison_incorrect:", comp_incorrect)
    print("comparison_Total:", comp_correct + comp_incorrect)
    print("comparison_Acc:", (comp_correct / (comp_correct + comp_incorrect)))
    print("-" * 100)
    if ru_correct + ru_incorrect != 0:
        print("rural_urban_correct:", ru_correct)
        print("rural_urban_incorrect:", ru_incorrect)
        print("rural_urban_Total:", ru_correct + ru_incorrect)
        print("rural_urban_Acc:", (ru_correct / (ru_correct + ru_incorrect)))
        print("-" * 100)
    print("total_correct:", correct)
    print("total_incorrect:", incorrect)
    print("total_Total:", correct + incorrect)
    print("total_Acc:", correct / (correct + incorrect))
    print("Geochat paper total acc:", (comp_correct + pre_correct + ru_correct) / (comp_correct + comp_incorrect + pre_correct + pre_incorrect + ru_correct + ru_incorrect))


def main():
    # # Table 8
    # table6_out_jsonl_path = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/output/lrben.jsonl"
    # evaluation_vqa(table6_out_jsonl_path)
    # print()
    # table5aid_out_jsonl_path = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/output/aid.jsonl"
    # evaluation_scene(table5aid_out_jsonl_path)
    # print()
    # table5uc_out_jsonl_path = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/output/UCmerced.jsonl"
    # evaluation_scene(table5uc_out_jsonl_path)
    # print()
    # table10_out_jsonl_path = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/output/region_captioning.jsonl"
    # evaluation_regioncaptioning(table10_out_jsonl_path)
    # print()
    table7_out_jsonl_path = "/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/output/referring_grounding.jsonl"
    evaluation_vg(table7_out_jsonl_path, images_dir="/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/dataset/GeoChat_Instruct/images/share/softwares/kartik/GeoChat_finetuning/final_images_llava")


if __name__ == "__main__":
    main()