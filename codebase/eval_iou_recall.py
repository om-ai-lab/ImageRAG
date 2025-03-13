import json
import re
import numpy as np


def iou(boxA, boxB):
    """
    计算两个边界框的 IoU，输入格式均为 [x1, y1, x2, y2]。
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def iogt(boxA, boxB):
    """计算 IOGT（Intersection over Ground Truth）"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    gtArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  # boxB 是 GT 框

    return interArea / gtArea if gtArea > 0 else 0.0


def evaluate_results(jsonl_file, iou_threshold=0.1):
    """
    评估预测结果：
      - 对于每个样本，从 jsonl 中读取 ground truth (gt_toi)，预测的视觉 cue (visual_cue)
        以及对应的 confidence (visual_cue_confidence)。
      - 对每个预测，若其 IoU >= iou_threshold，则认为该预测正确，其得分为该预测的 confidence，否则为 0。
      - 对一个样本，将所有预测的得分累加，得到该样本的得分。
      - 最终加权平均得分 = 所有样本得分之和 / 样本数。
      - 同时计算二值准确率：若一个样本中至少有一个预测正确（IoU>=阈值），则视为正确样本。
      - 公式：recall = (sum(正确预测的 confidence) / 样本数)
    """ 

    mr_list = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            per_sample_recall = 0
            data = json.loads(line.strip())
            gt_box = data.get("gt_toi", None)
            pred_boxes = data.get("visual_cue", [])
            conf_list = data.get("visual_cue_confidence", [])
            # print("gt_box:", gt_box, "visual_cue:", pred_boxes, "conf_list:", conf_list)

            img_size = data.get("img_size", None)
            width, height = img_size
            denormalized_boxes = [
                [box[0] / 1000 * width, box[1] / 1000 * height,
                 box[2] / 1000 * width, box[3] / 1000 * height]
                for box in pred_boxes
            ]
            # 只取前 3 个最高置信度的预测框
            # print(denormalized_boxes)
            # print(conf_list)
            sorted_indices = sorted(range(len(conf_list)), key=lambda i: conf_list[i], reverse=True)[:3]
            denormalized_boxes = [denormalized_boxes[i] for i in sorted_indices]
            conf_list = [conf_list[i] for i in sorted_indices]
            # print(denormalized_boxes)
            # print(conf_list)
            # print()
            
            if gt_box is None or len(pred_boxes) == 0 or len(conf_list) == 0:
                continue

            # 遍历每个预测框及其对应 confidence（顺序一一对应）
            for box, conf in zip(denormalized_boxes, conf_list):
                current_iou = iou(box, gt_box)
                if current_iou >= iou_threshold:
                    per_sample_recall += 1
                    
            per_sample_recall /= len(denormalized_boxes)
            mr_list.append(per_sample_recall)
            
    all_recall = np.array(mr_list)
    # print(all_recall)
    mean_reacall = all_recall.mean()
    print("jsonl_file:", jsonl_file)
    print("iou_threshold:", iou_threshold)
    print("总样本数:", len(all_recall))
    print("MR:", mean_reacall)

    return mean_reacall


if __name__ == "__main__":
    # jsonl_file = "/data9/zilun/zaguan/ImageRAG0214/codebase/inference/data/eval/cluster-answer-corrected-mme_8B_imagerag.jsonl"
    # jsonl_file = "/data9/zilun/zaguan/visualize/mmerealworldlite_zoom4kvqa10k_imagerag_cc_clip_0.5_0.1_0.5.jsonl"
    # jsonl_file = "/data9/zilun/zaguan/analyze/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.3_0.1_0.3_rerank.jsonl"
    # evaluate_results(jsonl_file)
    # # 0.1935483870967742

    # jsonl_file = "/data9/zilun/zaguan/analyze/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.3_0.2_0.5_rerank.jsonl"
    # # jsonl_file = "/data9/zilun/zaguan/ImageRAG0214/codebase/inference/data/eval/test.jsonl"
    # evaluate_results(jsonl_file)
    # 0.2

    # jsonl_file = "/data9/zilun/zaguan/analyze/new/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.5_cluster.jsonl"
    # evaluate_results(jsonl_file)
    # using iou
    # recall(thres0.3):0.1323529411764706
    # recall(thres0.2):0.17647058823529413
    # recall(thres0.1):0.3088235294117647

    ##################################
    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.0_0.0_0.0_cluster.jsonl"
    # evaluate_results(jsonl_file, iou_threshold=0.1)
    # evaluate_results(jsonl_file, iou_threshold=0.3)
    # evaluate_results(jsonl_file, iou_threshold=0.5)

    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/old_0306_52/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.5_cluster.jsonl"
    # evaluate_results(jsonl_file, iou_threshold=0.1)
    # evaluate_results(jsonl_file, iou_threshold=0.3)
    # evaluate_results(jsonl_file, iou_threshold=0.5)

    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.0_0.3_0.0_cluster.jsonl"
    # evaluate_results(jsonl_file, iou_threshold=0.1)
    # evaluate_results(jsonl_file, iou_threshold=0.3)
    # evaluate_results(jsonl_file, iou_threshold=0.5)

    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.5_cluster.jsonl"
    # evaluate_results(jsonl_file, iou_threshold=0.1)
    # evaluate_results(jsonl_file, iou_threshold=0.3)
    # evaluate_results(jsonl_file, iou_threshold=0.5)

    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.5_cluster.jsonl"
    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.0_0.0_cluster.jsonl"
    # jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.0_cluster.jsonl"
    jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.0_0.0_cluster.jsonl"
    evaluate_results(jsonl_file, iou_threshold=0.1)
    print()
    evaluate_results(jsonl_file, iou_threshold=0.3)
    print()
    print()
    jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.0_cluster.jsonl"
    evaluate_results(jsonl_file, iou_threshold=0.1)
    print()
    evaluate_results(jsonl_file, iou_threshold=0.3)
    print()
    print()
    jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.5_cluster.jsonl"
    evaluate_results(jsonl_file, iou_threshold=0.1)
    print()
    evaluate_results(jsonl_file, iou_threshold=0.3)
    print()
    print()
    jsonl_file = "/data1/zilun/ImageRAG0226/data/eval/mmerealworldlite_zoom4kvqa10k_imagerag_grid_clip_0.5_0.3_0.5_cluster.jsonl"
    evaluate_results(jsonl_file, iou_threshold=0.1)
    print()
    evaluate_results(jsonl_file, iou_threshold=0.3)
    
