import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from maskrcnn_benchmark.data import get_dataset_statistics
# from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
# from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
# from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall
# from mmdet.core.bbox.iou_calculators import bbox_overlaps

from .sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall



relations = ['__background__', 'over', 'not co-storage with', 'connect', 'parallelly parked on', 'intersect', 'co-storage with', 'converge','parallelly docked at', 'adjacent', 'within safe distance of', 'through', 'approach', 'away from', 'randomly parked on', 'run along', 'isolatedly parked on', 'around', 'randomly docked at', 'drive off',
             'drive toward', 'within danger distance of','supply to','isolatedly docked at','pass across','not run along','slightly emit','exhaust to','violently emit',
             'incorrectly parked on', 'pass under', 'directly transmit electricity to','indirectly transmit electricity to', 'pass through','within same line of', 'within different line of','directly connected to','indirectly connected to','driving in the same direction with',
             'driving in the opposite direction with', 'driving alongside with','driving in the same lane with','driving in the different lane with','working on','not working on','parked alongside with','not parked alongside with',
             'in the same parking with','in the different parking with','parking in the same apron with','parking in the different apron with','running along the same taxiway with','running along the different taxiway with',
             'running along the different runway with','docking at the same breakwater with','docking at the same dock with','docking at the different dock with','docked alongside with','not docked alongside with']
relation_id = {i:relation for i, relation in enumerate(relations)}

def do_vg_evaluation(
    # cfg,
    # dataset,
    gt_input,
    predictions,
    iou_thres = [0.5]
    # output_folder,
    # logger,
    # iou_types,
):
    # get zeroshot triplet
    # zeroshot_triplet = torch.load("/media/dell/data1/WTZ/SGG_Frame/maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()

    mode = 'sgdet' # mode = 'sgcls' mode = 'predcls' 
    iou_types = ["relations"]

    # num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    # multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    # iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        # img_info = dataset.get_img_info(image_id)
        # image_width = img_info["width"]
        # image_height = img_info["height"]
        # # recover original size which is before transform
        # predictions[image_id] = prediction.resize((image_width, image_height),val = True)
        
        if mode !=  'sgdet':
            # gt = dataset.get_groundtruth(image_id, evaluation=True)
            gt = gt_input[image_id]
            gt['bbox'] = prediction["pred_bboxes"]
            # gt.bbox = prediction.bbox
        else:
            # predictions[image_id].bbox = prediction.bbox
            # tmp = dataset.RS_test_get_groundtruth(image_id, evaluation=True)
            # gt= prediction.extra_fields["target"]
            gt = gt_input[image_id]
            # if "Small" in cfg.Type:
            #      gt.extra_fields["relation_tuple"] =  tmp.extra_fields["relation_tuple"]
        groundtruths.append(gt)

    result_str = '\n' + '=' * 100 + '\n'
    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall
        
        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy
        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, len(relations), relations, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K  TODO: 这relations含有background
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, len(relations), relations, print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        # prepare all inputs
        global_container = {}
        # global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        # global_container['multiple_preds'] = multiple_preds
        # global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        # global_container['attribute_on'] = attribute_on
        # global_container['num_attributes'] = num_attributes

        for groundtruth, prediction in zip(groundtruths, predictions):
            if len(groundtruth['gt_bboxes'])==0 or len(prediction['pred_bboxes'])==0 \
                or len(groundtruth['gt_triplet'])==0 or len(prediction['pred_triplet'])==0:
                continue
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)
        
        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)
        
        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        
        # if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'

    # logger.info(result_str)
    print(f'result_str:{result_str}')
    
    if "relations" in iou_types:
        # torch.save("/media/dell/data1/WTZ/SGG_Frame/checkpoints/RTPB-Dualtrans/Predcls/RTPB-Dualtrans1227/t", os.path.join(output_folder, 'result_dict.pytorch'))
        # output_folder = '/media/dell/data1/ljw/code/test3/SGG_VLM/sgg_instruction_generation/a-sgg_data/model_output/sgg_eval/'
        output_folder = None
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return float(np.mean(result_dict[mode + '_recall'][1000]))
    # elif "bbox" in iou_types:
    #     return float(mAp)
    else:
        return -1


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths':groundtruths, 'predictions':predictions}, os.path.join(output_folder, "eval_results.pytorch"))

        #with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
                ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
                ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
                })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    #unpack all inputs
    mode = global_container['mode']
    local_container = {}  # triplet: (target_id, target_cat, relation, obj_id_count, obj_cat)
    gt_rels = np.array([(triplet[0], triplet[3], triplet[2]) for triplet in groundtruth['gt_triplet']])
    gt_labels = np.array([(triplet[1], triplet[4]) for triplet in groundtruth['gt_triplet']]).flatten()
    gt_bboxes = np.vstack(groundtruth['gt_bboxes'])
    pred_rels = np.array([(triplet[0], triplet[3], triplet[2]) for triplet in prediction['pred_triplet']])
    pred_labels = np.array([(triplet[1], triplet[4]) for triplet in prediction['pred_triplet']]).flatten()
    pred_bboxes = np.vstack(prediction['pred_bboxes'])
    # local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()
    local_container['gt_rels'] = gt_rels
    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return
    
    local_container['gt_boxes'] = gt_bboxes  #convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = gt_labels 
    local_container['pred_rel_inds'] = pred_rels  # (#pred_rels, 2)
    local_container['rel_scores'] = np.ones_like(pred_rels)         # (#pred_rels, num_pred_class)

    # about relations
    
    local_container['pred_boxes'] =  pred_bboxes # prediction.convert('xyxy').bbox.detach().cpu().numpy()  # RS               # (#pred_objs, 4)
    local_container['pred_classes'] = pred_labels     # (#pred_objs, )
    local_container['obj_scores'] = np.ones_like(pred_labels)          # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)
    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])
    # change for RS
    elif mode == 'sgcls':
        # if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
        #     print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
        local_container['pred_boxes'] = local_container['gt_boxes']
        # local_container['pred_classes'] = local_container['gt_classes']
        # local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    # evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    # evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    # evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # # No Graph Constraint Zero-Shot Recall
    # evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

    return 



def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets) # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
        """
        from list of attribute indexs to [1,0,1,0,...,0,1] form
        """
        max_att = attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        without_attri_idx = 1 - with_attri_idx
        num_pos = int(with_attri_idx.sum())
        num_neg = int(without_attri_idx.sum())
        assert num_pos + num_neg == num_obj

        attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_att):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets