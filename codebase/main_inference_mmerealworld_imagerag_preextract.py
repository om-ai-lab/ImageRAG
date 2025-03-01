import pdb
import shutil

import pandas as pd
from tqdm import tqdm
import os
import uuid
import numpy as np
import pickle as pkl
from PIL import Image
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import argparse
import logging
import openai
import pytorch_lightning as pl
import json
import math
import regex as re
import shortuuid
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import yaml
import gc


from codebase.llm_template import paraphrase_template, keyword_template, text_expansion_template
from codebase.utils import (setup_vlm_model, set_up_paraphrase_model, setup_vqallm, setup_slow_text_encoder_model,
                            calculate_similarity_matrix, extract_vlm_img_text_feat, ranking_patch_t2p,
                            paraphrase_model_inference, text_expand_model_inference, setup_logger, meta_df2clsimg_dict,
                            img_reduce, select_visual_cue, ranking_patch_visualcue2patch, load_yaml, get_chunk,
                            convert_obb_to_region_str, obb2minhbb, sole_visualcue2mergedvisualcue, visualcue2imagepatch,
                            reduce_visual_cue_per_cls, setup_lrsd_vsd, setup_pub11_vsd, bbox_location, filter_visual_cue_basedon_T)
from codebase.sglang_util import get_paraphase_response, get_keyword_response, get_text_expansion_response
from codebase.utils import load_yaml
from codebase.patchify import cc_patchify, vit_patchify
from codebase.text_parser import extract_key_phrases

# export PYTHONPATH=$PYTHONPATH:/data1/zilun/grsm/ImageRAG_git
# export PYTHONPATH=$PYTHONPATH:/media/zilun/fanxiang4t/GRSM/ImageRAG_git
# export PYTHONPATH=$PYTHONPATH:/mnt/cfs/zilun/ImageRAG

Image.MAX_IMAGE_PIXELS = None


def collate_fn(batch):
    image_tensor, prompt, num_patches_list, question_list, line = zip(*batch)
    image_tensors = torch.cat(image_tensor, dim=0)
    return image_tensors, prompt, num_patches_list, question_list, line


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, use_dynamic, inference_mode, batch_size=1, num_workers=0, prompt=''):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = InternVLMMERSDataset(questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, prompt, inference_mode, use_dynamic)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             collate_fn=collate_fn)
    return data_loader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class InternVLMMERSDataset(Dataset):
    # InternVLMMERSDataset(questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, prompt, use_dynamic)
    def __init__(self, questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, test_prompt, mode="baseline", use_dynamic=True):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.dynamic_preprocess = dynamic_preprocess
        self.model_config = model_config
        self.test_prompt = test_prompt
        self.use_dynamic = use_dynamic
        self.mode = mode

    def __getitem__(self, index):
        line = self.questions[index]
        choices = line['Answer choices']
        image_file = line["Image"]
        qs = line["Text"]

        if self.mode == "baseline":
            choice_prompt = ' The choices are listed below: \n'
            for choice in choices:
                choice_prompt += choice + "\n"
            final_instruction = qs + choice_prompt + self.test_prompt + '\nThe best answer is:'
            prompt = final_instruction

        image_path = os.path.join(self.image_folder, image_file)

        if type(image_path) == str:
            image = Image.open(image_path).convert('RGB')
        # image file
        else:
            image = image_path
        if self.use_dynamic:
            images = self.dynamic_preprocess(
                image,
                image_size=self.model_config.vision_config.image_size,
                use_thumbnail=self.model_config.use_thumbnail,
                max_num=self.model_config.max_dynamic_patch
            )
            print("Use Dynamic")
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

        return pixel_values, prompt, pixel_values.size(0), line["Text"], line

    def __len__(self):
        return len(self.questions)


def image_rag(config, contrastive_vlm_pack, line, client, logger,
              lrsd_vectorstore, lrsd_vsd_label2imgname_dict, lrsd_vsd_imgname2feat_dict,
              pub11_vectorstore, pub11_label2imgname_dict, pub11_imgname2feat_dict,
              text_paraphrase, text_expand
              ):
    patch_saving_dir = os.path.join(config['work_dir'], config['patch_saving_dir'])
    fast_path_T = config['fast_path_T']
    paraphrase_model_config = config['paraphrase_model']
    kw_model_config = config['kw_model']
    text_expansion_model_config = config['text_expansion_model']

    choices = line['Answer choices']
    question = line["Text"]
    image_file = line["Image"]
    choice_prompt = ' The choices are listed below: \n'
    for choice in choices:
        choice_prompt += choice + "\n"
    question_with_test_template = question + choice_prompt + config["test_prompt"] + '\nThe best answer is:'

    image_path = os.path.join(config['input_image_dir'], image_file)
    input_uhr_image = Image.open(image_path)

    width, height = input_uhr_image.size
    logger.info("original image width and height: {}, {}".format(width, height))

    if text_paraphrase:
        paraphrase_result = get_paraphase_response(
            client,
            config['paraphrase_model']['model_path'],
            question,
            config['paraphrase_model']['generation_config']
        )
        logger.info("Paraphrased Text: {}".format(paraphrase_result))
    else:
        paraphrase_result = question

    if not kw_model_config:
        kw_model_config = paraphrase_model_config
        trail_budget = 20
        current_trail = 0
        while current_trail < trail_budget:
            print(paraphrase_result)
            query_keywords = get_keyword_response(client, kw_model_config['model_path'], paraphrase_result,
                                                  kw_model_config['generation_config'])
            print(query_keywords)
            try:
                query_keywords = re.findall(r'\[[^\]]*\]', query_keywords)[-1]
                query_keywords = eval(query_keywords)
                if len(query_keywords) > 0:
                    print("Last occurrence of []:", query_keywords)
                    break
                else:
                    print("No match found. Re-parse. Trail {}".format(current_trail))
                    current_trail += 1
                    continue
            except Exception as e:
                print("Bad query keywords: {}".format(query_keywords))
                continue

        if current_trail == trail_budget:
            kw_model_path = config["text_embed_model"]["model_path"]
            kw_model = KeyBERT(model=SentenceTransformer(kw_model_path))
            query_keywords = extract_key_phrases(paraphrase_result, paraphrase_result, kw_model)

    else:
        kw_model_path = kw_model_config['model_path']
        kw_model = KeyBERT(model=SentenceTransformer(kw_model_path))
        query_keywords = extract_key_phrases(paraphrase_result, paraphrase_result, kw_model)

    logger.info("Final Key Phrases: {}".format(query_keywords))

    patch_saving_dir = os.path.join(patch_saving_dir, config["patch_method"])
    if  config["patch_method"] == "vit":
        img_resize, original_image, coordinate_patchname_dict, image_save_dir = vit_patchify(image_path, patch_saving_dir, patch_size=config['model_input_image_size'])
    elif config["patch_method"] == "cc":
        img_resize, original_image, coordinate_patchname_dict, image_save_dir = cc_patchify(image_path, patch_saving_dir, c_denom=20)

    logger.info(
        "resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))

    fast_path_vlm, img_preprocess, text_tokenizer = contrastive_vlm_pack
    # fast_path_vlm_name = os.path.splitext(os.path.basename(config["fast_vlm_model"]["model_path"]))[0]
    fast_path_vlm_name = config["fast_vlm_model"]["model_name"]
    # Fast Path
    vlm_image_feats, vlm_text_feats, bbox_coordinate_list = extract_vlm_img_text_feat(
        question,
        query_keywords,
        coordinate_patchname_dict,
        patch_saving_dir,
        img_preprocess,
        text_tokenizer,
        fast_path_vlm,
        img_batch_size=50,
        feat_saving_dir=image_save_dir,
        fastvlm_encoder_name=fast_path_vlm_name
    )

    t2p_similarity = calculate_similarity_matrix(vlm_image_feats, vlm_text_feats, fast_path_vlm.logit_scale.exp())
    # visual_cue (topn, 4) -> [[x1, y1, x2, y2], ...]
    # pdb.set_trace()
    visual_cue, visual_cue_similarity = ranking_patch_t2p(bbox_coordinate_list, t2p_similarity, top_k=2)
    logger.info("Ranked Patch Shape: {}".format(visual_cue.shape))
    logger.info("visual_cue similarity: {}".format(visual_cue_similarity))

    visual_cue, visual_cue_similarity = filter_visual_cue_basedon_T(visual_cue, visual_cue_similarity, fast_path_T)
    
    # pdb.set_trace()
    # Slow Path
    if len(visual_cue) == 0:
        logger.info("fast path similarity does not meet the threshold {}, choose the slow path".format(fast_path_T))
        logger.info("<path>Slow</path>")
        if text_expand:
            if not text_expansion_model_config:
                text_expansion_model_config = paraphrase_model_config
                expanded_query_text_dict = get_text_expansion_response(client,
                                                                       text_expansion_model_config['model_path'],
                                                                       query_keywords,
                                                                       text_expansion_model_config['generation_config'])
            else:
                logger.info("Not Implemented")
                exit()
        else:
            expanded_query_text_dict = dict()
            for query_keyword in query_keywords:
                expanded_query_text_dict[query_keyword] = query_keyword

        # Uncomment if not use phrase expansion
        # expanded_query_text_list = query_keywords

        # embedding = GeoRSCLIPEmbeddings(checkpoint=config['fast_vlm_model']['model_path'], device=device)
        # vectorstore = Chroma(
        #     collection_name="mm_georsclip",
        #     embedding_function=embedding,
        #     persist_directory=config["vector_database"]["mm_vector_database_dir"],
        #     collection_metadata = {"hnsw:space": "cosine"}
        # )
        selected_label_names = []
        for expanded_query_text in expanded_query_text_dict:
            results = lrsd_vectorstore.similarity_search_with_score(
                expanded_query_text, k=3
            )
            for res, score in results:
                # * [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}] default l2
                logger.info(f"Query:={expanded_query_text_dict[expanded_query_text]} * [SIM={score:3f}] {res.page_content}")
                if score <= 0.5:
                    selected_label_names.append(res.page_content)

        if len(selected_label_names) > 0:
            selected_label_names = list(set(selected_label_names))
            logger.info("Selected labels from Text VSD: {}".format(selected_label_names))
            # label -> feats dict
            visual_cue_candidates_dict = dict()

            for label in selected_label_names:
                img_feat_selected_per_cls = []
                img_names = lrsd_vsd_label2imgname_dict[label]
                for img_name in img_names:
                    # feat = lrsd_vsd_imgname2feat_dict[img_name].unsqueeze(0)
                    feat = torch.from_numpy(lrsd_vsd_imgname2feat_dict[img_name])
                    img_feat_selected_per_cls.append(feat)
                img_feat_selected_per_cls = torch.cat(img_feat_selected_per_cls)
                visual_cue_candidates_dict[label] = img_feat_selected_per_cls
            reduced_visual_cue_per_cls = reduce_visual_cue_per_cls(visual_cue_candidates_dict, reduce_fn="mean", need_feat_normalize=True)
            visual_cue, visual_cue_similarity = select_visual_cue(vlm_image_feats, bbox_coordinate_list, reduced_visual_cue_per_cls, need_feat_normalize=True)
            return image_path, visual_cue, visual_cue_similarity, question_with_test_template, query_keywords
        else:
            logger.info("No label text pass the threshold. Cannot find label that match the keywords from query. Try Pub11 VSD.")
            # pub11_vectorstore, pub11_label2imgname_dict, pub11_imgname2feat_dict,
            selected_captions = []
            for expanded_query_text in expanded_query_text_dict:
                # expanded_query_text_sentence = "A photo of {}".format(expanded_query_text)
                expanded_query_text_sentence = expanded_query_text
                results = pub11_vectorstore.similarity_search_with_score(
                    expanded_query_text_sentence, k=3
                )
                for res, score in results:
                    # * [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}] default l2
                    logger.info(
                        f"Query:={expanded_query_text_dict[expanded_query_text]} * [SIM={score:3f}] {res.page_content}")
                    if score <= 0.5:
                        selected_captions.append(res.page_content)

            if len(selected_captions) > 0:
                visual_cue_candidates_dict = dict()
                for caption in selected_captions:
                    img_feat_selected_per_caption = []
                    img_names = pub11_label2imgname_dict[caption]
                    for img_name in img_names:
                        feat = torch.from_numpy(pub11_imgname2feat_dict[img_name])
                        img_feat_selected_per_caption.append(feat)
                    img_feat_selected_per_caption = torch.cat(img_feat_selected_per_caption)
                    visual_cue_candidates_dict[caption] = img_feat_selected_per_caption

                reduced_visual_cue_per_cls = reduce_visual_cue_per_cls(visual_cue_candidates_dict, reduce_fn="mean", need_feat_normalize=True)
                visual_cue, visual_cue_similarity = select_visual_cue(vlm_image_feats, bbox_coordinate_list, reduced_visual_cue_per_cls, need_feat_normalize=True)
                print(visual_cue, visual_cue_similarity)
                return image_path, visual_cue, visual_cue_similarity, question_with_test_template, query_keywords
            else:
                # w, h = original_image.size
                # visual_cue = [[0, 0, w, h]]
                # visual_cue_similarity = [1.0]
                # visual_cue = []
                # visual_cue_similarity = []
                logger.info("No caption text pass the threshold. Cannot find caption that match the keywords from query.")
                return image_path, visual_cue, visual_cue_similarity, question_with_test_template, query_keywords
    else:
        logger.info("<path>Fast</path>")
        print(visual_cue, visual_cue_similarity)
        return image_path, visual_cue, visual_cue_similarity, question_with_test_template, query_keywords
    


def inference_internvl(config, questions, ans_file_path, generative_vlm_pack, client, logger):
    InternVL_coord_pattern = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    # TODO: only works for InternVL model, Batch_Size=1
    ans_file = open(ans_file_path, "w")
    generative_vlm, generative_vlm_tokenizer, generative_vlm_generation_config, generative_vlm_dynamic_preprocess, generative_vlm_transform = generative_vlm_pack
    index = 0
    if config["mode"] == "baseline":
        data_loader = create_data_loader(
            questions, config["input_image_dir"],
            generative_vlm_tokenizer, generative_vlm_dynamic_preprocess,
            generative_vlm_transform, generative_vlm.config, config["use_dynamic"], config["mode"],
            config["batch_size"], prompt=config["test_prompt"]
        )
        for (image_tensors, prompts, num_patches_list, question_text_only_list, line) in tqdm(data_loader):
            with torch.inference_mode():
                responses = generative_vlm.batch_chat(
                    generative_vlm_tokenizer,
                    image_tensors,
                    num_patches_list=num_patches_list,
                    questions=prompts,
                    generation_config=generative_vlm_generation_config
                )
            for i, response in enumerate(responses):
                if index % 1 == 0:
                    print(f'Prompt: {prompts[i]}\n\n Output: {response}')
                line[i]['output'] = response
                ans_file.write(json.dumps(line[i]) + "\n")
                ans_file.flush()
                index += 1
        ans_file.close()

    elif config["mode"] == "baseline_detection":
        assert config["batch_size"] == 1
        for line in tqdm(questions):
            choices = line['Answer choices']
            question = line["Text"]
            image_file = line["Image"]
            choice_prompt = ' The choices are listed below: \n'
            for choice in choices:
                choice_prompt += choice + "\n"
            image_path = os.path.join(config['input_image_dir'], image_file)
            question_with_test_template = question + choice_prompt + config["test_prompt"] + '\nThe best answer is:'

            # if line["Category"] != "count":
            target_of_interest_prompt = 'What is the target of interest in this question? Answer in short phrases only.'
            target_of_interest_final_instruction = "Question: {}\n".format(question) + target_of_interest_prompt
            llm_params = dict(
                temperature=config['paraphrase_model']['generation_config']['temperature'],
                max_tokens=config['paraphrase_model']['generation_config']['max_tokens'],
                top_p=config['paraphrase_model']['generation_config']['top_p'],
                timeout=config['paraphrase_model']['generation_config']['timeout'],
                # do_sample=True
            )
            msg = [
                {
                    "role": "user",
                    "content": target_of_interest_final_instruction
                }
            ]
            llm_response = client.chat.completions.create(
                model="llm",
                messages=msg,
                **llm_params
            )
            toi_str = llm_response.choices[0].message.content.strip()
            line["target_of_interest"] = toi_str
            detection_final_instruction = '<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>.'.format(
                toi_str)
            # detection_final_instruction = '<image>\n[detection]Please locate the {} in the given image.'.format(toi_str)
            images, tile_num_list = [], []

            image = Image.open(image_path).convert('RGB')
            w, h = image.size
            if config["use_dynamic"]:
                print("use dynamic res")
                image_anyres = generative_vlm_dynamic_preprocess(
                    image,
                    max_num=generative_vlm.config.max_dynamic_patch,
                    image_size=generative_vlm.config.vision_config.image_size,
                    use_thumbnail=generative_vlm.config.use_thumbnail,
                )
            else:
                image_anyres = [image]
            images += image_anyres
            pixel_values = [generative_vlm_transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
            tile_num_list.append(len(image_anyres))

            # if line["Category"] != "count":
            with torch.inference_mode():
                # TODO: visual cues contain full image, duplicate
                bbox_response = generative_vlm.chat(
                    generative_vlm_tokenizer,
                    pixel_values,
                    detection_final_instruction,
                    generative_vlm_generation_config
                )

            predict_bbox = re.findall(InternVL_coord_pattern, bbox_response)
            try:
                predict_bbox = [float(predict_bbox[0][0]), float(predict_bbox[0][1]), float(predict_bbox[0][2]),
                                float(predict_bbox[0][3])]
            except:
                predict_bbox = None

            line["visual_cue"] = predict_bbox

            if predict_bbox:
                gc.collect()
                torch.cuda.empty_cache()
                visual_cues = [predict_bbox]
                for object_coord in visual_cues:
                    x1, y1, x2, y2 = object_coord
                    x1 = int(x1 / 1000 * w)
                    y1 = int(y1 / 1000 * h)
                    x2 = int(x2 / 1000 * w)
                    y2 = int(y2 / 1000 * h)
                    image_crop = image.crop((x1, y1, x2, y2))
                    images.append(image_crop)
                    tile_num_list.append(1)
                pixel_values = [generative_vlm_transform(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

                final_instruction = "<image>\n"
                final_instruction += "Additional information:\n"
                for i, bbox in enumerate(visual_cues):
                    final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(
                        i + 1, *bbox)
                final_instruction += "Look at the image and answer the question based on the provided additional information (location of sub-patches). \n"
                final_instruction += "Question: "
                final_instruction += question_with_test_template
                num_patches_list = [pixel_values.size(0) - len(visual_cues), len(visual_cues)]
                response = generative_vlm.chat(
                    generative_vlm_tokenizer,
                    pixel_values,
                    final_instruction,
                    generative_vlm_generation_config,
                    num_patches_list=num_patches_list
                )
            else:
                final_instruction = question_with_test_template
                response = generative_vlm.chat(
                    generative_vlm_tokenizer,
                    pixel_values,
                    final_instruction,
                    generative_vlm_generation_config,
                )

            with torch.inference_mode():
                # TODO: visual cues contain full image, duplicate

                print(
                    f'Prompt: {question_with_test_template}\n\n GT: {line["Ground truth"]} \n Output: {response}')
                line['output'] = response
                ans_file.write(json.dumps(line) + "\n")
                ans_file.flush()
                index += 1
        ans_file.close()

    elif config["mode"] == "detection_gt":
        assert config["batch_size"] == 1
        for line in tqdm(questions):
            choices = line['Answer choices']
            question = line["Text"]
            image_file = line["Image"]
            choice_prompt = ' The choices are listed below: \n'
            for choice in choices:
                choice_prompt += choice + "\n"
            image_path = os.path.join(config['input_image_dir'], image_file)
            question_with_test_template = question + choice_prompt + config["test_prompt"] + '\nThe best answer is:'

            images, tile_num_list = [], []

            image = Image.open(image_path).convert('RGB')
            w, h = image.size

            if config["use_dynamic"]:
                print("use dynamic res")
                image_anyres = generative_vlm_dynamic_preprocess(
                    image,
                    max_num=generative_vlm.config.max_dynamic_patch,
                    image_size=generative_vlm.config.vision_config.image_size,
                    use_thumbnail=generative_vlm.config.use_thumbnail,
                )
            else:
                image_anyres = [image]

            images += image_anyres
            pixel_values = [generative_vlm_transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
            tile_num_list.append(len(image_anyres))

            line["visual_cue"] = line["gt_toi"]
            predict_bbox = line["visual_cue"]
            predict_bbox = [int(predict_bbox[0]) / w * 1000, int(predict_bbox[1]) / h * 1000, int(predict_bbox[2]) / w * 1000, int(predict_bbox[3]) / h * 1000]

            if predict_bbox:
                gc.collect()
                torch.cuda.empty_cache()
                visual_cues = [predict_bbox]

                for i, object_coord in enumerate(visual_cues):
                    x1, y1, x2, y2 = object_coord
                    x1 = int(x1 / 1000 * w)
                    y1 = int(y1 / 1000 * h)
                    x2 = int(x2 / 1000 * w)
                    y2 = int(y2 / 1000 * h)
                    image_crop = image.crop((x1, y1, x2, y2))
                    images.append(image_crop)
                    tile_num_list.append(1)
                    if i == 0:
                        relative_pos = bbox_location(w, h, (x1, y1, x2, y2))
                pixel_values = [generative_vlm_transform(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

                final_instruction = "<image>\n"
                final_instruction += "Additional information:\n"
                for i, bbox in enumerate(visual_cues):
                    final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(i + 1, *bbox)
                # final_instruction += "Look at {} of the image and answer the question based on the provided additional information (location of sub-patches). \n".format(relative_pos)
                final_instruction += "Look at the image and answer the question based on the provided additional information (location of sub-patches). \n"
                final_instruction += "Question: "
                final_instruction += question_with_test_template
                num_patches_list = [pixel_values.size(0) - len(visual_cues), len(visual_cues)]

                response = generative_vlm.chat(
                    generative_vlm_tokenizer,
                    pixel_values,
                    final_instruction,
                    generative_vlm_generation_config,
                    num_patches_list=num_patches_list
                )

            else:
                final_instruction = question_with_test_template
                response = generative_vlm.chat(
                    generative_vlm_tokenizer,
                    pixel_values,
                    final_instruction,
                    generative_vlm_generation_config,
                )

            with torch.inference_mode():
                # TODO: visual cues contain full image, duplicate

                print(
                    f'Prompt: {question_with_test_template}\n\n GT: {line["Ground truth"]} \n Output: {response}')
                line['output'] = response
                line['final_instruction'] = final_instruction
                ans_file.write(json.dumps(line) + "\n")
                ans_file.flush()
                index += 1
        ans_file.close()

    elif config["mode"] == "imagerag":
        # imagerag does not support parallel inference
        assert config["batch_size"] == 1
        fast_vlm_model_name = config['fast_vlm_model']['model_name']
        fast_vlm_model_path = config['fast_vlm_model']['model_path']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fast_path_vlm, img_preprocess, text_tokenizer
        contrastive_vlm_pack = setup_vlm_model(fast_vlm_model_path, fast_vlm_model_name, device)
        lrsd_vectorstore, lrsd_vsd_label2imgname_dict, lrsd_vsd_imgname2feat_dict = setup_lrsd_vsd(config)
        pub11_vectorstore, pub11_vsd_label2imgname_dict, pub11_vsd_imgname2feat_dict = setup_pub11_vsd(config)

        for line in tqdm(questions):
            gc.collect()
            torch.cuda.empty_cache()
            image_path, visual_cues, visual_cues_similarity, question_with_test_template, query_keywords = image_rag(
                config, contrastive_vlm_pack, line, client, logger,
                lrsd_vectorstore, lrsd_vsd_label2imgname_dict, lrsd_vsd_imgname2feat_dict,
                pub11_vectorstore, pub11_vsd_label2imgname_dict, pub11_vsd_imgname2feat_dict,
                text_paraphrase=False, text_expand=False
            )

            
            if len(visual_cues) > 0:
                images, tile_num_list = [], []
                global_and_locals = []
                num_patches_list = []
                image = Image.open(image_path).convert('RGB')
                w, h = image.size
                global_and_locals.append(image)
                for object_coord in visual_cues:
                    image_crop = image.crop(object_coord)
                    global_and_locals.append(image_crop)

                for i, image in enumerate(global_and_locals):
                    if i == 0:
                        image = generative_vlm_dynamic_preprocess(
                            image,
                            max_num=generative_vlm.config.max_dynamic_patch,
                            # max_num=6,
                            image_size=generative_vlm.config.vision_config.image_size,
                            use_thumbnail=False,
                        )
                        images += image
                        tile_num_list.append(len(image))
                    else:
                        images.append(image)
                        tile_num_list.append(1)

                pixel_values = [generative_vlm_transform(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
                num_patches = pixel_values.size(0) - len(visual_cues)

                normalized_visual_cues = []
                final_instruction = "<image>\n"
                final_instruction += "Additional information:\n"
                for i, bbox in enumerate(visual_cues):
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1 / w * 1000)
                    y1 = int(y1 / h * 1000)
                    x2 = int(x2 / w * 1000)
                    y2 = int(y2 / h * 1000)
                    normalized_bbox = (x1, y1, x2, y2)
                    normalized_visual_cues.append(normalized_bbox)
                    final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(
                        i + 1, *normalized_bbox)
                final_instruction += "Look at the image and answer the question based on the provided additional information (location of sub-patches). \n"
                final_instruction += "Question: "
                final_instruction += question_with_test_template

                line["visual_cue"] = normalized_visual_cues
                line["visual_cue_confidence"] = visual_cues_similarity
                line["target_of_interest"] = query_keywords

                print(pixel_values.shape)
                print(tile_num_list)
                with torch.inference_mode():
                    # TODO: visual cues contain full image, duplicate
                    response = generative_vlm.chat(
                        generative_vlm_tokenizer,
                        pixel_values,
                        final_instruction,
                        generative_vlm_generation_config,
                        num_patches_list=tile_num_list
                    )
            else:
                images, tile_num_list = [], []
                image = Image.open(image_path).convert('RGB')
                image = generative_vlm_dynamic_preprocess(
                            image,
                            # max_num=generative_vlm.config.max_dynamic_patch,
                            max_num=6,
                            image_size=generative_vlm.config.vision_config.image_size,
                            use_thumbnail=False,
                        )
                images += image
                tile_num_list.append(len(image))
                pixel_values = [generative_vlm_transform(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
                final_instruction = question_with_test_template
                with torch.inference_mode():
                    # TODO: visual cues contain full image, duplicate
                    response = generative_vlm.chat(
                        generative_vlm_tokenizer,
                        pixel_values,
                        final_instruction,
                        generative_vlm_generation_config,
                        num_patches_list=tile_num_list
                    )
                    
                line["visual_cue"] = []
                line["visual_cue_confidence"] = []
                line["target_of_interest"] = query_keywords

            logger.info(f'Prompt: {question_with_test_template}\n\n GT: {line["Ground truth"]} \n Output: {response}')
            line['output'] = response
            ans_file.write(json.dumps(line) + "\n")
            ans_file.flush()
            index += 1
        ans_file.close()

    return ans_file_path


def inference():
    # os.environ['ALL_PROXY'] = ''
    # os.environ['all_proxy'] = ''

    pl.seed_everything(2024)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str,
                        default='/media/zilun/fanxiang4t/GRSM/ImageRAG0214/config/config_internvl8b_5hbb_448_dynamic_0-1000_mmerealworld-imagerag-zoom4kvqa10k-2epoch_local-nonlite.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to the log file.')
    parser.add_argument('--base_url', type=str, 
                        # default='http://127.0.0.1:30000/v1', 
                        default='http://192.168.0.251:30000/v1',
                        help='base url')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, "log.txt"))

    config = load_yaml(args.cfg_path)
    patch_saving_dir = config['patch_saving_dir']
    os.makedirs(patch_saving_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llmvqa_model_name = config['llmvqa_model']['model_name']
    llmvqa_model_path = config['llmvqa_model']['model_path']
    generative_vlm_generation_config = dict(
        max_new_tokens=config["llmvqa_model"]["generation_config"]["max_tokens"],
        do_sample=True if config["llmvqa_model"]["generation_config"]["temperature"] > 0 else False,
        temperature=config["llmvqa_model"]["generation_config"]["temperature"],
        top_p=config["llmvqa_model"]["generation_config"]["top_p"],
        num_beams=config["llmvqa_model"]["generation_config"]["num_beams"],
    )
    generative_vlm_pack = setup_vqallm(llmvqa_model_path, llmvqa_model_name, generative_vlm_generation_config,
                                       config["model_input_image_size"], device=device,
                                       load_in_8bit=config["llmvqa_model"]["load_in_8bit"])

    client = openai.Client(base_url=args.base_url, api_key="None")

    with open(config['question_file_path'], 'r') as file:
        questions = json.load(file)
    questions = [question for question in questions if question["Subtask"] == "Remote Sensing"]
    questions = get_chunk(questions, config['num_chunks'], config['chunk_idx'])
    answers_file = os.path.expanduser(config['answers_file_path'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if "InternVL" in llmvqa_model_path:
        answers_file = inference_internvl(config, questions, answers_file, generative_vlm_pack, client, logger)
        print(answers_file)
    else:
        print("Check checkpoint name")
        exit()


if __name__ == "__main__":
    inference()
