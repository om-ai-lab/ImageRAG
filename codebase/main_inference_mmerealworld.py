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

from codebase.llm_template import paraphrase_template, keyword_template, text_expansion_template
from codebase.utils import (setup_vlm_model, set_up_paraphrase_model, setup_vqallm, setup_slow_text_encoder_model,
                            calculate_similarity_matrix, extract_vlm_img_text_feat, ranking_patch_t2p,
                            paraphrase_model_inference, text_expand_model_inference, setup_logger, meta_df2clsimg_dict,
                            img_reduce, select_visual_cue, ranking_patch_visualcue2patch, load_yaml, get_chunk,
                            convert_obb_to_region_str, obb2minhbb, sole_visualcue2mergedvisualcue, visualcue2imagepatch,
                            reduce_visual_cue_per_cls, setup_text_vsd)
from codebase.sglang_util import get_paraphase_response, get_keyword_response, get_text_expansion_response
from codebase.utils import load_yaml
from codebase.cc_algo import img_2patch, vis_patches
from codebase.text_parser import extract_key_phrases



# export PYTHONPATH=$PYTHONPATH:/data1/zilun/grsm/ImageRAG_git
# export PYTHONPATH=$PYTHONPATH:/media/zilun/fanxiang4t/GRSM/ImageRAG_git

Image.MAX_IMAGE_PIXELS = None


def collate_fn(batch):
    image_tensor, prompt, num_patches_list, question_list, line = zip(*batch)
    image_tensors = torch.cat(image_tensor, dim=0)
    return image_tensors, prompt, num_patches_list, question_list, line


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, batch_size=1, num_workers=0, prompt=''):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = InternVLMMERSDataset(questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, prompt)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def organize_prompt(test_prompt, qs, choices, mode="vanilla", bboxes=None, fit_relative_pos_star=None):
    if "vanilla" not in mode:
        assert bboxes is not None

    choice_prompt = ' The choices are listed below: \n'
    for choice in choices:
        choice_prompt += choice + "\n"
    qs += choice_prompt + test_prompt + '\nThe best answer is:'
    
    if mode == "vanilla":
        prompt = qs
        
    elif mode == "vanilla_image_token":
        final_instruction = "<image>\n"
        # final_instruction += "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        final_instruction += qs
        prompt = final_instruction

    elif mode == "vanilla_relative_pos":
        # prompt = qs
        # fit_relative_pos_star = "Top-center"
        fit_relative_pos_star = "Bottom-right"
        final_instruction = "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        final_instruction += qs
        prompt = final_instruction

    elif mode == "vanilla_image_token_relative_pos":
        # prompt = qs
        final_instruction = "<image>\n"
        fit_relative_pos_star = "Bottom-right"
        final_instruction += "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        final_instruction += qs
        prompt = final_instruction

    elif mode == "withoutobb":
        final_instruction = "<image>\n"
        final_instruction += "Additional information:\n"
        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(i + 1, *bbox)
        final_instruction += "Look at {} of the image and answer the question: \n".format(fit_relative_pos_star.lower())
        final_instruction += qs
        prompt = final_instruction

    elif mode == "withobb":
        final_instruction = "<image>\n"
        final_instruction += "Additional information:\n"
        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(i + 1, *bbox)
        final_instruction += "Look at {} of the image and answer the question based on the provided additional information (location of sub-patches) \nQuestion: ".format(fit_relative_pos_star.lower())
        final_instruction += qs
        prompt = final_instruction

    return prompt


# Custom dataset class
class InternVLMMERSDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, dynamic_preprocess, transform, model_config, prompt, use_dynamic=True):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.dynamic_preprocess = dynamic_preprocess
        self.model_config = model_config
        self.prompt = prompt
        self.use_dynamic = use_dynamic

    def __getitem__(self, index):
        line = self.questions[index]
        choices = line['Answer choices']
        image_file = line["Image"]
        qs = line["Text"]
        choice_prompt = ' The choices are listed below: \n'
        for choice in choices:
            choice_prompt += choice + "\n"
        qs += choice_prompt + self.prompt + '\nThe best answer is:'
        prompt = qs
        # prompt = organize_prompt(self.prompt, qs, choices, mode=mode, bboxes=None, fit_relative_pos_star=None)
        

        image_path = os.path.join(self.image_folder, image_file)

        # pixel_values = self.load_image(
        #     image_path,
        #     transform=transform,
        #     input_size=self.model_config.vision_config.image_size,
        #     max_num=self.model_config.max_dynamic_patch,
        #     use_dynamic=True
        # ).to(torch.bfloat16).cuda()

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


def image_rag(config, contrastive_vlm_pack, line, client, logger, text_vectorstore, vsd_label2imgname_dict, vsd_imgname2label_dict, vsd_imgname2feat_dict, text_paraphrase, text_expand):
    # responses = image_rag(config, generative_vlm, generative_vlm_tokenizer, generative_vlm_generation_config, image_tensors, prompts, num_patches_list, question_text_only_list, line, client, logger, paraphrase=False)

    patch_saving_dir = config['patch_saving_dir']
    fast_path_T = int(config['fast_path_T'])
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
    # input_uhr_image.close()

    # pixel_values = generative_vlm_load_image(
    #     image_path,
    #     input_size=model.config.vision_config.image_size,
    #     max_num=model.config.max_dynamic_patch,
    #     use_dynamic=True
    # ).to(torch.bfloat16).cuda()

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
        while True:
            query_keywords = get_keyword_response(client, kw_model_config['model_path'], paraphrase_result, kw_model_config['generation_config'])
            try:
                query_keywords = eval(query_keywords)
                break
            except Exception as e:
                print("Bad query keywords: {}".format(query_keywords))
    else:
        kw_model_path = kw_model_config['model_path']
        kw_model = KeyBERT(model=SentenceTransformer(kw_model_path))
        query_keywords = extract_key_phrases(paraphrase_result, paraphrase_result, kw_model)

    logger.info("Final Key Phrases: {}".format(query_keywords))

    # patchify -> padded image and dict of bbox - patch save name
    img_resize, coordinate_patchname_dict = img_2patch(
        input_uhr_image,
        c_denom=2,
        dump_imgs=True,
        # dump_imgs=False, Not working
        patch_saving_dir=patch_saving_dir
    )
    logger.info(
        "resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))

    fast_path_vlm, img_preprocess, text_tokenizer = contrastive_vlm_pack
    # Fast Path
    vlm_image_feats, vlm_text_feats, bbox_coordinate_list = extract_vlm_img_text_feat(
        question,
        query_keywords,
        coordinate_patchname_dict,
        patch_saving_dir,
        img_preprocess,
        text_tokenizer,
        fast_path_vlm,
        img_batch_size=50
    )

    t2p_similarity = calculate_similarity_matrix(vlm_image_feats, vlm_text_feats, fast_path_vlm.logit_scale.exp())
    visual_cue, corresponding_similarity = ranking_patch_t2p(bbox_coordinate_list, t2p_similarity, top_k=5)
    logger.info("Ranked Patch Shape: {}".format(visual_cue.shape))
    logger.info("Corresponding similarity: {}".format(corresponding_similarity))

    # pdb.set_trace()

    # Slow Path
    if max(corresponding_similarity) < fast_path_T:
        logger.info("fast path similarity does not meet the threshold, choose the slow path")

        if text_expand:
            if not text_expansion_model_config:
                text_expansion_model_config = paraphrase_model_config
                expanded_query_text_dict = get_text_expansion_response(client, text_expansion_model_config['model_path'],
                                                                       query_keywords,
                                                                       text_expansion_model_config['generation_config'])
            else:
                print("Not Implemented")
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
            results = text_vectorstore.similarity_search_with_score(
                expanded_query_text, k=5
            )
            for res, score in results:
                # * [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}]
                print(f"Query:={expanded_query_text_dict[expanded_query_text]} * [SIM={score:3f}] {res.page_content}")
                selected_label_names.append(res.page_content)
        selected_label_name = list(set(selected_label_names))
        print(selected_label_names)

        # img_name_selected_per_cls = [vsd_label2imgname_dict[label] for label in selected_label_name]

        # label -> feats dict
        visual_cue_candidates_dict = dict()

        for label in selected_label_name:
            img_feat_selected_per_cls = []
            img_names = vsd_label2imgname_dict[label]
            for img_name in img_names:
                feat = vsd_imgname2feat_dict[img_name].unsqueeze(0)
                img_feat_selected_per_cls.append(feat)
            img_feat_selected_per_cls = torch.cat(img_feat_selected_per_cls)
            visual_cue_candidates_dict[label] = img_feat_selected_per_cls

        reduced_visual_cue_per_cls = reduce_visual_cue_per_cls(visual_cue_candidates_dict, reduce_fn="mean")

        # meta_vector_database_df_selected = meta_vector_database_df[meta_vector_database_df['label_list'].isin(selected_label_name)]
        # reduced_img_group = meta_df2clsimg_dict(meta_vector_database_df_selected, config['vector_database']['img_dir'])
        # visual_cue_candidates_dict = img_reduce(reduced_img_group, fast_path_vlm, img_preprocess)

        visual_cue, visual_cue_similarity = select_visual_cue(vlm_image_feats, bbox_coordinate_list, reduced_visual_cue_per_cls)

        # visual_cues.append(visual_cue)
    # else:
    #     visual_cues.append(visual_cue)
    # if len(visual_cues) == 0:
    #     visual_cues.append(np.array([0, 0, width, height]))

    return image_path, visual_cue, question_with_test_template


def inference_internvl(config, questions, ans_file_path, contrastive_vlm_pack, generative_vlm_pack, client, logger):
    # TODO: only works for InternVL model, Batch_Size=1
    ans_file = open(ans_file_path, "w")
    generative_vlm, generative_vlm_tokenizer, generative_vlm_generation_config, generative_vlm_dynamic_preprocess, generative_vlm_transform = generative_vlm_pack
    index = 0
    if config["mode"] == "baseline":
        data_loader = create_data_loader(
            questions, config["input_image_dir"],
            generative_vlm_tokenizer, generative_vlm_dynamic_preprocess,
            generative_vlm_transform, generative_vlm.config,
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

    elif config["mode"] == "imagerag":
        # imagerag does not support parallel inference
        text_vectorstore, vsd_label2imgname_dict, vsd_imgname2label_dict, vsd_imgname2feat_dict = setup_text_vsd(config)
        assert config["batch_size"] == 1
        for line in questions:
            image_path, visual_cues, question_with_test_template = image_rag(
                config, contrastive_vlm_pack, line, client, logger,
                text_vectorstore, vsd_label2imgname_dict, vsd_imgname2label_dict, vsd_imgname2feat_dict,
                text_paraphrase=False, text_expand=False
            )
            images, tile_num_list = [], []
            global_and_locals = []
            image = Image.open(image_path).convert('RGB')
            global_and_locals.append(image)
            for object_coord in visual_cues:
                image_crop = image.crop(object_coord)
                global_and_locals.append(image_crop)

            for i, image in enumerate(global_and_locals):
                if i == 0:
                    image = generative_vlm_dynamic_preprocess(
                        image,
                        max_num=generative_vlm.config.max_dynamic_patch,
                        image_size=generative_vlm.config.vision_config.image_size,
                        use_thumbnail=generative_vlm.config.use_thumbnail,
                    )
                    images += image
                    tile_num_list.append(len(image))
                else:
                    images.append(image)
                    tile_num_list.append(1)
            pixel_values = [generative_vlm_transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
            num_patches = pixel_values.size(0)

            final_instruction = "<image>\n"
            final_instruction += "Additional information:\n"
            for i, bbox in enumerate(visual_cues):
                final_instruction += "Sub-patch {} at location <box>[[{:.2f}, {:.2f}, {:.2f}, {:.2f}]]</box>: <image>\n".format(i + 1, *bbox)
            final_instruction += "Look at the image and answer the question based on the provided additional information (location of sub-patches) \n"
            final_instruction += "Question: "
            final_instruction += question_with_test_template

            with torch.inference_mode():
                # TODO: visual cues contain full image, duplicate
                responses = generative_vlm.batch_chat(
                    generative_vlm_tokenizer,
                    pixel_values,
                    num_patches_list=[num_patches],
                    questions=final_instruction,
                    generation_config=generative_vlm_generation_config
                )

            for i, (response) in enumerate(responses):
                if index % 100 == 0:
                    print(f'Prompt: {question_with_test_template}\n\n Output: {response}')
                line['output'] = response
                ans_file.write(json.dumps(line) + "\n")
                ans_file.flush()
                index += 1
        ans_file.close()

    return ans_file_path


def inference():
    pl.seed_everything(2024)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='/media/zilun/fanxiang4t/GRSM/ImageRAG_git/config/config_internvl8b_5hbb_448_dynamic_0-1000_mmerealworld.yaml', help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to the log file.')
    parser.add_argument('--base_url', type=str, default='http://127.0.0.1:30000/v1', help='base url')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, "log.txt"))

    config = load_yaml(args.cfg_path)
    patch_saving_dir = config['patch_saving_dir']
    os.makedirs(patch_saving_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_vlm_model_name = config['fast_vlm_model']['model_name']
    fast_vlm_model_path = config['fast_vlm_model']['model_path']
    llmvqa_model_name = config['llmvqa_model']['model_name']
    llmvqa_model_path = config['llmvqa_model']['model_path']
    # fast_path_vlm, img_preprocess, text_tokenizer
    contrastive_vlm_pack = setup_vlm_model(fast_vlm_model_path, fast_vlm_model_name, device)
    generative_vlm_generation_config = dict(
        max_new_tokens=config["llmvqa_model"]["generation_config"]["max_tokens"],
        do_sample=True if config["llmvqa_model"]["generation_config"]["temperature"] > 0 else False,
        temperature=config["llmvqa_model"]["generation_config"]["temperature"],
        top_p=config["llmvqa_model"]["generation_config"]["top_p"],
        num_beams=config["llmvqa_model"]["generation_config"]["num_beams"],
    )
    generative_vlm_pack = setup_vqallm(llmvqa_model_path, llmvqa_model_name, generative_vlm_generation_config, config["model_input_image_size"], device=device)
    client = openai.Client(base_url=args.base_url, api_key="None")

    with open(config['question_file_path'], 'r') as file:
        questions = json.load(file)
    questions = [question for question in questions if question["Subtask"] == "Remote Sensing"]
    questions = get_chunk(questions, config['num_chunks'], config['chunk_idx'])
    answers_file = os.path.expanduser(config['answers_file_path'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)


    if "InternVL" in llmvqa_model_path:
        answers_file = inference_internvl(config, questions, answers_file, contrastive_vlm_pack, generative_vlm_pack, client, logger)
        print(answers_file)


if __name__ == "__main__":
    inference()

    