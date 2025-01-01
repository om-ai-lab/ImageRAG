import pdb
import shutil

from pygments.lexer import default
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

from codebase.llm_template import paraphrase_template, keyword_template, text_expansion_template
from codebase.utils import (setup_vlm_model, set_up_paraphrase_model, setup_vqallm, setup_slow_text_encoder_model,
                            calculate_similarity_matrix, extract_vlm_img_text_feat, ranking_patch_t2p,
                            paraphrase_model_inference, text_expand_model_inference, setup_logger, meta_df2clsimg_dict,
                            img_reduce, select_visual_cue, ranking_patch_visualcue2patch, load_yaml, get_chunk,
                            convert_obb_to_region_str, obb2minhbb, sole_visualcue2mergedvisualcue)
from codebase.sglang_util import get_paraphase_response, get_keyword_response, get_text_expansion_response


from codebase.utils import load_yaml
from codebase.cc_algo import img_2patch, vis_patches
from codebase.text_parser import extract_key_phrases
Image.MAX_IMAGE_PIXELS = None


def imagerag_inference(image_path, question, config, contrastive_vlm, task_category, client, logger, paraphrase=False):
    patch_saving_dir = config['patch_saving_dir']
    fast_path_T = int(config['fast_path_T'])
    paraphrase_model_config = config['paraphrase_model']
    kw_model_config = config['kw_model']
    text_expansion_model_config = config['text_expansion_model']
    visual_cues = []

    input_uhr_image = Image.open(image_path)
    width, height = input_uhr_image.size
    logger.info("original image width and height: {}, {}".format(width, height))

    if "<rbox>" in question:
        pattern = r'\{(<.*?>)\}'
        # 使用正则表达式找到所有的矩形框
        matches = re.findall(pattern, question)
        for match in matches:
            # 在每个矩形框中，找到所有的数字
            numbers_str = re.findall(r'<(.*?)>', match)
            rbox = np.array(numbers_str, dtype=np.float32)
            polys = obb2minhbb(rbox)
            # need normalized cx, cy, w, h
            visual_cues.append(polys)
        merged_visual_cue = sole_visualcue2mergedvisualcue(visual_cues)
        visual_cues.append(merged_visual_cue)


    if "<ref>" in question:
        if paraphrase:
            paraphrase_result = get_paraphase_response(client, config['paraphrase_model']['model_path'], question, config['paraphrase_model']['generation_config'])
            logger.info("Paraphrased Text: {}".format(paraphrase_result))
        else:
            paraphrase_result = question

        if not kw_model_config:
            kw_model_config = paraphrase_model_config
            while True:
                query_keywords = get_keyword_response(client, kw_model_config['model_path'], paraphrase_result,
                                                      kw_model_config['generation_config'])
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
            c_denom=10,
            dump_imgs=True,
            patch_saving_dir=patch_saving_dir
        )
        logger.info(
            "resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))

        fast_path_vlm, img_preprocess, text_tokenizer = contrastive_vlm
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
        ranked_patch, corresponding_similarity = ranking_patch_t2p(bbox_coordinate_list, t2p_similarity, top_k=5)
        logger.info("Ranked Patch Shape: {}".format(ranked_patch.shape))
        logger.info("Corresponding similarity: {}".format(corresponding_similarity))

        # Slow Path
        if max(corresponding_similarity) < fast_path_T:
            from langchain.vectorstores import Chroma, FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("fast path similarity does not meet the threshold, choose the slow path")

            if not text_expansion_model_config:
                text_expansion_model_config = paraphrase_model_config
                expanded_query_text_dict = get_text_expansion_response(client, text_expansion_model_config['model_path'],
                                                                       query_keywords,
                                                                       text_expansion_model_config['generation_config'])
            else:
                print("Not Implemented")
                exit()
            # Uncomment if not use phrase expansion
            # expanded_query_text_list = query_keywords

            # embedding = GeoRSCLIPEmbeddings(checkpoint=config['fast_vlm_model']['model_path'], device=device)
            # vectorstore = Chroma(
            #     collection_name="mm_georsclip",
            #     embedding_function=embedding,
            #     persist_directory=config["vector_database"]["mm_vector_database_dir"],
            #     collection_metadata = {"hnsw:space": "cosine"}
            # )

            # Create Chroma vector store
            slow_text_emb_model_path = config["text_embed_model"]["model_path"]
            text_embeddings = HuggingFaceEmbeddings(model_name=slow_text_emb_model_path)
            text_vectorstore = Chroma(
                collection_name="vector_store4keyphrase_label_matching",
                embedding_function=text_embeddings,
                persist_directory=config["vector_database"]["text_vector_database_dir"],
                # Where to save data locally, remove if not necessary
            )

            meta_pkl_path = config["vector_database"]["meta_pkl_path"]
            meta_vector_database_df = pkl.load(open(meta_pkl_path, "rb"))
            labels_in_database = list(set(meta_vector_database_df["cls_list"].tolist()))
            meta = [{'type': 'text'}] * len(labels_in_database)
            text_vectorstore.add_texts(texts=labels_in_database, metadatas=meta)

            selected_label_name = []
            for expanded_query_text in expanded_query_text_dict:
                results = text_vectorstore.similarity_search_with_score(
                    expanded_query_text, k=5
                )
                for res, score in results:
                    # * [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}]
                    print(f"Query:={expanded_query_text_dict[expanded_query_text]} * [SIM={score:3f}] {res.page_content}")
                    selected_label_name.append(res.page_content)
                print()
            selected_label_name = list(set(selected_label_name))
            meta_vector_database_df_selected = meta_vector_database_df[
                meta_vector_database_df['cls_list'].isin(selected_label_name)]
            reduced_img_group = meta_df2clsimg_dict(meta_vector_database_df_selected, config['vector_database']['img_dir'])

            visual_cue_candidates_dict = img_reduce(reduced_img_group, fast_path_vlm, img_preprocess)
            visual_cue = select_visual_cue(vlm_image_feats, bbox_coordinate_list, visual_cue_candidates_dict)
            visual_cues.append(visual_cue)

    if len(visual_cues) == 0:
        visual_cues.append([0, 0, width, height])

    return visual_cues


def main_fit(questions, config, answers_file, contrastive_vlm_pack, generative_vlm_pack, client, logger, mode="zeroshot"):
    # TODO: only works for InternVL model, Batch_Size=1
    generative_vlm, generative_vlm_tokenizer, generative_vlm_generation_config, generative_vlm_load_image = generative_vlm_pack
    ans_file = open(answers_file, "w")
    for i in tqdm(range(0, len(questions), config['batch_size'])):
        count=i
        num_patches_list = []
        image_folder = []
        image_path_list = []
        question_list = []
        category_list = []
        batch_end = min(i + config['batch_size'], len(questions))
        item_list = []
        for j in range(i, batch_end):
            item_list.append(questions[j])
            image_file = questions[j]['image']
            if "fit" in config['input_image_dir'].lower():
                image_path = os.path.join(config['input_image_dir'], image_file)
            elif "star" in config['input_image_dir'].lower():
                image_path = os.path.join(config['input_image_dir'], image_file + ".png")
            # 判断问题类别,进而确定模板
            image_path_list.append(image_path)
            category = questions[j]['category']
            qs = questions[j]['question']
            if len(generative_vlm_pack) == 4:
                # batch inference, single image per sample (单图批处理)
                tensor_images = generative_vlm_load_image(image_path, input_size=config["model_input_image_size"], max_num=6, use_dynamic=config["use_dynamic"]).to(torch.bfloat16).cuda()
                image_folder.append(tensor_images)
                num_patches_list.append(tensor_images.size(0))
                question_list.append(qs)
                category_list.append(category)


        if mode == "zeroshot":
            with torch.inference_mode():
                image_folder = torch.cat(image_folder, dim=0)
                responses = generative_vlm.batch_chat(
                    generative_vlm_tokenizer,
                    image_folder,
                    num_patches_list=num_patches_list,
                    questions=question_list,
                    generation_config=generative_vlm_generation_config
                )
            for item, response, category in zip(item_list, responses, category_list):
                print(f'Category: {category}\nUser: {item}\nAssistant: {response}')
                ans_file.write(json.dumps({
                    "question_id": item["question_id"],
                    "image_id": item["image"],
                    "category": category,
                    "ground_truth": item["ground_truth"],
                    "answer": response,
                }) + "\n")
                count = count + 1
                ans_file.flush()

        elif mode == "imagerag":
            visual_cue_list = []
            for i in range(len(image_path_list)):
                image_path = image_path_list[i]
                question = question_list[i]
                item = item_list[i]
                task_category = category_list[i]
                image = image_folder[i]
                input_images = [image]
                # image_path, qs, config, contrastive_vlm, client, logger
                visual_cue = imagerag_inference(image_path, question, config, contrastive_vlm_pack, task_category, client, logger)
                visual_cue_list.append(visual_cue)
                input_images.append(visual_cue)
                input_images = torch.cat(input_images, dim=0)

                with torch.inference_mode():
                    response = generative_vlm.chat(
                        generative_vlm_tokenizer,
                        input_images,
                        question,
                        generation_config=generative_vlm_generation_config,
                        num_patches_list=num_patches_list,
                        history=None, return_history=True
                    )

                # # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
                # pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
                # pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
                # pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                # num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
                #
                # question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
                # response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                #                                num_patches_list=num_patches_list,
                #                                history=None, return_history=True)
                # print(f'User: {question}\nAssistant: {response}')
                #
                # question = 'What are the similarities and differences between these two images.'
                # response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                #                                num_patches_list=num_patches_list,
                #                                history=history, return_history=True)
                # print(f'User: {question}\nAssistant: {response}')


            # for item, response, category, visual_cue in zip(item_list, responses, category_list, visual_cue_list):
                print(f'Category: {task_category}\nUser: {item}\nAssistant: {response}')
                ans_file.write(json.dumps({
                    "question_id": item["question_id"],
                    "image_id": item["image"],
                    "category": task_category,
                    "ground_truth": item["ground_truth"],
                    "answer": response,
                }) + "\n")
                count = count + 1
                ans_file.flush()

    return answers_file

def main():
    pl.seed_everything(2024)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='/media/zilun/fanxiang4t/GRSM/ImageRAG_git/config/config_internvl8b_5obb1_512_nondynamic_100_star_local.yaml', help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to the log file.')
    parser.add_argument('--base_url', type=str, default='http://127.0.0.1:30000/v1', help='base url')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, "log.txt"))

    config = load_yaml(args.cfg_path)
    patch_saving_dir = config['patch_saving_dir']
    os.makedirs(patch_saving_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_input_image_size = int(config['model_input_image_size'])
    fast_vlm_model_name = config['fast_vlm_model']['model_name']
    fast_vlm_model_path = config['fast_vlm_model']['model_path']
    llmvqa_model_name = config['llmvqa_model']['model_name']
    llmvqa_model_path = config['llmvqa_model']['model_path']
    # fast_path_vlm, img_preprocess, text_tokenizer
    contrastive_vlm_pack = setup_vlm_model(fast_vlm_model_path, fast_vlm_model_name, device)
    # model, tokenizer, generation_config, load_image
    generative_vlm_pack = setup_vqallm(llmvqa_model_path, llmvqa_model_name, model_input_image_size=model_input_image_size, device=device)
    client = openai.Client(base_url=args.base_url, api_key="None")

    # setup inference data
    questions = [json.loads(q) for q in open(config['question_file_path'], "r")]
    questions = get_chunk(questions, config['fast_vlm_model']['num_chunks'], config['fast_vlm_model']['chunk_idx'])
    answers_file = os.path.expanduser(config['answers_file_path'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    main_fit(questions, config, answers_file, contrastive_vlm_pack, generative_vlm_pack, client, logger, mode="imagerag")








if __name__ == "__main__":
    main()