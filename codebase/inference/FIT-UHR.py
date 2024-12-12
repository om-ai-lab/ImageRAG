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

from codebase.utils import (setup_vlm_model, set_up_paraphrase_model, setup_vqallm, setup_slow_text_encoder_model,
                   calculate_similarity_matrix, extract_vlm_img_text_feat, ranking_patch_t2p, paraphrase_model_inference,
                   text_expand_model_inference, setup_logger, meta_df2clsimg_dict, img_reduce, select_visual_cue, ranking_patch_visualcue2patch, load_yaml)
from codebase.cc_algo import img_2patch, vis_patches
from codebase.text_parser import extract_key_phrases
from codebase.llm_template import paraphrase_template, keyword_template, text_expansion_template
from codebase.sglang_util import get_paraphase_response, get_keyword_response, get_text_expansion_response

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle

from geochat.model.builder import load_pretrained_model
# from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

Image.MAX_IMAGE_PIXELS = None

# export PYTHONPATH=$PYTHONPATH:/data1/zilun/grsm/ImageRAG_git


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


def convert_obb_to_region_str(rbox_np):
    angle = rbox_np[-1]
    polys = obb2poly_np_oc(rbox_np)
    x_left = np.clip(np.min(polys[:, [0, 2, 4, 6]], axis=1), 0, None)
    y_top = np.clip(np.min(polys[:, [1, 3, 5, 7]], axis=1), 0, None)
    x_right = np.max(polys[:, [0, 2, 4, 6]], axis=1)
    y_bottom = np.max(polys[:, [1, 3, 5, 7]], axis=1)
    region_str = f"<{int(x_left[0])}><{int(y_top[0])}><{int(x_right[0])}><{int(y_bottom[0])}>|<{int(angle)}>"
    return region_str


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def fituhr_inference(config):

    # init model
    model_path = config['fast_vlm_model']['model_path']
    model_name = config['fast_vlm_model']['model_name']

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name
    )

    # setup inference data
    questions = [json.loads(q) for q in open(config['question_file_path'], "r")]
    questions = get_chunk(questions, config['fast_vlm_model']['num_chunks'], config['fast_vlm_model']['chunk_idx'])
    answers_file = os.path.expanduser(config['answers_file_path'])
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for i in tqdm(range(0,len(questions), config['batch_size'])):
        input_batch=[]
        count=i
        image_folder=[]
        batch_end = min(i + config['batch_size'], len(questions))

        for j in range(i, batch_end):
            image_file = questions[j]['image']
            # 判断问题类别,进而确定模板
            category = questions[j]['category']
            qs = questions[j]['question']
            # 需要的区域任务要针对geochat进行OBB的格式转换
            if category in ["task4", "task5", "task6"]:
                # if 'fgrs' not in answers_file.split("/")[-1] and 'geochat' in answers_file.split("/")[-1]:
                if 'geochat' in answers_file.split("/")[-1]:
                    pattern = r'\{(.+?)\}'
                    matches = re.findall(pattern, qs)
                    for match in matches:
                        numbers_str = match
                        pattern = r'<(.+?)>'
                        numbers = re.findall(pattern, numbers_str)
                        rbox_np = np.array(numbers, dtype=float)
                        region_str = convert_obb_to_region_str(rbox_np)
                        qs = qs.replace(numbers_str, region_str)

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[config['conv_mode']].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()
            input_batch.append(input_ids)

            image = Image.open(os.path.join(config['input_image_dir'], image_file))

            image_folder.append(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            # keywords = [stop_str]
            # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [torch.cat(
            (torch.zeros((1, max_length - tensor.size(1)), dtype=tensor.dtype, device=tensor.get_device()), tensor),
            dim=1) for tensor in input_batch]
        final_input_tensors = torch.cat(final_input_list, dim=0)
        image_tensor_batch = \
        image_processor.preprocess(image_folder, crop_size={'height': 504, 'width': 504}, size={'shortest_edge': 504},
                                   return_tensors='pt')['pixel_values']

        with torch.inference_mode():
            output_ids = model.generate(final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False,
                                        temperature=config['fast_vlm_model']['temperature'], top_p=config['fast_vlm_model']['top_p'], num_beams=1, max_new_tokens=256,
                                        length_penalty=2.0, use_cache=True)

        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for k in range(0, len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            # ans_id = shortuuid.uuid()

            ans_file.write(json.dumps({
                "question_id": questions[count]["question_id"],
                "image_id": questions[count]["image"],
                "category": category,
                "ground_truth": questions[count]["ground_truth"],
                "answer": output,
            }) + "\n")
            count = count + 1
            ans_file.flush()


def main():
    pl.seed_everything(2024)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='/data1/zilun/grsm/ImageRAG_git/config/config_fituhr_inference _server.yaml', help='Path to the configuration file.')
    parser.add_argument('--query_text', type=str, default='Suppose the top of this image represents north. How many aircraft are heading northeast? What is the color of the building rooftop to their southeast?', help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to the log file.')
    parser.add_argument('--base_url', type=str, default='http://127.0.0.1:34000/v1', help='base url')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, "log.txt"))
    config = load_yaml(args.cfg_path)

    input_uhr_image_path = config['input_uhr_image_path']
    patch_saving_dir = config['patch_saving_dir']
    os.makedirs(patch_saving_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_input_image_size = int(config['model_input_image_size'])
    fast_path_T = int(config['fast_path_T'])

    if config['task_name'] == "fit_uhr":
        paraphrase_model_config = config['paraphrase_model']
        kw_model_config = config['kw_model']
        text_expansion_model_config = config['text_expansion_model']
        fast_vlm_model_path = config['fast_vlm_model']['model_path']
        llmvqa_model_path = config['llmvqa_model']['model_path']

        fituhr_inference(config)

    #     client = openai.Client(base_url=args.base_url, api_key="None")
    #     fast_path_vlm, img_preprocess, text_tokenizer = setup_vlm_model(fast_vlm_model_path, device)
    #     vqa_llm = setup_vqallm(llmvqa_model_path, model_input_image_size=384, device=device)
    #
    #     logger.info("Original Input Query: {}".format(args.query_text))
    #
    #     paraphrase_result = get_paraphase_response(client, paraphrase_model_config['model_path'], args.query_text, paraphrase_model_config['generation_config'])
    #     logger.info("Paraphrased Text: {}".format(paraphrase_result))
    #
    #     if not kw_model_config:
    #         kw_model_config = paraphrase_model_config
    #         while True:
    #             query_keywords = get_keyword_response(client, kw_model_config['model_path'], paraphrase_result,
    #                                                   kw_model_config['generation_config'])
    #             try:
    #                 query_keywords = eval(query_keywords)
    #                 break
    #             except Exception as e:
    #                 print("Bad query keywords: {}".format(query_keywords))
    #     else:
    #         kw_model_path = kw_model_config['model_path']
    #         kw_model = KeyBERT(model=SentenceTransformer(kw_model_path))
    #         query_keywords = extract_key_phrases(paraphrase_result, paraphrase_result, kw_model)
    #
    #     logger.info("Final Key Phrases: {}".format(query_keywords))
    #
    #     input_uhr_image = Image.open(input_uhr_image_path)
    #     width, height = input_uhr_image.size
    #     logger.info("original image width and height: {}, {}".format(width, height))
    #
    #     # patchify -> padded image and dict of bbox - patch save name
    #     img_resize, coordinate_patchname_dict = img_2patch(
    #         input_uhr_image,
    #         c_denom=10,
    #         dump_imgs=True,
    #         patch_saving_dir=patch_saving_dir
    #     )
    #     logger.info("resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))
    #
    #     # Fast Path
    #     vlm_image_feats, vlm_text_feats, bbox_coordinate_list = extract_vlm_img_text_feat(args.query_text, query_keywords, coordinate_patchname_dict, patch_saving_dir, img_preprocess, text_tokenizer, fast_path_vlm, img_batch_size=50)
    #     t2p_similarity = calculate_similarity_matrix(vlm_image_feats, vlm_text_feats, fast_path_vlm.logit_scale.exp())
    #     ranked_patch, corresponding_similarity = ranking_patch_t2p(bbox_coordinate_list, t2p_similarity, top_k=5)
    #     logger.info("Ranked Patch Shape: {}".format(ranked_patch.shape))
    #     logger.info("Corresponding similarity: {}".format(corresponding_similarity))

    # # Slow Path
    # if max(corresponding_similarity) < fast_path_T:
    #     from langchain.vectorstores import Chroma, FAISS
    #     from langchain_huggingface import HuggingFaceEmbeddings
    #     logger.info("fast path similarity does not meet the threshold, choose the slow path")
    #
    #     if not text_expansion_model_config:
    #         text_expansion_model_config = paraphrase_model_config
    #         expanded_query_text_dict = get_text_expansion_response(client, text_expansion_model_config['model_path'], query_keywords, text_expansion_model_config['generation_config'])
    #     else:
    #         print("Not Implemented")
    #         exit()
    #     # Uncomment if not use phrase expansion
    #     # expanded_query_text_list = query_keywords
    #
    #     # embedding = GeoRSCLIPEmbeddings(checkpoint=config['fast_vlm_model']['model_path'], device=device)
    #     # vectorstore = Chroma(
    #     #     collection_name="mm_georsclip",
    #     #     embedding_function=embedding,
    #     #     persist_directory=config["vector_database"]["mm_vector_database_dir"],
    #     #     collection_metadata = {"hnsw:space": "cosine"}
    #     # )
    #
    #     # Create Chroma vector store
    #     slow_text_emb_model_path = "/media/zilun/wd-161/hf_download/all-MiniLM-L6-v2"
    #     text_embeddings = HuggingFaceEmbeddings(model_name=slow_text_emb_model_path)
    #     text_vectorstore = Chroma(
    #         collection_name="vector_store4keyphrase_label_matching",
    #         embedding_function=text_embeddings,
    #         persist_directory=config["vector_database"]["text_vector_database_dir"],  # Where to save data locally, remove if not necessary
    #     )
    #
    #     meta_pkl_path = config["vector_database"]["meta_pkl_path"]
    #     meta_vector_database_df = pkl.load(open(meta_pkl_path, "rb"))
    #     labels_in_database = list(set(meta_vector_database_df["cls_list"].tolist()))
    #     meta = [{'type': 'text'}] * len(labels_in_database)
    #     text_vectorstore.add_texts(texts=labels_in_database, metadatas=meta)
    #
    #     selected_label_name = []
    #     for expanded_query_text in expanded_query_text_dict:
    #         results = text_vectorstore.similarity_search_with_score(
    #             expanded_query_text, k=5
    #         )
    #         for res, score in results:
    #             # * [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}]
    #             print(f"Query:={expanded_query_text_dict[expanded_query_text]} * [SIM={score:3f}] {res.page_content}")
    #             selected_label_name.append(res.page_content)
    #         print()
    #     selected_label_name = list(set(selected_label_name))
    #     meta_vector_database_df_selected = meta_vector_database_df[meta_vector_database_df['cls_list'].isin(selected_label_name)]
    #     reduced_img_group = meta_df2clsimg_dict(meta_vector_database_df_selected, config['vector_database']['img_dir'])
    #
    #     visual_cue_candidates_dict = img_reduce(reduced_img_group, fast_path_vlm, img_preprocess)
    #
    #     visual_cue = select_visual_cue(vlm_image_feats, bbox_coordinate_list, visual_cue_candidates_dict)
    #
    #     # visual_cue = [[0, 0, img_resize.width, img_resize.height]]
    #
    # response = vqa_llm.free_form_inference(img_resize, args.query_text, visual_cue)
    # logger.info(response)


if __name__ == "__main__":
    main()