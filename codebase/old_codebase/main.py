import pdb
import shutil

from pygments.lexer import default
from tqdm import tqdm
import os
import uuid
import numpy as np
import pickle as pkl
from PIL import Image
from utils import (setup_vlm_model, set_up_paraphrase_model, setup_vqallm, setup_slow_text_encoder_model,
                   calculate_similarity_matrix, extract_vlm_img_text_feat, ranking_patch, paraphrase_model_inference,
                   text_expand_model_inference, setup_logger)
from cc_algo import img_2patch, vis_patches
from text_parser import extract_key_phrases
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import yaml
import argparse
import logging
from llm_template import paraphrase_template, keyword_template


Image.MAX_IMAGE_PIXELS = None


def load_yaml(config_filepath):
    # Load the YAML file
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    # L.seed_everything(2024)
    # input config
    # input_uhr_image_path = "/media/zilun/fanxiang4t/GRSM/IRAG/imageRAG/data/dqx_21n_chengguo.tif"

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='../config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--query_text', type=str, default='Suppose the top of this image represents north. How many aircraft are heading northeast? What is the color of the building rooftop to their southeast?', help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='../log', help='Path to the log file.')

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

    paraphrase_model_path = config['paraphrase_model']['model_path']
    fast_vlm_model_path = config['fast_vlm_model']['model_path']
    llmvqa_model_path = config['llmvqa_model']['model_path']
    paraphrase_model, paraphrase_tokenizer = set_up_paraphrase_model(paraphrase_model_path, device)
    if not config['kw_model']:
        kw_model = paraphrase_model
        kw_tokenizer = paraphrase_tokenizer
    else:
        kw_model_path = config['kw_model']['model_path']
        kw_model = KeyBERT(model=SentenceTransformer(kw_model_path))
    fast_path_vlm, img_preprocess, text_tokenizer = setup_vlm_model(fast_vlm_model_path, device)
    vqa_llm = setup_vqallm(llmvqa_model_path, model_input_image_size=384, device=device)


    # query_text = "Suppose the top of this image represents north. How many aircrafts are heading northeast? What is the color of the building rooftop to their southeast?"
    logger.info("Original Input Query: {}".format(args.query_text))

    text2paraphrase = paraphrase_template.format(args.query_text)
    paraphrase_result = paraphrase_model_inference(paraphrase_model, paraphrase_tokenizer, text2paraphrase)
    logger.info("Paraphrased Text: {}".format(paraphrase_result))

    if not config['kw_model']:
        text2parse = keyword_template.format(paraphrase_result)
        query_keywords = paraphrase_model_inference(kw_model, kw_tokenizer, text2parse)
        query_keywords = eval(query_keywords)
    else:
        query_keywords = extract_key_phrases(paraphrase_result, args.query_text, kw_model)
    logger.info("Final Key Phrases: {}".format(query_keywords))

    input_uhr_image = Image.open(input_uhr_image_path)
    width, height = input_uhr_image.size
    logger.info("original image width and height: {}, {}".format(width, height))

    # patchify -> padded image and dict of bbox - patch save name
    img_resize, coordinate_patchname_dict = img_2patch(
        input_uhr_image,
        c_denom=10,
        dump_imgs=True,
        patch_saving_dir=patch_saving_dir
    )
    logger.info("resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))

    vlm_image_feats, vlm_text_feats, bbox_coordinate_list = extract_vlm_img_text_feat(args.query_text, query_keywords, coordinate_patchname_dict, patch_saving_dir, img_preprocess, text_tokenizer, fast_path_vlm, img_batch_size=50)
    t2p_similarity = calculate_similarity_matrix(vlm_image_feats, vlm_text_feats, fast_path_vlm.logit_scale.exp())
    ranked_patch, corresponding_similarity = ranking_patch(bbox_coordinate_list, t2p_similarity, top_k=5)
    logger.info("Ranked Patch Shape: {}".format(ranked_patch.shape))
    logger.info("Corresponding similarity: {}".format(corresponding_similarity))

    # for patch_coord in ranked_patch:
    #     img_name = coordinate_patchname_dict[tuple(patch_coord.tolist())]
    #     img_path = os.path.join(patch_saving_dir, img_name)
    #     img = Image.open(img_path)
    #     img.show()

    fast_path_vlm.to("cpu")
    kw_model.to("cpu")

    if max(corresponding_similarity) < fast_path_T:
        from langchain.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("fast path similarity does not meet the threshold, choose the slow path")
        # slow path
        slow_text_model_path = "/media/zilun/wd-161/hf_download/all-MiniLM-L6-v2"
        text_expand_model = paraphrase_model
        text_expand_tokenizer = paraphrase_tokenizer
        expanded_query_text_list = text_expand_model_inference(text_expand_model, text_expand_tokenizer, query_keywords)
        # slow_text_encoder = SentenceTransformer(slow_text_model_path)
        # query_keyword_embeddings = slow_text_encoder.encode(expanded_query_text_list)
        embeddings = HuggingFaceEmbeddings(model_name=slow_text_model_path)
        # Create chroma
        vectorstore = Chroma(
            collection_name="vector_store4keyphrase_label_matching",
            embedding_function=embeddings,
            persist_directory="/media/zilun/wd-161/ImageRAG_database/chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
        vectorstore.add_texts(texts=expanded_query_text_list)

        ranked_patch = [[0, 0, img_resize.width, img_resize.height]]

    response = vqa_llm.free_form_inference(img_resize, args.query_text, ranked_patch)
    logger.info(response)


if __name__ == "__main__":
    main()