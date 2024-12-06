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
                   text_expand_model_inference, setup_logger, img_reduce, load_yaml)
from cc_algo import img_2patch, vis_patches
from text_parser import extract_key_phrases
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import argparse
import logging
from llm_template import paraphrase_template, keyword_template, text_expansion_template
from sglang_util import get_paraphase_response, get_keyword_response, get_text_expansion_response
import openai


Image.MAX_IMAGE_PIXELS = None


def main():
    # L.seed_everything(2024)
    # input config
    # input_uhr_image_path = "/media/zilun/fanxiang4t/GRSM/IRAG/imageRAG/data/dqx_21n_chengguo.tif"

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='../config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--query_text', type=str, default='Suppose the top of this image represents north. How many aircraft are heading northeast? What is the color of the building rooftop to their southeast?', help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default='../log', help='Path to the log file.')
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

    paraphrase_model_config = config['paraphrase_model']
    kw_model_config = config['kw_model']
    text_expansion_model_config = config['text_expansion_model']
    fast_vlm_model_path = config['fast_vlm_model']['model_path']
    llmvqa_model_path = config['llmvqa_model']['model_path']

    client = openai.Client(base_url=args.base_url, api_key="None")
    fast_path_vlm, img_preprocess, text_tokenizer = setup_vlm_model(fast_vlm_model_path, device)
    vqa_llm = setup_vqallm(llmvqa_model_path, model_input_image_size=384, device=device)

    logger.info("Original Input Query: {}".format(args.query_text))

    paraphrase_result = get_paraphase_response(client, paraphrase_model_config['model_path'], args.query_text, paraphrase_model_config['generation_config'])
    logger.info("Paraphrased Text: {}".format(paraphrase_result))

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

    # Fast Path
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

    fast_path_vlm = fast_path_vlm.to("cpu")
    # kw_model.to("cpu")

    # Slow Path
    if max(corresponding_similarity) < fast_path_T:
        from langchain.vectorstores import Chroma, FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        import chromadb
        from vector_database.georsclip_embedding import GeoRSCLIPEmbeddings
        logger.info("fast path similarity does not meet the threshold, choose the slow path")

        if not text_expansion_model_config:
            text_expansion_model_config = paraphrase_model_config
            expanded_query_text_dict = get_text_expansion_response(client, text_expansion_model_config['model_path'], query_keywords, text_expansion_model_config['generation_config'])
        else:
            print("Not Implemented")
            exit()
        # Uncomment if not use phrase expansion
        # expanded_query_text_list = query_keywords
        embedding = GeoRSCLIPEmbeddings(checkpoint=config['fast_vlm_model']['model_path'], device=device)

        vectorstore = Chroma(
            collection_name="mm_georsclip",
            embedding_function=embedding,
            persist_directory=config["vector_database"]["mm_vector_database_dir"],
            collection_metadata = {"hnsw:space": "cosine"}
        )

        # Create Chroma vector store
        # slow_text_emb_model_path = "/media/zilun/wd-161/hf_download/all-MiniLM-L6-v2"
        # embeddings = HuggingFaceEmbeddings(model_name=slow_text_emb_model_path)
        # text_vectorstore = Chroma(
        #     collection_name="vector_store4keyphrase_label_matching",
        #     embedding_function=text_embeddings,
        #     persist_directory=config["vector_database"]["text_vector_database_dir"],  # Where to save data locally, remove if not necessary
        # )

        meta_pkl_path = config["vector_database"]["meta_pkl_path"]
        meta_vector_database_df = pkl.load(open(meta_pkl_path, "rb"))
        labels_in_database = list(set(meta_vector_database_df["cls_list"].tolist()))
        meta = [{'type': 'text'}] * len(labels_in_database)
        vectorstore.add_texts(texts=labels_in_database, metadatas=meta)

        selected_label_name = []
        for expanded_query_text in expanded_query_text_dict:
            results = vectorstore.similarity_search_with_score(
                expanded_query_text, k=5
            )
            for res, score in results:
                # * [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}]
                print(f"Query:={expanded_query_text_dict[expanded_query_text]} * [SIM={score:3f}] {res.page_content}")
                selected_label_name.append(res.page_content)
            print()
        selected_label_name = list(set(selected_label_name))
        meta_vector_database_df_selected = meta_vector_database_df[meta_vector_database_df['cls_list'].isin(selected_label_name)]
        reduced_img_group = img_reduce(meta_vector_database_df_selected, config['vector_database']['img_dir'])
        for cls in reduced_img_group:
            cls_img_paths = reduced_img_group[cls][:10]
            meta = [{'label': cls, 'type': 'image'}] * len(cls_img_paths)
            vectorstore.add_images(uris=cls_img_paths, metadatas=meta)
        print()

        ranked_patch = [[0, 0, img_resize.width, img_resize.height]]

    response = vqa_llm.free_form_inference(img_resize, args.query_text, ranked_patch)
    logger.info(response)


if __name__ == "__main__":
    main()