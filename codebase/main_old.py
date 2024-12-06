import pdb
import shutil
from tqdm import tqdm
import os
import uuid
import numpy as np
import pickle as pkl
from PIL import Image
from utils import (setup_vlm_model, set_up_paraphrase_model, setup_vqallm, setup_slow_text_encoder_model,
                   calculate_similarity_matrix, extract_vlm_img_text_feat, ranking_patch, paraphrase_model_inference,
                   text_expand_model_inference)
from cc_algo import img_2patch, vis_patches
from text_parser import extract_key_phrases
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
# import lightning as L

# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
# from langchain_ollama import ChatOllama
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_experimental.open_clip import OpenCLIPEmbeddings

Image.MAX_IMAGE_PIXELS = None



def main():
    # L.seed_everything(2024)
    # input config
    # input_uhr_image_path = "/media/zilun/fanxiang4t/GRSM/IRAG/imageRAG/data/dqx_21n_chengguo.tif"
    input_uhr_image_path = "/media/zilun/fanxiang4t/GRSM/IRAG/image/paper_teaser.png"
    patch_saving_dir = "patch_save_dir"
    os.makedirs(patch_saving_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_input_image_size = 336
    fast_path_T = 70

    kw_model_path = "/media/zilun/wd-161/hf_download/all-MiniLM-L6-v2"
    paraphrase_model_path = "/media/zilun/wd-161/hf_download/Qwen2.5-3B-Instruct"
    fast_vlm_model_path = "/media/zilun/wd-161/RS5M/RS5M_codebase/ckpt/RS5M_ViT-L-14-336.pt"
    llmvqa_model_path = "/media/zilun/fanxiang4t/GRSM/IRAG/imageRAG/ckpt/ckpt_llava-onevision-qwen2-0.5b-ov"

    kw_model = KeyBERT(model=SentenceTransformer(kw_model_path))
    paraphrase_model, paraphrase_tokenizer = set_up_paraphrase_model(paraphrase_model_path, device)
    fast_path_vlm, img_preprocess, text_tokenizer = setup_vlm_model(fast_vlm_model_path, device)
    vqa_llm = setup_vqallm(llmvqa_model_path, model_input_image_size=384, device=device)

    query_text = "Suppose the top of this image represents north. How many aircraft are heading northeast? What is the color of the building rooftop to their southeast?"
    # query_text = "Suppose the top of this image represents north. How many aircrafts are heading northeast? What is the color of the building rooftop to their southeast?"
    print(query_text)
    print()

    paraphrase_result = paraphrase_model_inference(paraphrase_model, paraphrase_tokenizer, query_text)
    # paraphrase_result = query_text
    print(paraphrase_result)
    print()

    query_keywords = extract_key_phrases(paraphrase_result, query_text, kw_model)
    # query_keywords = ["aircrafts", "building rooftop", "aircrafts heading northeast"]
    print("Question: {}".format(query_text))
    print("Final Key Phrases")
    print(query_keywords)
    print()

    input_uhr_image = Image.open(input_uhr_image_path)
    width, height = input_uhr_image.size
    print("original image width and height: {}, {}".format(width, height))

    # patchify -> padded image and dict of bbox - patch save name
    img_resize, coordinate_patchname_dict = img_2patch(
        input_uhr_image,
        c_denom=10,
        dump_imgs=True,
        patch_saving_dir=patch_saving_dir
    )
    print("resize image to width and height: {}, {}, for patchify.".format(img_resize.size[0], img_resize.size[1]))

    vlm_image_feats, vlm_text_feats, bbox_coordinate_list = extract_vlm_img_text_feat(query_text, query_keywords, coordinate_patchname_dict, patch_saving_dir, img_preprocess, text_tokenizer, fast_path_vlm, img_batch_size=50)
    t2p_similarity = calculate_similarity_matrix(vlm_image_feats, vlm_text_feats, fast_path_vlm.logit_scale.exp())
    ranked_patch, corresponding_similarity = ranking_patch(bbox_coordinate_list, t2p_similarity, top_k=5)
    print("Ranked Patch Shape: {}".format(ranked_patch.shape))
    print("Corresponding similarity: {}".format(corresponding_similarity))

    # for patch_coord in ranked_patch:
    #     img_name = coordinate_patchname_dict[tuple(patch_coord.tolist())]
    #     img_path = os.path.join(patch_saving_dir, img_name)
    #     img = Image.open(img_path)
    #     img.show()

    # paraphrase_model = paraphrase_model.to("cpu")
    fast_path_vlm = fast_path_vlm.to("cpu")

    if max(corresponding_similarity) < fast_path_T:
        from langchain.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        print("fast path similarity does not meet the threshold, choose the slow path")
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




    response = vqa_llm.free_form_inference(img_resize, query_text,
                                           ranked_patch
                                           )
    print(response)


if __name__ == "__main__":
    main()