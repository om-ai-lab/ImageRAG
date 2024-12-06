import os
import pdb

import open_clip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ColorJitter
from PIL import Image
from collections import Counter
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import yaml
import pickle as pkl


def load_yaml(config_filepath):
    # Load the YAML file
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def collect_fn(batch):
    batch_img_list = []
    bbox_coordinate_list = []
    for data in batch:
        batch_img, bbox_coordinate = data
        batch_img_list.append(batch_img.unsqueeze(0))
        bbox_coordinate_list.append(bbox_coordinate)
    batch_img_list = torch.cat(batch_img_list)
    return batch_img_list, bbox_coordinate_list


def extract_vlm_img_text_feat(query, key_text, coordinate_patchname_dict, patch_saving_dir, img_preprocess, text_tokenizer, fast_path_vlm, img_batch_size):
    device = fast_path_vlm.logit_scale.device
    patch_dataset = CCDataset(coordinate_patchname_dict, patch_saving_dir, img_preprocess)
    patch_dataloader = DataLoader(patch_dataset, pin_memory=True, batch_size=img_batch_size, num_workers=os.cpu_count() // 2, shuffle=False, collate_fn=collect_fn)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_feature_list = []
        bbox_coordinate_list = []
        for batch_img, bbox_coordinate in tqdm(patch_dataloader):
            batch_img = batch_img.to(device)
            image_features = fast_path_vlm.encode_image(batch_img)
            image_feature_list.append(image_features)
            bbox_coordinate_list.extend(bbox_coordinate)
        image_features = torch.cat(image_feature_list)

        # text_content = [text_tokenizer(query)] + [text_tokenizer(f"a photo of the {c}") for c in key_text]
        # text_content = [text_tokenizer(f"a photo of the {c}") for c in key_text]
        # text_content = [text_tokenizer(query)]
        prompt = "a photo includes"
        for c in key_text:
            # c = c.replace("the ", "")
            if c != key_text[-1]:
                prompt += "{}, ".format(c)
            else:
                prompt += "and {}.".format(c)
        print(prompt)
        # text_content = [text_tokenizer(query)] + [text_tokenizer(prompt)]
        text_content = [text_tokenizer(prompt)] + [text_tokenizer(f"a photo of the {c}") for c in key_text]
        text_inputs = torch.cat(text_content).to(device)
        text_features = fast_path_vlm.encode_text(text_inputs)

    return image_features, text_features, bbox_coordinate_list


def set_up_paraphrase_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def paraphrase_model_inference(model, tokenizer, query_text):
    prompt = query_text
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def setup_vlm_model(model_path, device):
    model, _, img_preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-L-14-336-quickgelu',
        pretrained='openai',
        precision="fp16",
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14-336-quickgelu')
    checkpoint = torch.load(model_path, map_location=device)
    msg = model.load_state_dict(checkpoint, strict=False)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    return model, img_preprocess, tokenizer


def setup_slow_text_encoder_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def setup_vqallm(llmvqa_model_path, model_input_image_size, device):
    from vqa_llm.vaq_llm import VQA_LLM
    import warnings
    warnings.filterwarnings("ignore")
    vqa_llm = VQA_LLM(
        model_path=llmvqa_model_path,
        device=device
    )
    return vqa_llm

def calculate_similarity_matrix(img_feats, text_feats, logit_scale_exp, need_feat_normalize=True):
    img_feats = img_feats.detach().cpu().type(torch.float32)
    text_feats = text_feats.detach().cpu().type(torch.float32)
    logit_scale_exp = logit_scale_exp.detach().cpu()
    with torch.no_grad(), torch.cuda.amp.autocast():
        if need_feat_normalize:
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
        text2patch_similarity = (logit_scale_exp * img_feats @ text_feats.t()).t()
    return text2patch_similarity


def ranking_patch_t2p(bbox_coordinate_list, t2p_similarity, top_k=10):
    values, index = t2p_similarity.topk(top_k)
    # should be 5 * 3 = 15 candidates
    # top1patch_per_keyphrase = values[:, :1].flatten().tolist()
    flat_values = values.flatten().tolist()
    flat_index = index.flatten().tolist()
    counter = Counter(flat_index)
    top_k_element_index = counter.most_common(top_k)
    top_k_element_index = [element for element, count in top_k_element_index]

    select_index_similarity_dict = dict()
    for i, i_element in enumerate(top_k_element_index):
        for j, j_element in enumerate(flat_index):
            if i_element == j_element:
                if i_element not in select_index_similarity_dict:
                    select_index_similarity_dict[i_element] = flat_values[j]
                else:
                    if select_index_similarity_dict[i_element] < flat_values[j]:
                        select_index_similarity_dict[i_element] = flat_values[j]

    index_select = list(select_index_similarity_dict.keys())
    similarity_select = list(select_index_similarity_dict.values())

    top1patch_value_per_keyphrase = values[:, :1].flatten().tolist()
    top1patch_index_per_keyphrase = index[:, :1].flatten().tolist()
    assert set(top_k_element_index) == set(index_select)

    select_top1_value_list = []
    select_top1_index_list = []
    for i, top1_index in enumerate(top1patch_index_per_keyphrase):
        if top1_index not in top_k_element_index:
            select_top1_value_list.append(top1patch_value_per_keyphrase[i])
            select_top1_index_list.append(top1_index)

    candidate_index = index_select + select_top1_index_list
    candidate_similarity = similarity_select + select_top1_value_list

    selected_bbox_coordinate_list = np.array(bbox_coordinate_list)[candidate_index]

    return selected_bbox_coordinate_list, candidate_similarity


def text_expand_model_inference(model, tokenizer, query_text):
    response_list = []
    for text in query_text:
        prompt =  "Could you write a few sentences (output in string) to explain the phrase I give you? Provide a detailed explanation including synonyms (if they exist) of the phrase. The explanation should be comprehensive enough to match the phrase with relevant class names based on sentence embedding similarity. Below are a few examples of phrases and their corresponding explanations: \n User: Airport \n Assistant: An airport is a facility where aircraft, such as airplanes and helicopters, take off, land, and are serviced. It typically consists of runways, taxiways, terminals for passenger and cargo handling, control towers for air traffic management, and hangars for aircraft maintenance. Airports serve as hubs for commercial aviation, connecting travelers and goods between local and international destinations. Synonyms for 'airport' include airfield, aerodrome, airstrip, and terminal (in certain contexts, referring specifically to the passenger facilities. \n Follow this structure for any phrase provided. User: {}".format(text)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response_list.append(response)
    return response_list


def setup_logger(log_path):
    logging.basicConfig(level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('my_logger')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def meta_df2clsimg_dict(meta_vector_database_df_selected, vectordatabase_dir):
    """
    Input: meta df
    Output: {
                cls1: [img_path1, img_path2, ... ],
                cls2: [img_path1, img_path2, ... ],
                ......
            }
    """
    meta_vector_database_df_selected['img_path_list'] = meta_vector_database_df_selected.apply(
        lambda row: os.path.join(vectordatabase_dir, row['dataset'], row['img_name_list']), axis=1
    )
    cls_img_dict = meta_vector_database_df_selected.groupby('cls_list')['img_path_list'].apply(list).to_dict()

    return cls_img_dict


def img_reduce(cls_img_dict, vlm, img_preprocess, reduce_fn="mean"):
    device = vlm.logit_scale.device

    def batch_inference_clip_vision_encoder(cls_img_dataloader):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_feature_list = []
            for batch_img, bbox_coordinate in tqdm(cls_img_dataloader):
                batch_img = batch_img.to(device)
                image_features = vlm.encode_image(batch_img)
                image_feature_list.append(image_features)
            image_features = torch.cat(image_feature_list)
        return image_features

    cls_feat_dict = dict()
    for class_label in tqdm(cls_img_dict):
        img_list =  cls_img_dict[class_label]
        cls_dataset = VanillaDataset(img_list, img_preprocess)
        cls_dataloader = DataLoader(cls_dataset, pin_memory=True, batch_size=20, num_workers=8, shuffle=False)
        cls_image_features = batch_inference_clip_vision_encoder(cls_dataloader)
        cls_feat_dict[class_label] = cls_image_features
    pkl.dump(cls_feat_dict, open("cls_img_dict.pkl", "wb"))

    if reduce_fn == "mean":
        for class_label in tqdm(cls_feat_dict):
            cls_feat_dict[class_label] = cls_feat_dict[class_label].mean(-1)

    return cls_feat_dict

def ranking_patch_visualcue2patch(bbox_coordinate_list, visualcue2patch_similarity, top_k=10):
    values, index = visualcue2patch_similarity.topk(top_k)
    # should be 5 * 3 = 15 candidates
    # top1patch_per_keyphrase = values[:, :1].flatten().tolist()
    flat_values = values.flatten().tolist()
    flat_index = index.flatten().tolist()
    counter = Counter(flat_index)
    top_k_element_index = counter.most_common(top_k)
    top_k_element_index = [element for element, count in top_k_element_index]

    select_index_similarity_dict = dict()
    for i, i_element in enumerate(top_k_element_index):
        for j, j_element in enumerate(flat_index):
            if i_element == j_element:
                if i_element not in select_index_similarity_dict:
                    select_index_similarity_dict[i_element] = flat_values[j]
                else:
                    if select_index_similarity_dict[i_element] < flat_values[j]:
                        select_index_similarity_dict[i_element] = flat_values[j]

    index_select = list(select_index_similarity_dict.keys())
    similarity_select = list(select_index_similarity_dict.values())

    top1patch_value_per_keyphrase = values[:, :1].flatten().tolist()
    top1patch_index_per_keyphrase = index[:, :1].flatten().tolist()
    assert set(top_k_element_index) == set(index_select)

    select_top1_value_list = []
    select_top1_index_list = []
    for i, top1_index in enumerate(top1patch_index_per_keyphrase):
        if top1_index not in top_k_element_index:
            select_top1_value_list.append(top1patch_value_per_keyphrase[i])
            select_top1_index_list.append(top1_index)

    candidate_index = index_select + select_top1_index_list
    candidate_similarity = similarity_select + select_top1_value_list

    selected_bbox_coordinate_list = np.array(bbox_coordinate_list)[candidate_index]
    return selected_bbox_coordinate_list, candidate_similarity

def select_visual_cue(vlm_image_feats, bbox_coordinate_list, visual_cue_candidates_dict, need_feat_normalize=True):
    # (166, 512)
    patch_feats = vlm_image_feats.detach().cpu().type(torch.float32)
    # (3, 512)
    visual_cue_feats = visual_cue_candidates_dict.values().detach().cpu().type(torch.float32)
    with torch.no_grad(), torch.cuda.amp.autocast():
        if need_feat_normalize:
            patch_feats /= patch_feats.norm(dim=-1, keepdim=True)
            visual_cue_feats /= visual_cue_feats.norm(dim=-1, keepdim=True)
        visualcue2patch_similarity = (patch_feats @ visual_cue_feats.t()).t()

    visual_cues = ranking_patch_visualcue2patch(bbox_coordinate_list, visualcue2patch_similarity)
    return visual_cues


class VanillaDataset(Dataset):
    def __init__(self, img_path_list, transform):
        self.img_path_list = img_path_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path_list)


class CCDataset(Dataset):
    def __init__(self, coordinate_patchname_dict, img_dir, transform):
        self.img_dir = img_dir
        self.coordinate_patchname_dict = coordinate_patchname_dict
        self.data = list(coordinate_patchname_dict.keys())
        self.transform = transform

    def __getitem__(self, index):
        bbox_coordinate = self.data[index]
        sample_name = self.coordinate_patchname_dict[bbox_coordinate]
        img_path = os.path.join(self.img_dir, sample_name)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, bbox_coordinate

    def __len__(self):
        return len(self.data)