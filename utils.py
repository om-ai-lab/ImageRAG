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
    patch_dataset = VanillaDataset(coordinate_patchname_dict, patch_saving_dir, img_preprocess)
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
                prompt += " {},".format(c)
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
    prompt = "Question: '{}'. Please deduce the key factors in this question that is important. Output a sentence.".format(query_text)
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
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14-336-quickgelu')
    checkpoint = torch.load(model_path, map_location=device)
    msg = model.load_state_dict(checkpoint, strict=False)
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


def ranking_patch(bbox_coordinate_list, t2p_similarity, top_k=10):
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


class VanillaDataset(Dataset):
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