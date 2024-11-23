from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
import warnings
warnings.filterwarnings("ignore")
import torch
from copy import deepcopy

# model_path = "/data9/shz/ckpt/llava-onevision-qwen2-0.5b-ov/"
model_path = "/data9/shz/project/llava-ov/LLaVA-NeXT/checkpoints/search_llava-onevision-qwen2-0.5b-ov-2"
model = LlavaQwenForCausalLM.from_pretrained(model_path, device_map="cuda:1")

# model.lm_head.weight = deepcopy(model.model.embed_tokens.weight)
# model.save_pretrained("/data9/shz/project/llava-ov/LLaVA-NeXT/checkpoints/search_llava-onevision-qwen2-0.5b-ov-2")