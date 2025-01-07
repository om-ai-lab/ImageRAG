import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12, use_dynamic=True):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if use_dynamic:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        print("Use Dynamic")
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, load_image, model_config, prompt):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.load_image = load_image
        self.model_config = model_config
        self.prompt = prompt

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

        image_path = os.path.join(self.image_folder, image_file)
        pixel_values = load_image(
            image_path,
            input_size=self.model_config.vision_config.image_size,
            max_num=self.model_config.max_dynamic_patch,
            use_dynamic=True
        ).to(torch.bfloat16).cuda()

        return pixel_values, prompt, pixel_values.size(0), line["Text"], line

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    image_tensor, prompt, num_patches_list, question_list, line = zip(*batch)
    image_tensors = torch.cat(image_tensor, dim=0)
    return image_tensors, prompt, num_patches_list, question_list, line


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=0, prompt=''):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )

    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = [question for question in questions if question["Subtask"] == "Remote Sensing"]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, load_image, model.config, args.batch_size, prompt=args.test_prompt)

    index = 0
    for (image_tensors, prompts, num_patches_list, question_text_only_list, line) in tqdm(data_loader):
        with torch.inference_mode():
            responses = model.batch_chat(
                tokenizer,
                image_tensors,
                num_patches_list=num_patches_list,
                questions=prompts,
                generation_config=generation_config
            )
        for i, response in enumerate(responses):
            if index % 100 == 0:
                print(f'Prompt: {prompts[i]}\n\n Output: {response}')
            line[i]['output'] = response
            ans_file.write(json.dumps(line[i]) + "\n")
            ans_file.flush()
            index += 1
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/media/zilun/fanxiang4t/GRSM/ImageRAG_git/checkpoint/InternVL2_5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/media/zilun/wd-161/hf_download/MME-RealWorld/rs_subset")
    parser.add_argument("--question-file", type=str, default="/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/eval/MME_RealWorld.json")
    parser.add_argument("--answers-file", type=str, default="/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/answer_mme-realworld-rs.jsonl")
    parser.add_argument("--conv-mode", type=str, default="internvl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")

    parser.add_argument(
        "--test-prompt",
        type=str,
        default="Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.",
    )
    args = parser.parse_args()
    print(args)

    eval_model(args)
    