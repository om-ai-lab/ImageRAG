import numpy as np
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, process_images_with_multi_ratio
from llava.conversation import conv_templates
import torch


class VQA_LLM:
    def __init__(self, model_path, conv_type="qwen_1_5", device="cuda:0"):
        model_name = model_path.split("/")[-1]
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, device_map=device)
        self.device = device
        self.conv_type = conv_type
        self.patch_size = self.image_processor.crop_size['width']
        self._init_templates()
    
    def _init_templates(self):
        self.additional_template = 'Additional visual information to focus on: '
        self.patch_template = 'a sub-patch ' + DEFAULT_IMAGE_TOKEN + ' at location [{:.3f},{:.3f},{:.3f},{:.3f}]'
        self.seg_tag = "; "
    
    def get_patch(self, bbox, image_width, image_height, patch_size=224, patch_scale=None):
        object_width = int(np.ceil(bbox[2]))
        object_height = int(np.ceil(bbox[3]))

        object_center_x = int(bbox[0] + bbox[2]/2)
        object_center_y = int(bbox[1] + bbox[3]/2)

        if patch_scale is None:
            patch_width = max(object_width, patch_size)
            patch_height = max(object_height, patch_size)
        else:
            patch_width = int(object_width*patch_scale)
            patch_height = int(object_height*patch_scale)

        left = max(0, object_center_x-patch_width//2)
        right = min(left+patch_width, image_width)

        top = max(0, object_center_y-patch_height//2)
        bottom = min(top+patch_height, image_height)

        return [left, top, right, bottom]

    def normalize_bbox(self, bbox, image_width, image_height):
        normalized_bbox = [bbox[0]/image_width, bbox[1]/image_height, bbox[2]/image_width, bbox[3]/image_height]
        normalized_bbox = [np.clip(_, 0, 1) for _ in normalized_bbox]
        return normalized_bbox
    
    def get_prompt_and_patches(self, image: Image.Image, question, bbox_list):
        # get mid information
        mid_information = ''
        all_patches = []
        image_sizes = []
        if len(bbox_list) > 0:
            mid_information = self.additional_template
            for i in range(len(bbox_list)):
                resized_bbox = self.get_patch(bbox_list[i], image.width, image.height, self.patch_size)
                normalized_bbox = self.normalize_bbox(resized_bbox, image.width, image.height)
                mid_information += self.patch_template.format(normalized_bbox[0], normalized_bbox[1], normalized_bbox[2], normalized_bbox[3])
                if i != len(bbox_list)-1:
                    mid_information += self.seg_tag
                all_patches.append(image.crop(resized_bbox))
                image_sizes.append((resized_bbox[2]-resized_bbox[0], resized_bbox[3]-resized_bbox[1]))

        conv = conv_templates[self.conv_type].copy()
        qs = DEFAULT_IMAGE_TOKEN + '\n' + mid_information + '\n' + question	
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        return prompt, all_patches, image_sizes

    
    def free_form_inference(self, image: Image.Image, question, bbox_list):
        prompt, all_patches, image_sizes = self.get_prompt_and_patches(image, question, bbox_list)
        image_and_patches = [image] + all_patches   
        image_tensor = process_images_with_multi_ratio(image_and_patches, self.image_processor, ['anyres', 'pad'], self.model.config)
        image_tensor = [t.to(dtype=torch.float16, device=self.device) for t in image_tensor]
        image_sizes.insert(0, image.size)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)
        
        output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return outputs

        
