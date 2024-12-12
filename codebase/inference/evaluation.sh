########  Evaluation
model_name='skysensegpt-fullft-1e6'
#model_path='/media/zilun/fanxiang4t/GRSM/GeoChat/skysensegpt-lora-geochat'
model_path='/media/zilun/fanxiang4t/GRSM/SkySenseGPT/result/skysensegpt-fullft-1e6'
#model_base='/media/zilun/mx500/2-year-work/MLLM_without_forgetting/LLaVA/checkpoints/llava-v1.5-7b'

#model_name='geochat-7b'
#model_path='/media/zilun/fanxiang4t/GRSM/GeoChat/model/geochat-7B'
#model_base=''

img_path='/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/FIT/FIT-RS/FIT-RS_Instruction/FIT-RS_Img/imgv2_split_512_100_vaild'
hrben_img_path='/media/zilun/fanxiang4t/GRSM/evaluation_dataset/VQA_VG/GeoChat-Bench/dataset/HRBEN/Images'
FIT_RSFG_Bench_path='/media/zilun/fanxiang4t/GRSM/ov/inference/rsvqa/FIT-RSFG/FIT-RSFG-Bench'
FIT_RSFC_Bench_path='/media/zilun/fanxiang4t/GRSM/ov/inference/rsvqa/FIT-RSRC'
result_path='/media/zilun/fanxiang4t/GRSM/ov/inference/rsvqa/result'


## FIT-RSFG Scene Classification
#python geochat/eval/batch_geochat_sceneclassification.py \
#    --model-path ${model_path} \
#    --question-file ${FIT_RSFG_Bench_path}/test_FITRS_imageclassify_eval.jsonl \
#    --answers-file ${result_path}/${model_name}/FITRS_imageclassify_eval_${model_name}.jsonl \
#    --image-folder ${img_path}


## FIT-RSFG ImageCaption
#python geochat/eval/batch_geochat_caption.py \
#    --model-path ${model_path} \
#    --question-file ${FIT_RSFG_Bench_path}/test_FITRS_image_caption_eval.jsonl \
#    --answers-file ${result_path}/${model_name}/FITRS_image_caption_answer_${model_name}.jsonl \
#    --image-folder ${img_path}


## FIT-RSFG RegionCaption
#python geochat/eval/batch_geochat_caption.py \
#    --model-path ${model_path} \
#    --question-file ${FIT_RSFG_Bench_path}/test_FITRS_region_caption_eval.jsonl \
#    --answers-file ${result_path}/${model_name}/FITRS_region_caption_answer_${model_name}.jsonl \
#    --image-folder ${img_path}


## FIT-RSFG VQA
#python geochat/eval/batch_geochat_vqa.py \
#    --model-path ${model_path} \
#    --question-file ${FIT_RSFG_Bench_path}/test_FITRS_vqa_eval.jsonl \
#    --answers-file ${result_path}/${model_name}/FITRS_vqa_eval_${model_name}.jsonl \
#    --image-folder ${img_path}


# FIT-RSFG ComplexComprehension
python geochat/eval/batch_geochat_complex_compre.py \
    --model-path ${model_path} \
    --question-file ${FIT_RSFG_Bench_path}/test_FITRS_complex_comprehension_eval.jsonl \
    --answers-file ${result_path}/${model_name}/FITRS_complex_comprehension_eval_${model_name}.jsonl \
    --image-folder ${img_path}

## FIT-RSRC Single-Choice
#python geochat/eval/batch_fitrsrc_single_choice_qa.py \
#    --model-path ${model_path} \
#    --question-file ${FIT_RSFC_Bench_path}/FIT-RSRC_Questions_2k.jsonl \
#    --answers-file ${result_path}/${model_name}/FIT-RSRC_singlechoice_eval_${model_name}.jsonl \
#    --image-folder ${img_path}


####### Evaluation ######
## eval FIT-RSFG Caption
#python FIT-Eval/pycocoevalcap/eval_custom_caption.py \
#       --root_path ${result_path} \
#       --model_answers_file_list \
#       ${result_path}/${model_name}/FITRS_image_caption_answer_${model_name}.jsonl \
#       ${result_path}/${model_name}/FITRS_region_caption_answer_${model_name}.jsonl


# eval FIT-RSFG ComplexComprehension
python FIT-Eval/eval_complex_comprehension.py \
       --answer-file ${result_path}/${model_name}/FITRS_complex_comprehension_eval_0.001_${model_name}.jsonl


######
## VQA-HRBEN
#python geochat/eval/batch_geochat_vqa_hrben.py \
#    --model-path ${model_path} \
#    --question-file ${FIT_RSFG_Bench_path}/hrben.jsonl \
#    --answers-file ${result_path}/${model_name}/HRBEN_answers_fgrs_${model_name}.jsonl \
#    --image-folder ${hrben_img_path}/Data


## eval HRBEN(RSVQA-HR)
#python FIT-Eval/eval_vqa_HRBEN.py \
#    --answer-file ${result_path}/${model_name}/HRBEN_answers_fgrs_${model_name}.jsonl \
#    --output-file ${result_path}/${model_name}/HRBEN_answers_fgrs_${model_name}_combined.jsonl \
#    --questions-file FIT-Eval/HRBEN/USGS_split_test_phili_questions.json \
#    --answers-gt-file FIT-Eval/HRBEN/USGS_split_test_phili_answers.json
