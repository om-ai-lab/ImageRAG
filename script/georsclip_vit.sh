#!/bin/bash

# 定义参数范围和步长
start_path_T=0.3
end_path_T=0.7
increment_path_T=0.2

start_lrsd_T=0.1
end_lrsd_T=0.3
increment_lrsd_T=0.1

start_crsd_T=0.1
end_crsd_T=0.5
increment_crsd_T=0.2

# 循环生成所有可能的参数组合
for path_T in $(seq $start_path_T $increment_path_T $end_path_T); do
    for lrsd_T in $(seq $start_lrsd_T $increment_lrsd_T $end_lrsd_T); do
        for crsd_T in $(seq $start_crsd_T $increment_crsd_T $end_crsd_T); do
            # 构建完整的命令并运行
            CUDA_VISIBLE_DEVICES=$2 python codebase/main_inference_mmerealworld_imagerag_preextract.py \
                --cfg_path config/config_mmerealworld-imagerag-zoom4kvqa10k2epoch_vit_georsclip_server.yaml \
                --path_T $path_T \
                --lrsd_T $lrsd_T \
                --crsd_T $crsd_T \
                --reduce_fn $1
            echo "Finished processing path_T=$path_T, lrsd_T=$lrsd_T, crsd_T=$crsd_T"
        done
    done
done