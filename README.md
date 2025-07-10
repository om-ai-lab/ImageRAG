# ImageRAG
<img src="teaser.png" alt="imagerag" width="90%">

<font size=4><div align='center' > [[🤗 Checkpoint, cache and data](https://huggingface.co/omlab/ImageRAG)] [[📄 Paper](https://ieeexplore.ieee.org/document/11039502)] [[📝 Blog (in Mandarin)](https://mp.weixin.qq.com/s/BcFejPAcxh4Rx_rh1JvRJA)]</div></font> 
## ✨ Highlight

Ultrahigh resolution (UHR) remote sensing imagery (RSI) (e.g. 10,000 X 10,000 pixels) poses a significant challenge for current RS vision-language models (RSVLMs). If one chooses to resize the UHR image to the standard in
put image size, the extensive spatial and contextual information that UHR images contain will be neglected. Otherwise, the original size of these images often exceeds the token limits of standard RSVLMs, making it difficult to pro
cess the entire image and capture long-range dependencies to answer the query based on the abundant visual context. 

* Three crucial aspects for MLLMs to effectively handle UHR RSI are:
  
    * Managing small targets, ensuring that the model can accurately be aware and analyze fine details 
  within images
  
    * Processing the UHR image in a way that integrates with MLLMs without significantly increasing 
  the number of image tokens, which would lead to high computational costs
  
    * Achieving these goals while minimizing the need for additional training or specialized 
  annotation.


*  We contribute the ImageRAG framework, which offers several key advantages as follows:

    * It retrieves and emphasizes relevant visual context from the UHR image based on the text query, allowing the MLLM to focus on important details, even tiny ones.
      
    * It integrates various external knowledge sources (store in vector database) to guide the model, enhancing the understanding of the query and UHR RSI
      
    * ImageRAG requires only a small amount of training, making it a practical solution for efficiently handling UHR RSI.



## 🚀 Update
🔥🔥🔥 Last Updated on 2025.07.10 🔥🔥🔥

* **TODO**: Validate the codebase using uploaded data in isolate enviromemt

* **2025.07.10**: Upload checkpoint, cache and dataset.

* **2025.06.25**: Upload codebase and scripts.
  
* **2025.05.24**: ImageRAG is accepted by IEEE Geoscience and Remote Sensing Magazine 🎉
  * IEEE Early Access (we prefer this version): https://ieeexplore.ieee.org/document/11039502
    
  * Arxiv: https://arxiv.org/abs/2411.07688


## 📖 Setup Codebase and Data

* Clone this repo:
   ```bash
   git clone https://github.com/om-ai-lab/ImageRAG.git
   ```
   
* Download data, caches and checkpoints for ImageRAG from huggingface: 
   * https://huggingface.co/omlab/ImageRAG
      
   * Using the [hf mirror](https://hf-mirror.com/) if you encounter connection problems:
       ```bash
       ./hfd.sh omlab/ImageRAG --local-dir ImageRAG_hf
       ```
   * Merge two repos:
      ```bash
      mv ImageRAG_hf/cache ImageRAG_hf/checkpoint ImageRAG_hf/data  ImageRAG/
      ```
   * Unzip all zip files:
      * cache/patch/mmerealworld.zip
      * cache/vector_database/crsd_vector_database.zip
      * cache/vector_database/lrsd_vector_database.zip

   * The ImageRAG directory structure should look like this:
        ```bash
            /training/zilun/ImageRAG

            ├── codebase                        
                ├── inference
                ├── patchify
                ├── main_inference_mmerealworld_imagerag_preextract.py
                ......                                                     
            ├── config                        
                ├── config_mmerealworld-baseline-zoom4kvqa10k2epoch_server.yaml 
                ├── config_mmerealworld-detectiongt-zoom4kvqa10k2epoch_server.yaml 
                ......                                                      
            ├── data                        
                ├── dataset
                    ├── MME-RealWorld
                        ├── remote_sensing
                            ├── remote_sensing
                                ├── 03553_Toronto.png 
                                ......
                ├── crsd_clip_3M.pkl
                ......
            ├── cache                        
                ├── patch
                    ├── mmerealworld
                        ├── vit
                        ├── cc
                        ├── grid
                ├── vector_database 
                    ├── crsd_vector_database
                    ├── lrsd_vector_database
            ├── checkpoint                        
                ├── InternVL2_5-8B_lora32_vqa10k_zoom4k_2epoch_merged
                ......     
            ├── script                        
                ├── clip_cc.sh
                ......
        ```

## 📖 Setup Env

```bash
conda create -n imagerag python=3.10
conda activate imagerag
cd /training/zilun/ImageRAG
export PYTHONPATH=$PYTHONPATH:/training/zilun/ImageRAG
# Install torch, torchvision and flash attention accroding to your cuda version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ninja
MAX_JOBS=16 pip install flash-attn --no-build-isolation
```

```bash
pip install requirement.txt
```

```bash
python
>>> import nltk
>>> nltk.download('stopwords')

python -m spacy download en_core_web_sm
```

## 📖 Setup SGLang (Docker)
* Host Qwen2.5-32B-Instruct using SGLang for text parsing module

```bash
# Pull sglang docker (we use mirror just for speeding up the download process)
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:latest
docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:latest  docker.io/lmsysorg/sglang:latest

# docker load -i sglang.tar

bash script/sglang_start.sh
```


## 📖 Feature Extraction (Optional, use Ray to parallelize the process)
* Necessary if you need to run ImageRAG in cutomized data
  
```bash
ray start --head --port=6379
# extract patch features (e.g. MMERealworld-RS)
python codebase/ray_feat_extract_patch.py --ray_mode auto --num_runner 8
# extract image & text features for vector database (Section V-D, external data)
python codebase/ray_feat_extract_vectorstore.py --ray_mode auto --num_runner 8
# ray stop (optional)
```

## 📖 Inference

* See imagerag_result directory for result examples. 

### Run Baseline Inference (No ImageRAG, No GT, No Inference while detecting)
```bash
# inference
CUDA_VISIBLE_DEVICES=0 python codebase/main_inference_mmerealworld_imagerag_preextract.py --cfg_path config/config_mmerealworld-baseline-zoom4kvqa10k2epoch_server.yaml

# eval inference result
python codebase/inference/MME-RealWorld-RS/eval_your_results.py --results_file data/eval/mmerealworld_zoom4kvqa10k2epoch_baseline.jsonl
```

### Run Regular VQA Task Inference in Parallel

```bash
# {clip, georsclip, remoteclip, mcipclip} x {vit, cc, grid} x {rerank, mean, cluster} x {0, ... ,7}
bash script/georsclip_grid.sh rerank 0
```

### Run Inferring VQA Task Inference (No ImageRAG, No Inference while detecting, BBoxes are needed)
```bash
# inference
CUDA_VISIBLE_DEVICES=0 python codebase/main_inference_mmerealworld_imagerag_preextract.py --cfg_path config/config_mmerealworld-detectiongt-zoom4kvqa10k2epoch_server.yaml

# eval inference result
python codebase/inference/MME-RealWorld-RS/eval_your_results.py --results_file data/eval/mmerealworld_zoom4kvqa10k2epoch_baseline.jsonl
```

## 👨‍🏫 Contact
zilun.zhang@zju.edu.cn

## 🖊️ Citation
```bash
@ARTICLE{11039502,
  author={Zhang, Zilun and Shen, Haozhan and Zhao, Tiancheng and Guan, Zian and Chen, Bin and Wang, Yuhao and Jia, Xu and Cai, Yuxiang and Shang, Yongheng and Yin, Jianwei},
  journal={IEEE Geoscience and Remote Sensing Magazine}, 
  title={Enhancing Ultrahigh Resolution Remote Sensing Imagery Analysis With ImageRAG: A new framework}, 
  year={2025},
  volume={},
  number={},
  pages={2-27},
  keywords={Image resolution;Visualization;Benchmark testing;Training;Image color analysis;Standards;Remote sensing;Analytical models;Accuracy;Vehicle dynamics},
  doi={10.1109/MGRS.2025.3574742}}

```
