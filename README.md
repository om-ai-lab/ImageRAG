# ImageRAG

## Setup

```bash
# Install torch, torchvision and flash attention accroding to your cuda version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install ninja
MAX_JOBS=16 pip install flash-attn --no-build-isolation
```

```bash
pip install req.txt
```

```bash
python
>>> import nltk
>>> nltk.download('stopwords')

python -m spacy download en_core_web_sm
```


## Run Inference

```bash
conda activate imagerag
cd /data1/zilun/ImageRAG0226
export PYTHONPATH=$PYTHONPATH:/data1/zilun/ImageRAG0226
# {clip, georsclip, remoteclip, mcipclip} x {vit, cc, grid} x {rerank, mean, cluster} x {0, ... ,7}
bash script/georsclip_grid.sh rerank 7

```


## Run Baseline Inference (No ImageRAG, No GT, No Inference while detecting)
```bash
conda activate imagerag
cd /data1/zilun/ImageRAG0226

CUDA_VISIBLE_DEVICES=0 python codebase/main_inference_mmerealworld_imagerag_preextract.py --cfg_path /data1/zilun/ImageRAG0226/config/config_mmerealworld-baseline-zoom4kvqa10k5epoch_server.yaml

python codebase/inference/MME-RealWorld-RS/eval_your_results.py --results_file /data1/zilun/ImageRAG0226/data/eval/mmerealworld_zoom4kvqa10k5epoch_baseline.jsonl
```


## Run GT Inference (No ImageRAG, No Inference while detecting, BBoxes are needed)
```bash
conda activate imagerag
cd /data1/zilun/ImageRAG0226

CUDA_VISIBLE_DEVICES=0 python codebase/main_inference_mmerealworld_imagerag_preextract.py --cfg_path /data1/zilun/ImageRAG0226/config/config_mmerealworldlite-detectiongt-zoom4kvqa10k2epoch_server.yaml

python codebase/inference/MME-RealWorld-RS/eval_your_results.py --results_file /data1/zilun/ImageRAG0226/data/eval/mmerealworld_zoom4kvqa10k5epoch_baseline.jsonl
```

## Ray Feature Extraction
```bash
conda activate imagerag
cd /data1/zilun/ImageRAG0226
ray start --head --port=6379
ray stop
python codebase/patch_ray_feat_extract.py --ray_mode auto --num_runner 6
```

## Docker
```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:latest
docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:latest  docker.io/lmsysorg/sglang:latest

# docker load -i sglang.tar

bash script/sglang_start.sh
```