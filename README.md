# ImageRAG

## Setup

```bash
git clone https://github.com/om-ai-lab/ImageRAG.git
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

## Init SGLang (Docker)
```bash
# Pull the mirror of sglang, just for speeding up the download process.
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:latest
docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:latest  docker.io/lmsysorg/sglang:latest

# docker load -i sglang.tar

bash script/sglang_start.sh
```

## Run Inference in Parallel

```bash
# {clip, georsclip, remoteclip, mcipclip} x {vit, cc, grid} x {rerank, mean, cluster} x {0, ... ,7}
bash script/georsclip_grid.sh rerank 7
```


## Run Baseline Inference (No ImageRAG, No GT, No Inference while detecting)
```bash
# inference
CUDA_VISIBLE_DEVICES=0 python codebase/main_inference_mmerealworld_imagerag_preextract.py --cfg_path config/config_mmerealworld-baseline-zoom4kvqa10k2epoch_server.yaml

# eval inference result
python codebase/inference/MME-RealWorld-RS/eval_your_results.py --results_file data/eval/mmerealworld_zoom4kvqa10k2epoch_baseline.jsonl
```


## Run GT Inference (No ImageRAG, No Inference while detecting, BBoxes are needed)
```bash
# inference
CUDA_VISIBLE_DEVICES=0 python codebase/main_inference_mmerealworld_imagerag_preextract.py --cfg_path config/config_mmerealworld-detectiongt-zoom4kvqa10k2epoch_server.yaml

# eval inference result
python codebase/inference/MME-RealWorld-RS/eval_your_results.py --results_file data/eval/mmerealworld_zoom4kvqa10k2epoch_baseline.jsonl
```

## Ray Feature Extraction
```bash
ray start --head --port=6379
ray stop
# extract patch features
python codebase/ray_feat_extract_patch.py --ray_mode auto --num_runner 8
# extract features for vector database
python codebase/ray_feat_extract_vectorstore.py --ray_mode auto --num_runner 8

```

