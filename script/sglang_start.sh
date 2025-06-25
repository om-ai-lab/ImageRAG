sudo docker run --gpus '"device=7"' \
    --shm-size 64g \
    -p 30000:30000 \
    -v /training/zilun/model/Qwen2.5-32B-Instruct:/training/zilun/model/Qwen2.5-32B-Instruct \
    --env "HF_ENDPOINT=https://hf-mirror.com" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path /training/zilun/model/Qwen2.5-32B-Instruct \
        --host 0.0.0.0 \
        --port 30000 \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --disable-cuda-graph
