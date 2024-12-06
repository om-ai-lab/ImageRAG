python -m sglang.launch_server \
    --model-path /media/zilun/wd-161/hf_download/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --chunked-prefill-size 4096 \
    --disable-cuda-graph
