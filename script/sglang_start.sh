python -m sglang.launch_server \
    --model-path /media/zilun/wd-161/hf_download/Qwen2-1.5B-Instruct \
    --host 0.0.0.0 \
    --port $1 \
    --mem-fraction-static $2 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --chunked-prefill-size 4096 \
    --disable-cuda-graph
