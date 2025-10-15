PT_CHECKPOINT_PATH=$1
VOCODER_DIR=$2
OUTPUT_DIR=$3

export CUDA_VISIBLE_DEVICES=5

F5_TRT_LLM_CHECKPOINT_PATH="$OUTPUT_DIR/trt_f5_ckpt"
F5_TRT_LLM_ENGINE_PATH="$OUTPUT_DIR/trt_f5_engine"
VOCOS_ONNX_PATH="$OUTPUT_DIR/onnx_vocoder.onnx"
VOCOS_TRT_ENGINE_PATH="$OUTPUT_DIR/vocoder_engine.plan"

PKG=$(python3 -c "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])");

mkdir -p $F5_TRT_LLM_CHECKPOINT_PATH
mkdir -p $F5_TRT_LLM_ENGINE_PATH

is_dir_empty() {
    local dir="$1"
    if [ -d "$dir" ]; then
        if [ -z "$(find "$dir" -mindepth 1 -print -quit)" ]; then
            return 0 
        else
            return 1  
        fi
    else
        echo "Ошибка: '$dir' не существует или не является директорией" >&2
        return 2
    fi
}

if is_dir_empty $F5_TRT_LLM_CHECKPOINT_PATH; then
    echo "Converting checkpoint"

    python3 exporting/convert_checkpoint.py \
        --timm_ckpt "$PT_CHECKPOINT_PATH" \
        --output_dir "$F5_TRT_LLM_CHECKPOINT_PATH" 
fi

if is_dir_empty $F5_TRT_LLM_ENGINE_PATH; then
    echo "Building engine"

    cp -r patch/* $PKG/tensorrt_llm/models

    trtllm-build --checkpoint_dir $F5_TRT_LLM_CHECKPOINT_PATH \
        --max_batch_size 8 \
        --output_dir $F5_TRT_LLM_ENGINE_PATH --remove_input_padding disable
fi

if [ ! -f "$VOCOS_ONNX_PATH" ]; then
    echo "Exporting vocos vocoder"
    python3 exporting/export_vocoder_to_onnx.py --vocoder-path $VOCODER_DIR --output-path $VOCOS_ONNX_PATH
fi

if [ ! -f "$VOCOS_TRT_ENGINE_PATH" ]; then
    echo "Building vocos vocoder engine"
    bash exporting/export_vocos_trt.sh $VOCOS_ONNX_PATH $VOCOS_TRT_ENGINE_PATH
fi

# if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
#     echo "TRT-LLM: offline decoding benchmark test"
#     batch_size=1
#     split_name=wenetspeech4tts
#     backend_type=trt
#     log_dir=./log_benchmark_batch_size_${batch_size}_${split_name}_${backend_type}
#     rm -r $log_dir
#     ln -s model_repo_f5_tts/f5_tts/1/f5_tts_trtllm.py ./
#     torchrun --nproc_per_node=1 \
#     benchmark.py --output-dir $log_dir \
#     --batch-size $batch_size \
#     --enable-warmup \
#     --split-name $split_name \
#     --model-path $F5_TTS_HF_DOWNLOAD_PATH/$model/model_1200000.pt \
#     --vocab-file $F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt \
#     --vocoder-trt-engine-path $VOCOS_TRT_ENGINE_PATH \
#     --backend-type $backend_type \
#     --tllm-model-dir $F5_TTS_F5_TRT_LLM_ENGINE_PATH || exit 1
# fi
