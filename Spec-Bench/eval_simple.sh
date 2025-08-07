Vicuna_PATH='./models/vicuna-7b-v1.3'
Medusa_PATH=./models/medusa-vicuna-7b-v1.3

MODEL_NAME=vicuna-7b-v1.3
TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16"

echo "Starting medusa inference"
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype

#echo "Starting pld inference"
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype

echo "Finished!"