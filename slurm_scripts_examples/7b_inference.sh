#!/bin/sh

#SBATCH --job-name=llm_infrnce
#SBATCH --error=/userspace/kma/reward/llm_training/logs/float_type/llm_infrnce_7b_test.err
#SBATCH --output=/userspace/kma/reward/llm_training/logs/float_type/llm_infrnce_7b_test.log
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-socket=1
#SBATCH --no-requeue
#SBATCH -o /userspace/kma/reward/llm_training/logs/float_type/llm_infrnce_7b_test.log
#SBATCH --nodelist=ngpu04

echo "# SLURM_JOBID  = ${SLURM_JOBID}"
echo "# SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "# SLURM_NNODES = ${SLURM_NNODES}"
echo "# SLURM_NTASKS = ${SLURM_NTASKS}"
echo "# SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "# SLURMTMPDIR = ${SLURMTMPDIR}"
echo "# Submission directory = ${SLURM_SUBMIT_DIR}"
echo "Starting SLURM job on $(hostname) with GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo "Current directory: $(pwd)"


echo "Loading env and variables"
. "/storage2/gdv/miniconda3/etc/profile.d/conda.sh"
conda activate /storage2/gdv/llm_gpu_env


export LD_LIBRARY_PATH=/storage2/gdv/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/userspace/kma/reward/llm_training/cache/ru-qwen-audio/cache
export HF_HOME=/userspace/kma/reward/llm_training/cache/ru-qwen-audio/hf_home
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export TRITON_CACHE_DIR=/userspace/kma/reward/llm_training/cache/triton_cached
export RAY_CACHE_DIR=/userspace/kma/reward/llm_training/cache/ray_cached
export RAY_CACHE=/userspace/kma/reward/llm_training/cache/ray_cached
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_DIR=/userspace/kma/reward/llm_training/cache/vllm_stats_dir
export VLLM_USE_V1=0
export PYTHONPATH=/userspace/kma/reward/evaluation-framework_08_25/evaluation-framework
export TORCH_DYNAMO_CACHE_DIR=/userspace/kma/torchdynamo_cached

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_TIMEOUT=600

VLLM_HOST="127.0.0.1"
VLLM_PORT=9998
SERVER_READY_WAIT_SECONDS=350
MODEL_PATH="/userspace/kma/reward/llm_training/models/binary_fine-tuned-qwen-qwen2_5-Coder-7b-instruct_10/qwen2_5-coder-7b-instruct/v0-20250820-145727/checkpoint-1954"

nvidia-smi
nvcc -V
python -V
python -c "import torch, transformers, datasets, tokenizers; print(f'torch.version = {torch.__version__}, CUDA = {torch.cuda.is_available()}, transformers.version = {transformers.__version__}, datasets.version = {datasets.__version__}, tokenizers.version = {tokenizers.__version__}')"

VLLM_PID=""
cleanup() {
    echo "Caught EXIT signal. Attempting to kill VLLM server (PID: $VLLM_PID)..."
    if [ -n "$VLLM_PID" ]; then
        kill $VLLM_PID
        echo "VLLM server (PID: $VLLM_PID) killed."
    else
        echo "VLLM server PID not found or not set."
    fi
    echo "Cleanup complete."
}

trap cleanup EXIT

echo "Starting vLLM server with model '$MODEL_PATH' on $VLLM_HOST:$VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.75 \
    --served-model-name Qwen2.5-Coder-7B-Instruct \
    --max-model-len 16000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --api-key "token-abc123" \
    --max-num-batched-tokens 200000\
    --disable-log-requests \
    --dtype auto &

VLLM_PID=$!

echo "vLLM server started with PID: $VLLM_PID"
echo "Giving vLLM server $SERVER_READY_WAIT_SECONDS seconds to initialize..."
sleep $SERVER_READY_WAIT_SECONDS

python -c "import torch, transformers, datasets; print(f'torch.version = {torch.__version__}, CUDA = {torch.cuda.is_available()}, transformers.version = {transformers.__version__}, datasets.version = {datasets.__version__}')"

python /userspace/kma/reward/evaluation-framework_08_25/evaluation-framework/scripts/multi_gpu_inference.py --dataset_dir /storage2/gdv/reward_model_dataset_07_08/zero_shot_subset_with_diffs --save_file_name float_type_7b_zero_shot_subset_with_diffs_test.csv --target_type float_type --zero_shot True 

CLIENT_EXIT_CODE=$?

if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo "Client script finished successfully."
else
    echo "Client script failed with exit code $CLIENT_EXIT_CODE."
    exit $CLIENT_EXIT_CODE
fi

echo "Script finished. Trap will ensure VLLM server is killed."