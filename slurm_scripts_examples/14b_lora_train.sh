#!/bin/sh

#SBATCH --job-name=llm_trng
#SBATCH --error=/userspace/kma/reward/llm_training/logs/binary/llm_trng_14b_lora.err
#SBATCH --output=/userspace/kma/reward/llm_training/logs/binary/llm_trng_14b_lora.log
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --no-requeue
#SBATCH -o /userspace/kma/reward/llm_training/logs/binary/llm_trng_14b_lora.log
#SBATCH --nodelist=ngpu09

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
. "/storage0/bi/miniconda/etc/profile.d/conda.sh"
conda activate pycuda118

export LD_LIBRARY_PATH=/storage2/gdv/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/userspace/kma/reward/llm_training/cache/ru-qwen-audio/cache
export HF_HOME=/userspace/kma/reward/llm_training/cache/ru-qwen-audio/hf_home
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export TRITON_CACHE_DIR=/userspace/kma/reward/llm_training/cache/triton_cached
export PYTHONPATH=/storage2/gdv/llm_inference
export TORCH_DYNAMO_CACHE_DIR=/userspace/kma/reward/llm_training/cache/torchdynamo_cached
export MODELSCOPE_CACHE=/userspace/kma/reward/llm_training/cache/swift_cache

nvidia-smi
nvcc -V
python -V
python -c "import torch, transformers, datasets, tokenizers; print(f'torch.version = {torch.__version__}, CUDA = {torch.cuda.is_available()}, transformers.version = {transformers.__version__}, datasets.version = {datasets.__version__}, tokenizers.version = {tokenizers.__version__}')"



MKL_THREADING_LAYER=GNU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
NPROC_PER_NODE=10
swift sft \
    --model_type 'qwen2_5-coder-14b-instruct' \
    --model_id_or_path "/userspace/dra/models/Qwen2.5-Coder-14B-Instruct"  \
    --sft_type lora \
    --dataset '/userspace/kma/reward/llm_training/data/diff_data/train_binary_type_reward_model_dataset.jsonl' \
    --val_dataset '/userspace/kma/reward/llm_training/data/diff_data/val_binary_type_reward_model_dataset.jsonl' \
    --dtype bf16 \
    --auto_find_batch_size false \
    --batch_size 1 \
    --max_length 15000 \
    --learning_rate 3e-5 \
    --lr_scheduler_type constant \
    --optim adafactor \
    --num_train_epochs 3 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 5 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir /userspace/kma/reward/llm_training/models/binary_fine-tuned-qwen-qwen2_5-coder-14b-instruct_10-v2 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05