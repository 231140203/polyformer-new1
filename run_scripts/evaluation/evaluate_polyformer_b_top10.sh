#!/bin/bash

# 1. 改为单卡 (修改这里)
export MASTER_PORT=6099
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0,5,6,2,4,3
model='polyformer_b'
num_bins=64
batch_size=8

# 2. 直接指定你的 nnunet 数据集 (修改这里)
dataset='nnunet'
split='nnunet_val_mini'

# 3. 直接强行指向你的 last 权重 (修改这里)
ckpt_path=../../run_scripts/finetune/${model}_checkpoints/100_5e-5_512/checkpoint_best_mdice.pt

data=../../datasets/finetune/${dataset}/${split}.tsv
result_path=../../results_${model}/${dataset}/last_model
vis_dir=${result_path}/vis/${split}
result_dir=${result_path}/result/${split}

# 4. 只保留这一个执行块，底下的全删了
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${ckpt_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=${batch_size} \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --num-bins=${num_bins} \
    --vis \
    --vis_dir=${vis_dir} \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"