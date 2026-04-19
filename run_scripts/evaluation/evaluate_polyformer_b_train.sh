#!/bin/bash
# ----------------------------------------------------------------
# 功能：评价单个指定的 Epoch 模型。支持 nnunet 数据库。
# ----------------------------------------------------------------

export MASTER_PORT=6092
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0,5,6,2,4,3
model='polyformer_b'
num_bins=64
batch_size=8

# 【已修改为你的 nnunet 数据库设置】
dataset='nnunet'
split='nnunet_val'

# 模型路径检查：必须明确传入存在的 pt 文件，否则直接报错阻断！
if [ -n "$1" ] && [ -f "$1" ]; then
    ckpt_path=$1
    echo "🎯 [INFO] 正在评测指定模型: $ckpt_path"
else
    echo "❌ [ERROR] 未传入参数，或指定的模型文件 ($1) 不存在！"
    echo "🛑 评测进程已主动终止，防止脏数据产生。"
    exit 1  # 状态码非 0 退出，这样外面的监控脚本就能立刻捕获到异常！
fi



data=../../datasets/finetune/${dataset}/${split}.tsv
result_path=../../results_${model}/${dataset}/last_model
vis_dir=${result_path}/vis/${split}
result_dir=${result_path}/result/${split}

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
    --vis_dir=${vis_dir} \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"