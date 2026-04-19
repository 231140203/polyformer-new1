#!/bin/bash
# ----------------------------------------------------------------
# 功能：使用 nnunet_test.tsv 评估最终模型表现
# ----------------------------------------------------------------

export MASTER_PORT=6092
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

# 1. 核心绝对路径绑定（防止任何相对路径报错）
PROJECT_ROOT="/root/autodl-tmp/polygon-transformer"
cd $PROJECT_ROOT  # 强制切换到项目根目录

user_dir="${PROJECT_ROOT}/polyformer_module"
bpe_dir="${PROJECT_ROOT}/utils/BPE"
evaluate_py="${PROJECT_ROOT}/evaluate.py"

selected_cols=0,5,6,2,4,3
model='polyformer_b'
num_bins=64
batch_size=2  # 保持较小的 batch_size 防止测试时爆显存

# 2. 🎯 【核心修改】指向测试集
dataset='nnunet'
split='nnunet_test'  # 自动寻找 datasets/finetune/nnunet/nnunet_test.tsv

# 3. 🎯 【权重路径】指向你最满意的模型
# 注意：如果你之前挑选出了 best 模型，请把 checkpoint_last.pt 改成 checkpoint_best_mdice.pt
ckpt_path="${PROJECT_ROOT}/run_scripts/finetune/${model}_checkpoints/100_5e-5_512/checkpoint_best_mdice.pt"

# 4. 设置独立的测试结果输出路径（绝不和 val 的结果混淆）
data="${PROJECT_ROOT}/datasets/finetune/${dataset}/${split}.tsv"
result_path="${PROJECT_ROOT}/results_${model}/${dataset}/test_results"
vis_dir="${result_path}/vis"
result_dir="${result_path}/result"

echo "==================================================="
echo "🚀 开始在测试集 (${split}.tsv) 上执行最终评估..."
echo "📦 加载权重: $ckpt_path"
echo "📂 结果将保存在: $result_path"
echo "==================================================="

# 5. 启动评测
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ${evaluate_py} \
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