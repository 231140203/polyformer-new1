#!/bin/bash

# =====================================================================
# 【绝对防误删系统级保险】
# 拦截 Ctrl+C (SIGINT) 和 kill (SIGTERM)
# 一旦按下 Ctrl+C，脚本瞬间自杀退出，绝不往下执行任何 rm 语句！
# =====================================================================
trap "echo -e '\n🛑 收到中断信号 (Ctrl+C)！守护进程立刻安全退出，文件已冻结保留！'; exit 1" SIGINT SIGTERM

PROJECT_ROOT="/root/autodl-tmp/polygon-transformer"
TARGET_DIR="${PROJECT_ROOT}/run_scripts/finetune/polyformer_b_checkpoints/100_5e-5_512"
EVAL_SCRIPT_DIR="${PROJECT_ROOT}/run_scripts/evaluation"
EVAL_SCRIPT_NAME="evaluate_polyformer_b_train.sh"

RESULT_FILE="${PROJECT_ROOT}/results_polyformer_b/nnunet/last_model/nnunet_val_result.txt"
CURVE_LOG="${PROJECT_ROOT}/eval_curve.csv"

# 【修复 1：变量名统一为 MDICE】
BEST_MDICE=0.0

echo "🚀 启动 [积压熔断 + 防 Ctrl+C + mDice选优] 监控守护进程..."

# 初始化包含 10 个指标的 CSV 表头
if [ ! -f "$CURVE_LOG" ]; then
    echo "Epoch,mDice,oDice,mIoU,oIoU,AP,F_score,Prec0.5,Prec0.7,Prec0.9" > "$CURVE_LOG"
fi

# 【修复 2：去掉多余的内层 while true，只保留这一个主循环】
while true; do
    
    # 1. 找出所有的排队模型 (确保确实有文件，防止 ls 报错输出给变量)
    ALL_CKPTS=$(ls -v $TARGET_DIR/checkpoint_epoch_*.pt 2>/dev/null)
    
    # 【修复 3：用 wc -w 安全地统计文件个数】
    CKPT_COUNT=$(echo "$ALL_CKPTS" | wc -w)

    if [ "$CKPT_COUNT" -ge 1 ]; then
        # 2. 永远只挑“最新”的那个去评测 (用 tail -n 1)
        # 注意：ALL_CKPTS 包含多行，tail -n 1 会取最后一行，也就是最大的 Epoch
        CURRENT_CKPT=$(echo "$ALL_CKPTS" | tail -n 1)
        EPOCH_NUM=$(echo "$CURRENT_CKPT" | grep -oP 'checkpoint_epoch_\K\d+')
        
        # 3. 【积压熔断清理机制】
        if [ "$CKPT_COUNT" -gt 1 ]; then
            echo "⚠️ 警告：检测到队列积压 (共 $CKPT_COUNT 个待测模型)！"
            echo "🔥 触发熔断机制：放弃中间 Epoch，优先追赶最新进度..."
            
            # 提取出除了最后一个（最新模型）之外的所有旧模型
            OBSOLETE_CKPTS=$(echo "$ALL_CKPTS" | head -n -1)
            for old_ckpt in $OBSOLETE_CKPTS; do
                rm -f "$old_ckpt"
                echo "🗑️ 抛弃过期未测模型: $old_ckpt"
            done
        fi

        echo "------------------------------------------------------"
        echo "[$(date '+%H:%M:%S')] 🔍 开始优先评测最新进度: Epoch $EPOCH_NUM"
        
        # 评测前强制清理旧成绩单
        rm -f "$RESULT_FILE"

        cd "$EVAL_SCRIPT_DIR"
        bash "$EVAL_SCRIPT_NAME" "$CURRENT_CKPT"
        EVAL_STATUS=$?  # 获取 Bash 的执行状态码
        cd "$PROJECT_ROOT"

        # 【第二重保险】：防 Ctrl+C 拦截
        if [ $EVAL_STATUS -eq 130 ]; then
            echo "🛑 评测进程被用户 (Ctrl+C) 中断！监控脚本随之退出，绝对不删文件！"
            exit 130
        fi

        # 只有正常运行完毕且成绩单真实存在，才继续
        if [ $EVAL_STATUS -eq 0 ] && [ -f "$RESULT_FILE" ]; then
            # 抓取指标
            MD=$(grep -oP 'mDice: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            OD=$(grep -oP 'oDice: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            MI=$(grep -oP 'mIoU score: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            OI=$(grep -oP 'oIoU score: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            AP=$(grep -oP 'ap det score: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            F1=$(grep -oP 'f score: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            P5=$(grep -oP 'prec@0.5: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            P7=$(grep -oP 'prec@0.7: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)
            P9=$(grep -oP 'prec@0.9: \K[0-9.]+' "$RESULT_FILE" | tail -n 1)

            # 写入 CSV
            echo "${EPOCH_NUM},${MD:-0},${OD:-0},${MI:-0},${OI:-0},${AP:-0},${F1:-0},${P5:-0},${P7:-0},${P9:-0}" >> "$CURVE_LOG"
            echo "📊 [Epoch $EPOCH_NUM] 评测成功: mDice=${MD:-0} | oDice=${OD:-0}"

            # 比较选拔最佳模型 (使用 mDice)
            IS_BETTER=$(awk -v curr="${MD:-0}" -v best="$BEST_MDICE" 'BEGIN {print (curr > best)}')

            if [ "$IS_BETTER" -eq 1 ]; then
                BEST_MDICE=$MD
                echo "🌟 发现新记录！mDice 提升至: $BEST_MDICE，正在更新 checkpoint_best.pt"
                cp "$CURRENT_CKPT" "$TARGET_DIR/checkpoint_best_mdice.pt" 
            fi

            # 【终极开火许可】安全清理当前测完的模型
            rm -f "$CURRENT_CKPT"
            echo "[$(date '+%H:%M:%S')] 🗑️ 已安全清理完成: Epoch $EPOCH_NUM"
        else
            echo "❌ 评测异常 (可能因显存溢出、OOM或配置错误)。"
            echo "🛑 防护开启：文件 $CURRENT_CKPT 已原样保留！等待修复。"
            sleep 60
        fi
    else
        echo "[$(date '+%H:%M:%S')] 💤 暂无模型排队，等待 5 分钟..."
        sleep 300
    fi
done