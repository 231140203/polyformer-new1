#!/bin/bash

# ================= 配置区 =================
# 目标文件夹路径
TARGET_DIR="$HOME/autodl-tmp/polygon-transformer/run_scripts/finetune/polyformer_b_checkpoints/100_5e-5_512"
# ==========================================

echo "======================================================"
echo "⚡ 启动 [高频闪电清理] 守护进程 (每30秒巡逻)"
echo "📂 监控目录: $TARGET_DIR"
echo "⚠️ 目标: 只要出现无用 pt 文件，立刻抹杀！"
echo "======================================================"

while true; do
    if [ -d "$TARGET_DIR" ]; then
        # 【极其重要：1分钟保护锁】
        # -mmin +1 的意思是：寻找修改时间在 1 分钟之前的文件。
        # AutoDL 的 4090D 机器用的是企业级固态硬盘，写入 2GB 文件大概需要几秒到十几秒。
        # 我们给它 1 分钟的时间，确保 PyTorch 已经彻底把文件写完了，然后再删。
        # 否则如果你在它刚写到 500MB 的时候删掉，PyTorch 会当场崩溃！
        
        # 1. 猎杀 checkpoint_last.pt
        find "$TARGET_DIR" -maxdepth 1 -name "checkpoint_last.pt" -mmin +1 -exec rm -f {} \;
        
        # 2. 猎杀 checkpoint_best.pt
        find "$TARGET_DIR" -maxdepth 1 -name "checkpoint_best.pt" -mmin +1 -exec rm -f {} \;
        
        # 3. 猎杀 checkpoint.best_score_0.0000.pt
        find "$TARGET_DIR" -maxdepth 1 -name "checkpoint.best_score_0.0000.pt" -mmin +1 -exec rm -f {} \;
        
    else
        echo "❌ 警告：未找到目标目录 $TARGET_DIR ，请检查路径！"
    fi
    
    # 每 30 秒巡视一次。去掉了烦人的 echo，让它在后台安静地干活，不刷屏。
    sleep 30
done