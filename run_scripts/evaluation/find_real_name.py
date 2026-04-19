import os
import base64
import io
import numpy as np
from PIL import Image

# ==========================================
# 1. 破案配置
# ==========================================
TARGET_UID = "21437_0"  # 你想查的那个内部 ID
TSV_FILE = '/root/autodl-tmp/polygon-transformer/datasets/finetune/msd/msd_val.tsv'
ORIG_IMG_DIR = '/root/autodl-tmp/polygon-transformer/datasets/images/msd/'

print(f"🔍 正在 TSV 文件中提取 {TARGET_UID} 的图像特征...")

# ==========================================
# 2. 从 TSV 中提取目标图像的像素矩阵
# ==========================================
target_img_np = None
with open(TSV_FILE, 'r', encoding='utf8') as f:
    for line in f:
        # 找到那一行
        if line.startswith(TARGET_UID + '\t'):
            parts = line.split('\t')
            # 提取 base64
            b64 = next((c for c in parts if len(c) > 1000), "")
            try:
                img_bytes = base64.urlsafe_b64decode(b64)
            except:
                img_bytes = base64.b64decode(b64)
            
            # 转换为灰度矩阵(L)用于精确比对
            target_img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"))
            break

if target_img_np is None:
    print(f"❌ 在 TSV 中找不到 ID: {TARGET_UID}")
    exit()

print(f"✅ 成功提取目标图像，尺寸为: {target_img_np.shape}")
print(f"🕵️‍♂️ 开始在原图文件夹中进行“滴血认亲”比对 (可能需要十几秒)...")

# ==========================================
# 3. 暴力扫盘：寻找 100% 像素匹配的真实文件
# ==========================================
found = False
# 遍历真实的图片文件夹
for filename in os.listdir(ORIG_IMG_DIR):
    if filename.endswith(".png"):
        filepath = os.path.join(ORIG_IMG_DIR, filename)
        try:
            # 读取真实图片
            test_img_np = np.array(Image.open(filepath).convert("L"))
            
            # 如果尺寸一样，且每一个像素值都完全相等！
            if target_img_np.shape == test_img_np.shape:
                if np.array_equal(target_img_np, test_img_np):
                    print("\n" + "="*50)
                    print(f"🎉 BINGO！破案了！")
                    print(f"内部 ID 【{TARGET_UID}】 的真实名字就是：👉 {filename} 👈")
                    print("="*50 + "\n")
                    found = True
                    break
        except Exception as e:
            pass

if not found:
    print("❌ 扫遍了所有文件，没有找到一模一样的图片。")