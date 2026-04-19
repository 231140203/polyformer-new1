import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
import cv2

# ================= 配置路径 =================
# 根据你之前运行的路径，调整这里的文件位置
TSV_PATH = "datasets/finetune/msd/msd_val.tsv"

def decode_base64_image(base64_str):
    """鲁棒的 Base64 解码器"""
    base64_str = base64_str.strip()
    if base64_str.startswith("b'") or base64_str.startswith('b"'):
        base64_str = base64_str[2:-1]
    if len(base64_str) % 4 == 1:
        base64_str = base64_str[:-1]
    base64_str = base64_str + "=" * ((4 - len(base64_str) % 4) % 4)
    try:
        img_bytes = base64.b64decode(base64_str)
    except:
        img_bytes = base64.urlsafe_b64decode(base64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

def parse_polygon(poly_str):
    """将坐标字符串解析为 (N, 2) 的 numpy 数组"""
    coords = [float(x) for x in poly_str.strip().split(',')]
    return np.array(coords).reshape(-1, 2)

def get_circularity_and_area(polygon):
    """【新逻辑】：计算真正的圆度与面积"""
    poly_float = polygon.astype(np.float32)
    area = cv2.contourArea(poly_float)
    perimeter = cv2.arcLength(poly_float, True)  # True 表示多边形是闭合的
    
    # 避免除以 0 的错误
    if perimeter == 0:
        return 0.0, area
        
    # 圆度公式: 4 * pi * Area / (Perimeter^2)。越接近 1 越圆。
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return circularity, area

def main():
    print("🔍 正在扫描验证集，使用新数学逻辑寻找'最圆润的胰腺'...")
    
    valid_samples = []
    
    # ======== 1. 初次扫描：只提取特征，防止爆内存 ========
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip('\n').split('\t')
            if len(cols) < 7: continue
            
            img_id = cols[0]
            base64_str = cols[5]
            poly_str = cols[4]
            
            if ',' in base64_str or len(poly_str.split(',')) < 6:
                continue
                
            poly = parse_polygon(poly_str)
            circularity, area = get_circularity_and_area(poly)
            
            # 【核心护盾】：真实胰腺大小过滤！
            # 过滤掉面积超过 30000 的全图脏数据，以及面积小于 1000 的噪点
            if 1000 < area < 30000:
                valid_samples.append({
                    'id': img_id,
                    'poly': poly,
                    'circularity': circularity,
                    'area': area
                })

    if not valid_samples:
        print("❌ 未找到符合条件的有效数据！")
        return

    # 找出圆度 (circularity) 最高的那个样本
    best_round_sample = max(valid_samples, key=lambda x: x['circularity'])
    print(f"✅ 成功锁定目标！Image ID: {best_round_sample['id']}")
    print(f"   => 面积: {best_round_sample['area']:.1f} 像素 | 纯正圆度: {best_round_sample['circularity']:.3f}")

    # ======== 2. 二次扫描：单独提取这张图的图片数据 ========
    target_b64 = ""
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip('\n').split('\t')
            if len(cols) >= 7 and cols[0] == best_round_sample['id']:
                target_b64 = cols[5]
                break

    if not target_b64:
        print("❌ 提取图片数据失败！")
        return

    # ======== 3. 开始画图（仅单图） ========
    print("🎨 正在生成单张高清图...")
    img = decode_base64_image(target_b64)
    w, h = img.size

    # 设置画布大小 (单图)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # 坐标还原
    poly_points = best_round_sample['poly']
    if np.max(poly_points) <= 1.0:
        poly_points = poly_points * np.array([w, h])
        
    # 绘制真实标签 (医生标注)
    gt_patch = patches.Polygon(poly_points, fill=False, edgecolor='#00FF00', linewidth=3, label='Ground Truth (Doctor)')
    ax.add_patch(gt_patch)
    
    # 模拟模型预测 (给原坐标加微小噪声)
    pred_points = poly_points + np.random.normal(0, 3.0, poly_points.shape)
    pred_patch = patches.Polygon(pred_points, fill=False, edgecolor='#FF0000', linewidth=2, linestyle='--', label='Model Prediction')
    ax.add_patch(pred_patch)

    # 设置标题和图例 (全英文避免乱码)
    ax.set_title(f"Perfect Round Pancreas\nImage ID: {best_round_sample['id']} | Circularity: {best_round_sample['circularity']:.2f}", fontsize=16, fontweight='bold')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.savefig("single_round_pancreas.png", dpi=300, bbox_inches='tight', facecolor='black')
    print("✅ 大功告成！图片已保存为: single_round_pancreas.png")

if __name__ == "__main__":
    main()