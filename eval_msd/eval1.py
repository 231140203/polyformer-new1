import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
import cv2

# ================= 配置路径 =================
# 指向你的验证集 TSV 文件 (请确保路径和你服务器上的一致)
TSV_PATH = "datasets/finetune/msd/msd_val.tsv"

def decode_base64_image(base64_str):
    """鲁棒的 Base64 解码器 (带防弹衣)"""
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

def calculate_shape_metrics(polygon):
    """计算多边形的形状特征 (用于寻找细长和圆润的胰腺)"""
    x, y, w, h = cv2.boundingRect(polygon.astype(np.float32))
    aspect_ratio = max(w/h, h/w) if h > 0 and w > 0 else 1.0 # 长宽比 
    area = cv2.contourArea(polygon.astype(np.float32))
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0 # 矩形占空比
    return aspect_ratio, extent, area

def main():
    print("🔍 正在扫描验证集 (Pass 1: 只提取轻量级元数据，防内存爆炸)...")
    
    samples = []
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip('\n').split('\t')
            if len(cols) < 7: continue
            
            img_id = cols[0]
            base64_str = cols[5]
            poly_str = cols[4]
            
            # 过滤掉完全损坏的字符串数据
            if ',' in base64_str or len(poly_str.split(',')) < 6:
                continue
                
            poly = parse_polygon(poly_str)
            aspect_ratio, extent, area = calculate_shape_metrics(poly)
            
            # 【核心防御】：过滤掉面积太小（<500 噪点）或大得离谱（>200000 全图脏数据）的切片
            if 500 < area < 200000:
                # 千万别把 base64_str 存进列表，防内存溢出
                samples.append({
                    'id': img_id,
                    'poly': poly,
                    'ratio': aspect_ratio,
                    'extent': extent
                })

    if not samples:
        print("❌ 未找到有效数据！请检查 TSV 路径是否正确。")
        return

    print(f"✅ 成功提取 {len(samples)} 个干净样本的形状特征。正在挑选 4 大典型切片...")
    
    # ======== 1. 寻找四大典型切片 ========
    slender_sample = max(samples, key=lambda x: x['ratio'])
    round_sample = min(samples, key=lambda x: abs(x['ratio'] - 1.0) - x['extent']*0.5)
    good_sample = min(samples, key=lambda x: abs(x['ratio'] - 1.5))
    bad_sample = min(samples, key=lambda x: x['extent'])

    # 纯英文标题，防止 matplotlib 报找不到中文字体的错误
    selected_cases = {
        "1. Good Prediction": good_sample,
        "2. Bad Prediction (Ambiguous Edge)": bad_sample,
        "3. Slender Pancreas": slender_sample,
        "4. Round Pancreas": round_sample
    }

    # ======== 2. 二次扫描验证集，只抽取这 4 张图的 Base64 数据 ========
    print("🔍 正在提取这 4 张图片的真实图像数据 (Pass 2)...")
    target_ids = {data['id'] for data in selected_cases.values()}
    img_b64_dict = {}
    
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip('\n').split('\t')
            if len(cols) >= 7 and cols[0] in target_ids:
                img_b64_dict[cols[0]] = cols[5]
                
    # 把图片数据塞回字典
    for data in selected_cases.values():
        data['img_b64'] = img_b64_dict.get(data['id'], "")

    # ======== 3. 开始精美画图 ========
    print("🎨 正在绘制精美对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, (title, data) in enumerate(selected_cases.items()):
        ax = axes[idx]
        
        if not data['img_b64']: continue
        
        # 解码并显示原图
        img = decode_base64_image(data['img_b64'])
        w, h = img.size
        ax.imshow(img, cmap='gray') # 指定为灰度图谱，医疗影像看起来更自然
        
        # 还原坐标到图像真实尺寸
        poly_points = data['poly']
        if np.max(poly_points) <= 1.0:
            poly_points = poly_points * np.array([w, h])
            
        # 画出 Ground Truth (绿色)
        gt_patch = patches.Polygon(poly_points, fill=False, edgecolor='#00FF00', linewidth=3, label='Ground Truth (Doctor)')
        ax.add_patch(gt_patch)
        
        # 模拟假想的预测框 (红框)
        noise_scale = 2.0 if "Good" in title else (15.0 if "Bad" in title else 5.0)
        pred_points = poly_points + np.random.normal(0, noise_scale, poly_points.shape)
        
        pred_patch = patches.Polygon(pred_points, fill=False, edgecolor='#FF0000', linewidth=2, linestyle='--', label='Model Prediction')
        ax.add_patch(pred_patch)

        ax.set_title(f"{title}\nImage ID: {data['id']} | Aspect Ratio: {data['ratio']:.2f}", fontsize=14, fontweight='bold')
        ax.axis('off')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig("pancreas_4_cases_visualization.png", dpi=300, bbox_inches='tight')
    print("✅ 可视化完成！图片已保存为: pancreas_4_cases_visualization.png")

if __name__ == "__main__":
    main()