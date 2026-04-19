import os
import glob
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

# ==========================================
# 1. 核心配置与目标病人
# ==========================================
TARGET_PATIENT = "21437" # 你想要可视化的目标病人 ID

CONFIG = {
    'dice_csv':      '/root/autodl-tmp/polygon-transformer/run_scripts/evaluation/per_slice_dice.csv',
    'mini_tsv':      '/root/autodl-tmp/polygon-transformer/datasets/finetune/msd/msd_val_mini.tsv',
    'full_tsv':      '/root/autodl-tmp/polygon-transformer/datasets/finetune/msd/msd_val.tsv',
    'poly_pred_dir': '/root/autodl-tmp/polygon-transformer/results_polyformer_b/msd/last_model/vis/msd_val_mini',
    'save_dir':      f'/root/autodl-tmp/polygon-transformer/run_scripts/evaluation/vis_patient_{TARGET_PATIENT}/',
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ==========================================
# 2. 读取全量 TSV，抽取该病人所有切片
# ==========================================
print("=" * 55)
print(f"Step 1: 扫描题库，提取病人 {TARGET_PATIENT} 的所有序列")
print("=" * 55)

patient_lines = []
tsv_data = {}
global_xs, global_ys = [], [] # 用于收集全局胰腺坐标

with open(CONFIG['full_tsv'], 'r', encoding='utf8') as f:
    for line in f:
        uid = line.split('\t')[0]
        # 匹配该病人 ID (例如 "21437_0", "21437_15")
        if uid.startswith(f"{TARGET_PATIENT}_"):
            patient_lines.append(line)
            parts = line.strip('\n').split('\t')
            
            # 提取多边形坐标，为了算全局中心
            try:
                coords = list(map(float, parts[4].split(',')))
                if len(coords) > 0:
                    global_xs.extend(coords[0::2])
                    global_ys.extend(coords[1::2])
            except:
                pass
            
            b64 = next((c for c in parts if len(c) > 1000), "")
            # 为了排序，提取切片编号
            slice_idx = int(uid.split('_')[1]) if '_' in uid else 0
            tsv_data[uid] = {'polygon': parts[4], 'base64': b64, 'idx': slice_idx}

# 按切片的物理顺序(0, 1, 2, 3...)重新排序
patient_lines.sort(key=lambda x: int(x.split('\t')[0].split('_')[1]))
target_uids = [line.split('\t')[0] for line in patient_lines]

print(f"✅ 成功找到病人 {TARGET_PATIENT} 的 {len(patient_lines)} 张切片！")

# 写入 mini_tsv，方便一会去跑模型预测
with open(CONFIG['mini_tsv'], 'w', encoding='utf8') as f:
    f.writelines(patient_lines)
print(f"✅ 已将这 {len(patient_lines)} 张切片写入 mini TSV。")

# ==========================================
# 3. 读取成绩单，匹配 Dice 分数
# ==========================================
df = pd.read_csv(CONFIG['dice_csv'], names=['id', 'dice'])
df['id'] = df['id'].astype(str)
df = df.drop_duplicates(subset='id')
dice_dict = dict(zip(df['id'], df['dice']))

# ==========================================
# 4. 【核心魔法】：计算该病人的全局稳定裁剪框
# ==========================================
def get_global_stable_crop(img_h, img_w, padding=50, min_size=128):
    """计算整个病人 3D 胰腺在 2D 视角下的最大全局包围盒，确保翻页时镜头不动"""
    if not global_xs or not global_ys:
        return 0, img_h, 0, img_w # 如果全都没器官，直接返回全屏
        
    center_x = (min(global_xs) + max(global_xs)) / 2
    center_y = (min(global_ys) + max(global_ys)) / 2
    
    actual_size = max(max(global_xs) - min(global_xs), max(global_ys) - min(global_ys))
    final_size = max(actual_size + padding * 2, min_size)
    half_size = final_size / 2
    
    x1 = max(0, int(center_x - half_size))
    x2 = min(img_w, int(center_x + half_size))
    y1 = max(0, int(center_y - half_size))
    y2 = min(img_h, int(center_y + half_size))
    return y1, y2, x1, x2

# ==========================================
# 5. 可视化绘制函数
# ==========================================
# ==========================================
# 5. 可视化绘制函数（带智能追踪扩展）
# ==========================================
def plot_and_save(uid, global_box):
    poly_dice = dice_dict.get(uid, 1.0)
    print(f"  🔍 处理 -> {uid} | Dice: {poly_dice:.4f}")

    if not tsv_data[uid]['base64']: return
    
    try:
        b64 = tsv_data[uid]['base64']
        try: img_bytes = base64.urlsafe_b64decode(b64)
        except: img_bytes = base64.b64decode(b64)
        img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    except: return

    img_h, img_w = img_np.shape[:2]
    try:
        coords = list(map(float, tsv_data[uid]['polygon'].split(',')))
        gt_points = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
    except:
        gt_points = []

    # 1. 获取基础全局镜头
    y1, y2, x1, x2 = global_box

    # 2. 获取预测结果（包括彩色 overlay 和纯黑白 mask）
    pattern_all = os.path.join(CONFIG['poly_pred_dir'], f"{uid}_*.png")
    all_paths = glob.glob(pattern_all)
    overlay_paths = [p for p in all_paths if "overlayed" in p]
    pure_paths = [p for p in all_paths if "overlayed" not in p]

    if not overlay_paths or not pure_paths:
        print(f"     ⚠️ 找不到预测图，跳过。")
        return
        
    pred_img = np.array(Image.open(overlay_paths[0]).convert("RGB"))
    pred_mask = np.array(Image.open(pure_paths[0]).convert("L"))
    
    if pred_mask.shape[:2] != (img_h, img_w):
        pred_img = np.array(Image.fromarray(pred_img).resize((img_w, img_h)))
        pred_mask = np.array(Image.fromarray(pred_mask).resize((img_w, img_h)))

    # 3. 【核心魔法：动态追踪护航】
    # 检查模型是不是跑到镜头外面去了？如果跑出去了，强行拉开镜头！
    nz_y, nz_x = np.where(pred_mask > 0)
    if len(nz_x) > 0:
        # 向外扩展 30 个像素的宽容度
        x1 = min(x1, max(0, np.min(nz_x) - 30))
        x2 = max(x2, min(img_w, np.max(nz_x) + 30))
        y1 = min(y1, max(0, np.min(nz_y) - 30))
        y2 = max(y2, min(img_h, np.max(nz_y) + 30))

    # 4. 裁剪
    crop_orig = img_np[y1:y2, x1:x2]
    crop_pred = pred_img[y1:y2, x1:x2]

    # 原图上画 GT
    fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(img_w/100, img_h/100), dpi=100)
    ax_tmp.imshow(img_np)
    if gt_points:
        poly_patch = Polygon(gt_points, closed=True, edgecolor='lime', facecolor='lime', alpha=0.45, linewidth=2)
        ax_tmp.add_patch(poly_patch)
    ax_tmp.axis('off'); ax_tmp.set_xlim(0, img_w); ax_tmp.set_ylim(img_h, 0)
    fig_tmp.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig_tmp.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig_tmp); buf.seek(0)
    gt_rendered = np.array(Image.fromarray(np.array(Image.open(buf).convert("RGB"))).resize((img_w, img_h)))
    crop_gt = gt_rendered[y1:y2, x1:x2]

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(crop_orig, interpolation='bicubic');  axes[0].set_title("Image");       axes[0].axis('off')
    axes[1].imshow(crop_gt, interpolation='bicubic');    axes[1].set_title("Label");       axes[1].axis('off')
    axes[2].imshow(crop_pred, interpolation='bicubic');  axes[2].set_title("PolyFormer");  axes[2].axis('off')

    color = 'red' if poly_dice < 0.5 else 'green'
    if len(gt_points) == 0: color = 'gray' 
    
    axes[2].text(0.04, 0.96, f"{poly_dice:.4f}", transform=axes[2].transAxes, color=color, fontsize=18, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    plt.tight_layout()
    slice_idx_str = str(tsv_data[uid]['idx']).zfill(3)
    plt.savefig(os.path.join(CONFIG['save_dir'], f"Seq_{slice_idx_str}_{uid}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    
# ==========================================
# 6. 主执行逻辑
# ==========================================
print("\n" + "=" * 55)
print("Step 2: 开始生成全局稳定序列")
print("=" * 55)

# 算出全局稳定框
img_h, img_w = 256, 256 # 默认 CT 大小，可在下面被覆盖
global_box = get_global_stable_crop(img_h, img_w, padding=40, min_size=120)

for uid in target_uids:
    plot_and_save(uid, global_box)

print(f"\n🎉 完美！病人 {TARGET_PATIENT} 的全序列切片已保存在: {CONFIG['save_dir']}")