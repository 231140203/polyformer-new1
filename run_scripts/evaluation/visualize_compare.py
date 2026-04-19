import os
import glob
import base64
import io
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

# ==========================================
# 1. 核心配置 (请确保路径和你的 sh 脚本输出一致)
# ==========================================
CONFIG = {
    # 你的完整成绩单
    'dice_csv':      '/root/autodl-tmp/polygon-transformer/run_scripts/evaluation/per_slice_dice.csv',
    # 第一步：要生成的包含 20 个题目的目标 tsv
    'mini_tsv':      '/root/autodl-tmp/polygon-transformer/datasets/finetune/nnunet/nnunet_val_mini.tsv',
    # 你当前的完整题库 (视你当前在跑 val 还是 test 而定)
    'full_tsv':      '/root/autodl-tmp/polygon-transformer/datasets/finetune/nnunet/nnunet_val.tsv', 
    # 数据集 mapping (用于根据 6290_0 追溯病人 case_id)
    'mapping_json':  '/root/autodl-tmp/Polyformer_Export/mapping.json',
    # 第三步：读取预测图的文件夹 (你的 evaluate_polyformer_b_top10.sh 跑完后生成的 vis 文件夹)
    'poly_pred_dir': '/root/autodl-tmp/polygon-transformer/results_polyformer_b/nnunet/last_model/vis/nnunet_val_mini', 
    # 最终保存 20 张高清对比图的目录
    'save_dir':      '/root/autodl-tmp/polygon-transformer/run_scripts/evaluation/vis_results_diverse/',
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)
FIX_ROTATION_180 = True

# ==========================================
# 2. 读取 Mapping (提取真实病人 ID)
# ==========================================
print("=" * 55)
print("Step 1: 读取题库，映射病人并判断器官形态...")
mapping = {}
if os.path.exists(CONFIG['mapping_json']):
    with open(CONFIG['mapping_json'], 'r') as f:
        mapping = json.load(f)

# ==========================================
# 3. 扫描 TSV，计算长宽比分类器官形态
# ==========================================
shape_dict = {}
full_tsv_lines = {}

with open(CONFIG['full_tsv'], 'r', encoding='utf8') as f:
    for line in f:
        parts = line.strip('\n').split('\t')
        uid = parts[0].strip()  # 例如 6290_0
        full_tsv_lines[uid] = line
        try:
            coords = list(map(float, parts[4].split(',')))
            xs, ys = coords[0::2], coords[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if min(w, h) > 0:
                aspect = max(w, h) / min(w, h)
                # 定义：长宽比 > 1.8 为细长，< 1.5 为圆润
                if aspect > 1.8:
                    shape_dict[uid] = 'Elongated'
                elif aspect < 1.5:
                    shape_dict[uid] = 'Round'
                else:
                    shape_dict[uid] = 'Normal'
        except:
            pass

# ==========================================
# 4. 严苛挑选：20 个不同病人，好坏各半，长圆各半
# ==========================================
print("Step 2: 严格挑选 20 个不同病人的极致案例...")
df = pd.read_csv(CONFIG['dice_csv'], header=None, usecols=[0, 1], names=['id', 'dice'])
df['id'] = df['id'].astype(str).str.strip()
df['dice'] = pd.to_numeric(df['dice'], errors='coerce')
df = df.dropna(subset=['dice', 'id'])

# 映射 Patient ID
def get_patient_id(uid):
    img_id = uid.split('_')[0]
    return mapping.get(img_id, {}).get('case_id', f'Unknown_{img_id}')

df['patient_id'] = df['id'].apply(get_patient_id)
df['shape'] = df['id'].map(shape_dict)
df = df.dropna(subset=['shape'])

# 按照 Dice 排序
df = df.sort_values('dice', ascending=False)

seen_patients = set()
selected_cases = []

def pick_cases(sub_df, count, rank_type, shape_type):
    picked = 0
    for _, row in sub_df.iterrows():
        # 确保同一个病人只出现一次
        if row['patient_id'] not in seen_patients:
            selected_cases.append({
                'id': row['id'], 'dice': row['dice'], 
                'rank': rank_type, 'shape': shape_type, 'patient_id': row['patient_id']
            })
            seen_patients.add(row['patient_id'])
            picked += 1
            if picked >= count: break

# 1. 挑 5 个圆润极好，5 个圆润极差
df_round = df[df['shape'] == 'Round']
pick_cases(df_round, 5, 'Good', 'Round')
pick_cases(df_round.iloc[::-1], 5, 'Bad', 'Round')

# 2. 挑 5 个细长极好，5 个细长极差
df_elong = df[df['shape'] == 'Elongated']
pick_cases(df_elong, 5, 'Good', 'Elongated')
pick_cases(df_elong.iloc[::-1], 5, 'Bad', 'Elongated')

final_df = pd.DataFrame(selected_cases)
print(f"✅ 成功挑选出 {len(final_df)} 个符合条件的独立案例！")
print(final_df[['patient_id', 'shape', 'rank', 'dice']].to_string(index=False))

# ==========================================
# 5. 生成 mini_tsv (配合你的第一步)
# ==========================================
with open(CONFIG['mini_tsv'], 'w', encoding='utf8') as f:
    for uid in final_df['id']:
        if uid in full_tsv_lines:
            f.write(full_tsv_lines[uid])
print(f"✅ 已将 {len(final_df)} 个案例写入 {CONFIG['mini_tsv']}，可以执行 sh 脚本了！")

# ==========================================
# 工具函数：自适应长方形包围盒 (完美解决边缘被切掉)
# ==========================================
def get_dynamic_robust_crop(gt_points, pred_xs, pred_ys, img_h, img_w, padding=60):
    xs = [p[0] for p in gt_points] if gt_points else []
    ys = [p[1] for p in gt_points] if gt_points else []
    
    # 🌟 核心升级：将预测图中的红绿框/多边形坐标边界也囊括进来！
    if len(pred_xs) > 0 and len(pred_ys) > 0:
        xs.extend([np.min(pred_xs), np.max(pred_xs)])
        ys.extend([np.min(pred_ys), np.max(pred_ys)])
        
    if not xs or not ys: return 0, img_h, 0, img_w
    
    # 独立计算 X 和 Y 的边界，生成“长方形”而不是正方形！
    x1 = max(0, int(min(xs) - padding))
    x2 = min(img_w, int(max(xs) + padding))
    y1 = max(0, int(min(ys) - padding))
    y2 = min(img_h, int(max(ys) + padding))
    return y1, y2, x1, x2

# ==========================================
# 6. 高清制图核心 (配合你的第三步)
# ==========================================
print("\nStep 3: 开始生成超高清对比图...")
for _, row in final_df.iterrows():
    slice_id = row['id']
    poly_dice = row['dice']
    file_prefix = f"{row['rank']}_{row['shape']}_{slice_id}_{row['patient_id']}"
    
    if slice_id not in full_tsv_lines: continue
    
    # 1. 极限防御：安全解析 Base64 (防止坏图报错)
    parts = full_tsv_lines[slice_id].strip('\n').split('\t')
    b64 = next((c for c in parts if len(c) > 1000), "")
    try:
        b64 += "=" * ((4 - len(b64) % 4) % 4)
        try: img_bytes = base64.urlsafe_b64decode(b64)
        except: img_bytes = base64.b64decode(b64)
        img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    except Exception as e:
        print(f"  ⚠️ [{slice_id}] Base64 图像损坏，跳过。")
        continue

    img_h, img_w = img_np.shape[:2]
    coords = list(map(float, parts[4].split(',')))
    gt_points = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]

    # 2. 读取模型生成的预测图
    pattern = os.path.join(CONFIG['poly_pred_dir'], f"{slice_id}_*pred_overlayed.png")
    pred_paths = glob.glob(pattern)
    if not pred_paths:
        print(f"  ⚠️ [{slice_id}] 找不到预测图。如果你在执行第一步，这是正常的，请去跑 evaluate.sh。")
        continue
        
    pred_img = np.array(Image.open(pred_paths[0]).convert("RGB"))
    if pred_img.shape[:2] != (img_h, img_w):
        pred_img = np.array(Image.fromarray(pred_img).resize((img_w, img_h)))

    # 3. 180度旋转矫正
    if FIX_ROTATION_180:
        img_np = np.rot90(img_np, 2)
        pred_img = np.rot90(pred_img, 2)
        gt_points = [(img_w - x, img_h - y) for x, y in gt_points]
        
    # 🌟 核心修改：提取预测图里画了红绿框/掩码的坐标范围
    # 原理：将带有预测标记的图与原图相减，有色彩差异的像素就是预测画图的区域！
    diff = np.abs(pred_img.astype(np.int32) - img_np.astype(np.int32)).sum(axis=-1)
    pred_ys, pred_xs = np.where(diff > 15)
        
    # 4. 矩形自适应裁剪 (🌟 把提取到的预测坐标系一起传进去 🌟)
    y1, y2, x1, x2 = get_dynamic_robust_crop(gt_points, pred_xs, pred_ys, img_h, img_w, padding=50)
    
    # 5. 画图 (提高 DPI 到 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    # 原图
    axes[0].imshow(img_np[y1:y2, x1:x2]); axes[0].axis('off'); axes[0].set_title("Image")
    
    # 金标准 (高精度渲染 Label)
    fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(img_w/100, img_h/100), dpi=200)
    ax_tmp.imshow(img_np)
    ax_tmp.add_patch(Polygon(gt_points, closed=True, edgecolor='lime', facecolor='lime', alpha=0.45, linewidth=2))
    ax_tmp.axis('off'); ax_tmp.set_xlim(0, img_w); ax_tmp.set_ylim(img_h, 0)
    fig_tmp.subplots_adjust(0, 0, 1, 1); buf = io.BytesIO()
    fig_tmp.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0); plt.close(fig_tmp); buf.seek(0)
    gt_rendered = np.array(Image.fromarray(np.array(Image.open(buf).convert("RGB"))).resize((img_w, img_h)))
    
    axes[1].imshow(gt_rendered[y1:y2, x1:x2]); axes[1].axis('off'); axes[1].set_title("Label")
    
    # 预测图
    axes[2].imshow(pred_img[y1:y2, x1:x2]); axes[2].axis('off'); axes[2].set_title("Polyformer")
    
    # 标注分数
    axes[2].text(0.04, 0.96, f"{poly_dice:.4f}", transform=axes[2].transAxes, 
                 color='red', fontsize=20, fontweight='bold', va='top', ha='left',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_dir'], f"{file_prefix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  🔍 成功保存: {file_prefix} (Dice: {poly_dice:.4f})")

print("\n🎉 完美！代码执行完毕！")