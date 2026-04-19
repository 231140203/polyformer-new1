import os
import glob
import pickle
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ================= 1. 路径配置 (请核对) =================
PRED_DIR = '/root/autodl-tmp/polygon-transformer/results_polyformer_b/nnunet/test_results/result'
ORIGINAL_NII_DIR = '/root/autodl-tmp/nnUNet_raw/Dataset009_PancreasFull/imagesTs'
OUTPUT_3D_DIR = '/root/autodl-tmp/Polyformer_3D_Predictions'

# 🌟【核心替换】：使用 refs.p 作为终极密码本！
P_FILE_PATH = '/root/autodl-tmp/polygon-transformer/refer/data/nnunet/refs(unc).p'

os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

# ================= 2. 载入终极密码本 =================
print("="*50)
print("📖 正在读取 pancreas_refs.p 密码本...")
try:
    with open(P_FILE_PATH, 'rb') as f:
        ref_p_list = pickle.load(f)
        
    # 建立 ref_id -> (case_id, slice_z) 的绝对映射
    ref_mapping = {}
    for item in ref_p_list:
        ref_id = str(item['ref_id'])
        filename = item['file_name']  # 例如: "pancreas_225_slice034.jpg"
        
        # 切分提取 case_id 和 z轴楼层
        case_id = filename.split('_slice')[0]  # "pancreas_225"
        slice_z = int(filename.split('_slice')[1].split('.jpg')[0])  # 34
        
        ref_mapping[ref_id] = {
            'case_id': case_id,
            'slice_z': slice_z
        }
    print(f"✅ 成功读取！共建立 {len(ref_mapping)} 条映射记录。")
except Exception as e:
    print(f"❌ 读取 .p 文件失败: {e}")
    exit()

# ================= 3. 扫描并匹配预测文件 =================
print("\n" + "="*50)
print(f"📂 正在扫描预测目录: {PRED_DIR}")
pred_files = glob.glob(os.path.join(PRED_DIR, "*.pt"))

patient_tasks = {}
missing_count = 0

for pf in pred_files:
    # 提取流水号，例如 7125
    ref_id = os.path.basename(pf).split('_')[0] 
    
    if ref_id not in ref_mapping:
        missing_count += 1
        continue
        
    # 通过密码本获取真实的病人和楼层
    case_id = ref_mapping[ref_id]['case_id']  
    slice_z = ref_mapping[ref_id]['slice_z'] 
    
    if case_id not in patient_tasks:
        patient_tasks[case_id] = []
    patient_tasks[case_id].append((slice_z, pf))

print(f"🚨 匹配报告：")
print(f"   - 成功找到归属的 .pt 文件数: {len(pred_files) - missing_count}")
print(f"   - 匹配失败的 .pt 文件数: {missing_count}")
print(f"🔍 最终共汇总成 {len(patient_tasks)} 个 3D 病例，准备重建...")

# ================= 4. 精准还原 3D =================
if len(patient_tasks) > 0:
    for case_id, slice_info_list in tqdm(patient_tasks.items(), desc="3D 拼装进度"):
        orig_path = os.path.join(ORIGINAL_NII_DIR, f"{case_id}_0000.nii.gz")
        if not os.path.exists(orig_path): 
            print(f"\n⚠️ 找不到原图: {orig_path}，跳过该病人。")
            continue
            
        orig_itk = sitk.ReadImage(orig_path)
        orig_shape = sitk.GetArrayFromImage(orig_itk).shape  # (Z, Y, X)
        
        # 盖一栋全黑的大楼（没有切片的地方天然是0）
        pred_volume = np.zeros(orig_shape, dtype=np.uint8)
        
        for slice_z, pf in slice_info_list:
            try:
                data = torch.load(pf, map_location='cpu')
                mask_2d = data.numpy() if torch.is_tensor(data) else (data['mask'].numpy() if isinstance(data, dict) else np.array(data))
                
                # 将导出的 180 度翻转，翻转回真实的医学朝向！
                mask_2d = np.rot90(mask_2d, -2)
                
                # 放入对应楼层
                pred_volume[slice_z] = mask_2d
            except Exception as e: 
                pass
                
        # 转化为医学 NIfTI 并抄袭物理坐标
        pred_itk = sitk.GetImageFromArray(pred_volume)
        pred_itk.CopyInformation(orig_itk)
        
        sitk.WriteImage(pred_itk, os.path.join(OUTPUT_3D_DIR, f"{case_id}.nii.gz"))

    print(f"\n🎉 大功告成！所有 3D 预测结果已保存至 {OUTPUT_3D_DIR}")
    print("👉 下一步：可以直接运行 nnUNetv2_evaluate_folder 算成绩啦！")
else:
    print("\n🛑 拼装失败：没有任何文件匹配成功。")