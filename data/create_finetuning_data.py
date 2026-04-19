from refer.refer import REFER
import numpy as np
from PIL import Image
import random
import os
from tqdm import tqdm

import pickle
from poly_utils import is_clockwise, revert_direction, check_length, reorder_points, \
    approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string

max_length = 400
data_root = './refer/data'

# 【修改 1】: 将数据集指向你刚刚建好的 msd
datasets = ['msd']

# 【修改 2】: 指向你的 MSD 图片真实存放路径
image_dir = './datasets/images/msd'

combined_train_data = []

for dataset in datasets:
    splits = ['train', 'val']
    splitBy = 'unc'

    save_dir = f'datasets/finetune/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    for split in splits:
        num_pts = []
        max_num_pts = 0
        file_name = os.path.join(save_dir, f"{dataset}_{split}.tsv")
        print("creating ", file_name)

        writer = open(file_name, 'w')
        print(f"Loading dataset {dataset} with splitBy {splitBy}...")

        refer = REFER(data_root, dataset, splitBy)

        ref_ids = refer.getRefIds(split=split)

        for this_ref_id in tqdm(ref_ids):
            this_img_id = refer.getImgIds(this_ref_id)
            this_img = refer.Imgs[this_img_id[0]]
            fn = this_img['file_name']

            # ---------------- 找图逻辑 ----------------
            img_path = os.path.join(image_dir, fn)
            if not os.path.exists(img_path):
                print(f"⚠️ 跳过：找不到图片 {fn}")
                continue

            # load image
            img = Image.open(img_path).convert("RGB")

            # convert image to string
            img_base64 = image_to_base64(img, format='jpeg')

            # load mask
            ref = refer.loadRefs(this_ref_id)
            ref_mask = np.array(refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1  # 255
            annot_img = Image.fromarray(annot.astype(np.uint8), mode="P")
            annot_base64 = image_to_base64(annot_img, format='png')

            # 处理多边形逻辑
            polygons = refer.getPolygon(ref[0])['polygon']
            polygons_processed = []
            for polygon in polygons:
                # make the polygon clockwise
                if not is_clockwise(polygon):
                    polygon = revert_direction(polygon)

                # reorder the polygon so that the first vertex is the one closest to image origin
                polygon = reorder_points(polygon)
                polygons_processed.append(polygon)

            polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
            polygons_interpolated = interpolate_polygons(polygons)
            polygons = approximate_polygons(polygons, 5, max_length)

            pts_string = polygons_to_string(polygons)
            pts_string_interpolated = polygons_to_string(polygons_interpolated)

            # load box
            box = refer.getRefBox(this_ref_id)  # x,y,w,h
            x, y, w, h = box
            box_string = f'{x},{y},{x + w},{y + h}'

            max_num_pts = max(max_num_pts, check_length(polygons))
            num_pts.append(check_length(polygons))

            # load text
            ref_sent = refer.Refs[this_ref_id]
            for i, (sent, sent_id) in enumerate(zip(ref_sent['sentences'], ref_sent['sent_ids'])):
                uniq_id = f"{this_ref_id}_{i}"
                instance = '\t'.join(
                    [uniq_id, str(this_img_id[0]), sent['sent'], box_string, pts_string, img_base64, annot_base64,
                     pts_string_interpolated]) + '\n'
                writer.write(instance)
                combined_train_data.append(instance)

        writer.close()

# 保存最终的 Shuffled 总表
random.shuffle(combined_train_data)
# 文件名改为 msd_train_shuffled.tsv
file_name = os.path.join("datasets/finetune/msd_train_shuffled.tsv")
print("creating ", file_name)
writer = open(file_name, 'w')
writer.writelines(combined_train_data)
writer.close()