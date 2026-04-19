# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# ------------------------------------------------------------------------

import json
from itertools import chain
import os
import torch
import torch.distributed as dist
import numpy as np
from skimage import draw
from PIL import Image
from utils.vis_utils import overlay_predictions
from torchvision.utils import save_image

SMOOTH = 1e-6

def check_length(polygons):
    length = 0
    for polygon in polygons:
        length += len(polygon)
    return length

def eval_refcoco(task, generator, models, sample, **kwargs):
    def _computeIoU(pred_seg, gd_seg):
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))
        return I, U

    def _calculate_ap_score(hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def convert_pts(coeffs):
        pts = []
        for i in range(len(coeffs) // 2):
            pts.append([coeffs[2 * i + 1], coeffs[2 * i]])  # y, x
        return np.array(pts, np.int32)

    def get_mask_from_codes(codes, img_size):
        masks = [np.zeros(img_size)]
        for code in codes:
            if len(code) > 0:
                try:
                    mask = draw.polygon2mask(img_size, convert_pts(code))
                    mask = np.array(mask, np.uint8)
                except:
                    mask = np.zeros(img_size)
                masks.append(mask)
        mask = sum(masks)
        mask = mask > 0
        return mask.astype(np.uint8)

    def _calculate_score(hyps, hyps_det, refs, sample, n_poly_pred, n_poly_gt, vis=True, vis_dir=None):
        if vis:
            os.makedirs(vis_dir, exist_ok=True)

        def compute_jf(pred_mask, gt_mask):
            I, U = _computeIoU(pred_mask, gt_mask)
            this_iou = 0.0 if U == 0 else I * 1.0 / U
            prec = (I + SMOOTH) / (pred_mask.sum() + SMOOTH)
            rec = (I + SMOOTH) / (gt_mask.sum() + SMOOTH)
            this_f = 2 * prec * rec / (prec + rec)
            return this_iou, this_f, I, U

        IoU, F_score, cum_I, cum_U = [], [], [], []
        pred_masks_list = [] # 🌟 新增：用于收集真实的掩码矩阵
        
        bboxes = hyps_det
        b = len(hyps)
        bboxes = torch.tensor(np.stack(bboxes, 0))
        bboxes = bboxes.to(sample['w_resize_ratios'].device)
        ap_scores = _calculate_ap_score(bboxes.float(), sample['region_coords'].float())
        
        for i in range(b):
            hyps_i = hyps[i]
            gt_mask = refs[i]
            # 🌟 终于抓到真实的预测矩阵了！
            pred_mask = get_mask_from_codes(hyps_i, gt_mask.shape[0:2])
            pred_masks_list.append(pred_mask) 
            
            this_iou, this_f, this_I, this_U = compute_jf(pred_mask, gt_mask)
            IoU.append(this_iou); F_score.append(this_f); cum_I.append(this_I); cum_U.append(this_U)

            if vis:
                def pre_caption(caption):
                    import re
                    caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
                    caption = re.sub(r"\s{2,}", ' ', caption).rstrip('\n')
                    return caption

                gt_box = sample['region_coords'][i].cpu().numpy()
                pred_box = bboxes[i].cpu().numpy()
                pred_box[::2] *= sample['w_resize_ratios'][i].cpu().numpy()
                pred_box[1::2] *= sample['h_resize_ratios'][i].cpu().numpy()
                gt_box[::2] *= sample['w_resize_ratios'][i].cpu().numpy()
                gt_box[1::2] *= sample['h_resize_ratios'][i].cpu().numpy()
                uniq_id = sample["id"][i]
                text = pre_caption(sample["text"][i])
                img = (sample["net_input"]['patch_images'][i] + 1) / 2
                img_ndarray = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                pred_overlayed = Image.fromarray(overlay_predictions(img_ndarray, pred_mask, hyps_i, pred_box).astype(np.uint8))
                pred_overlayed.save(os.path.join(vis_dir, f"{uniq_id}_{text}_pred_overlayed.png"))
                gt_overlayed = Image.fromarray(overlay_predictions(img_ndarray, gt_mask, None, gt_box).astype(np.uint8))
                gt_overlayed.save(os.path.join(vis_dir, f"{uniq_id}_{text}_gt_overlayed.png"))
                save_image(img, os.path.join(vis_dir, f"{uniq_id}_{text}.png"))

        # 🌟 返回时带上宝贵的预测掩码列表
        return torch.tensor(IoU), torch.tensor(F_score), ap_scores, torch.tensor(cum_I), torch.tensor(cum_U), pred_masks_list

    gen_out = task.inference_step(models, sample)
    hyps, hyps_det, n_poly_pred, poly_len = [], [], [], []
    b = len(gen_out)
    
    for i in range(b):
        gen_out_i = np.array(gen_out[i])
        gen_out_i = gen_out_i[gen_out_i != -1]
        gen_out_i_det = gen_out_i[:4]
        gen_out_i_det[::2] *= sample['w'][i].cpu().numpy()
        gen_out_i_det[1::2] *= sample['h'][i].cpu().numpy()

        polygons_pred = np.append(gen_out_i[4:], [2])
        idx_list = [idx for idx, val in enumerate(polygons_pred) if val == 2]
        polygons_pred *= task.cfg.patch_image_size
        
        polygons = []
        prev_idx = 0
        for idx in idx_list:
            if prev_idx != idx and prev_idx != len(polygons_pred):
                polygons.append(polygons_pred[prev_idx: idx])
            prev_idx = idx + 1

        poly_len.append(check_length(polygons))
        n_poly_pred.append(len(polygons))
        hyps.append(polygons); hyps_det.append(gen_out_i_det)
        
    gt = sample['label']
    results = [{"uniq_id": sample_id} for sample_id in sample["id"].tolist()]

    # 🌟 接收掩码列表
    iou_scores, f_scores, ap_scores, cum_I, cum_U, pred_masks_list = _calculate_score(
        hyps, hyps_det, gt, sample, n_poly_pred, sample['n_poly'], vis=kwargs['vis'], vis_dir=kwargs['vis_dir'])
    
    result_dir = kwargs['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    # =========================================================================
    # 🌟 终极修复：拆开 Batch 循环保存！给每个病人都保存带 0,1,2 分类的 Mask
    # =========================================================================
    for i in range(b):
        uniq_id = sample["id"][i]
        if torch.is_tensor(uniq_id): uniq_id = uniq_id.item()
        if isinstance(uniq_id, bytes): uniq_id = uniq_id.decode('utf-8')
        uniq_id = str(uniq_id)
        
        text = sample["text"][i].lower() if "text" in sample else ""
        class_id = 1
        if "tumor" in text or "lesion" in text or "mass" in text:
            class_id = 2
            
        final_mask = pred_masks_list[i].astype(np.uint8) * class_id
        
        torch.save({
            "iou_scores": iou_scores[i].item() if torch.is_tensor(iou_scores[i]) else iou_scores[i],
            "ap_scores": ap_scores[i].item() if torch.is_tensor(ap_scores[i]) else ap_scores[i],
            "n_poly_pred": n_poly_pred[i],
            "n_poly_gt": sample['n_poly'][i] if 'n_poly' in sample else 1,
            "poly_len": poly_len[i],
            "uniq_id": uniq_id,
            "mask": final_mask,  # <--- 将 0,1,2 的完美掩码保存进去！
            "text": text
        }, os.path.join(result_dir, f'{uniq_id}.pt'))
    # =========================================================================

    return results, iou_scores, f_scores, ap_scores, cum_I, cum_U

def eval_step(task, generator, models, sample, **kwargs):
    if task.cfg._name == 'refcoco':
        return eval_refcoco(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError

def merge_results(task, cfg, logger, score_cnt, score_sum, f_score_sum=None, ap_det_score_sum=None, prec_score_sum=None, cum_I_sum=None, cum_U_sum=None, results=None):
    if task.cfg._name == 'image_gen':
        pass
    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(f_score_sum.data)
            dist.all_reduce(cum_I_sum.data)
            dist.all_reduce(cum_U_sum.data)
            for prec_score in prec_score_sum: dist.all_reduce(prec_score.data)
            dist.all_reduce(ap_det_score_sum.data)
            dist.all_reduce(score_cnt.data)
            
        if score_cnt.item() > 0:
            mIoU = score_sum.item() / score_cnt.item()
            oIoU = cum_I_sum.item() / cum_U_sum.item()
            mDice = (2 * mIoU) / (1 + mIoU)
            oDice = (2 * cum_I_sum.item()) / (cum_I_sum.item() + cum_U_sum.item())

            txt = "sample_cnt: {}, mIoU: {}, oIoU: {}, mDice: {}, oDice: {}\n".format(
                score_cnt, round(mIoU, 4), round(oIoU, 4), round(mDice, 4), round(oDice, 4))
            logger.info(txt)
            
            output_path = os.path.join(cfg.common_eval.results_path, "{}_result.txt".format(cfg.dataset.gen_subset))
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(txt)

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw)