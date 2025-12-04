import os
import csv
import glob
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from matrics_calculator import MetricsCalculator


def load_image_as_np(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def load_mask_as_np(path, target_size=None):
    mask = Image.open(path).convert("L")
    if target_size is not None:
        mask = mask.resize(target_size, Image.NEAREST)
    mask = np.array(mask, dtype=np.float32)
    if mask.max() > 1.0:
        mask /= 255.0
    mask = np.clip(mask, 0.0, 1.0)
    return mask[..., None]



def save_visualization(img_in, img_out, mask_unedit, img_name, vis_dir):
    """保存 input/output/difference/mask 可视化对比图"""
    os.makedirs(vis_dir, exist_ok=True)

    # 计算背景区域差异（只在未编辑部分显示）
    diff = np.abs(img_in - img_out) * mask_unedit
    diff_gray = diff.mean(axis=2)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img_in)
    ax[0].set_title("Input (original)")
    ax[1].imshow(img_out)
    ax[1].set_title("Output (model)")
    ax[2].imshow(diff_gray, cmap="magma")
    ax[2].set_title("Diff (unedit region)")
    ax[3].imshow(mask_unedit.squeeze(), cmap="gray")
    ax[3].set_title("Mask (unedit=white)")

    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, os.path.splitext(img_name)[0] + "_bgvis.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)

def numerical_sort_key(path):
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    # 提取文件名中的数字部分作为排序依据
    digits = ''.join(filter(str.isdigit, name))
    return int(digits) if digits else name

def evaluate_background_consistency(
    input_dir, output_dir, mask_dir, result_csv_path,
    vis_dir=None, max_vis=10, device="cuda"
):
    """
    批量评估模型在未编辑区域 (背景保持) 的效果并输出可视化。
    """

    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    metrics_calculator = MetricsCalculator(device=torch.device(device))
    input_images = sorted(glob.glob(os.path.join(input_dir, "*")), key=numerical_sort_key)
    output_images = sorted(glob.glob(os.path.join(output_dir, "*")), key=numerical_sort_key)
    mask_images   = sorted(glob.glob(os.path.join(mask_dir, "*")), key=numerical_sort_key)

    assert len(input_images) == len(output_images) == len(mask_images), "输入数量不一致！"

    results = []
    for idx, (in_path, out_path, mask_path) in enumerate(tqdm(
        zip(input_images, output_images, mask_images),
        total=len(input_images),
        desc="Evaluating background consistency"
    )):
        try:
            # ---- 文件名提取 ----
            img_name = os.path.basename(in_path)

            # ---- 以输入图的尺寸为基准对齐 ----
            with Image.open(in_path) as im:
                W, H = im.size
            target_size = (W, H)

            img_in = load_image_as_np(in_path, target_size)
            img_out = load_image_as_np(out_path, target_size)
            mask = load_mask_as_np(mask_path, target_size)
            mask_unedit = 1.0 - mask  # 反转 mask：1=未编辑区域

            # ---- 指标计算 ----
            psnr = metrics_calculator.calculate_psnr(img_out, img_in,
                                                     mask_pred=mask_unedit,
                                                     mask_gt=mask_unedit)
            ssim = metrics_calculator.calculate_ssim(img_out, img_in,
                                                     mask_pred=mask_unedit,
                                                     mask_gt=mask_unedit)
            lpips = metrics_calculator.calculate_lpips(img_out, img_in,
                                                       mask_pred=mask_unedit,
                                                       mask_gt=mask_unedit)
            struct_dist = metrics_calculator.calculate_structure_distance(
                img_out, img_in, mask_pred=mask_unedit, mask_gt=mask_unedit)

            results.append({
                "image_name": img_name,
                "psnr_unedit_part": psnr,
                "ssim_unedit_part": ssim,
                "lpips_unedit_part": lpips,
                "structure_distance_unedit_part": struct_dist
            })

            # ---- 输出部分可视化 ----
            if vis_dir and idx < max_vis:
                save_visualization(img_in, img_out, mask_unedit, img_name, vis_dir)

        except Exception as e:
            print(f"[WARN] {os.path.basename(in_path)} 计算失败: {e}")
            continue

    # ---- 保存 CSV ----
    fieldnames = list(results[0].keys()) if results else []
    with open(result_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ---- 计算平均值 ----
    if results:
        avg = {k: np.mean([r[k] for r in results if isinstance(r[k], (int, float))])
               for k in results[0] if k != "image_name"}
        avg["image_name"] = "AVERAGE"
        with open(result_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(avg)

        print("\n背景保持指标计算完成！")
        print("平均结果：")
        for k, v in avg.items():
            if k != "image_name":
                print(f"{k}: {v:.4f}")

        if vis_dir:
            print(f"\n已保存可视化样例到 {vis_dir}")
    else:
        print("没有有效的评测结果！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate background (unedit part) consistency.")
    parser.add_argument("--input_dir", required=True, help="Path to input (original) images.")
    parser.add_argument("--output_dir", required=True, help="Path to model output images.")
    parser.add_argument("--mask_dir", required=True, help="Path to masks (1=edit part).")
    parser.add_argument("--result_csv", default="results/background_metrics.csv", help="Output CSV path.")
    parser.add_argument("--vis_dir", default="results/vis_bg", help="Folder to save visualization images.")
    parser.add_argument("--max_vis", type=int, default=10, help="Max number of visualizations to save.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    args = parser.parse_args()

    evaluate_background_consistency(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        result_csv_path=args.result_csv,
        vis_dir=args.vis_dir,
        max_vis=args.max_vis,
        device=args.device,
    )