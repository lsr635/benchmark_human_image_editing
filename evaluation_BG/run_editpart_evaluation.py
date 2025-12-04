import os
import csv
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from matrics_calculator import MetricsCalculator
from evaluate import calculate_metric


def load_mask(mask_path, size=(512, 512)):
    """加载掩码：白=编辑区域"""
    mask = Image.open(mask_path).convert("L").resize(size)
    mask = np.array(mask, dtype=np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    return mask


def list_images(directory):
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def visualize_samples(image_results, output_dir, token_dir, mask_dir, vis_dir, num_samples=10):
    """
    可视化四幅图：
      1. Token (reference)
      2. Output (model)
      3. Processed Token (完整原图)
      4. Processed Output (仅mask区域)
    """
    os.makedirs(vis_dir, exist_ok=True)
    sample_images = random.sample(image_results, min(num_samples, len(image_results)))

    for item in sample_images:
        img_name = item["image_name"]
        output_path = os.path.join(output_dir, img_name)
        token_path = os.path.join(token_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        if not all(os.path.exists(p) for p in [output_path, token_path, mask_path]):
            continue

        output_img = Image.open(output_path).convert("RGB").resize((512, 512))
        token_img = Image.open(token_path).convert("RGB").resize((512, 512))
        mask = load_mask(mask_path, (512, 512)).squeeze()  # 白=编辑区域

        output_np = np.array(output_img, dtype=np.float32)
        token_np = np.array(token_img, dtype=np.float32)
        mask_3c = np.stack([mask]*3, axis=-1)

        # === Processed Token: 原图 ===
        processed_token = token_np.copy()

        # === Processed Output: 仅显示编辑区域，其余灰化 ===
        gray_bg = np.ones_like(output_np) * 100
        processed_output = output_np * mask_3c + gray_bg * (1 - mask_3c)
        processed_output = np.clip(processed_output, 0, 255)

        processed_token_img = Image.fromarray(processed_token.astype(np.uint8))
        processed_output_img = Image.fromarray(processed_output.astype(np.uint8))

        # === 绘制 4 张图像 ===
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(token_img)
        axes[0].set_title("Token (reference)")
        axes[0].axis("off")

        axes[1].imshow(output_img)
        axes[1].set_title("Output (model)")
        axes[1].axis("off")

        axes[2].imshow(processed_token_img)
        axes[2].set_title("Processed Token (original)")
        axes[2].axis("off")

        axes[3].imshow(processed_output_img)
        axes[3].set_title("Processed Output (mask region)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"vis_{img_name}"), bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(sample_images)} visualization samples to {vis_dir}")


def evaluate_dataset(
    output_dir,
    token_dir,
    mask_dir,
    metrics,
    prompts_json=None,
    image_size=(512, 512),
    device="cuda",
):
    """评估 ACE++ output 的编辑部分与 token reference 的相似度。"""
    os.makedirs(output_dir, exist_ok=True)

    metrics_calculator = MetricsCalculator(device=device)
    prompts = {}

    if prompts_json and os.path.exists(prompts_json):
        with open(prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    image_names = list_images(output_dir)
    results = []

    print(f"\n=== Evaluating {len(image_names)} images (ACE++ edited region vs token reference) ===")

    for img_name in tqdm(image_names):
        output_path = os.path.join(output_dir, img_name)
        token_path = os.path.join(token_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        if not (os.path.exists(output_path) and os.path.exists(token_path) and os.path.exists(mask_path)):
            print(f" Missing file: {img_name}, skipping.")
            continue

        output_img = Image.open(output_path).convert("RGB").resize(image_size)
        token_img = Image.open(token_path).convert("RGB").resize(image_size)
        mask = load_mask(mask_path, image_size)  # 白=编辑区域

        # ✅ 为避免 None 报错，src_mask 全为 1（整图参与评估）
        src_mask = np.ones_like(mask)

        prompt_text = prompts.get(img_name, "")
        image_result = {"image_name": img_name}

        for metric in metrics:
            value = calculate_metric(
                metrics_calculator=metrics_calculator,
                metric=metric,
                src_image=token_img,    # 参考人物图
                tgt_image=output_img,   # 模型输出编辑结果
                src_mask=src_mask,      # token整图有效
                tgt_mask=mask,          # output仅编辑部分比较
                src_prompt=prompt_text,
                tgt_prompt=prompt_text,
            )
            image_result[metric] = value

        results.append(image_result)

    return results


def save_results_csv(filepath, results, metrics):
    if not results:
        print("No results to save.")
        return

    summary = compute_averages(results, metrics)
    average_row = {"image_name": "average"}
    average_row.update(summary)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name"] + metrics)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow(average_row)

    print(f"Results saved to {filepath}")
    print("Average scores:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if np.isfinite(v) else f"{k}: nan")


def compute_averages(results, metrics):
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in results if isinstance(r[metric], (float, int))]
        summary[metric] = float(np.mean(values)) if values else float("nan")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ACE++ output edited region vs. token reference image.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--token_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--vis_dir", required=True)
    parser.add_argument("--prompts_json", type=str, default=None)
    parser.add_argument("--save_csv", default="./editpart_results.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_vis", type=int, default=10)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "clip_similarity_target_image_edit_part",
            "lpips_edit_part",
            "structure_distance_edit_part",
        ],
        help="Metrics to evaluate edited regions"
    )
    args = parser.parse_args()

    results = evaluate_dataset(
        output_dir=args.output_dir,
        token_dir=args.token_dir,
        mask_dir=args.mask_dir,
        metrics=args.metrics,
        prompts_json=args.prompts_json,
        device=args.device,
    )

    save_results_csv(args.save_csv, results, args.metrics)

    visualize_samples(
        results,
        output_dir=args.output_dir,
        token_dir=args.token_dir,
        mask_dir=args.mask_dir,
        vis_dir=args.vis_dir,
        num_samples=args.num_vis
    )