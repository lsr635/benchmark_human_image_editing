import os, sys, glob
import pandas as pd
import torch

sys.path.insert(0, '/project/imgtextmod/imagenhub/liusr/ImagenHub/src')

from imagen_hub.metrics.dreambooth_metric import MetricCLIP_I, MetricDINO
from imagen_hub.utils import load_image


def batch_evaluate_dreambooth(results_dir, input_dir, token_dir, csv_path, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_metric = MetricCLIP_I(device=device)
    dino_metric = MetricDINO(device=device)

    generated_images = sorted(glob.glob(os.path.join(results_dir, "*.jpg")))
    results = []

    for gen_img_path in generated_images:
        sample_id = os.path.splitext(os.path.basename(gen_img_path))[0]
        token_path = os.path.join(token_dir, f"{sample_id}.jpg")
        input_path = os.path.join(input_dir, f"{sample_id}.jpg")

        if not (os.path.isfile(token_path) and os.path.isfile(input_path)):
            print(f"warn no paired image: {sample_id}")
            continue

        gen_img = load_image(gen_img_path)
        token_img = load_image(token_path)
        input_img = load_image(input_path)

        clip_token = clip_metric.evaluate(token_img, gen_img)
        clip_input = clip_metric.evaluate(input_img, gen_img)
        dino_token = dino_metric.evaluate(token_img, gen_img)
        dino_input = dino_metric.evaluate(input_img, gen_img)

        results.append(
            {
                "sample_id": sample_id,
                "model": model_name,
                "clip_gen_vs_token": clip_token,
                "clip_gen_vs_input": clip_input,
                "dino_gen_vs_token": dino_token,
                "dino_gen_vs_input": dino_input,
            }
        )

    if results:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"DreamBooth metrics are saved in: {csv_path}")
    else:
        print("No results.")


if __name__ == "__main__":
    results_dir = "/project/imgtextmod/imagenhub/liusr/ImagenHub/results/ImagenHub_Subject-Driven_IE/DreamEdit" 
    input_dir = "/project/imgtextmod/imagenhub/liusr/ImagenHub/results/ImagenHub_Subject-Driven_IE/input"
    token_dir = "/project/imgtextmod/imagenhub/liusr/ImagenHub/results/ImagenHub_Subject-Driven_IE/token"
    csv_path="/project/imgtextmod/imagenhub/liusr/benchmark/eval_results/dreambooth_scores_DreamEdit.csv"
    model_name="DreamEdit"

    batch_evaluate_dreambooth(results_dir, input_dir, token_dir, csv_path, model_name)