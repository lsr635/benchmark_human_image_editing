import os, sys, glob
from PIL import Image
import pandas as pd

# Add ImagenHub/src to the python path
sys.path.insert(0, '/project/imgtextmod/imagenhub/liusr/ImagenHub/src')

from imagen_hub.metrics.dreamsim_metric import MetricDreamSim
from imagen_hub.utils import load_image

def batch_evaluate_dreamsim(results_dir, input_dir, token_dir, csv_path, model_name):
    """
    Use DreamSim to evaluate the similarity of generated image and reference image
    """
    # initalize model
    dreamsim_model = MetricDreamSim(device="cuda")

    # get all generated images
    generated_images = glob.glob(os.path.join(results_dir, "*.jpg"))
    generated_images.sort()

    results = []

    for gen_img_path in generated_images:
        img_name = os.path.basename(gen_img_path)
        sample_id = img_name.replace('.jpg','')

        print(f"Processing {img_name}...")

        generated_image = load_image(gen_img_path)
        reference_path = os.path.join(token_dir, f"{sample_id}.jpg")
        source_path = os.path.join(input_dir, f"{sample_id}.jpg")

        if not os.path.isfile(reference_path) or not os.path.isfile(source_path):
            print(f"warn: lack of image: {sample_id}")
            continue

        reference_image = load_image(reference_path)
        source_image = load_image(source_path)

        score_ref = dreamsim_model.evaluate(reference_image, generated_image)
        score_src = dreamsim_model.evaluate(source_image, generated_image)

        results.append(
            {
                "sample_id": sample_id,
                "model": model_name,
                "PhotoSwap_vs_reference": score_ref,
                "PhotoSwap_vs_source": score_src,
            }
        )

    if results:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"dreamsim results are saved in: {csv_path}")
    else:
        print(f"No results.")

if __name__ == "__main__":
    results_dir = "/project/imgtextmod/imagenhub/liusr/ImagenHub/results/ImagenHub_Subject-Driven_IE/DreamEdit" 
    input_dir = "/project/imgtextmod/imagenhub/liusr/ImagenHub/results/ImagenHub_Subject-Driven_IE/input"
    token_dir = "/project/imgtextmod/imagenhub/liusr/ImagenHub/results/ImagenHub_Subject-Driven_IE/token"
    csv_path = "/project/imgtextmod/imagenhub/liusr/benchmark/eval_results/dreamsim_scores_DreamEdit.csv"
    model_name = "DreamEdit"

    batch_evaluate_dreamsim(results_dir, input_dir, token_dir, csv_path, model_name)

