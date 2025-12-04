# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import glob
import io
import os
from contextlib import suppress

from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS

from examples.examples import all_examples
from inference.ace_plus_diffusers import ACEPlusDiffuserInference
inference_dict = {
    "ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference
}

fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)
def _read_bytes(path):
    if path is None:
        return None
    with suppress(Exception):
        return FS.get_object(path)
    with open(path, "rb") as local_f:
        return local_f.read()


def run_one_case(pipe,
                input_image = None,
                input_mask = None,
                input_reference_image = None,
                save_path = "examples/output/example.png",
                instruction = "",
                output_h = 1024,
                output_w = 1024,
                seed = -1,
                sample_steps = None,
                guide_scale = None,
                repainting_scale = None,
                model_path = None,
                **kwargs):
    if input_image is not None:
        image_bytes = _read_bytes(input_image)
        if image_bytes is not None:
            input_image = Image.open(io.BytesIO(image_bytes))
            input_image = pillow_convert(input_image, "RGB")
    if input_mask is not None:
        mask_bytes = _read_bytes(input_mask)
        if mask_bytes is not None:
            input_mask = Image.open(io.BytesIO(mask_bytes))
            input_mask = pillow_convert(input_mask, "L")
    if input_reference_image is not None:
        ref_bytes = _read_bytes(input_reference_image)
        if ref_bytes is not None:
            input_reference_image = Image.open(io.BytesIO(ref_bytes))
            input_reference_image = pillow_convert(input_reference_image, "RGB")

    image, seed = pipe(
        reference_image=input_reference_image,
        edit_image=input_image,
        edit_mask=input_mask,
        prompt=instruction,
        output_height=output_h,
        output_width=output_w,
        sampler='flow_euler',
        sample_steps=sample_steps or pipe.input.get("sample_steps", 28),
        guide_scale=guide_scale or pipe.input.get("guide_scale", 50),
        seed=seed,
        repainting_scale=repainting_scale or pipe.input.get("repainting_scale", 1.0),
        lora_path = model_path
    )
# ================================================= 
    with suppress(Exception):
        with FS.put_to(save_path) as local_path:
            image.save(local_path)
            return local_path, seed
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    image.save(save_path)
    return save_path, seed
# ======================================================

def run():
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--instruction',
                        dest='instruction',
                        help='The instruction for editing or generating!',
                        default="")
    parser.add_argument('--output_h',
                        dest='output_h',
                        help='The height of output image for generation tasks!',
                        type=int,
                        default=1024)
    parser.add_argument('--output_w',
                        dest='output_w',
                        help='The width of output image for generation tasks!',
                        type=int,
                        default=1024)
    parser.add_argument('--input_reference_image',
                        dest='input_reference_image',
                        help='The input reference image!',
                        default=None
                        )
    parser.add_argument('--input_image',
                        dest='input_image',
                        help='The input image!',
                        default=None
                        )
    parser.add_argument('--input_mask',
                        dest='input_mask',
                        help='The input mask!',
                        default=None
                        )
    parser.add_argument('--save_path',
                        dest='save_path',
                        help='The save path for output image!',
                        default='examples/output_images/output.png'
                        )
    parser.add_argument('--seed',
                        dest='seed',
                        help='The seed for generation!',
                        type=int,
                        default=-1)

    parser.add_argument('--step',
                        dest='step',
                        help='The sample step for generation!',
                        type=int,
                        default=None)

    parser.add_argument('--guide_scale',
                        dest='guide_scale',
                        help='The guide scale for generation!',
                        type=int,
                        default=None)

    parser.add_argument('--repainting_scale',
                        dest='repainting_scale',
                        help='The repainting scale for content filling generation!',
                        type=int,
                        default=None)

    parser.add_argument('--task_type',
                        dest='task_type',
                        choices=['portrait', 'subject', 'local_editing'],
                        help="Choose the task type.",
                        default='')

    parser.add_argument('--task_model',
                        dest='task_model',
                        help='The models list for different tasks!',
                        default="./models/model_zoo.yaml")
# ==========================================================================================
    parser.add_argument('--dataset_path',
                        dest='dataset_path',
                        help='Directory that contains prepared samples (e.g. 0000/input.png).',
                        default=None)

    parser.add_argument('--case_id',
                        dest='case_id',
                        help='Folder name inside dataset_path to run. Defaults to the first folder.',
                        default=None)

    parser.add_argument('--output_dir',
                        dest='output_dir',
                        help='Directory to store generated images when using dataset_path.',
                        default=None)

    parser.add_argument('--run_all',
                        dest='run_all',
                        action='store_true',
                        help='Process every case folder under dataset_path.',
                        default=False)

    parser.add_argument('--max_cases',
                        dest='max_cases',
                        type=int,
                        help='Optional limit when using --run_all to only process the first N cases.',
                        default=None)

# =============================================================================================
    parser.add_argument('--infer_type',
                        dest='infer_type',
                        choices=['diffusers'],
                        default='diffusers',
                        help="Choose the inference scripts. 'native' refers to using the official implementation of ace++, "
                             "while 'diffusers' refers to using the adaptation for diffusers")

    parser.add_argument('--cfg_folder',
                        dest='cfg_folder',
                        help='The inference config!',
                        default="./config")

    cfg = Config(load=True, parser_ins=parser)

    model_yamls = glob.glob(os.path.join(cfg.args.cfg_folder, '*.yaml'))
    model_choices = dict()
    for i in model_yamls:
        model_cfg = Config(load=True, cfg_file=i)
        model_name = model_cfg.NAME
        model_choices[model_name] = model_cfg

    if cfg.args.infer_type == "native":
        infer_name = "ace_plus_native_infer"
    elif cfg.args.infer_type == "diffusers":
        infer_name = "ace_plus_diffuser_infer"
    else:
        raise ValueError("infer_type should be native or diffusers")

    assert infer_name in model_choices

    # choose different model
    task_model_cfg = Config(load=True, cfg_file=cfg.args.task_model)

    task_model_dict = {}
    for task_name, task_model in task_model_cfg.MODEL.items():
        task_model_dict[task_name] = task_model


    # choose the inference scripts.
    pipe_cfg = model_choices[infer_name]
    infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE_PLUS")
    pipe = inference_dict[infer_name]()
    pipe.init_from_cfg(pipe_cfg)

# ======================================================================
# changes
# ======================================================================
    cases_for_batch = []
    dataset_case_dir = None
    initial_instruction = cfg.args.instruction

    if cfg.args.run_all and cfg.args.dataset_path is None:
        raise ValueError("--run_all requires --dataset_path to be specified.")

    if cfg.args.dataset_path is not None:
        dataset_root = cfg.args.dataset_path
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"dataset_path `{dataset_root}` does not exist or is not a directory.")
        candidate_dirs = sorted(
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        )
        if not candidate_dirs:
            raise ValueError(f"dataset_path `{dataset_root}` has no subfolders.")

        def _candidate(case_dir, *path_names):
            for path_name in path_names:
                candidate = os.path.join(case_dir, path_name)
                if os.path.exists(candidate):
                    return candidate
            return None

        def _build_case(case_id):
            case_dir = os.path.join(dataset_root, case_id)
            if not os.path.isdir(case_dir):
                raise FileNotFoundError(f"case_id `{case_id}` does not exist under `{dataset_root}`.")
            case_inputs = {
                "input_image": _candidate(case_dir, "input.png", "add.png"),
                "input_mask": _candidate(case_dir, "mask.png", "label_mask.png"),
                "input_reference_image": _candidate(case_dir, "subject.png", "token_add.png", "token.png"),
            }
            instruction_value = initial_instruction.strip() if isinstance(initial_instruction, str) else initial_instruction
            if not instruction_value:
                prompt_path = _candidate(case_dir, "prompt.txt", "prompt_addition_qwen.txt", "prompt_add.txt")
                if prompt_path is not None:
                    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                        instruction_value = prompt_file.read().strip()
            case_inputs["instruction"] = instruction_value or ""

            output_dir = cfg.args.output_dir
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                case_inputs["save_path"] = os.path.join(output_dir, f"{case_id}.png")
            else:
                base_path = cfg.args.save_path or "examples/output_images/output.png"
                base_dir, base_name = os.path.split(base_path)
                if not base_dir:
                    base_dir = "examples/output_images"
                os.makedirs(base_dir, exist_ok=True)
                name_root, ext = os.path.splitext(base_name if base_name else "output.png")
                if not ext:
                    ext = ".png"
                case_inputs["save_path"] = os.path.join(base_dir, f"{name_root}_{case_id}{ext}")
            return case_inputs

        if cfg.args.run_all:
            if cfg.args.case_id:
                requested = [cid.strip() for cid in str(cfg.args.case_id).split(',') if cid.strip()]
                missing = [cid for cid in requested if cid not in candidate_dirs]
                if missing:
                    raise FileNotFoundError(f"Requested case_ids {missing} not found under `{dataset_root}`.")
                case_ids = requested
            else:
                case_ids = candidate_dirs
            if cfg.args.max_cases is not None:
                if cfg.args.max_cases <= 0:
                    raise ValueError("--max_cases must be a positive integer when provided.")
                case_ids = case_ids[:cfg.args.max_cases]
            if not case_ids:
                raise ValueError("No cases available to process after applying filters.")
            for cid in case_ids:
                cases_for_batch.append((cid, _build_case(cid)))
        else:
            case_id = cfg.args.case_id or candidate_dirs[0]
            dataset_case_dir = os.path.join(dataset_root, case_id)
            case_inputs = _build_case(case_id)
            if cfg.args.save_path:
                case_inputs["save_path"] = cfg.args.save_path
            if cfg.args.input_image is None:
                cfg.args.input_image = case_inputs["input_image"]
            if cfg.args.input_mask is None:
                cfg.args.input_mask = case_inputs["input_mask"]
            if cfg.args.input_reference_image is None:
                cfg.args.input_reference_image = case_inputs["input_reference_image"]
            cfg.args.instruction = case_inputs["instruction"]
            cfg.args.save_path = case_inputs["save_path"]

# ===============================================================================
    run_examples = (
        not cfg.args.run_all
        and cfg.args.instruction == ""
        and cfg.args.input_image is None
        and cfg.args.input_reference_image is None
    )

    if run_examples:
        params = {
            "output_h": cfg.args.output_h,
            "output_w": cfg.args.output_w,
            "sample_steps": cfg.args.step,
            "guide_scale": cfg.args.guide_scale
        }
        # run examples

        for example in all_examples:
            example["model_path"] = FS.get_from(task_model_dict[example["task_type"].upper()]["MODEL_PATH"])
            example.update(params)
            if example["edit_type"] == "repainting":
                example["repainting_scale"] = 1.0
            else:
                example["repainting_scale"] = task_model_dict[example["task_type"].upper()].get("REPAINTING_SCALE", 1.0)
            print(example)
            local_path, seed = run_one_case(pipe, **example)

    else:
        task_key = cfg.args.task_type.upper()
        if task_key not in task_model_dict:
            raise KeyError(f"task_type `{cfg.args.task_type}` not found in {cfg.args.task_model}. Available: {list(task_model_dict)}")
        task_entry = task_model_dict[task_key]
        model_path_local = FS.get_from(task_entry["MODEL_PATH"])
        default_repainting = cfg.args.repainting_scale
        if default_repainting is None:
            default_repainting = task_entry.get("REPAINTING_SCALE", 1.0)

        common_params = {
            "output_h": cfg.args.output_h,
            "output_w": cfg.args.output_w,
            "sample_steps": cfg.args.step,
            "guide_scale": cfg.args.guide_scale,
            "seed": cfg.args.seed,
        }

        if cfg.args.run_all:
            if not cases_for_batch:
                raise ValueError("No cases gathered for batch processing. Check dataset_path/run_all settings.")
            for case_id, case_inputs in cases_for_batch:
                params = {
                    "input_image": case_inputs["input_image"],
                    "input_mask": case_inputs["input_mask"],
                    "input_reference_image": case_inputs["input_reference_image"],
                    "save_path": case_inputs["save_path"],
                    "instruction": case_inputs.get("instruction", cfg.args.instruction),
                    "repainting_scale": default_repainting,
                    "model_path": model_path_local,
                }
                params.update(common_params)
                local_path, seed = run_one_case(pipe, **params)
                print(f"{case_id}: {local_path} {seed}")
            return

        params = {
            "input_image": cfg.args.input_image,
            "input_mask": cfg.args.input_mask,
            "input_reference_image": cfg.args.input_reference_image,
            "save_path": cfg.args.save_path,
            "instruction": cfg.args.instruction,
            "repainting_scale": default_repainting,
            "model_path": model_path_local,
        }
        params.update(common_params)
        local_path, seed = run_one_case(pipe, **params)
        print(local_path, seed)

if __name__ == '__main__':
    run()