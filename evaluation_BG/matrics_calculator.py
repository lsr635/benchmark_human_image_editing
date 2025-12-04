import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Resize

from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class VitExtractor:
    BLOCK_KEY = "block"
    ATTN_KEY = "attn"
    PATCH_IMD_KEY = "patch_imd"
    QKV_KEY = "qkv"
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name: str, device: torch.device):
        self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers: List[torch.utils.hooks.RemovableHandle] = []
        self.layers_dict: Dict[str, List[int]] = {}
        self.outputs_dict: Dict[str, List[torch.Tensor]] = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()
        self.device = device

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = list(range(12))
        self.layers_dict[VitExtractor.ATTN_KEY] = list(range(12))
        self.layers_dict[VitExtractor.QKV_KEY] = list(range(12))
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = list(range(12))
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(
                    block.register_forward_hook(self._get_block_hook())
                )
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(
                    block.attn.attn_drop.register_forward_hook(
                        self._get_attn_hook()
                    )
                )
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(
                    block.attn.qkv.register_forward_hook(self._get_qkv_hook())
                )
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(
                    block.attn.register_forward_hook(self._get_patch_imd_hook())
                )

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, _input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, _input, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, _input, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_patch_output(model, _input, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_patch_output

    def _forward_with_hooks(self, input_img: torch.Tensor):
        self._register_hooks()
        try:
            self.model(input_img.to(self.device))
        finally:
            self._clear_hooks()

    def get_feature_from_input(self, input_img: torch.Tensor):
        self._forward_with_hooks(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img: torch.Tensor):
        self._forward_with_hooks(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img: torch.Tensor):
        self._forward_with_hooks(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape: Tuple[int, ...]):
        _, _, _, w = input_img_shape
        return w // self.get_patch_size()

    def get_height_patch_num(self, input_img_shape: Tuple[int, ...]):
        _, _, h, _ = input_img_shape
        return h // self.get_patch_size()

    def get_patch_num(self, input_img_shape: Tuple[int, ...]):
        return 1 + self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(
            input_img_shape
        )

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv: torch.Tensor, input_img_shape: Tuple[int, ...]):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = (
            qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num)
            .permute(1, 2, 0, 3)[0]
            .contiguous()
        )
        return q

    def get_keys_from_qkv(self, qkv: torch.Tensor, input_img_shape: Tuple[int, ...]):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = (
            qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num)
            .permute(1, 2, 0, 3)[1]
            .contiguous()
        )
        return k

    def get_values_from_qkv(self, qkv: torch.Tensor, input_img_shape: Tuple[int, ...]):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = (
            qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num)
            .permute(1, 2, 0, 3)[2]
            .contiguous()
        )
        return v

    def get_keys_from_input(self, input_img: torch.Tensor, layer_num: int):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img: torch.Tensor, layer_num: int):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = self.attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

    @staticmethod
    def attn_cosine_sim(x: torch.Tensor, eps: float = 1e-8):
        x = x[0]
        norm1 = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
        sim_matrix = (x @ x.permute(0, 2, 1)) / factor
        return sim_matrix

class LossG(torch.nn.Module):
    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.extractor = VitExtractor(model_name=cfg["dino_model_name"], device=device)

        imagenet_norm = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
        global_resize_transform = Resize(cfg["dino_global_patch_size"], antialias=True)

        self.global_transform = transforms.Compose(
            [
                global_resize_transform,
                imagenet_norm,
            ]
        )

        # 使用 get 防止缺 key
        self.lambdas = dict(
            lambda_global_cls=cfg.get("lambda_global_cls", 0.0),
            lambda_global_ssim=0.0,
            lambda_entire_ssim=0.0,
            lambda_entire_cls=0.0,
            lambda_global_identity=0.0,
        )

    def update_lambda_config(self, step: Optional[int]):
        if step is None:
            return

        cls_warmup = self.cfg.get("cls_warmup", 0)
        if step == cls_warmup:
            self.lambdas["lambda_global_ssim"] = self.cfg.get("lambda_global_ssim", 0.0)
            self.lambdas["lambda_global_identity"] = self.cfg.get(
                "lambda_global_identity", 0.0
            )

        entire_A_every = self.cfg.get("entire_A_every", 1)
        if entire_A_every > 0 and step % entire_A_every == 0:
            self.lambdas["lambda_entire_ssim"] = self.cfg.get("lambda_entire_ssim", 0.0)
            self.lambdas["lambda_entire_cls"] = self.cfg.get("lambda_entire_cls", 0.0)
        else:
            self.lambdas["lambda_entire_ssim"] = 0.0
            self.lambdas["lambda_entire_cls"] = 0.0

    def forward(self, outputs: Dict, inputs: Dict):
        step = inputs.get("step")
        self.update_lambda_config(step)
        losses = {}
        loss_G = 0.0

        if self.lambdas["lambda_global_ssim"] > 0:
            losses["loss_global_ssim"] = self.calculate_global_ssim_loss(
                outputs["x_global"], inputs["A_global"]
            )
            loss_G += losses["loss_global_ssim"] * self.lambdas["lambda_global_ssim"]

        if self.lambdas["lambda_entire_ssim"] > 0:
            losses["loss_entire_ssim"] = self.calculate_global_ssim_loss(
                outputs["x_entire"], inputs["A"]
            )
            loss_G += losses["loss_entire_ssim"] * self.lambdas["lambda_entire_ssim"]

        if self.lambdas["lambda_entire_cls"] > 0:
            losses["loss_entire_cls"] = self.calculate_crop_cls_loss(
                outputs["x_entire"], inputs["B_global"]
            )
            loss_G += losses["loss_entire_cls"] * self.lambdas["lambda_entire_cls"]

        if self.lambdas["lambda_global_cls"] > 0:
            losses["loss_global_cls"] = self.calculate_crop_cls_loss(
                outputs["x_global"], inputs["B_global"]
            )
            loss_G += losses["loss_global_cls"] * self.lambdas["lambda_global_cls"]

        if self.lambdas["lambda_global_identity"] > 0:
            losses["loss_global_id_B"] = self.calculate_global_id_loss(
                outputs["y_global"], inputs["B_global"]
            )
            loss_G += (
                losses["loss_global_id_B"] * self.lambdas["lambda_global_identity"]
            )

        losses["loss"] = loss_G
        return losses

    def calculate_global_ssim_loss(
        self, outputs: List[torch.Tensor], inputs: List[torch.Tensor]
    ):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(
                    a, layer_num=11
                )
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b, layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(
        self, outputs: List[torch.Tensor], inputs: List[torch.Tensor]
    ):
        loss = 0.0
        for a, b in zip(outputs, inputs):
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(
        self, outputs: List[torch.Tensor], inputs: List[torch.Tensor]
    ):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a, 11)
            keys_b = self.extractor.get_keys_from_input(b, 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss


# ============================================================
# MetricsCalculator（新的实现）
# ============================================================
class MetricsCalculator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.clip_metric_calculator = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze", reduction="none"
        ).to(device)
        self.structure_distance_metric_calculator = LossG(
            cfg={
                "dino_model_name": "dino_vitb8",
                "dino_global_patch_size": 224,
                "lambda_global_cls": 10.0,
                "lambda_global_ssim": 1.0,
                "lambda_global_identity": 1.0,
                "entire_A_every": 75,
                "lambda_entire_cls": 10.0,
                "lambda_entire_ssim": 1.0,
                "cls_warmup": 0,
            },
            device=device,
        )

        self._ssim_kernel_cache: Dict = {}
        self._ssim_padding_cache: Dict = {}

    # ------- 基础工具函数 -------
    @staticmethod
    def _to_numpy_image(img) -> np.ndarray:
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
        else:
            arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr

    @staticmethod
    def _ensure_mask_shape(mask: np.ndarray, spatial_shape: Tuple[int, int]) -> np.ndarray:
        if mask.ndim == 2:
            mask = mask[..., None]
        elif mask.ndim == 3 and mask.shape[2] > 1:
            mask = np.mean(mask, axis=2, keepdims=True)

        mask = mask.astype(np.float32)
        if mask.shape[0] != spatial_shape[0] or mask.shape[1] != spatial_shape[1]:
            raise ValueError("mask 的尺寸与图像不一致")
        return np.clip(mask, 0.0, 1.0)

    def _combine_masks(
        self,
        mask_pred: Optional[np.ndarray],
        mask_gt: Optional[np.ndarray],
        spatial_shape: Tuple[int, int],
    ) -> np.ndarray:
        if mask_pred is None and mask_gt is None:
            return np.ones((*spatial_shape, 1), dtype=np.float32)
        if mask_pred is None or mask_gt is None:
            raise ValueError("mask_pred 和 mask_gt 必须同时提供或同时为 None")

        mask_pred = self._ensure_mask_shape(np.array(mask_pred), spatial_shape)
        mask_gt = self._ensure_mask_shape(np.array(mask_gt), spatial_shape)
        combined = (mask_pred * mask_gt) > 0.5
        return combined.astype(np.float32)

    @staticmethod
    def _expand_mask_to_channels(mask: np.ndarray, channels: int) -> np.ndarray:
        if mask.shape[2] == channels:
            return mask
        return np.repeat(mask, channels, axis=2)

    def _mask_to_tensor(
        self, mask: np.ndarray, size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        tensor = torch.from_numpy(mask.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        if size is not None and tensor.shape[-2:] != size:
            tensor = F.interpolate(
                tensor, size=size, mode="bilinear", align_corners=False
            )
            tensor = tensor.clamp(0.0, 1.0)
        return tensor

    @staticmethod
    def _masked_pixel_count(mask: np.ndarray) -> float:
        return float(mask[..., 0].sum())

    # ------- MSE & PSNR -------
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None) -> float:
        img_pred = self._to_numpy_image(img_pred)
        img_gt = self._to_numpy_image(img_gt)
        if img_pred.shape != img_gt.shape:
            raise ValueError("预测图与目标图尺寸不一致")

        mask = self._combine_masks(mask_pred, mask_gt, img_pred.shape[:2])
        if self._masked_pixel_count(mask) <= 0:
            return float("nan")

        mask_c = self._expand_mask_to_channels(mask, img_pred.shape[2])
        diff = (img_pred - img_gt) * mask_c
        denominator = mask_c.sum()
        mse = (diff**2).sum() / (denominator + 1e-8)
        return float(mse)

    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None) -> float:
        mse = self.calculate_mse(img_pred, img_gt, mask_pred, mask_gt)
        if not math.isfinite(mse):
            return float("nan")
        if mse <= 0.0:
            return float("inf")
        psnr = 10.0 * math.log10(1.0 / mse)
        return float(psnr)

    # ------- LPIPS -------
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None) -> float:
        img_pred = self._to_numpy_image(img_pred)
        img_gt = self._to_numpy_image(img_gt)
        mask = self._combine_masks(mask_pred, mask_gt, img_pred.shape[:2])
        masked_pixels = self._masked_pixel_count(mask)
        if masked_pixels <= 0:
            return float("nan")

        mask_tensor = self._mask_to_tensor(mask)
        mask_tensor_3c = mask_tensor.repeat(1, img_pred.shape[2], 1, 1)

        pred_tensor = (
            torch.from_numpy(img_pred.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        gt_tensor = (
            torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )

        pred_for_metric = pred_tensor * mask_tensor_3c + gt_tensor * (1 - mask_tensor_3c)

        with torch.no_grad():
            lpips_map = self.lpips_metric_calculator(
                pred_for_metric * 2.0 - 1.0, gt_tensor * 2.0 - 1.0
            )
            if lpips_map.ndim == 4:
                mask_resized = self._mask_to_tensor(mask, size=lpips_map.shape[-2:])
                weighted = lpips_map * mask_resized
                score = weighted.sum() / (mask_resized.sum() + 1e-8)
                score = score.detach().cpu().item()
            else:
                total_pixels = mask.shape[0] * mask.shape[1]
                base_score = lpips_map.mean().detach().cpu().item()
                score = base_score * total_pixels / masked_pixels

        self.lpips_metric_calculator.reset()
        return float(score)

    # ------- SSIM -------
    def _get_gaussian_kernel(
        self, channels: int, kernel_size: int = 11, sigma: float = 1.5
    ) -> torch.Tensor:
        key = (channels, kernel_size, sigma, self.device)
        if key in self._ssim_kernel_cache:
            return self._ssim_kernel_cache[key]

        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords = coords - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self._ssim_kernel_cache[key] = kernel
        self._ssim_padding_cache[key] = kernel_size // 2
        return kernel

    def _compute_ssim_map(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        channels = pred.shape[1]
        kernel = self._get_gaussian_kernel(channels)
        pad = self._ssim_padding_cache[(channels, kernel.shape[-1], 1.5, self.device)]

        mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
        mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = (
            F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy
        )

        c1 = 0.01**2
        c2 = 0.03**2

        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / (denominator + 1e-8)
        ssim_map = ssim_map.mean(dim=1, keepdim=True)
        return torch.clamp(ssim_map, -1.0, 1.0)

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None) -> float:
        img_pred = self._to_numpy_image(img_pred)
        img_gt = self._to_numpy_image(img_gt)
        mask = self._combine_masks(mask_pred, mask_gt, img_pred.shape[:2])
        masked_pixels = self._masked_pixel_count(mask)
        if masked_pixels <= 0:
            return float("nan")

        mask_tensor = self._mask_to_tensor(mask)

        pred_tensor = (
            torch.from_numpy(img_pred.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        gt_tensor = (
            torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )

        mask_3c = mask_tensor.repeat(1, pred_tensor.shape[1], 1, 1)
        pred_for_metric = pred_tensor * mask_3c + gt_tensor * (1 - mask_3c)

        with torch.no_grad():
            ssim_map = self._compute_ssim_map(pred_for_metric, gt_tensor)
            if mask_tensor.shape[-2:] != ssim_map.shape[-2:]:
                mask_tensor = self._mask_to_tensor(mask, size=ssim_map.shape[-2:])
            score = (ssim_map * mask_tensor).sum() / (mask_tensor.sum() + 1e-8)
            score = score.detach().cpu().item()

        return float(score)

    # ------- 结构距离 -------
    def calculate_structure_distance(
        self, img_pred, img_gt, mask_pred=None, mask_gt=None
    ) -> float:
        img_pred = self._to_numpy_image(img_pred)
        img_gt = self._to_numpy_image(img_gt)
        mask = self._combine_masks(mask_pred, mask_gt, img_pred.shape[:2])
        if self._masked_pixel_count(mask) <= 0:
            return float("nan")

        pred_tensor = (
            torch.from_numpy(img_pred.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        gt_tensor = (
            torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )

        mask_tensor = self._mask_to_tensor(mask).repeat(1, pred_tensor.shape[1], 1, 1)
        pred_for_metric = pred_tensor * mask_tensor + gt_tensor * (1 - mask_tensor)

        with torch.no_grad():
            structure_distance = (
                self.structure_distance_metric_calculator.calculate_global_ssim_loss(
                    gt_tensor, pred_for_metric
                )
            )

        return float(structure_distance.data.cpu().item())

    # ------- CLIP -------
    def calculate_clip_similarity(self, img, txt, mask=None) -> float:
        img_np = np.array(img)

        if mask is not None:
            mask_np = self._combine_masks(mask, mask, img_np.shape[:2])
            if self._masked_pixel_count(mask_np) <= 0:
                return float("nan")

            mask_bool = mask_np[..., 0] > 0.5
            y_idx, x_idx = np.where(mask_bool)
            y0, y1 = y_idx.min(), y_idx.max() + 1
            x0, x1 = x_idx.min(), x_idx.max() + 1

            img_np = img_np[y0:y1, x0:x1]
            mask_crop = mask_bool[y0:y1, x0:x1][..., None].astype(np.float32)
            img_np = (img_np.astype(np.float32) * mask_crop).astype(np.uint8)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            score = self.clip_metric_calculator(img_tensor, txt)

        return float(score.cpu().item())