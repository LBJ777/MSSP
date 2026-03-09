"""
data/transforms.py
------------------
Image pre-processing pipelines for MSSP.
Copied from DRIFT_new for standalone operation.
"""

from __future__ import annotations

import random
from io import BytesIO
from typing import Optional, Tuple, List

from PIL import Image
import torch
import torchvision.transforms as T

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# [-1, 1] normalization for ADM UNet input
ADM_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
ADM_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)


class JPEGCompression:
    def __init__(self, quality_range: Tuple[int, int] = (75, 95)) -> None:
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianBlur:
    def __init__(self, sigma_range: Tuple[float, float] = (0.1, 2.0)) -> None:
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        import PIL.ImageFilter as IF
        sigma = random.uniform(*self.sigma_range)
        return img.filter(IF.GaussianBlur(radius=sigma))


def get_transforms(
    split: str,
    image_size: int = 256,
    noise_type: Optional[str] = None,
    jpeg_quality_range: Tuple[int, int] = (75, 95),
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
    no_flip: bool = False,
    no_crop: bool = False,
    no_resize: bool = False,
    normalize_for_adm: bool = False,
) -> T.Compose:
    """Return transform pipeline for the given split.

    Args:
        normalize_for_adm: If True, normalize to [-1, 1] for ADM UNet input.
            If False, use ImageNet normalization (default).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(
            f"split must be 'train', 'val', or 'test', got '{split}'."
        )

    is_train = split == "train"
    steps: List = []

    if not no_resize:
        steps.append(T.Resize((image_size, image_size)))

    if is_train:
        if noise_type == "jpg":
            steps.append(JPEGCompression(quality_range=jpeg_quality_range))
        elif noise_type == "blur":
            steps.append(GaussianBlur(sigma_range=blur_sigma_range))
        if not no_crop:
            steps.append(T.RandomCrop(image_size))
        if not no_flip:
            steps.append(T.RandomHorizontalFlip())
    else:
        if not no_crop:
            steps.append(T.CenterCrop(image_size))

    steps.append(T.ToTensor())

    if normalize_for_adm:
        steps.append(T.Normalize(mean=list(ADM_MEAN), std=list(ADM_STD)))
    else:
        steps.append(T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)))

    return T.Compose(steps)


def get_adm_transforms(image_size: int = 256) -> T.Compose:
    """Get transforms that output images in [-1, 1] for ADM UNet input."""
    return get_transforms(
        split="test",
        image_size=image_size,
        normalize_for_adm=True,
    )


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert ImageNet normalisation for visualisation."""
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif tensor.ndim == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)

    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
