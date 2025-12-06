import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any
import torchvision.io
from config import ImageMeta

if TYPE_CHECKING:
    from config import DefaultDataSetConfig

    DefaultDataSetConfigType = DefaultDataSetConfig
else:
    DefaultDataSetConfigType = Any


def image_preprocess(
    image: torch.Tensor, config: DefaultDataSetConfigType
) -> tuple[torch.Tensor, ImageMeta]:
    """
    image: (B, C, H, W) or (C, H, W) shape, RGB format
    out: (B, C, H, W)
    Preprocess image tensor for model input.
    It performs the following operations:
    1. Resize image to config.image_size
    2. Pad image to config.INP_SIZE
    3. Convert image from RGB to BGR
    4. Normalize image according to config.IMAGE_MEAN
    """
    h, w = image.shape[-2:]
    # Add batch dimension if missing
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # 1. Resize image to config.image_size
    image = F.interpolate(
        image, size=config.image_size, mode="bilinear", align_corners=True
    )
    target_h, target_w = config.INP_SIZE

    # 2. Pad image to config.INP_SIZE
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = F.pad(
        image,
        [pad_left, pad_right, pad_top, pad_bottom],
        mode="constant",
        value=0,
    )

    # 3. Convert image from RGB to BGR
    image = image.flip(dims=[1])

    # 4. Normalize image according to config.IMAGE_MEAN
    image = image - config.IMAGE_MEAN.view(1, 3, 1, 1)

    window = (pad_top, pad_left, pad_top + h, pad_left + w)
    meta = ImageMeta(window)

    return image, meta


def load_single(
    path: str, config: DefaultDataSetConfigType
) -> tuple[torch.Tensor, ImageMeta]:
    """Load single image from path and preprocess it"""
    image = torchvision.io.decode_image(
        path, mode=torchvision.io.ImageReadMode.RGB
    ).float()  # [C, H, W]
    image, meta = image_preprocess(image, config)  # [C, H, W]
    return image, meta
