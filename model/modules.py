import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor
from typing import List, Tuple, TYPE_CHECKING, Any
import math

if TYPE_CHECKING:
    from config import AnchorConf

    def roi_align(
        input: Tensor,
        boxes: Tensor | list[Tensor],
        output_size: Tuple[int, int],
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
        aligned: bool = False,
    ) -> Tensor: ...
else:
    from torchvision.ops import roi_align

    AnchorConf = Any


TYPE_FPN_OUTPUT = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def generate_anchors(
    scales: list[int],
    ratios: list[float],
    shape: tuple[int, int],
    feature_stride: int,
    anchor_stride: int,
) -> torch.Tensor:
    """
    PyTorch version of anchor generation.
    """
    # Convert to tensors
    scales_tensor = tensor(scales, dtype=torch.float32)
    ratios_tensor = tensor(ratios, dtype=torch.float32)

    # Get all combinations of scales and ratios
    scales_grid, ratios_grid = torch.meshgrid(
        scales_tensor, ratios_tensor, indexing="ij"
    )
    scales_flat = scales_grid.flatten()
    ratios_flat = ratios_grid.flatten()

    # Calculate base anchor dimensions
    heights = scales_flat / torch.sqrt(ratios_flat)
    widths = scales_flat * torch.sqrt(ratios_flat)

    # Generate shift coordinates
    height, width = shape
    shifts_y = torch.arange(0, height, anchor_stride) * feature_stride
    shifts_x = torch.arange(0, width, anchor_stride) * feature_stride

    # Create grid of shifts
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    # Number of shifts and anchors
    num_shifts = shift_x.shape[0]
    num_anchors = heights.shape[0]

    # Combine shifts and anchors (broadcasting)
    # Shape: (num_shifts * num_anchors, 4) where 4 is (center_y, center_x, height, width)
    all_anchors = torch.zeros((num_shifts * num_anchors, 4), dtype=torch.float32)

    # Expand shifts and anchors for broadcasting
    shift_x_expanded = shift_x.repeat_interleave(num_anchors)
    shift_y_expanded = shift_y.repeat_interleave(num_anchors)
    widths_expanded = widths.repeat(num_shifts)
    heights_expanded = heights.repeat(num_shifts)

    # Calculate box corners (y1, x1, y2, x2)
    all_anchors[:, 0] = shift_y_expanded - 0.5 * heights_expanded  # y1
    all_anchors[:, 1] = shift_x_expanded - 0.5 * widths_expanded  # x1
    all_anchors[:, 2] = shift_y_expanded + 0.5 * heights_expanded  # y2
    all_anchors[:, 3] = shift_x_expanded + 0.5 * widths_expanded  # x2

    return all_anchors


def generate_pyramid_anchors(
    scales: List[List[int]],
    ratios: List[float],
    feature_shapes: List[Tuple[int, int]],
    feature_strides: List[int],
    anchor_stride: int,
) -> Tensor:
    """
    Generate anchors at different levels of a feature pyramid.

    Args:
        scales: List of anchor scale lists for each pyramid level
        ratios: Anchor aspect ratios (width/height)
        feature_shapes: Feature map shapes [(height, width), ...] for each pyramid level
        feature_strides: Feature stride for each pyramid level
        anchor_stride: Anchor stride on feature map

    Returns:
        torch.Tensor: All anchors concatenated with shape [total_anchors, 4]
    """

    anchors_list: List[torch.Tensor] = []

    for i, scale in enumerate(scales):
        level_anchors = generate_anchors(
            scales=scale,
            ratios=ratios,
            shape=feature_shapes[i],
            feature_stride=feature_strides[i],
            anchor_stride=anchor_stride,
        )
        anchors_list.append(level_anchors)

    # Concatenate along the first dimension
    pyramid_anchors = torch.cat(anchors_list, dim=0)

    return pyramid_anchors


def apply_box_deltas(boxes, deltas):
    """
    Alternative implementation using dimension-agnostic approach.
    Works for any shape as long as the last dimension is 4.

    Args:
        boxes: Tensor of shape [..., 4] where each 4-element vector is (y1, x1, y2, x2)
        deltas: Tensor of shape [..., 4] where each 4-element vector is (dy, dx, log(dh), log(dw))

    Returns:
        Transformed bounding boxes in (y1, x1, y2, x2) format,
        same shape as input boxes
    """
    # Unbind the last dimension
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    dy, dx, log_dh, log_dw = deltas.unbind(dim=-1)

    # Calculate height and width
    h = y2 - y1
    w = x2 - x1

    # Calculate center points and apply deltas
    cy = y1 + h * 0.5 + dy * h
    cx = x1 + w * 0.5 + dx * w
    h = h * torch.exp(log_dh)
    w = w * torch.exp(log_dw)

    # Convert back to corner format
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = y1 + h
    x2 = x1 + w

    # Stack along the last dimension
    return torch.stack([y1, x1, y2, x2], dim=-1)


def clip_boxes(boxes: Tensor, window: tuple[int, int, int, int]):
    """
    Clip bounding boxes to stay within the given window boundaries.
    Supports arbitrary batch dimensions (batched or non-batched input).

    Args:
        boxes: Tensor of shape [..., 4] where last dimension contains y1, x1, y2, x2
               Supports both batched [B, N, 4] and non-batched [N, 4] inputs.
        window: Tuple of (y_min, x_min, y_max, x_max) defining the clipping boundaries.

    Returns:
        Tensor of same shape as input with clipped coordinates.
    """
    y_min, x_min, y_max, x_max = window

    # Extract coordinates from the last dimension
    y1 = boxes[..., 0].clamp(y_min, y_max)
    x1 = boxes[..., 1].clamp(x_min, x_max)
    y2 = boxes[..., 2].clamp(y_min, y_max)
    x2 = boxes[..., 3].clamp(x_min, x_max)

    # Ensure y2 >= y1 and x2 >= x1 (box validity)
    y2 = torch.maximum(y2, y1)
    x2 = torch.maximum(x2, x1)

    # Stack coordinates back to original shape
    return torch.stack([y1, x1, y2, x2], dim=-1)


def pyramid_roi_align(
    boxes: torch.Tensor,
    feature_maps: List[torch.Tensor],
    pool_size: int | Tuple[int, int],
    image_shape: Tuple[int, int, int],
    backbone_strides: List[int],
) -> torch.Tensor:
    """
    Performs ROI Align on multiple levels of the feature pyramid (FPN).

    This function implements ROI pooling using Feature Pyramid Networks (FPN) as described in:
    "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)

    Do not support multi-batch input.

    Parameters:
    -----------
    boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates
    feature_maps: List of feature maps from different pyramid levels.
                    Each has shape [batch, channels, height, width]
    pool_size : Union[int, Tuple[int, int]]
        Height and width of the output pooled regions. Usually [7, 7] or 7.
    image_shape : Tuple[int, int, int]
        Shape of input image in pixels [height, width, channels].
    backbone_strides : List[int]
        Strides for each feature map in the backbone.

    Returns:
    --------
    torch.Tensor
        Pooled regions in shape: [batch, num_boxes, channels, pool_height, pool_width].
        or [total_boxes, channels, pool_height, pool_width] if boxes have variable length.

    Notes:
    ------
    - Input boxes should be in normalized coordinates (range [0, 1]).
    - Feature maps are expected from pyramid levels P2 to P5.
    """
    # Convert pool_size to tuple if it's an integer
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)

    # Get batch size
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1] if boxes.dim() == 3 else boxes.shape[0]

    # Calculate ROI areas and assign to pyramid levels
    # Split boxes into coordinates: [batch, num_boxes, 4] -> [batch, num_boxes]
    if boxes.dim() == 3:
        y1, x1, y2, x2 = boxes.chunk(4, dim=2)
        roi_height = (y2 - y1).squeeze(2)
        roi_width = (x2 - x1).squeeze(2)
    else:
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
        roi_height = y2 - y1
        roi_width = x2 - x1

    # Equation 1 from FPN paper: k = ⌊k₀ + log₂(√(wh)/224)⌋
    # Compute which pyramid level each ROI should be assigned to
    image_area = torch.tensor(
        float(image_shape[0] * image_shape[1]),
        device=boxes.device,
        dtype=torch.float32,
    )

    # Reference area (224x224 is ImageNet reference size)
    reference_area = 224.0 * 224.0

    # Compute ROI level in the pyramid
    # 4 corresponds to P4 level in the pyramid (stride 16)
    roi_level = 4 + torch.log2(
        torch.sqrt(roi_height * roi_width) / (224.0 / torch.sqrt(image_area))
    )
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)  # Limit to levels P2 to P5

    # Initialize containers for results
    pooled_features = []
    box_indices = []
    batch_indices = []

    # Process each pyramid level (P2 to P5)
    for level_idx, pyramid_level in enumerate(range(2, 6)):
        # Find which ROIs belong to this pyramid level
        level_mask = roi_level == pyramid_level

        # Skip if no ROIs assigned to this level
        if not level_mask.any():
            continue

        # Get indices of ROIs for this level
        # We need to track both batch index and box index
        level_indices = torch.nonzero(level_mask, as_tuple=True)

        if boxes.dim() == 3:
            # boxes shape: [batch, num_boxes, 4]
            batch_idx = level_indices[0]
            roi_idx = level_indices[1]

            # Get the boxes for this level
            level_boxes = boxes[batch_idx, roi_idx]

            # Store mapping from ROI index to pyramid level
            batch_indices.append(batch_idx)
            box_indices.append(roi_idx)

        else:
            # boxes shape: [num_boxes, 4] (legacy format)
            roi_indices = level_indices[0]
            level_boxes = boxes[roi_indices]
            box_indices.append(roi_indices)
            batch_indices.append(torch.zeros_like(roi_indices))

        # Detach boxes to prevent gradient flow to ROI proposals
        level_boxes = level_boxes.detach()

        # Convert normalized coordinates to pixel coordinates
        img_height, img_width = image_shape[:2]
        scale = torch.tensor(
            [img_width, img_height, img_width, img_height],
            device=level_boxes.device,
            dtype=torch.float32,
        )

        # Apply scale to convert from normalized to pixel coordinates
        pixel_boxes = level_boxes * scale

        # Reorder from (y1, x1, y2, x2) to (x1, y1, x2, y2) for torchvision.ops.roi_align
        pixel_boxes = pixel_boxes[:, [1, 0, 3, 2]]
        # Get batch indices for ROI Align
        if boxes.dim() == 3:
            # Use the actual batch indices
            batch_idx_tensor = batch_indices[-1].float()
        else:
            # Legacy: all boxes belong to batch 0
            batch_idx_tensor = torch.zeros(
                level_boxes.size(0), dtype=torch.float32, device=level_boxes.device
            )

        # Concatenate batch indices with boxes: [batch_idx, x1, y1, x2, y2]
        boxes_with_batch = torch.cat(
            [batch_idx_tensor.unsqueeze(1), pixel_boxes], dim=1
        )

        # Get spatial scale for this feature map (1/stride)
        spatial_scale = 1.0 / backbone_strides[level_idx]

        # Apply ROI Align
        # Use the full batch feature map
        level_feature_map = feature_maps[level_idx]
        level_features = roi_align(
            level_feature_map,
            boxes_with_batch,
            output_size=pool_size,
            spatial_scale=spatial_scale,
            aligned=False,  # Turning this off actually make the result same to the original implementation
        )
        pooled_features.append(level_features)

    # Concatenate features from all pyramid levels
    if pooled_features:
        pooled_output = torch.cat(pooled_features, dim=0)

        # Legacy behavior for single batch
        box_to_level = torch.cat(box_indices, dim=0)
        sorted_indices = torch.argsort(box_to_level)
        pooled_output = pooled_output[sorted_indices]

    else:
        # Handle edge case: no ROIs
        if boxes.dim() == 3:
            total_boxes = batch_size * num_boxes
        else:
            total_boxes = boxes.shape[0]

        num_channels = feature_maps[0].shape[1] if feature_maps else 0
        pooled_output = torch.zeros(
            (total_boxes, num_channels, pool_size[0], pool_size[1]),
            device=boxes.device,
            dtype=feature_maps[0].dtype if feature_maps else boxes.dtype,
        )

        if boxes.dim() == 3:
            pooled_output = pooled_output.view(
                batch_size, num_boxes, num_channels, pool_size[0], pool_size[1]
            )

    return pooled_output


def coordinates_roi(
    boxes: torch.Tensor,
    coordinates: torch.Tensor,
    pool_size: Tuple[int, int],
    image_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Implements ROI Pooling/Align on multiple feature maps with batch support.

    Args:
        boxes: [batch_size, num_boxes, 4] or [num_boxes, 4] in normalized
               coordinates (y1, x1, y2, x2). If 2D, same boxes are applied to all batches.
        coordinates: [batch_size, channels, height, width] coordinates
        pool_size: int or (height, width) of output pooled regions. Usually [7, 7]
        image_shape: [height, width, channels] of input image in pixels.
                    If None, assumes boxes are already in pixel coordinates.
        spatial_scale: Direct scale factor (1/stride). If None, auto-calculates.
        stride: Feature map stride relative to input image. Used if spatial_scale is None.

    Returns:
        Pooled regions: [batch_size, num_boxes, channels, pool_h, pool_w] if multi-batch,
                        or [num_boxes, channels, pool_h, pool_w] if single batch.
    """
    if coordinates.dim() == 3:
        coordinates = coordinates.unsqueeze(0)
    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(0)

    batch_size, channels, feat_h, feat_w = coordinates.shape
    num_boxes = boxes.size(1)

    # Detach boxes to stop gradient propagation
    boxes = boxes.detach()

    # Convert normalized boxes to pixel coordinates
    h, w = image_shape[:2]
    scale = torch.tensor([h, w, h, w], device=boxes.device, dtype=torch.float32)
    boxes = boxes * scale

    # Reorder boxes from (y1, x1, y2, x2) to (x1, y1, x2, y2)
    # Create a view to avoid memory copy
    boxes = boxes[:, :, [1, 0, 3, 2]]

    # Prepare batch indices: [batch_idx, box1, box2, ...] for each batch
    batch_indices = (
        torch.arange(batch_size, device=boxes.device, dtype=torch.float32)
        .view(batch_size, 1, 1)
        .expand(batch_size, num_boxes, 1)
    )

    # Concatenate batch indices with boxes: [batch_size, num_boxes, 5]
    boxes_with_ind = torch.cat([batch_indices, boxes], dim=2)

    # Reshape to [batch_size * num_boxes, 5] for roi_align
    boxes_with_ind = boxes_with_ind.reshape(-1, 5)

    img_h, img_w = image_shape[:2]
    spatial_scale = min(feat_h / img_h, feat_w / img_w)

    # Use roi_align for pooling
    pooled_features = roi_align(
        coordinates,
        boxes_with_ind,
        output_size=pool_size,
        spatial_scale=spatial_scale,
        aligned=False,  # Turning this off actually make the result same to the original implementation
    )

    # Reshape back to [batch_size, num_boxes, channels, pool_h, pool_w]
    pooled_features = pooled_features.view(
        batch_size, num_boxes, channels, pool_size[0], pool_size[1]
    )

    return pooled_features


def intersect1d(tensor1, tensor2):
    unique1 = torch.unique(tensor1)
    unique2 = torch.unique(tensor2)

    mask = torch.isin(unique1, unique2)
    return unique1[mask]


def filter_boxes_by_area(boxes: Tensor, area_threshold: float) -> Tensor:
    """
    Filters boxes by area threshold, and returns a boolean mask.

    Note: all boxes are assumed to be normalized coordinates (y1, x1, y2, x2).
    """
    y1, x1, y2, x2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    areas = (y2 - y1) * (x2 - x1)
    mask = areas > area_threshold
    return mask


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding."""

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)  # type: ignore
        self.stride = torch.nn.modules.utils._pair(stride)  # type: ignore

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = (
            (out_width - 1) * self.stride[0] + self.kernel_size[0] - in_width
        )
        pad_along_height = (
            (out_height - 1) * self.stride[1] + self.kernel_size[1] - in_height
        )
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)


class Bottleneck(nn.Module):
    EXPANSION = 4
    BN_EPS = 0.001
    BN_MOMENTUM = 0.01

    def __init__(self, inp_dim: int, hid_dim: int, stride=1):
        """
        Note: Output dim is hid_dim * self.expansion
        """
        super(Bottleneck, self).__init__()

        # 1x1 conv
        self.conv1 = nn.Conv2d(
            inp_dim,
            hid_dim,
            kernel_size=1,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(hid_dim, eps=self.BN_EPS, momentum=self.BN_MOMENTUM)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            hid_dim,
            hid_dim,
            kernel_size=3,
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(hid_dim, eps=self.BN_EPS, momentum=self.BN_MOMENTUM)

        # 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(
            hid_dim,
            hid_dim * self.EXPANSION,
            kernel_size=1,
        )
        self.bn3 = nn.BatchNorm2d(
            hid_dim * self.EXPANSION, eps=self.BN_EPS, momentum=self.BN_MOMENTUM
        )
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inp_dim != hid_dim * self.EXPANSION:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inp_dim,
                    hid_dim * self.EXPANSION,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(
                    hid_dim * self.EXPANSION, eps=self.BN_EPS, momentum=self.BN_MOMENTUM
                ),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    """
    Here is a nice article: https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    """

    BN_EPS = 0.001
    BN_MOMENTUM = 0.01

    def __init__(
        self,
        architecture: str,
        out_channels: int,
        bilinear_upsampling=False,
        input_channels: int = 3,
    ):
        super().__init__()
        assert architecture in ["resnet50", "resnet101"]

        """Bottom-up pathway(ResNet)"""
        self.inplanes = 64
        self.input_channels = input_channels
        self.build_resnet(architecture)

        """Top-down pathway"""
        self.out_channels = out_channels
        self.bilinear_upsampling = bilinear_upsampling
        self.build_topdown()

    def forward(self, x: Tensor) -> TYPE_FPN_OUTPUT:
        """
        The output order is from P2 to P6
        """
        resnet_outputs = []

        # Inference of bottom-up pathway(just normal ResNet)
        for i, ci in enumerate(self.bottom_up_blocks):
            x = ci(x)
            if i == 0:
                continue  # conv1, which isn't needed for FPN
            resnet_outputs.append(x)
        # Inference of top-down pathway
        FPN_outputs = []
        for i in range(5, 1, -1):
            resnet_output = resnet_outputs[i - 2]
            lateral_features = self.lateral_convs[i - 2](resnet_output)
            if i == 5:
                # The first block
                prev_features = lateral_features
            else:
                prev_features = lateral_features + F.interpolate(
                    prev_features,  # Use the "Sum" from the previous loop # type: ignore
                    scale_factor=2,
                    mode="bilinear" if self.bilinear_upsampling else "nearest",
                )
            topdown_output = self.fpn_convs[i - 2](prev_features)
            FPN_outputs.append(topdown_output)
        # P6
        FPN_outputs.reverse()
        FPN_outputs.append(self.M6(FPN_outputs[-1]))
        return tuple(FPN_outputs)

    def build_resnet(self, architecture: str):
        layers_map = {"resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3]}
        layers = layers_map[architecture]
        # Stage 1
        C1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=self.BN_EPS, momentum=self.BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SamePad2d(3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.bottom_up_blocks = nn.Sequential()
        self.bottom_up_blocks.append(C1)
        # Stage 2
        self.bottom_up_blocks.append(self.build_resnet_block(64, layers[0]))
        # Stage 3, 4, 5
        for i in range(3, 6):
            self.bottom_up_blocks.append(
                self.build_resnet_block(64 * 2 ** (i - 2), layers[i - 2], stride=2)
            )

    def build_resnet_block(
        self, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """
        Constructs a layer of Bottleneck blocks.
        The first block handles the stride and potential channel expansion (downsample).
        The output dim is planes * Bottleneck.expansion
        """
        layers = nn.Sequential()

        # First block: might downsample
        layers.append(Bottleneck(self.inplanes, planes, stride))

        # Update inplanes to match the output of the Bottleneck (expansion * planes)
        self.inplanes = planes * Bottleneck.EXPANSION

        # Subsequent blocks: stride is always 1, no downsample needed
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return layers

    def build_topdown(self):
        self.M6 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # Build for C2, C3, C4, C5
        for i in range(2, 6):
            # Lateral: 1x1 conv to reduce channels to out_channels
            lat = nn.Conv2d(
                64 * 2 ** (i - 2) * Bottleneck.EXPANSION,
                self.out_channels,
                kernel_size=1,
            )
            # FPN Output: 3x3 conv to smooth aliasing
            fpn = nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,  # Equivalent to padding='same' with stride 1
            )

            self.lateral_convs.append(lat)
            self.fpn_convs.append(fpn)


class Depth(nn.Module):
    EN_EPS = 0.001
    EN_MOMENTUM = 0.01
    DECONV_PARAM = [(128, 128), (256, 128), (256, 128), (256, 128), (256, 64)]

    def __init__(self, num_output_channels=1):
        super(Depth, self).__init__()

        self.num_output_channels = num_output_channels
        self.convs = nn.Sequential()
        for _ in range(5):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128, eps=self.EN_EPS, momentum=self.EN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
            )

        self.deconvs = nn.Sequential()
        for in_channels, out_channels in self.DECONV_PARAM:
            self.deconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(
                        out_channels, eps=self.EN_EPS, momentum=self.EN_MOMENTUM
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        self.depth_pred = nn.Conv2d(
            64, num_output_channels, kernel_size=3, stride=1, padding=1
        )

    def crop_feature_maps(self, x: TYPE_FPN_OUTPUT) -> TYPE_FPN_OUTPUT:
        """
        Crop each feature maps 25% from top and down. (May because of padding)
        """
        return tuple(
            [*x[:2]] + [i[:, :, i.shape[2] // 8 : -i.shape[2] // 8] for i in x[2:]]
        )  # type: ignore

    def pad_depth_pred(self, x: Tensor) -> Tensor:
        """
        Pad the cropped 25% back.

        Original code:
        x = torch.nn.functional.interpolate(x, size=(480, 640), mode="bilinear")
        zeros = torch.zeros((len(x), self.num_output_channels, 80, 640))
        x = torch.cat([zeros, x, zeros], dim=2)
        """
        x = F.interpolate(x, size=(480, 640), mode="bilinear", align_corners=False)
        x = F.pad(x, (0, 0, 80, 80), "constant", 0)
        return x

    def forward(self, feature_maps: TYPE_FPN_OUTPUT) -> Tensor:
        feature_maps = self.crop_feature_maps(feature_maps)

        for i, (conv, deconv) in enumerate(zip(self.convs, self.deconvs)):
            if i == 0:
                x = deconv(conv(feature_maps[i]))
                continue
            x = deconv(torch.cat([conv(feature_maps[i]), x], dim=1))  # type: ignore

            # Mystical crop
            if i == 1:
                x = x[:, :, 5:35]

        x = self.depth_pred(x)  # type: ignore
        x = self.pad_depth_pred(x)
        return x


class RPN(nn.Module):
    """Region Proposal Network (RPN).

    The RPN is a lightweight neural network that runs on the feature maps
    to generate region proposals. For each predefined anchor (candidate box),
    it predicts:
      1. Whether the anchor contains an object (objectness score)
      2. How to adjust the anchor to better fit the object (bounding box deltas)

    A nice article about how the anchors were selected:
    https://towardsmachinelearning.org/region-proposal-network/

    Args:
        anchors_per_location (int): Number of anchors at each feature map location.
                                    Typically 9 (3 scales × 3 aspect ratios).
        dim (int): Input channel dimension (e.g., 256 from FPN's feature maps).

    Outputs:
        # rpn_class_logits: [batch, N, 2] - raw logits for foreground/background classification
        rpn_probs:        [batch, N, 2] - softmax probabilities for FG/BG
        rpn_bbox:         [batch, N, 4] - bbox adjustment deltas [dy, dx, log(dh), log(dw)]
        where N = H × W × anchors_per_location (total anchors across all positions)
    """

    def __init__(self, anchors_per_location: int, dim: int):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location  # K anchors per position

        # Shared convolutional layer: 3x3 with same padding (padding=1)
        # This extracts features with local context around each anchor position
        # Input: [B, dim, H, W] → Output: [B, 512, H, W]
        self.conv_shared = nn.Conv2d(dim, 512, kernel_size=3, padding="same", stride=1)
        self.relu = nn.ReLU(inplace=True)  # Non-linearity after shared conv

        # Classification branch: 1x1 conv → 2 × K channels (K = anchors_per_location)
        # For each anchor, predicts 2 scores: foreground vs background
        # Output shape: [B, 2*K, H, W]
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1)

        # BBox regression branch: 1x1 conv → 4 × K channels
        # For each anchor, predicts 4 deltas: (dy, dx, log(dh), log(dw))
        # These deltas adjust the anchor to better fit the object
        # Output shape: [B, 4*K, H, W]
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)  # Softmax over FG/BG dimension

    def forward(self, x):
        """
        Forward pass through RPN.

        Args:
            x (Tensor): Feature maps from backbone/FPN [batch, dim, height, width]

        Returns:
            list: [rpn_probs, rpn_bbox]
        """
        # Shared feature extraction: apply 3x3 conv + ReLU
        # Captures context around each position to make better predictions
        x = self.relu(self.conv_shared(x))  # Output: [B, 512, H, W]

        # ========== CLASSIFICATION BRANCH ==========
        # Predict objectness scores (FG/BG) for each anchor
        logits = self.conv_class(x)  # Raw logits: [B, 2*K, H, W]

        # Reshape: [B, 2*K, H, W] → [B, H, W, 2*K] → [B, H*W*K, 2]
        # Permute changes memory layout to channel-last format
        logits = logits.permute(0, 2, 3, 1).contiguous()
        batch_size, h, w = logits.shape[:3]

        # Final reshape: flatten spatial dimensions and anchors
        # Each anchor gets its own row in the batch dimension
        rpn_class_logits = logits.view(batch_size, -1, 2)

        # Apply softmax to get probabilities: foreground vs background
        # dim=2 applies softmax over the last dimension (FG/BG)
        rpn_probs = self.softmax(rpn_class_logits)

        # ========== BBOX REGRESSION BRANCH ==========
        # Predict adjustment deltas for each anchor
        bbox = self.conv_bbox(x)  # Raw deltas: [B, 4*K, H, W]

        # Reshape: [B, 4*K, H, W] → [B, H, W, 4*K] → [B, H*W*K, 4]
        bbox = bbox.permute(0, 2, 3, 1).contiguous()
        rpn_bbox = bbox.view(batch_size, -1, 4)

        return [rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################
class Classifier(nn.Module):
    def __init__(
        self,
        depth,
        image_shape: Tuple[int, int, int],
        anchor_conf: AnchorConf,
    ):
        """
        Note: we don't support multiple bathes yet, cause the tensor fed in Conv2d has shape [num_boxes, channels, pool_height, pool_width].
        """
        super(Classifier, self).__init__()
        self.depth = depth
        self.anchor_conf = anchor_conf
        self.pool_size: tuple[int, int] = (anchor_conf.pool_size, anchor_conf.pool_size)
        self.image_shape = image_shape
        self.num_classes = anchor_conf.num_classes
        self.num_parameters = anchor_conf.num_parameters

        self.conv1 = nn.Conv2d(
            self.depth + 64, 1024, kernel_size=self.pool_size, stride=1
        )
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, self.num_classes * 4)

        self.linear_parameters = nn.Linear(1024, self.num_classes * self.num_parameters)

    def forward(self, mrcnn_feature_maps, rois, ranges, pool_features=True):
        mrcnn_feature_maps = pyramid_roi_align(
            rois,
            mrcnn_feature_maps,
            self.pool_size,
            self.image_shape,
            self.anchor_conf.backbone_strides,
        )
        ranges = coordinates_roi(
            rois,
            ranges,
            self.pool_size,
            self.image_shape,
        )
        mrcnn_feature_maps = mrcnn_feature_maps.squeeze(0)
        ranges = ranges.squeeze(0)
        roi_features = torch.cat([mrcnn_feature_maps, ranges], dim=1)

        mrcnn_feature_maps = self.conv1(roi_features)
        mrcnn_feature_maps = self.bn1(mrcnn_feature_maps)
        mrcnn_feature_maps = self.relu(mrcnn_feature_maps)
        mrcnn_feature_maps = self.conv2(mrcnn_feature_maps)
        mrcnn_feature_maps = self.bn2(mrcnn_feature_maps)
        mrcnn_feature_maps = self.relu(mrcnn_feature_maps)

        mrcnn_feature_maps = mrcnn_feature_maps.view(-1, 1024)
        mrcnn_class_logits: Tensor = self.linear_class(mrcnn_feature_maps)
        mrcnn_probs: Tensor = self.softmax(mrcnn_class_logits)

        mrcnn_bbox: Tensor = self.linear_bbox(mrcnn_feature_maps)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        mrcnn_parameters: Tensor = self.linear_parameters(mrcnn_feature_maps)

        mrcnn_parameters = mrcnn_parameters.view(
            mrcnn_parameters.size()[0], -1, self.num_parameters
        )
        if pool_features:
            return [
                mrcnn_class_logits,
                mrcnn_probs,
                mrcnn_bbox,
                mrcnn_parameters,
                roi_features,
            ]
        else:
            return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_parameters]


class Mask(nn.Module):
    def __init__(
        self,
        depth,
        pool_size,
        image_shape,
        num_classes,
        num_parameters_channels,
        backbone_strides,
    ):
        """
        Do not support multiple batches yet.
        """
        super(Mask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_parameters_channels = num_parameters_channels
        self.backbone_strides = backbone_strides

        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1, padding="same")
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(
            256, num_classes + num_parameters_channels, kernel_size=1, stride=1
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: list[Tensor], rois: Tensor):
        roi_features = pyramid_roi_align(
            rois,
            x,
            self.pool_size,
            self.image_shape,
            self.backbone_strides,
        )

        roi_features = roi_features.squeeze(0)
        x = self.conv1(roi_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)

        x = self.sigmoid(x)
        return x, roi_features
