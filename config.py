from typing import List, Tuple
import torch


class ImageMeta:
    def __init__(self, window: tuple[int, int, int, int]) -> None:
        self.window = window  # (y1, x1, y2, x2) in image coordinates. The part of the image that contains the image excluding the padding.


class AnchorConf:
    scales: List[List[int]] = [[32], [64], [128], [256], [512]]  # RPN_ANCHOR_SCALES
    ratios: List[float] = [0.5, 1, 2]  # RPN_ANCHOR_RATIOS
    feature_shapes: List[Tuple[int, int]] = [
        (160, 160),
        (80, 80),
        (40, 40),
        (20, 20),
        (10, 10),
    ]
    feature_strides: List[int] = [4, 8, 16, 32, 64]
    anchor_stride: int = 1

    roi_proposal_count = 1000
    rpn_bbox_std = torch.tensor([0.1, 0.1, 0.2, 0.2])  # Original: RPN_BBOX_STD_DEV
    rpn_nms_threshold = 0.7  # RPN_NMS_THRESHOLD
    pre_nms_limit = 6000
    num_classes = 8  # NUM_CLASSES
    backbone_strides: List[int] = [4, 8, 16, 32, 64]
    pool_size = 7
    mask_pool_size = 14  # MASK_POOL_SIZE
    num_parameters = 3  # NUM_PARAMETERS
    detection_min_confidence = 0.7  # DETECTION_MIN_CONFIDENCE
    # Non-maximum suppression threshold for detection
    detection_nms_threshold = 0.3  # DETECTION_NMS_THRESHOLD
    # Max number of final detections
    detection_max_instances = 100  # DETECTION_MAX_INSTANCES

    NUM_PARAMETER_CHANNELS = 0

    anchor_normals_tensor = torch.tensor(
        [
            [-0.9192, 0.1282, -0.0120],
            [0.0186, 0.8950, 0.3132],
            [-0.0028, 0.4440, -0.8347],
            [0.9193, 0.1334, -0.0179],
            [-0.6042, 0.6867, 0.2610],
            [0.0097, 0.1625, 0.8836],
            [0.6270, 0.6712, 0.2511],
        ]
    )

    min_roi_area = 0.0  # Normalized. Original implementation does this mystically in detection_target_layer function. Setting this can speed up inference a lot.


class DefaultCamParams:
    # METADATA = np.array([571.87, 571.87, 320, 240, 640, 480, 0, 0, 0, 0])
    FX = 571.87  # focal length x
    FY = 571.87  # focal length y
    CX = 320  # principal point x
    CY = 240  # principal point y
    W = 640  # image width
    H = 480  # image height

    def __init__(self) -> None:
        self.URANGE_UNIT = (
            ((torch.arange(self.W, requires_grad=False).float() + 0.5) / self.W)
            .view((1, -1))
            .repeat(self.H, 1)
        )
        self.VRANGE_UNIT = (
            ((torch.arange(self.H, requires_grad=False).float() + 0.5) / self.H)
            .view((-1, 1))
            .repeat(1, self.W)
        )
        self.ONES = torch.ones(self.URANGE_UNIT.shape, requires_grad=False)

        self.urange = (self.URANGE_UNIT * self.W - self.CX) / self.FX
        self.vrange = (self.VRANGE_UNIT * self.H - self.CY) / self.FY
        self.ranges = torch.stack([self.urange, self.ONES, -self.vrange], dim=-1)

        # Mystical transpose for ranges
        # TODO: figure out what's going on here.
        self._ranges = self.ranges.transpose(1, 2).transpose(0, 1)
        zeros = torch.zeros(3, (self.W - self.H) // 2, self.W)
        self._ranges = torch.cat([zeros, self._ranges, zeros], dim=1)
        self._ranges = torch.nn.functional.interpolate(
            self._ranges.unsqueeze(0), size=(160, 160), mode="bilinear"
        )


class DefaultDataSetConfig:
    IMAGE_MEAN = torch.tensor([123.7, 116.8, 103.9], dtype=torch.float32)
    INP_SIZE = (
        640,
        640,
    )  # Your image will first be resized to self.image_size, then padded to self.INP_SIZE. This value depends on the network input size, so you may not want to change it. Check datasets.image_preprocess for more details.

    def __init__(self, image_size: Tuple[int, int]) -> None:
        self.image_size = image_size


class Conf:
    ResNet_type = "resnet101"
    bilinear_upsample = False
    anchor_conf = AnchorConf()

    cam_params = DefaultCamParams()
    data_set_config = DefaultDataSetConfig((cam_params.H, cam_params.W))
    image_shape: tuple[int, int, int] = data_set_config.INP_SIZE + (3,)  # [H, W, C]
