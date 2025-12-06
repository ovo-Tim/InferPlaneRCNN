import torch
import torch.nn as nn
from torch import Tensor
from .modules import (
    FPN,
    generate_pyramid_anchors,
    Depth,
    RPN,
    apply_box_deltas,
    clip_boxes,
    Classifier,
    intersect1d,
    Mask,
    TYPE_FPN_OUTPUT,
    filter_boxes_by_area,
)
from typing import Literal, Any, TYPE_CHECKING
from torchvision.ops import nms

if TYPE_CHECKING:
    from config import AnchorConf, DefaultCamParams, ImageMeta

    CamParamsType = DefaultCamParams
    ImageMetaType = ImageMeta
else:
    AnchorConf = Any
    CamParamsType = Any
    ImageMetaType = Any


class MaskRCNN(nn.Module):
    def __init__(
        self,
        anchor_conf: AnchorConf,
        cam_params: CamParamsType,
        image_shape: tuple[int, int, int],
        ResNet_type: Literal["resnet101", "resnet50"] = "resnet101",
        bilinear_upsample: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        image_shape: [H, W, C]

        Note: We do not support batched input yet, cause tensor like [batch, num_rois, C, H, W] is too hard to handle. :(
        """
        super().__init__(*args, **kwargs)
        self.image_shape = image_shape  # INP_SIZE [H, W, C]
        self.scale = torch.tensor(
            [
                self.image_shape[0],
                self.image_shape[1],
                self.image_shape[0],
                self.image_shape[1],
            ]
        )
        self.cam_params = cam_params
        self.anchor_conf = anchor_conf

        self.fpn = FPN(ResNet_type, 256, bilinear_upsample)
        self.depth = Depth()
        self.rpn = RPN(len(anchor_conf.ratios) * len(anchor_conf.scales[0]), 256)
        self.classifier = Classifier(256, self.image_shape, anchor_conf)
        self.mask = Mask(
            256,
            self.anchor_conf.mask_pool_size,
            self.image_shape,
            self.anchor_conf.num_classes,
            self.anchor_conf.NUM_PARAMETER_CHANNELS,
            self.anchor_conf.backbone_strides,
        )

        ## Coordinate feature
        self.coordinates = nn.Conv2d(3, 64, kernel_size=1, stride=1)

        anchors = generate_pyramid_anchors(
            anchor_conf.scales,
            anchor_conf.ratios,
            anchor_conf.feature_shapes,
            anchor_conf.feature_strides,
            anchor_conf.anchor_stride,
        )
        self.register_buffer("anchors", anchors, persistent=False)
        if TYPE_CHECKING:
            self.anchors = anchors

    def forward(
        self, x: torch.Tensor, image_meta: ImageMetaType
    ) -> tuple[Tensor, Tensor]:
        """
        x: input image tensor of shape (B, C, H, W). Usually, H=W=640
        Return:
            detections shaped: [N, (y1, x1, y2, x2, class_id, score, param1, param2, param3)] (Normalized)
            detection_masks: [num_detections, Anchor normal ID, H, W] (Normalized)
        """
        feature_maps: TYPE_FPN_OUTPUT = self.fpn(x)
        mrcnn_feature_maps = feature_maps[:-1]
        depth_np: Tensor = self.depth(feature_maps[::-1])
        depth_np = depth_np.squeeze(1)

        # PREDICT_BOUNDARY is always false, so we just ignore.
        ranges = self.coordinates(self.cam_params._ranges * 10)
        rpn_output = [self.rpn(feature_map) for feature_map in feature_maps]

        ## Concatenate layer outputs
        ## Convert from list of lists of level outputs to list of lists
        ## of outputs across levels.
        ## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        rpn_output = list(zip(*rpn_output))
        rpn_output = [torch.cat(list(o), dim=1) for o in rpn_output]
        rpn_probs, rpn_bbox = rpn_output

        # This layer only performs quick filtering...
        rpn_rois = self.proposal_layer(
            rpn_probs,
            rpn_bbox,
        )  # Note the result is normalized. Shape: [1, 1000, 4]

        # Original implementation has a mystical detection_target_layer function here.
        # Since gt_boxes are dummy data when performing inference, it should do nothing. But it turns out it does cut off the number of ROIs.
        # It turns out this function just first filter out the ROIs that are too small and then randomly add back some off them(as negative samples)???
        # It should be only called when training.

        rpn_rois = rpn_rois[
            filter_boxes_by_area(rpn_rois, self.anchor_conf.min_roi_area)
        ].unsqueeze(0)  # Fix here to support multiple batches.

        # ...then we do fine-grained classification here
        (
            mrcnn_class_logits_final,
            mrcnn_class_final,
            mrcnn_bbox_final,
            mrcnn_parameters_final,
            roi_features,
        ) = self.classifier(mrcnn_feature_maps, rpn_rois[0], ranges, pool_features=True)

        detections, indices = self.refine_detections(
            rpn_rois,
            mrcnn_class_final,
            mrcnn_bbox_final,
            mrcnn_parameters_final,
            image_meta.window,
        )
        detections[:, :4] = detections[:, :4] / self.scale
        detection_boxes = detections[:, :4].unsqueeze(0)
        detection_masks, _ = self.mask(mrcnn_feature_maps, detection_boxes)
        roi_features = roi_features[indices]

        return detections, detection_masks

    def proposal_layer(
        self,
        rpn_probs: Tensor,
        rpn_bbox: Tensor,
    ):
        """
        Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinment detals to anchors.

        (I tried to support multiple batches. Let's hope it works.)

        Inputs:
            rpn_probs: [batch, anchors, (bg prob, fg prob)] or [anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))] or [anchors, (dy, dx, log(dh), log(dw))]

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """
        # Check if we have a batch or not
        if rpn_probs.dim() == 2:
            rpn_probs = rpn_probs.unsqueeze(0)
        if rpn_bbox.dim() == 2:
            rpn_bbox = rpn_bbox.unsqueeze(0)

        foreground_probs = rpn_probs[:, :, 1]
        rpn_bbox *= self.anchor_conf.rpn_bbox_std

        pre_nms_limit = min(self.anchor_conf.pre_nms_limit, self.anchors.size(0))
        all_batch_proposals = []
        for batch in range(rpn_probs.shape[0]):
            batch_probs = foreground_probs[batch]
            batch_bbox = rpn_bbox[batch]

            scores, order = batch_probs.sort(descending=True)
            order = order[:pre_nms_limit]
            scores = scores[:pre_nms_limit]
            batch_bbox_selected = batch_bbox[order, :]  # [pre_nms_limit, 4]
            anchors_selected = self.anchors[order, :]  # [pre_nms_limit, 4]

            proposals = apply_box_deltas(anchors_selected, batch_bbox_selected)
            proposals = clip_boxes(proposals, (0, 0, *self.image_shape[:2]))

            # Convert boxes from (y1, x1, y2, x2) to (x1, y1, x2, y2) for torchvision.ops.nms
            boxes_for_nms = proposals[:, [1, 0, 3, 2]]
            keep = nms(boxes_for_nms, scores, self.anchor_conf.rpn_nms_threshold)
            keep = keep[: self.anchor_conf.roi_proposal_count]
            proposals = proposals[keep, :]

            # Perform normalization
            normalized_boxes = proposals / self.scale

            all_batch_proposals.append(normalized_boxes)

        return torch.stack(all_batch_proposals, dim=0)

    def refine_detections(
        self,
        rois: Tensor,
        probs: Tensor,
        deltas: Tensor,
        parameters: Tensor,
        window: tuple[int, int, int, int],
    ) -> tuple[Tensor, Tensor]:
        """Refine classified proposals and filter overlaps and return final detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            parameters: [N, num_classes, 3] Plane parameters
            window: (y1, x1, y2, x2) in image coordinates. The part of the image
                that contains the image excluding the padding.
            config: Configuration object
            return_indices: If True, return indices and original rois
            use_nms: 0=no NMS, 1=per-class NMS, 2=global NMS
            one_hot: Whether probs is one-hot (unused, kept for compatibility)

        Returns:
            detections shaped: ([N, (y1, x1, y2, x2, class_id, score, param1, param2, param3)], indices)
        """
        device = rois.device

        class_scores, class_ids = torch.max(probs, dim=1)
        idx = torch.arange(class_ids.size(0), device=device)
        deltas_specific = deltas[idx, class_ids]
        class_parameters = parameters[idx, class_ids]

        # Apply bounding box deltas
        std_dev = self.anchor_conf.rpn_bbox_std.detach().clone().view(1, 4)

        refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

        # Convert to image coordinates
        refined_rois = refined_rois * self.scale

        # Clip to window and round to integers
        refined_rois = clip_boxes(refined_rois, window).squeeze(0)
        refined_rois = torch.round(refined_rois)

        # Filter out invalid boxes
        keep_bool = class_ids > 0

        # Filter by confidence
        keep_bool = keep_bool & (
            class_scores >= self.anchor_conf.detection_min_confidence
        )

        # Filter boxes with zero or negative area
        keep_bool = (
            keep_bool
            & (refined_rois[..., 2] > refined_rois[..., 0])
            & (refined_rois[..., 3] > refined_rois[..., 1])
        )

        # Early return if no valid boxes
        if not keep_bool.any():
            empty = torch.zeros((0, 10), device=device)
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty_long

        keep: Tensor = torch.where(keep_bool)[0]

        # Global NMS (all classes together)
        pre_nms_scores = class_scores[keep]
        pre_nms_rois = refined_rois[keep]

        # Sort by score
        sorted_scores, sort_idx = pre_nms_scores.sort(descending=True)
        sorted_rois = pre_nms_rois[sort_idx]

        # Convert to (x1, y1, x2, y2) for torchvision NMS
        rois_for_nms = sorted_rois[:, [1, 0, 3, 2]]
        nms_keep_indices = nms(
            rois_for_nms, sorted_scores, self.anchor_conf.detection_nms_threshold
        )

        # Map back to original indices
        nms_keep = keep[sort_idx[nms_keep_indices]]
        keep = intersect1d(keep, nms_keep)

        # Keep top detections
        if len(keep) > self.anchor_conf.detection_max_instances:
            top_scores = class_scores[keep]
            _, top_indices = torch.topk(
                top_scores, self.anchor_conf.detection_max_instances
            )
            keep = keep[top_indices]

        # Apply plane anchors
        # Original function: applyAnchorsTensor
        class_parameters = (
            class_parameters[..., :3]
            + self.anchor_conf.anchor_normals_tensor[class_ids.data - 1]
        )

        # Prepare final result
        result = torch.cat(
            [
                refined_rois[keep],
                class_ids[keep].unsqueeze(1).float(),
                class_scores[keep].unsqueeze(1),
                class_parameters[keep],
            ],
            dim=1,
        )

        return result, keep
