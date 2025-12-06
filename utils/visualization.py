import torch
import cv2
import numpy as np
import colorsys
import random


def random_color_hsl(h_range=(0, 1), s_range=(0.5, 0.8), l_range=(0.4, 0.7)):
    h = random.uniform(*h_range)
    s = random.uniform(*s_range)
    l = random.uniform(*l_range)  # noqa: E741

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r * 255, g * 255, b * 255)


def visualize_mask(
    masks: torch.Tensor, detections: torch.Tensor, image_path: str, alpha: float = 0.5
):
    """
    Docstring for visualize_mask

    :param masks: [num_detections, Anchor normal ID, H, W]
    :type masks: torch.Tensor
    :param detections: [N, (y1, x1, y2, x2, class_id, score, param1, param2, param3)]
    :type detections: torch.Tensor
    """
    image = cv2.imread(str(image_path))
    assert image is not None
    img_w, img_h = image.shape[1], image.shape[0]
    for mask, detection in zip(masks, detections):
        y1, x1, y2, x2, class_id, score = detection[:6].tolist()

        inp_size = max(img_h, img_w)
        y1, x1, y2, x2 = (
            int(y1 * inp_size),
            int(x1 * inp_size),
            int(y2 * inp_size),
            int(x2 * inp_size),
        )
        # Convert to image coordinates
        y_padped = (inp_size - img_h) // 2
        x_padped = (inp_size - img_w) // 2

        y1, x1, y2, x2 = (
            y1 - y_padped,
            x1 - x_padped,
            y2 - y_padped,
            x2 - x_padped,
        )

        box_w, box_h = x2 - x1, y2 - y1

        # Skip low-confidence detections
        if score < 0.5:
            continue

        # Select mask for the predicted class (convert to int)
        class_idx = int(class_id)
        if class_idx >= mask.shape[0]:
            continue  # Skip if class index is out of bounds

        instance_mask = mask[class_idx]  # Shape: [H, W]

        # Convert mask to numpy
        mask_np = instance_mask.detach().cpu().numpy()

        # Resize mask to match original image dimensions
        mask_resized = cv2.resize(
            mask_np, (box_w, box_h), interpolation=cv2.INTER_LINEAR
        )

        # Threshold to get binary mask
        binary_mask = mask_resized > 0.5  # Common threshold for segmentation masks

        # Generate a random color for this instance
        color = random_color_hsl()

        # Create colored mask (same size as image)
        colored_mask = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        colored_mask[binary_mask] = color

        alpha_mask = np.zeros((box_h, box_w), dtype=np.float32)
        alpha_mask[binary_mask] = alpha

        alpha_mask_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

        # output = alpha * foreground + (1-alpha) * background
        blended = (1 - alpha_mask_3d) * image[y1:y2, x1:x2].astype(
            np.float32
        ) + alpha_mask_3d * colored_mask.astype(np.float32)

        image[y1:y2, x1:x2] = blended.astype(np.uint8)

        # Add label with class and score
        label = f"Plane {class_idx}: {score:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    cv2.imshow("Mask", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
