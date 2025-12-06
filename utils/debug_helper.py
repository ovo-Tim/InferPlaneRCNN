from torch import Tensor, nn
import torch


def debug_tensor(tensor: Tensor, text=""):
    print(text, tensor.shape, tensor.mean(), tensor.std())


def debug_model(model: nn.Module, text=""):
    for name, param in model.named_parameters():
        print(text, name, param.mean())


def count_boxes_by_area(boxes: Tensor, area_threshold: float):
    boxes = boxes.squeeze(0)
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    mask = areas > area_threshold
    count = torch.sum(mask).item()
    return count
