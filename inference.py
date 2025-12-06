from utils.datasets import load_single
from model.model import MaskRCNN
from config import Conf
import torch
from utils.pretrained_weight_map import load_mapped_weights, get_map_dict
from utils.visualization import visualize_mask

torch.set_default_dtype(torch.float32)
if __name__ == "__main__":
    model = MaskRCNN(Conf.anchor_conf, Conf.cam_params, Conf.image_shape)
    image, meta = load_single("test/image_2.png", Conf.data_set_config)

    weights = torch.load(
        "./checkpoints/checkpoint.pth", map_location=torch.device("cpu")
    )

    load_mapped_weights(
        model,
        weights,
        get_map_dict(model.state_dict().keys(), weights.keys()),  # type: ignore
    )

    model.eval()

    res = model.forward(image, meta)
    visualize_mask(res[1], res[0], "test/image_2.png")
