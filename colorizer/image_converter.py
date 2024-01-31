import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab


def convert_lab_to_rgb(l: torch.Tensor, ab: torch.tensor) -> np.array:
    """
    Convert Lab to RGB

    Parameters:
    l (torch.Tensor): L channel
    ab (torch.Tensor): ab channels

    Returns:
    numpy.array: RGB image
    """
    l = (l + 1.0) * 50.0
    ab = ab * 110.0
    lab = torch.cat([l, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in lab:
        rgb_imgs.append(lab2rgb(img))
    return np.stack(rgb_imgs, axis=0)


def convert_rgb_to_lab(img: np.array) -> np.array:
    """
    Convert RGB to Lab image

    Parameters:
    img (np.array): RGB image

    Returns:
    np.array: Lab image
    """
    return rgb2lab(img).astype("float32")
