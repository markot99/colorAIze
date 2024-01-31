import numpy as np
import PIL
import torch
from torchvision import transforms

from colorizer.model import ColorizationModel
from colorizer.image_converter import convert_lab_to_rgb


class Colorizer:
    """Class for converting images"""

    def __init__(self, model_path: str):
        """
        Constructor for Colorizer class

        Parameters:
        model_path (str): Path to model
        """
        self.model = ColorizationModel()
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.model.get_device(),
            )
        )
        self.model_name = model_path.split("/")[-1].split(".")[0]
        self.model.eval()

    def colorize(self, img: PIL.Image) -> PIL.Image:
        """
        Colorize image

        Parameters:
        img (PIL.Image): Image to colorize

        Returns:
        PIL.Image: Colorized image
        """

        # convert image to tensor
        img = transforms.ToTensor()(img)

        # select only first channel (on grayscale images are r == g == b)
        img = img[:1]

        # normalize image to [-1, 1]
        img = img * 2.0 - 1.0

        with torch.no_grad():
            preds = self.model.generator(img.unsqueeze(0).to(self.model.get_device()))

        # convert image to rgb
        colorized = convert_lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]

        # restore image normalization
        np_img = (colorized * 255).astype(np.uint8)

        # convert image to PIL.Image
        return PIL.Image.fromarray(np_img)
