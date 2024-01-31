import numpy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from colorizer.image_converter import convert_rgb_to_lab


class ColorizationDataset(Dataset):
    """Dataset for colorization"""

    def __init__(self, img_paths: list):
        """
        Constructor for ColorizationDataset class

        Parameters:
        img_paths (list): List of paths to images
        """
        self.size = 256
        self.transforms = transforms.Resize((self.size, self.size), Image.BICUBIC)
        self.img_paths = img_paths

    def __getitem__(self, idx: int) -> dict:
        """
        Get image from dataset

        Parameters:
        idx (int): Index of image

        Returns:
        dict: Dictionary containing L and ab channels
        """
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = numpy.array(img)

        img_lab = convert_rgb_to_lab(img)
        img_lab = transforms.ToTensor()(img_lab)
        
        # normalize
        l = img_lab[[0], ...] / 50.0 - 1.0
        ab = img_lab[[1, 2], ...] / 110.0

        return {"l": l, "ab": ab}

    def __len__(self) -> int:
        """
        Get length of dataset

        Returns:
        int: Length of dataset
        """
        return len(self.img_paths)
