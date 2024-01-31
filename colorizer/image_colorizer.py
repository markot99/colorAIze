import os
from io import BytesIO

import numpy as np
import PIL
import requests

from colorizer.colorizer import Colorizer


class ImageColorizer:
    """Class for  the colorization of images"""

    def __init__(self, colorizer: Colorizer, img_path=None, img_url=None, img: np.array = None):
        """
        Constructor for ImageColorizer class

        Parameters:
        colorizer (Colorizer): Colorizer class
        img_path (str): Path to image
        img_url (str): URL to image
        img (np.array): Image as numpy array
        """
        self.colorizer = colorizer

        if img_path is not None:
            self.img = PIL.Image.open(img_path).convert("RGB")

        if img_url is not None:
            response = requests.get(img_url, timeout=5)

            if response.status_code != 200:
                print("Error: Image URL not found")
                exit()

            self.img = PIL.Image.fromarray(np.array(PIL.Image.open(BytesIO(response.content)).convert("RGB")))

        if img is not None:
            self.img = PIL.Image.fromarray(img)

        self.original_width = self.img.width
        self.original_height = self.img.height
        self.prepared_width = self.get_next_number(self.original_width)
        self.prepared_height = self.get_next_number(self.original_height)

    def get_next_number(self, number: int) -> int:
        """
        Get next number divisible by 256

        Parameters:
        number (int): Number

        Returns:
        int: Next number divisible by 256
        """
        rest = number % 256
        if rest == 0:
            return number
        return number + (256 - rest)

    def get_prepared_image(self) -> PIL.Image:
        """
        Prepare image for colorization (make borders divisible by 256)

        Returns:
        PIL.Image: Prepared image
        """
        new_image = PIL.Image.new("RGB", (self.prepared_width, self.prepared_height), (255, 255, 255))

        # convert to grayscale
        new_image = new_image.convert("L")

        new_image.paste(
            self.img,
            (
                (self.prepared_width - self.original_width) // 2,
                (self.prepared_height - self.original_height) // 2,
            ),
        )
        return new_image

    def get_colorized_image(self) -> PIL.Image:
        """
        Get colorized image

        Returns:
        PIL.Image: Colorized image
        """
        prepared_image = self.get_prepared_image()
        return self.colorizer.colorize(prepared_image)

    def get_post_processed_image(self) -> PIL.Image:
        """
        Get post processed image (crop image to original size)

        Returns:
        PIL.Image: Post processed image
        """

        colorized_image = self.get_colorized_image()

        # crop image
        start_x = (self.prepared_width - self.original_width) // 2
        start_y = (self.prepared_height - self.original_height) // 2

        box = (
            start_x,
            start_y,
            start_x + self.original_width,
            start_y + self.original_height,
        )
        newimg = colorized_image.crop(box)
        return newimg

    def save_colorized_image(self, filename: str) -> None:
        """
        Save image to path

        Parameters:
        filename (str): Filename of the colorized image
        """
        img = self.get_post_processed_image()

        # get date and time for filename

        out_folder = "colorized"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        filename = out_folder + "/" + filename + ".jpg"
        img.save(filename)
