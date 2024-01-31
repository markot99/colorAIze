import os

import cv2
import numpy as np

from colorizer.colorizer import Colorizer
from colorizer.image_colorizer import ImageColorizer


class VideoColorizer:
    """Class for video colorization"""

    def __init__(self, colorizer: Colorizer, video_path: str) -> None:
        """
        Constructor for VideoColorizer class

        Parameters:
        colorizer (Colorizer): Colorizer class
        video_path (str): Path to video
        """
        self.colorizer = colorizer
        self.video_path = video_path

        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            print("Cannot open video")
            exit()

        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.count_of_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video.release()
        cv2.destroyAllWindows()

    def save_colorized_video(self, filename: str) -> None:
        """
        Colorize video and save it to a file in the current directory

        Parameters:
        filename (str): Filename of the colorized video
        """
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            print("Cannot open video")
            exit()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out_folder = "colorized"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        filename = out_folder + "/" + filename + ".mp4"
        out = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))

        frame_count = 1
        while True:
            ret, frame = self.video.read()
            if ret:
                img = ImageColorizer(self.colorizer, img=frame).get_post_processed_image()
                numpy_image = np.array(img)
                out.write(numpy_image)
            else:
                break
            print("Colorized frame", frame_count, "of", self.count_of_frames)
            frame_count += 1

        self.video.release()
        out.release()
        cv2.destroyAllWindows()
