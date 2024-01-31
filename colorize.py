import argparse
import datetime
import os

from matplotlib import pyplot as plt

from colorizer.colorizer import Colorizer
from colorizer.image_colorizer import ImageColorizer
from colorizer.video_colorizer import VideoColorizer


def check_weights_argument(weight: str, weights: str) -> None:
    """
    Check if the weights argument is valid

    Parameters:
    weight (str): Path to model weight
    weights (str): Path to the model weights folder
    """
    # model and models are mutually exclusive
    if weight is not None and weights is not None:
        print("Please specify either weight or weights")
        exit()

    # check if model exists
    if weight is not None:
        if not os.path.exists(weight):
            print("Weight " + weight + " does not exist")
            exit()

    # check if weights folder exists
    if weights is not None:
        if not os.path.exists(weights):
            print("Weights folder " + weights + " does not exist")
            exit()

        # check if models folder contains weights
        weights = os.listdir(weights)
        if len(weights) == 0:
            print("Weights folder is empty")
            exit()


def check_image_and_video_arguments(image_path: str, image_url: str, video_path: str) -> None:
    """
    Check if the image and video arguments are valid

    Parameters:
    image_path (str): Path to image
    image_url (str): URL to image
    video_path (str): Path to video
    """
    if (image_path is not None) and not os.path.exists(image_path):
        print(image_path)
        print("Image path does not exist")
        return

    if (video_path is not None) and not os.path.exists(video_path):
        print("Video path does not exist")
        exit()


def colorize(colorizer: Colorizer, image_path: str, image_url: str, video_path: str, out_filename: str) -> None:
    """
    Colorize image or video with the given colorizer

    Parameters:
    colorizer (Colorizer): Colorizer class
    image_path (str): Path to image
    image_url (str): URL to image
    video_path (str): Path to video
    out_filename (str): Filename of the colorized video
    """
    if video_path is not None:
        video_colorizer = VideoColorizer(colorizer, video_path)
        video_colorizer.save_colorized_video(out_filename)

    if image_url is not None:
        image_colorizer = ImageColorizer(colorizer, img_url=image_url)
        image_colorizer.save_colorized_image(out_filename)

    if image_path is not None:
        image_colorizer = ImageColorizer(colorizer, img_path=image_path)
        image_colorizer.save_colorized_image(out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize images")

    parser.add_argument("--image_path", type=str, help="Path to image", default=None)
    parser.add_argument("--image_url", type=str, help="URL to image", default=None)
    parser.add_argument("--video_path", type=str, help="Path to video", default=None)
    parser.add_argument("--weight", type=str, help="Path to model weight", default=None)
    parser.add_argument("--weights", type=str, help="Path to the model weights folder", default=None)
    parser.add_argument("--view", type=bool, help="View the image in a plot side by side", default=False)

    args = parser.parse_args()

    check_weights_argument(args.weight, args.weights)
    check_image_and_video_arguments(args.image_path, args.image_url, args.video_path)

    out_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.weight is not None:
        print("Using model: " + args.weight)
        colorizer = Colorizer(args.weight)
        colorize(
            colorizer,
            args.image_path,
            args.image_url,
            args.video_path,
            out_filename + "_weight_" + colorizer.model_name,
        )
        print("Finished Colorization")
        exit()

    if args.weights is not None:
        weights = os.listdir(args.weights)
        for weight in weights:
            colorizer = Colorizer(args.weights + "/" + weight)
            colorize(
                colorizer,
                args.image_path,
                args.image_url,
                args.video_path,
                out_filename + "_weight_" + colorizer.model_name,
            )
        print("Finished Colorization")
        exit()

    print("No weight or weights specified")
    exit()
