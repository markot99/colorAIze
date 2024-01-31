import csv
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader as TorchDataloader

from colorizer.loss_statistics import LossStatistics
from colorizer.model import ColorizationModel

# suppress warnings
warnings.filterwarnings(
    "ignore",
    message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in 7 negative Z values that have been clipped to zero",
)


def save_losses(loss_statistics: LossStatistics, epoch: int, model_filename: str) -> None:
    """
    Save training losses to csv file

    Parameters:
    loss_statistics (LossStatistics): loss statistics
    epoch (int): Epoch number
    model_filename (str): Filename of model
    """
    csv_filename = "models/losses.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    fieldnames = [
        "filename",
        "epoch",
        "loss_d_fake",
        "loss_d_real",
        "loss_d",
        "loss_g_gan",
        "loss_g_l1",
        "loss_g",
    ]

    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", encoding="UTF-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_filename, "a", encoding="UTF-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(
            {
                "filename": model_filename,
                "epoch": epoch,
                "loss_d_fake": loss_statistics.loss_d_fake.avg,
                "loss_d_real": loss_statistics.loss_d_real.avg,
                "loss_d": loss_statistics.loss_d.avg,
                "loss_g_gan": loss_statistics.loss_g_gan.avg,
                "loss_g_l1": loss_statistics.loss_g_l1.avg,
                "loss_g": loss_statistics.loss_g.avg,
            }
        )


def save_model(model: ColorizationModel, loss_statistics: LossStatistics, epoch) -> None:
    """
    Save model and training losses

    Parameters:
    model (ColorizationModel): Model to save
    loss_statistics (LossStatistics): loss statistics
    epoch (int): Epoch number
    """
    date_string = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = "models/" + str(date_string) + ".pt"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    torch.save(model.state_dict(), model_filename)
    save_losses(loss_statistics, epoch, model_filename)


def train_model(
    model: ColorizationModel,
    train_dataloader: TorchDataloader,
    validation_dataloader: TorchDataloader,
    epochs: int,
    display_every: int = 100,
):
    """
    Train model

    Parameters:
    model (ColorizationModel): Model to train
    train_dataloader (TorchDataloader): Dataloader for training data
    validation_dataloader (TorchDataloader): Dataloader for validation data
    epochs (int): Number of epochs to train
    display_every (int): Display training losses every x iterations
    """
    data = next(iter(validation_dataloader))
    for e in range(epochs):
        start = time.time()
        loss_statistics = LossStatistics()
        i = 0
        for data in train_dataloader:
            model.setup_input(data)
            model.optimize()
            loss_statistics.update(model, data["l"].size(0))
            i += 1
            if i % display_every == 0:
                loss_statistics.print()

            print(
                f"\rIteration {i}/{len(train_dataloader)}: {(time.time() - start)/i:.3f} s/iter",
                end="",
            )
            print(
                f"Estimated time for completing epoch {e+1}: {(time.time() - start)*(len(train_dataloader)-i)/i/60:.3f} minutes",
                end="",
            )

        end = time.time()
        print("++++++++++++++++++++++++++ Duration: " + str(end - start))

        save_model(model, loss_statistics, e + 1)
