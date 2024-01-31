import glob
import random

from torch.utils.data import DataLoader as TorchDataloader

from colorizer.colorization_dataset import ColorizationDataset
from colorizer.model import ColorizationModel
from colorizer.model_trainer import train_model

train_data_path = "./data/train"
val_data_path = "./data/val"

if __name__ == "__main__":
    train_data_paths = glob.glob(train_data_path + "/*/*.JPEG")
    random.shuffle(train_data_paths)
    print("Number of training images: " + str(len(train_data_paths)))

    val_data_paths = glob.glob(val_data_path + "/*.JPEG")
    random.shuffle(val_data_paths)
    print("Number of validation images: " + str(len(val_data_paths)))

    train_datset = ColorizationDataset(train_data_paths)
    train_dataloader = TorchDataloader(train_datset, batch_size=16, num_workers=4, pin_memory=True)

    validation_dataset = ColorizationDataset(val_data_paths)
    validation_dataloader = TorchDataloader(validation_dataset, batch_size=16, num_workers=4, pin_memory=True)

    model = ColorizationModel()

    # Optional load existing model
    # model.load_state_dict(torch.load("<filename>", map_location=device))

    train_model(model, train_dataloader, validation_dataloader, 20)
