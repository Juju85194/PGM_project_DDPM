import torch
from models.unet import Unet
from trainers.trainer import Trainer
from config import Config
from datasets.dataset import get_dataloader
from torchvision.utils import save_image
from schedulers.scheduler import compute_diffusion_vars
import os

def train():
    # Set the device
    config = Config()
    device = config.device

    # Load the model and send to device
    model = Unet(
        dim=config.image_size,
        channels=config.channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    # Load the dataloader
    dataloader = get_dataloader(config.dataset_name, config.image_size, config.batch_size)

    # Train the model
    trainer = Trainer(config, model, dataloader, device)
    trainer.train()

if __name__ == "__main__":
    train()