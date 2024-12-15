from inspect import isfunction
import torch
import matplotlib.pyplot as plt
from utils.visualizer import get_transform
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from schedulers.scheduler import extract

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def compare_samples(samples_list, titles, image_size, channels, save_path):
    fig, axs = plt.subplots(1, len(samples_list), figsize=(15, 5))

    for i, samples in enumerate(samples_list):
        # Get the last image from the samples (final generated image)
        sample = samples[-1][0]

        # Reshape and normalize the image
        img = torch.tensor(sample).view(channels, image_size, image_size)
        img = (img + 1) / 2  # Rescale to [0, 1]

        # Display the image
        axs[i].imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def compare_schedules(config):
    # Plotting alpha_bar over time
    plt.figure(figsize=(10, 5))
    for schedule_fn in config.beta_schedule_fns:
        betas = schedule_fn(config.timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)
        plt.plot(np.linspace(0, 1, config.timesteps), alphas_cumprod, label=schedule_fn.__name__)
    plt.title("$\\bar \\alpha_t$ Over Time for Different Schedules")
    plt.xlabel("t/T")
    plt.ylabel("$\\bar \\alpha_t$")
    plt.legend()
    plt.savefig(Path(config.results_folder) / "alpha_bar_comparison.png")
    plt.show()

    # Plotting noisy images at different time steps
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    transform = get_transform(config.image_size)
    x_start = transform(image).unsqueeze(0)

    timesteps_to_plot = np.linspace(0, config.timesteps - 1, 11, dtype=int).tolist()
    for schedule_fn in config.beta_schedule_fns:
      betas = schedule_fn(timesteps=config.timesteps)
      alphas = 1. - betas
      alphas_cumprod = torch.cumprod(alphas, axis=0)
      sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
      noisy_images = []
      for t in timesteps_to_plot:
          # Calculate noise
          noise = torch.randn_like(x_start)
          sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, torch.tensor([t]), x_start.shape)
          noisy_image = x_start * (1 - sqrt_one_minus_alphas_cumprod_t) + noise * sqrt_one_minus_alphas_cumprod_t
          noisy_images.append(noisy_image)

      # Plot the noisy images for this schedule
      fig, axs = plt.subplots(1, len(timesteps_to_plot), figsize=(15, 5))
      for i, t in enumerate(timesteps_to_plot):
          img = noisy_images[i].squeeze().permute(1, 2, 0).numpy()
          img = (img + 1) / 2  # Rescale to [0, 1]
          axs[i].imshow(img, cmap='gray')
          axs[i].set_title(f't={t}')
          axs[i].axis('off')
      plt.suptitle(f"Noisy Images at Different Timesteps for {schedule_fn.__name__}")
      plt.tight_layout()
      plt.savefig(Path(config.results_folder) / f"noisy_images_{schedule_fn.__name__}.png")
      plt.show()