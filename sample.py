import torch
from models.unet import Unet
from config import Config
import matplotlib.pyplot as plt
from utils.visualizer import get_reverse_transform, save_gif, get_transform
from schedulers.scheduler import extract, compute_diffusion_vars
from tqdm.auto import tqdm
from pathlib import Path
from torchvision.utils import save_image
from utils.helpers import compare_samples
from PIL import Image
import requests
import torchmetrics
from dataset.dataset import get_dataloader
from datasets import load_dataset
import torch.nn.functional as F
from datasets.utils import disable_progress_bar

# sampling functions
def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
def p_sample_loop(model, shape, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
        imgs.append(img.detach().cpu().numpy())
        if i % 50 == 0:
          torch.cuda.empty_cache()

    return imgs

def sample(model, batch_size, channels, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), timesteps=timesteps, betas=betas, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas=sqrt_recip_alphas, posterior_variance=posterior_variance)

def show_sample_image(model, image_size, batch_size, channels, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, samples_folder):
    # sample 64 images
    samples = sample(model, batch_size, channels, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)

    # show a random one
    reverse_transform = get_reverse_transform(image_size)
    random_index = 5
    sample_image = reverse_transform(torch.tensor(samples[-1][random_index]))
    plt.imshow(sample_image, cmap="gray")

    # Save the image
    sample_image_path = samples_folder / "sample.png"
    plt.savefig(sample_image_path)
    print(f"Sample image saved to {sample_image_path}")

def calculate_mse(img_pred, img_gt):
    return torch.mean((img_pred - img_gt)**2).item()

def calculate_fid(model, dataset, batch_size, channels, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device):
    fid = torchmetrics.image.fid.FrechetInceptionDistance(feature=2048).to(device)

    # Prepare real images
    real_images = []
    for batch in dataset:
        real_images.append(batch["pixel_values"])
        if len(real_images) * batch_size >= 100:
            break
    real_images = torch.cat(real_images, dim=0).to(device)
    # Reshape real images to 3 channels
    real_images = real_images.repeat(1, 3, 1, 1)

    # Resize real images
    real_images = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)

    # Convert images to be in the range [0, 255] and of type uint8
    real_images = (real_images * 255).byte()

    # Prepare generated images
    generated_images = []
    for _ in range((500 + batch_size - 1) // batch_size):
        samples = sample(model, batch_size, channels, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
        # Get the last image from the samples (final generated image)
        imgs = [torch.tensor(s[-1]).view(channels, image_size, image_size) for s in samples]
        imgs = torch.stack(imgs).to(device)

        # Reshape generated images to 3 channels
        imgs = imgs.repeat(1, 3, 1, 1)

        # Resize generated images
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)

        # Convert images to be in the range [0, 255] and of type uint8
        imgs = (imgs * 255).byte()

        generated_images.append(imgs)

    generated_images = torch.cat(generated_images, dim=0)

    # Update FID metric
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    return fid.compute().item()

def main():
    config = Config()
    samples_folder = Path(config.samples_folder)
    samples_folder.mkdir(exist_ok=True)
    device = config.device
    torch.manual_seed(config.seed)

    # Load the Fashion MNIST dataset and get a held-out image
    disable_progress_bar()
    dataset = load_dataset(config.dataset_name)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]  # Using 'test' split as validation for simplicity here
    dataloader = get_dataloader(config.dataset_name, config.image_size, config.batch_size)

    # Get a held-out image from the validation set
    image_gt = val_dataset[0]['image']
    transform = get_transform(config.image_size)
    image_gt = transform(image_gt).unsqueeze(0).to(device) # Add a batch dimension

    # Dictionary to store samples and metrics for each schedule
    schedules_data = {}

    for schedule_fn in config.beta_schedule_fns:
        model = Unet(
            dim=config.image_size,
            channels=config.channels,
            dim_mults=(1, 2, 4,)
        )
        model.load_state_dict(torch.load(f"results/model_{schedule_fn.__name__}.pth", map_location=device))
        model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        betas = schedule_fn(timesteps=config.timesteps)
        _, _, _, sqrt_recip_alphas, _, sqrt_one_minus_alphas_cumprod, posterior_variance = compute_diffusion_vars(betas)

        # Generate samples
        samples = sample(model, config.sample_batch_size, config.channels, config.image_size, config.timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)

        # Calculate metric
        if config.metric == "mse":
            img_pred = torch.tensor(samples[-1][0]).view(config.channels, config.image_size, config.image_size)
            img_pred = (img_pred + 1) / 2
            metric_value = calculate_mse(img_pred.to(device), image_gt.squeeze(0))
        elif config.metric == "fid":
            metric_value = calculate_fid(model, dataloader, config.sample_batch_size, config.channels, config.image_size, config.timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device)
        else:
            raise ValueError("Metric not supported")

        # Store samples and metric
        schedules_data[schedule_fn.__name__] = {"samples": samples, "metric": metric_value}

        # Save GIF
        save_gif(f'{schedule_fn.__name__}.gif', samples, config.image_size, config.channels, samples_folder)

    # Compare samples visually
    compare_samples([data["samples"] for data in schedules_data.values()], list(schedules_data.keys()), config.image_size, config.channels, samples_folder / "comparison.png")

    # Print metrics
    for name, data in schedules_data.items():
        print(f"Metric ({config.metric}) for {name}: {data['metric']}")

if __name__ == "__main__":
    main()