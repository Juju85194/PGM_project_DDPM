import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from utils.visualizer import plot, save_gif
from utils.helpers import num_to_groups
from schedulers.scheduler import extract, compute_diffusion_vars
from torchvision.utils import save_image
import os

def p_losses(denoise_model, x_start, t, alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

class Trainer:
    def __init__(self, config, denoise_model, dataloader, device):
        self.config = config
        self.denoise_model = denoise_model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = Adam(denoise_model.parameters(), lr=config.lr)
        self.results_folder = Path(config.results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.timesteps = config.timesteps
        self.betas = config.beta_schedule_fn(timesteps=self.timesteps)
        self.alphas, self.alphas_cumprod, self.alphas_cumprod_prev, self.sqrt_recip_alphas, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.posterior_variance = compute_diffusion_vars(self.betas)

    def train(self):
        for epoch in range(self.config.epochs):
          self.denoise_model.train()
          for step, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(self.device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

            loss = p_losses(self.denoise_model, batch, t, self.alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.config.loss_type)

            if step % self.config.log_every == 0:
              print("Loss:", loss.item())

            loss.backward()
            self.optimizer.step()

            # save generated images
            if step != 0 and step % self.config.save_and_sample_every == 0:
              self.denoise_model.eval()
              milestone = step // self.config.save_and_sample_every
              batches = num_to_groups(self.config.save_n_samples, self.config.sample_batch_size)
              all_images_list = list(map(lambda n: sample(self.denoise_model, batch_size=n, channels=self.config.channels, timesteps=self.config.timesteps, betas=self.betas, sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas=self.sqrt_recip_alphas, posterior_variance=self.posterior_variance), batches))
              all_images = torch.cat(all_images_list, dim=0)
              all_images = (all_images + 1) * 0.5
              save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 6)
              samples = sample(self.denoise_model, batch_size=64, channels=self.config.channels, timesteps=self.config.timesteps, betas=self.betas, sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas=self.sqrt_recip_alphas, posterior_variance=self.posterior_variance)
              save_gif(f'diffusion-{milestone}.gif', samples, self.config.image_size, self.config.channels)

        # Save the model
        torch.save(self.denoise_model.state_dict(), os.path.join(self.results_folder, "model.pth"))