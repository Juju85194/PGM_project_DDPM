import torch
from models.unet import Unet
from config import Config
import matplotlib.pyplot as plt
from utils.visualizer import get_reverse_transform
from schedulers.scheduler import compute_diffusion_vars

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
        imgs.append(img.cpu().numpy())
    return imgs

def sample(model, batch_size, channels, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), timesteps=timesteps, betas=betas, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas=sqrt_recip_alphas, posterior_variance=posterior_variance)

def show_sample_image(model, image_size, batch_size, channels, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    # sample 64 images
    samples = sample(model, batch_size, channels, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)

    # show a random one
    reverse_transform = get_reverse_transform(image_size)
    random_index = 5
    plt.imshow(reverse_transform(torch.tensor(samples[-1][random_index])), cmap="gray")

def main():
    config = Config()
    device = config.device
    model = Unet(
        dim=config.image_size,
        channels=config.channels,
        dim_mults=(1, 2, 4,)
    )
    model.load_state_dict(torch.load(f"results/model.pth", map_location=device))
    model.eval()
    betas = config.beta_schedule_fn(timesteps=config.timesteps)
    _, _, _, sqrt_recip_alphas, _, sqrt_one_minus_alphas_cumprod, posterior_variance = compute_diffusion_vars(betas)
    show_sample_image(model, config.image_size, config.sample_batch_size, config.channels, config.timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)

if __name__ == "__main__":
    main()