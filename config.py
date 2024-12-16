from schedulers.scheduler import linear_beta_schedule, cosine_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, linear_variance_beta_schedule, softsign_beta_schedule, tanh_beta_schedule
from pathlib import Path

class Config:
    def __init__(self):
        self.image_size = 28
        self.channels = 1
        self.batch_size = 128
        self.dataset_name = "fashion_mnist"
        self.epochs = 6
        self.loss_type = "l2" # l1 or l2 or huber
        self.log_every = 100
        self.save_and_sample_every = 1000
        self.results_folder = "./results"
        self.samples_folder = Path(self.results_folder) / "samples"
        self.lr = 1e-3
        self.timesteps = 1000
        self.beta_schedule_fns = [linear_beta_schedule, cosine_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, linear_variance_beta_schedule]
        self.save_n_samples = 4
        self.sample_batch_size = 64
        self.device = "cuda"
        self.metric = "mse" # mse or fid
        self.seed = 42