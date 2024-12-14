from schedulers.scheduler import linear_beta_schedule, cosine_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule

class Config:
    def __init__(self):
        self.image_size = 28
        self.channels = 1
        self.batch_size = 128
        self.dataset_name = "fashion_mnist"
        self.epochs = 6
        self.loss_type = "huber"
        self.log_every = 100
        self.save_and_sample_every = 1000
        self.results_folder = "./results"
        self.lr = 1e-3
        self.timesteps = 300
        self.beta_schedule_fn = linear_beta_schedule
        self.save_n_samples = 4
        self.sample_batch_size = 64
        self.device = "cuda"