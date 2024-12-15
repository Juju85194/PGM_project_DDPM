# DDPM Experimentation: Exploring Beta Scheduling in Denoising Diffusion Probabilistic Models

This repository provides a structured framework for experimenting with Denoising Diffusion Probabilistic Models (DDPMs), with a particular focus on exploring the effects of different beta scheduling strategies. It is based on the code from the blog post "The Annotated Diffusion Model" and the paper "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).

## Repository Structure

The repository is organized as follows:

```
ddpm-experiment/
├── models/          # Contains the U-Net model definition.
│   └── unet.py       
├── schedulers/      # Contains beta scheduling functions.
│   └── scheduler.py
├── dataset/        # Dataset loading and preprocessing.
│   └── dataset.py
├── trainers/        # Training loop and logic.
│   └── trainer.py
├── utils/           # Helper functions and visualization tools.
│   ├── helpers.py
│   └── visualizer.py
├── config.py         # Configuration parameters.
├── train.py          # Main training script.
├── sample.py         # Script for generating samples.
└── requirements.txt  # Project dependencies.
```

## Usage

### Training

1. **Configuration:**
    *   Modify the  `config.py`  file to adjust hyperparameters, including the beta scheduling function and its parameters. You can choose from  `linear_beta_schedule`,  `cosine_beta_schedule`,  `quadratic_beta_schedule`,  `sigmoid_beta_schedule`, or define your own in  `schedulers/scheduler.py`.
    *   Specify the metric to use for evaluation: either "mse" (Mean Squared Error) or "fid" (Fréchet Inception Distance).
    *   You can add several beta scheduling functions to the list `beta_schedule_fns` to compare them.
2. **Run Training:**
    *   Execute the  `train.py`  script:

        ```bash
        python train.py
        ```

    *   Training progress, including loss values, will be printed to the console.
    *   Generated samples will be saved periodically in the  `results/samples`  directory (or the directory you specified in  `config.py`).
    *   The trained model will be saved to the  `results`  directory as  `model.pth`.
    *   Plots comparing the beta schedules (alpha bar over time and noisy images at different timesteps) will be generated in the `results` folder.

### Sampling

1. **Load a Trained Model:**
    *   Make sure you have trained models in the  `results`  directory (one for each schedule you want to compare, named as `model_{schedule_fn_name}.pth`).

2. **Generate Samples and Compare:**
    *   Run the  `sample.py`  script:

        ```bash
        python sample.py
        ```

    *   This will generate samples from each trained model using the same seed for a fair comparison.
    *   A comparison image (`comparison.png`) will be created in `results/samples`, showing the generated samples side-by-side.
    *   GIFs of the sampling process for each schedule will be saved in the `results/samples` folder.
    *   The chosen metric (MSE or FID) will be calculated and printed for each schedule.

## Experimenting with Beta Scheduling

The main purpose of this repository is to facilitate experimentation with different beta schedules. Here's how you can do it:

1. **Modify Existing Schedules:**
    *   Change the  `beta_schedule_fns`  in  `config.py`  to a list containing one or more of the predefined functions in  `schedulers/scheduler.py`.
    *   Adjust the parameters of the chosen schedule within  `config.py`.

2. **Create New Schedules:**
    *   Implement new beta scheduling functions in  `schedulers/scheduler.py`.
    *   Update  `config.py`  to include your newly defined schedule function in the `beta_schedule_fns` list.

3. **Run and Compare:**
    *   Train models using  `train.py`  with different schedules.
    *   Generate samples and compare results using  `sample.py`.
    *   Visually compare the generated samples (using the comparison image and GIFs) and analyze the effects of different schedules on sample quality using the chosen metric (MSE or FID). The plots generated during training (alpha bar over time, noisy image plot) will help you understand the impact of different schedules on the diffusion process.

## Metrics

*   **Mean Squared Error (MSE):**  Measures the pixel-wise difference between a generated image and a held-out image from the validation set. Lower values are better.
*   **Fréchet Inception Distance (FID):**  A more perceptually relevant metric that compares the distribution of generated images to the distribution of real images in the feature space of a pre-trained Inception network. Lower values are better.
