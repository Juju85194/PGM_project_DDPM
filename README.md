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
    *   Modify the `config.py` file to adjust hyperparameters, including the beta scheduling function and its parameters. You can choose from `linear_beta_schedule`, `cosine_beta_schedule`, `quadratic_beta_schedule`, `sigmoid_beta_schedule`, or define your own in `schedulers/scheduler.py`.

2. **Run Training:**
    *   Execute the `train.py` script:

        ```bash
        python train.py
        ```

    *   Training progress, including loss values, will be printed to the console.
    *   Generated samples will be saved periodically in the `results` directory (or the directory you specified in `config.py`).
    *   The trained model will be saved to the `results` directory as `model.pth`.

### Sampling

1. **Load a Trained Model:**
    *   Make sure you have a trained model in the `results` directory (or specify the path in `sample.py`).

2. **Generate Samples:**
    *   Run the `sample.py` script:

        ```bash
        python sample.py
        ```

    *   This will generate and display a random sample from the model.
    *   You can modify `sample.py` to generate multiple samples or create a GIF of the denoising process.

## Experimenting with Beta Scheduling

The main purpose of this repository is to facilitate experimentation with different beta schedules. Here's how you can do it:

1. **Modify Existing Schedules:**
    *   Change the `beta_schedule_fn` in `config.py` to one of the predefined functions in `schedulers/scheduler.py`.
    *   Adjust the parameters of the chosen schedule within `config.py`.

2. **Create New Schedules:**
    *   Implement new beta scheduling functions in `schedulers/scheduler.py`.
    *   Update `config.py` to use your newly defined schedule function.

3. **Run and Compare:**
    *   Train models using `train.py` with different schedules.
    *   Generate samples using `sample.py` from the trained models.
    *   Visually compare the generated samples and analyze the effects of different schedules on sample quality and the denoising process. You can also incorporate quantitative metrics for a more rigorous comparison.
