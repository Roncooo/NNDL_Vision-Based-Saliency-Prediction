# Vision-Based Saliency Prediction

**Authors:** Gianluca Caregnato, Francesco Roncolato, Giuly Wang

## Overview
This repository contains the codebase for our Deep Learning project focused on Vision-Based Saliency Prediction. The pipeline is built strictly in **PyTorch**, emphasizing a highly modular, registry-driven architecture to allow seamless swapping of datasets, model backbones, and loss functions.

## Datasets
* [SALICON](https://www.kaggle.com/datasets/roshan401/salicon)
* [MIT1003](https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html)

## Project Structure (WIP)
The codebase relies on a strict interface contract across the following modules:

* `config.py` - Centralized configurations for experiments.
* `data.py` - Dataset loaders, preprocessing, and augmentations (Outputs: `image [3,H,W]`, `gt [1,H,W]`).
* `models.py` - Model architectures utilizing a `@register_model` pattern.
* `losses.py` - Interchangeable and composable loss functions (MSE, KL, CC).
* `metrics.py` - Evaluation metrics dict for the eval loop.
* `train.py` - Generic, config-driven training loop.
* `run.ipynb` - Main notebook for running experiments and generating report figures.

## Installation & Setup
> **[TODO]:** Add conda/pip environment instructions, e.g., `requirements.txt` or `environment.yml`.

## Usage
> **[TODO]:** Add quickstart commands for training and evaluation once `train.py` is fully wired up.
