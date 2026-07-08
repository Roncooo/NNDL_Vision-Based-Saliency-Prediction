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
We recommend using `conda` for environment management:
```bash
conda env create -f environment.yml
conda activate nndl_saliency
python -m ipykernel install --user --name nndl_saliency --display-name "Python (NNDL Saliency)"
```

Alternatively, if you prefer using `pip`:
```bash
pip install -r requirements.txt
```

## Usage

Running experiments is completely streamlined thanks to our configuration-driven pipeline:

1. **Configure:** Open `config.py` to define your experiment settings. This is where you specify the dataset, model architecture (from the registry), loss function, learning rate, and batch size.
2. **Run:** Open `run.ipynb`, ensure your kernel is set to **Python (NNDL Saliency)**, and run the cells. 

The notebook will automatically instantiate the chosen modules, execute the training loop, and generate the plots.
