# Satellite Road Extraction Evaluation


This repository provides an end-to-end deep learning framework designed to evaluate existing satellite road extraction datasets and methods.

* **Supported Datasets:**
    * [DeepGlobe Road Extraction Dataset](https://competitions.codalab.org/competitions/18467)
    * [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/)
* **Supported Models:**
    * U-Net (`unet.py`)
    * Attention U-Net (`attention_unet.yaml`)
    * LinkNet34 (`linknet.py`) (https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge)
    * D-LinkNet34 (`dlinknet.py`) (https://github.com/snakers4/spacenet-three)

## Repository Structure

```text
.
├── configs/                 # Configuration files for experiments
│   ├── dataset/             
│   ├── logger/              
│   ├── model/               
│   ├── trainer/             
│   └── run.yaml            
├── jobs/                    # Shell scripts for job submission
├── src/                     
│   ├── datasets/            # Dataset loaders and utility functions
│   ├── models/              # Neural network architecture implementations
│   ├── logger.py            # W&B and console logging setup
│   ├── predict.py           # Inference script
│   ├── run.py               # Main entry point for training
│   ├── trainer.py           # Training/Validation loop
│   └── utils.py             
├── pyproject.toml           
└── uv.lock
```

## Usage Instructions

### 1. Environment Setup

To initialize the environment using the provided `uv.lock` file:

```bash
uv sync
```

### Running Experiments

Experiments are executed via the `src.run` module. You can swap the model and dataset dynamically.

```bash
# Example: Train LinkNet34 on Massachusetts Roads
uv run python -m src.run model=linknet34 dataset=massachusetts-roads

# Example: Overriding hyperparameters
uv run python -m src.run model=unet trainer.lr=0.0001 trainer.batch_size=16
```
