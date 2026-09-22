# Dual-STNet: A Dual-Branch Spatio-Temporal Network with Explicit Feature Decoupling for Pre-stack AVA Inversion

## Overview

Dual-STNet is a deep-learning-based seismic inversion project. Its goal is to simultaneously invert three elastic properties of the subsurface medium from pre-stack seismic data:

- P-wave velocity (Vp)
- S-wave velocity (Vs)
- Density

The project uses a semi-supervised learning framework composed of two main parts:

- **Inverse model**: predicts the elastic properties (Vp, Vs, Density) from seismic data.
- **Forward model**: reconstructs the seismic response from the predicted elastic properties, used for self-supervised training.

## Repository Structure

```text
Dual-STNet/ 
└── SEAM/
    ├── train_main.py
    ├── core/
    │   └── functions.py
    ├── model/
    │   └── vitCA_and_MutiScale_ModernTCN2D.py
    └── data/
```


## Requirements

- Python 3
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- scikit-image
- tqdm

## Data

The data files are located in:

```text
SEAM/data/
```

The project uses NumPy `.npy` files to store seismic data and elastic property data.

## Usage
### SEAM

```bash
cd SEAM
python train_main.py
```

## Testing / Evaluation

### Model evaluation

`train_main.py` provides a test/evaluation mode for a trained model. To evaluate with a saved checkpoint:

```bash
python train_main.py -test_checkpoint <checkpoint_name>
```

This command must be run from the corresponding dataset directory, for example:
```bash
cd SEAM
python train_main.py -test_checkpoint <checkpoint_name>
```

Note:

- Using `-test_checkpoint` requires an existing trained checkpoint.
- The repository does not include a `checkpoints/` directory or any pretrained models.
- You must first train a model, or provide your own checkpoint, before running the evaluation.
