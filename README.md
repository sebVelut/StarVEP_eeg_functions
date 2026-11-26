# StarVEP EEG Functions

Utilities for loading, preprocessing, splitting, and converting EEG data from the
**StArVEP dataset** (available from Zenodo) for machine-learning experiments.
The package was notably developed and used with [Gromo](https://github.com/growingnet/gromo)

> [!IMPORTANT]
> This package is experimental and is primarily oriented toward the
> `dryburst100.set` StArVEP recordings used in the original experiments.
> Paths and external project dependencies may therefore require adaptation.

## What this package provides

- Loading EEGLAB `.set` recordings with MNE.
- Band-pass filtering and epoch extraction.
- Sliding-window creation over the 2.2-second code epochs.
- Conversion of burst code labels into binary sample labels.
- xDAWN spatial filtering and Riemannian-based alignment.
- Class balancing by domain with `imbalanced-learn`.
- Train, validation, and test splits for a single-participant/single-session
  experiment.
- PyTorch `TensorDataset` and Gromo-compatible `DataLoader` creation.
- Riemannian and Euclidean alignment utilities.

## Repository layout

| Path | Purpose |
|---|---|
| `STLDataLoader.py` | Main MNE-based loader for StArVEP `.set` data. |
| `get_dataloader.py` | Preprocessing, dataset splitting, and PyTorch/Gromo DataLoader helpers. |
| `utils.py` | Additional experimental covariance, windowing, balancing, and loading utilities. |
| `Alignments/` | Riemannian alignment and covariance/alignment implementations. |
| `pyproject.toml` | Package metadata and Python dependencies. |

## Data requirements

The main loader expects one EEGLAB file per participant, named as:

```text
<SUBJECT>_dryburst100.set
```

For example:

```text
S01_dryburst100.set
S02_dryburst100.set
```

Place these files in the dataset directory configured by `config["path"]`.
Annotations in the recordings are used to identify the trial epochs and visual
code associated with each trial. The EEG is resampled and band-pass filtered
before epoching.


## Installation

Python 3.10 or newer is required.

Create and activate a virtual environment, then install this folder in editable
mode:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

The declared dependencies include PyTorch, MNE, pyRiemann, scikit-learn,
imbalanced-learn, OmegaConf, Weights & Biases, and Gromo. Installing the package
through `pyproject.toml` also requires access to the Gromo package expected by
`get_dataloader.py`.

Or you can install it in your environment with `pip install "git+https://github.com/sebVelut/StarVEP_eeg_functions.git`.

## Quick usage

The recommended entry point is `get_dataloader_eeg()`:

```python
import torch
from get_dataloader import get_dataloader_eeg

config = {
    "path": "path/to/StarVEP/data",
    "participant": "S01",
    "fmin": 1.0,
    "fmax": 25.0,
    "window_size": 0.35,
    "sample_freq": 500,
    "fps": 60,
    "timewise": "time_sample",
    "n_class": 5,
    "ncal": 4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_dataloader_eeg(
    config,
    device,
    batch_size=32,
    num_workers=4,
)

for eeg, labels in train_loader:
    print(eeg.shape, labels.shape)
```

To construct the datasets without DataLoaders:

```python
from get_dataloader import get_dataset

train_dataset, val_dataset, test_dataset = get_dataset(config, device)
print(len(train_dataset), len(val_dataset), len(test_dataset))
```

## Configuration parameters

| Parameter | Example | Description |
|---|---:|---|
| `path` | `/data/starvep` | Directory containing participant `.set` files. |
| `participant` | `S01` | Participant identifier used to build the filename. |
| `fmin` | `1.0` | Lower frequency cutoff in Hz. |
| `fmax` | `40.0` | Upper frequency cutoff in Hz. |
| `window_size` | `0.35` | Sliding-window duration in seconds. |
| `sample_freq` | `500` | Sampling frequency after resampling. |
| `fps` | `60` | Visual-code update frequency. |
| `timewise` | `time_sample` | Window progression unit: samples or frames. |
| `n_class` | `5` | Number of visual codes/classes. |
| `ncal` | `4` | Number of calibration blocks used by the current experiment setup. |

The implementation also assumes a 2.2-second trial duration. This value is used
for epoch extraction and for calculating the calibration, validation, and test
boundaries.

## Preprocessing pipeline

The `get_dataset()` workflow performs the following steps:

1. `STLDataLoader.load_data()` loads and filters participant recordings.
2. `STLDataLoader.get_epochs()` extracts epochs and builds sliding EEG windows.
3. Event codes are mapped to binary labels.
4. `features_preproc()` standardizes calibration data and applies xDAWN.
5. A Riemannian mean-based alignment is fitted on calibration windows and
   applied to the participant windows.
6. `get_train_test_data()` separates calibration, validation, and test regions.
7. `balance()` randomly undersamples each split/domain to balance binary labels.
8. Data is converted to CPU-based `torch.Tensor` objects and wrapped in
   `TensorDataset`.

Tensors are deliberately created on CPU. With multiple DataLoader workers,
CUDA tensors can trigger initialization errors during process forking; the
training loop should move the corresponding batch to the selected device.

## Main APIs

### `STLDataLoader`

```python
from STLDataLoader import STLDataLoader

loader = STLDataLoader(
    path="/data/starvep",
    fmin=1.0,
    fmax=40.0,
    window_size=0.35,
    sample_freq=500,
    fps=60,
    timewise="time_sample",
    participants=["S01"],
    n_class=5,
)
raw = loader.load_data()
X, y, domains, codes, labels_code = loader.get_epochs(raw)
```

The window shape is of the form:

```text
(participants, windows, channels, samples_per_window)
```

Labels and domain arrays are exposed at the same window granularity.

### `Aligner`

The alignment module can align real-valued EEG matrices or covariance matrices:

```python
from Alignments.aligner import Aligner

aligner = Aligner(estimator="lwf", metric="real")
X_aligned = aligner.fit_transform(X_calibration)
X_aligned_test = aligner.transform(X_test)
```

Use `metric="riemann"` for covariance-matrix inputs and `metric="real"` for
multi-channel time-series inputs.


## Experimental and compatibility notes

- `utils.py` contains historical and workspace-specific absolute Windows paths.
  They must be updated or removed before using it outside the original machine.
- `get_dataloader.py` imports Gromo's `make_dataloader` and can use an OmegaConf
  `DictConfig`, although it currently works with compatible mapping objects.
- The xDAWN, alignment, and split routines assume a particular StArVEP coding
  and calibration protocol. They should not be treated as a universal EEG
  preprocessing pipeline.
- Some imports use fallbacks for both standalone-package and `hydra_script`
  repository layouts.
- The package metadata has a `gromo` dependency, but Gromo is primarily a
  runtime integration dependency in this context.

## Related projects

- Gromo: growing neural-network modules and containers.
- StArcVEP Zenodo dataset:  and its EEGLAB annotations.
