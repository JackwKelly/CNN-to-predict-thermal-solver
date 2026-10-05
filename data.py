"""Load the HDF5 dataset and produce a deterministic train/val/test split."""
import h5py
import numpy as np
import torch

import config


def load_splits(h5_path=config.H5_PATH):
    """Return {"train"|"val"|"test": (X, Y, sim_ids)}.

    X, Y are float32 tensors of shape (n, 1, 100, 100); sim_ids is a numpy array.
    The split depends only on config.SPLIT_SEED and the dataset size, so train.py and
    plot_results.py always agree on which simulations are in the test set.
    """
    with h5py.File(h5_path, "r") as h5:
        sources = h5["sources"][:]
        final = h5["final"][:]
        sim_ids = h5["sim_ids"][:]

    n = len(sources)
    order = np.random.default_rng(config.SPLIT_SEED).permutation(n)
    n_train = int(config.TRAIN_FRAC * n)
    n_val = int(config.VAL_FRAC * n)
    index = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }

    X = torch.from_numpy(sources).float().unsqueeze(1)   # the data that the CNN ingest is the input source block arrangement and the final 2D thermal distributions.
    Y = torch.from_numpy(final).float().unsqueeze(1)
    return {name: (X[idx], Y[idx], sim_ids[idx]) for name, idx in index.items()}
