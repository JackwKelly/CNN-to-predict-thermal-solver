"""Plot (1) true vs predicted temperature for one test simulation and (2) a parity diagram.

Usage:
    python plot_results.py [--index 0]     # index into the test set

Outputs: results/true_vs_pred.png, results/parity.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data import load_splits
from model import ThermalCNN
from train import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--h5", default=config.H5_PATH)
    parser.add_argument("--ckpt", default=config.CKPT_PATH)
    parser.add_argument("--index", type=int, default=0, help="which test-set sample to show")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ThermalCNN(ckpt["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    X, Y, sim_ids = load_splits(args.h5)["test"]
    if not 0 <= args.index < len(X):
        raise SystemExit(f"--index must be in [0, {len(X) - 1}] because that is the size of the test set")
    pred = predict(model, X, device).numpy()[:, 0]
    true = Y.numpy()[:, 0]
    source = X.numpy()[:, 0]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: true vs predicted, same colour scale -------------------
    i = args.index
    vmax = max(true[i].max(), pred[i].max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, field, title in zip(axes, (true[i], pred[i]), ("True (simulation)", "Predicted (CNN)")):
        im = ax.imshow(field, cmap="inferno", vmin=0.0, vmax=vmax, origin="lower")
        ax.contour(source[i], levels=[0.5], colors="white", linewidths=0.8)   # source block outlines
        ax.set_title(title)
        ax.set_xlabel("x (cell)")
        ax.set_ylabel("y (cell)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Temperature (source = 1)")
    rmse = np.sqrt(np.mean((true[i] - pred[i]) ** 2))
    fig.suptitle(f"Test sample {i} (simulation {sim_ids[i]}), RMSE = {rmse:.4f}")

    pred_path = config.RESULTS_DIR / f"true_vs_pred_{i}.png"
    fig.savefig(pred_path, dpi=150)
   

    plt.close(fig)

    # ---- Figure 2: parity diagram over every pixel of the test set --------
    t, p = true.ravel(), pred.ravel()
    r2 = 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
    rmse_all = np.sqrt(np.mean((p - t) ** 2))
    lim = [0.0, max(t.max(), p.max())]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    hb = ax.hexbin(t, p, gridsize=120, extent=(*lim, *lim), bins="log", mincnt=1, cmap="viridis")
    ax.plot(lim, lim, "r--", lw=1, label="ideal (y = x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("True temperature")
    ax.set_ylabel("Predicted temperature")
    ax.set_title(f"Parity plot, test set ({len(true)} fields)\nR$^2$ = {r2:.4f}, RMSE = {rmse_all:.4f}")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, shrink=0.8, label="pixel count (log)")
    fig.savefig(config.RESULTS_DIR / "parity.png", dpi=150)
    plt.close(fig)

    print(f"Saved {pred_path} and {config.RESULTS_DIR / 'parity.png'}")


if __name__ == "__main__":
    main()
