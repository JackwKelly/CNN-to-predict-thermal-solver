"""Train ThermalCNN on the HDF5 dataset.

Usage:
    python train.py [--epochs 40] [--batch-size 32] [--lr 1e-3] [--no-augment]

The best model (lowest validation MSE) is saved to checkpoints/best.pt and evaluated
once on the held-out test set at the end.
"""
import argparse
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from data import load_splits
from model import ThermalCNN


def random_dihedral(x, y):
    """Random 90-degree rotation + optional flip applied to a whole batch.

    Valid augmentation: the square domain with T=0 on every edge is symmetric under
    these transforms, so the transformed pair is also a correct (source, result) pair.
    """
    k = random.randint(0, 3)
    if k:
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    if random.random() < 0.5:
        x, y = x.flip(3), y.flip(3)
    return x, y


@torch.no_grad()
def predict(model, X, device, batch_size=128):
    model.eval()
    out = [model(X[i:i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--h5", default=config.H5_PATH)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits(args.h5)
    X_tr, Y_tr, _ = splits["train"]
    X_va, Y_va, _ = splits["val"]
    X_te, Y_te, _ = splits["test"]
    print(f"device={device}  train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = ThermalCNN(args.base_channels).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    config.CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            if not args.no_augment:
                x, y = random_dihedral(x, y)
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            running += loss.item()
        scheduler.step()

        train_loss = running / len(loader)
        val_loss = loss_fn(predict(model, X_va, device), Y_va).item()
        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "base_channels": args.base_channels,
                        "val_loss": val_loss}, config.CKPT_PATH)
            flag = "  * saved"
        print(f"epoch {epoch:3d}/{args.epochs}  train MSE {train_loss:.3e}  val MSE {val_loss:.3e}{flag}")

    # Final evaluation of the best checkpoint on the untouched test set.
    ckpt = torch.load(config.CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    pred = predict(model, X_te, device)
    mse = loss_fn(pred, Y_te).item()
    r2 = 1.0 - ((pred - Y_te) ** 2).sum().item() / ((Y_te - Y_te.mean()) ** 2).sum().item()
    print(f"\nTest set: MSE {mse:.3e}  RMSE {mse ** 0.5:.4f}  R^2 {r2:.4f}")


if __name__ == "__main__":
    main()
