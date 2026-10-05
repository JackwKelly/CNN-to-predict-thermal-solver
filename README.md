# Thermal-conduction CNN surrogate

This is a very toy example, and in very early stages

The idea is that this functions as an analogue of a CFD + ML workflow except we have a 2D heat-conduction solver which generates data,
and a PyTorch CNN which learns to predict the final temperature field from the source layout.

## workflow

```bash
pip install -r requirements.txt
chmod +x run_simulations.sh

./run_simulations.sh 2000 20     # 2000 simulations, 20 running in parallel -> data/csv/
python concat_to_hdf5.py  --out /output_file_path.h5  # has to be .h5       # all CSVs -> data/dataset.h5  default output path is /data/....
python train.py                  # trains, saves checkpoints/best.pt, prints test metrics
python plot_results.py --index 0 # results/true_vs_pred.png and results/parity.png
```

## Files

| File | Purpose |

| `config.py` | Grid size, iteration count, diffusion number, paths, split seed |
| `simulate.py` | One simulation: random source blocks, explicit heat-equation solver, writes two CSVs |
| `run_simulations.sh` | Runs `simulate.py` for N ids, P at a time (via `xargs -P`) |
| `concat_to_hdf5.py` | Merges the CSVs into `data/dataset.h5` (`sources`, `final`, `sim_ids`) |
| `model.py` | `ThermalCNN` class (small U-Net, takes a 100x100 mask) |
| `data.py` | HDF5 loading and the deterministic train/val/test split |
| `train.py` | Training loop, checkpointing, test-set metrics |
| `plot_results.py` | True vs predicted plot for one test sample, plus parity diagram |

## Physics

Fourier's law plus energy conservation gives dT/dt = alpha * laplacian(T), solved with
an explicit 5-point stencil (r = alpha*dt/dx^2 = 0.2; the scheme needs r <= 0.25).
Source cells are held at T = 1, and the outer edge of the grid is held at T = 0.


