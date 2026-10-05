"""Concatenates all simulation CSVs into a single HDF5 dataset.

HDF5 layout (N = number of simulations):
    sources  (N, 100, 100) uint8     1 = source block, 0 = empty   <- CNN input
    final    (N, 100, 100) float32   temperature after N_ITERS     <- CNN target
    sim_ids  (N,)          int64

Usage:
    python concat_to_hdf5.py [--csv-dir data/csv] [--out data/dataset.h5]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--csv-dir", type=Path, default=config.CSV_DIR)
    parser.add_argument("--out", type=Path, default=config.H5_PATH) # should be a FILE path
    args = parser.parse_args()

    source_files = sorted(args.csv_dir.glob("sim_*_source.csv"))
    if not source_files:
        raise SystemExit(f"No sim_*_source.csv files found in {args.csv_dir}. Run run_simulations.sh first.")

    n, g = len(source_files), config.GRID_SIZE
    sources = np.zeros((n, g, g), dtype=np.uint8)
    final = np.zeros((n, g, g), dtype=np.float32)
    sim_ids = np.zeros(n, dtype=np.int64)

    for i, src_path in enumerate(source_files):
        sim_id = int(src_path.stem.split("_")[1])
        fin_path = args.csv_dir / f"sim_{sim_id:05d}_final.csv"
        if not fin_path.exists():
            raise SystemExit(f"Missing {fin_path.name}: simulation {sim_id} did not finish.")

        s = np.loadtxt(src_path, delimiter=",", dtype=np.uint8)
        f = np.loadtxt(fin_path, delimiter=",", dtype=np.float32)
        if s.shape != (g, g) or f.shape != (g, g):
            raise SystemExit(f"Unexpected shape in simulation {sim_id}: {s.shape}, {f.shape}")

        sources[i], final[i], sim_ids[i] = s, f, sim_id
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  loaded {i + 1}/{n}")

    if args.out.parent.is_file():
        args.out.parent.unlink()  # remove a stale file blocking the output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)

    
    with h5py.File(args.out, "w") as h5:
        h5.create_dataset("sources", data=sources, compression="gzip")
        h5.create_dataset("final", data=final, compression="gzip")
        h5.create_dataset("sim_ids", data=sim_ids)
    print(f"Wrote {n} simulations to {args.out}")


if __name__ == "__main__":
    main()
