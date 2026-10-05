"""

This python script runs the 2D heat convection simulation. 

The domain consists of a 100x100 grid. The outer boundaries are held at T=0 (heat sinks) and the source blocks 
are held at T = s

saves the domain results after the specified number of iterations in a csv file, saved in an output folder in the csv folder in the working directory.

the parsed simulation id is used to seed the random number generation
    
    
"""
import argparse
from pathlib import Path

import numpy as np

import config


def generate_sources(rng, grid=config.GRID_SIZE, max_blocks=5, min_size=4, max_size=20):
    """randomises the number of, and location of source blocks
    """
    mask = np.zeros((grid, grid), dtype=np.uint8)
    for _ in range(rng.integers(1, max_blocks + 1)):
        h = int(rng.integers(min_size, max_size + 1))
        w = int(rng.integers(min_size, max_size + 1))
        r0 = int(rng.integers(1, grid - h))   # block occupies rows r0 .. r0+h-1 <= grid-2
        c0 = int(rng.integers(1, grid - w))
        mask[r0:r0 + h, c0:c0 + w] = 1
    return mask


def run_simulation(source_mask, n_iters=config.N_ITERS, r=config.DIFFUSION_NUMBER):
    """ The actual Fouriers law time marching scheme"""
    if not 0.0 < r <= 0.25:
        raise ValueError(f"r={r} is unstable for the explicit 2D scheme (need 0 < r <= 0.25)")

    src = source_mask.astype(bool)
    T = np.zeros(source_mask.shape, dtype=np.float64)
    T[src] = config.SOURCE_TEMP

    for _ in range(n_iters):
        laplacian = (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2]
                     - 4.0 * T[1:-1, 1:-1])
        T[1:-1, 1:-1] += r * laplacian   # laplacian is a fresh array, so this is a clean Euler step
        T[src] = config.SOURCE_TEMP      # sources stay fixed; outer edge is never touched (stays 0)
    return T


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--id", type=int, required=True, help="simulation id (also seeds the random sources)")
    parser.add_argument("--out-dir", type=Path, default=config.CSV_DIR)
    parser.add_argument("--n-iters", type=int, default=config.N_ITERS)
    parser.add_argument("--seed-offset", type=int, default=0, help="change to generate a fresh dataset")
    args = parser.parse_args()

    rng = np.random.default_rng([args.seed_offset, args.id])
    source = generate_sources(rng)
    final = run_simulation(source, n_iters=args.n_iters)

    args.out_dir.mkdir(parents=True, exist_ok=True)   # makes the csv output inside the csv folder
    stem = args.out_dir / f"sim_{args.id:05d}"
    np.savetxt(f"{stem}_source.csv", source, fmt="%d", delimiter=",")
    np.savetxt(f"{stem}_final.csv", final, fmt="%.6g", delimiter=",")


if __name__ == "__main__":
    main()
