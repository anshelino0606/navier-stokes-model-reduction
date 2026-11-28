from pathlib import Path
import sys
import time
import argparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from maths.ns2d import make_model, fom_step_flat
from maths.ns2d_io import load_snapshot_matrix

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description="Measure the runtime of the full fine-grid FOM.")

    parser.add_argument( "--reference", type=Path, default=REPO / "data/raw/ns2d/reference.npz")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()

    if args.dt <= 0:
        parser.error("--dt must be positive")

    t_ref, U_ref, model = load_snapshot_matrix(args.reference, as_jax=True)

    t_ref = np.asarray(t_ref)

    train_path = REPO / "data/raw/ns2d/training.npz"
    t_train, _, _ = load_snapshot_matrix(train_path, as_jax=True)
    t_train = np.asarray(t_train)

    t0 = float(t_train[-1])
    i0 = int(np.argmin(np.abs(t_ref - t0)))

    if abs(float(t_ref[i0]) - t0) > 1e-7:
        raise ValueError("Reference does not contain the training endpoint.")

    t_end = float(t_ref[-1])
    total_steps = int(round((t_end - t0) / args.dt))

    if total_steps < 1:
        raise ValueError("Prediction interval is too short.")

    u0 = jnp.asarray(U_ref[:, i0])

    print("\nFULL FINE-GRID FOM TIMING")
    print("-------------------------")
    print(f"Grid:             {model.nx} x {model.ny}")
    print(f"Time interval:    {t0:.4f} to {t_end:.4f}")
    print(f"Time step:        {args.dt}")
    print(f"Number of steps: {total_steps}")
    print(f"Repeats:          {args.repeats}")

    runtimes = []

    for repeat in range(args.repeats):
        u = u0

        if repeat == 0:
            u = fom_step_flat(u, args.dt, model)
            jax.block_until_ready(u)

            start = time.perf_counter()

            for _ in range(total_steps - 1):
                u = fom_step_flat(u, args.dt, model)

            jax.block_until_ready(u)

        else:

            start = time.perf_counter()
            for _ in range(total_steps):
                u = fom_step_flat(u, args.dt, model)

            jax.block_until_ready(u)

        elapsed = time.perf_counter() - start
        runtimes.append(elapsed)

        print(f"Run {repeat + 1}: {elapsed:.3f} seconds")

    measured = runtimes[1:] if len(runtimes) > 1 else runtimes

    print("\nRESULTS")
    print("-------")
    print(f"Average runtime: {np.mean(measured):.3f} seconds")
    print(f"Minimum runtime: {np.min(measured):.3f} seconds")
    print(f"Maximum runtime: {np.max(measured):.3f} seconds")


if __name__ == "__main__":
    main()
