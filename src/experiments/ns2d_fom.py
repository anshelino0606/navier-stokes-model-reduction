import argparse
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from maths.ns2d import (
    energy, energy_terms, initial_velocity, make_model, simulate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--lx", type=float, default=4 * np.pi)
    parser.add_argument("--ly", type=float, default=2 * np.pi)
    parser.add_argument("--nu", type=float, default=0.02)
    parser.add_argument("--forcing-mode", type=int, default=4)
    parser.add_argument("--forcing-amplitude", type=float, default=1.0)
    parser.add_argument("--initial-amplitude", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/raw/ns2d/snapshots.npz"))
    args = parser.parse_args()

    model = make_model(
        nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
        nu=args.nu, forcing_mode=args.forcing_mode,
        amplitude=args.forcing_amplitude,
    )
    u0 = initial_velocity(model, args.initial_amplitude)
    tic = perf_counter()
    times, states = simulate(u0, args.dt, args.steps, model, args.save_every)
    states.block_until_ready()
    elapsed = perf_counter() - tic

    diagnostics = jax.vmap(lambda u: (
        energy(u, model),
        *energy_terms(u, model),
    ))(states)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t=np.asarray(times),
        velocity=np.asarray(states),
        energy=np.asarray(diagnostics[0]),
        injection=np.asarray(diagnostics[1]),
        dissipation=np.asarray(diagnostics[2]),
        convection_work=np.asarray(diagnostics[3]),
        nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
        nu=args.nu, forcing_mode=args.forcing_mode,
        forcing_amplitude=args.forcing_amplitude, dt=args.dt,
        save_every=args.save_every,
    )
    print(f"Saved {len(times)} frames to {out} in {elapsed:.3f}s (includes compilation)")
    print(f"Initial/final energy: {float(diagnostics[0][0]):.8g} / {float(diagnostics[0][-1]):.8g}")
    print(f"Largest sampled convective work: {np.abs(np.asarray(diagnostics[3])).max():.3e}")


if __name__ == "__main__":
    main()
