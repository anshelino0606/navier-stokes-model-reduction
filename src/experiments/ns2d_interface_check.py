import argparse
from pathlib import Path

import jax.numpy as jnp

from maths.ns2d import F_h_flat, apply_K_flat, apply_M_flat, weighted_dot_flat
from maths.ns2d_io import load_snapshot_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/ns2d/training.npz"))
    args = parser.parse_args()

    t, S, model = load_snapshot_matrix(args.data, as_jax=True)
    a = S[:, -1]
    v = a / jnp.sqrt(weighted_dot_flat(a, a, 0.0, model))
    rhs = F_h_flat(a, model)
    f_mass = apply_M_flat(rhs, model)
    eta2 = jnp.maximum(jnp.vdot(rhs, f_mass).real - jnp.vdot(v, f_mass).real ** 2, 0.0)
    print("Snapshots:", S.shape, "time interval:", float(t[0]), float(t[-1]))
    print("Mass norm of sample mode:", float(jnp.sqrt(v @ apply_M_flat(v, model))))
    print("Weighted norm (alpha=0.1):", float(jnp.sqrt(weighted_dot_flat(v, v, 0.1, model))))
    print("Dissipation quadratic form:", float(v @ apply_K_flat(v, model)))
    print("F_h flattened size:", rhs.shape, "residual indicator:", float(jnp.sqrt(eta2)))


if __name__ == "__main__":
    main()
