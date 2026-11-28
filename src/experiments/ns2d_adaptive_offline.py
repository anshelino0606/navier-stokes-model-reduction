
import argparse
from pathlib import Path

import numpy as np

from maths.ns2d_io import load_snapshot_matrix
from maths.ns2d_ae import (compute_pod_basis,perform_enrichment,get_alpha,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=Path("data/raw/ns2d/training.npz"),)
    parser.add_argument("--initial-rank", type=int, default=10)
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--max-modes", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--initial-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--out",type=Path,default=Path("data/results/ns2d/adaptive.npz"),)

    args = parser.parse_args()

    if args.initial_rank < 1 or args.max_rank < args.initial_rank:
        parser.error("Require max_rank >= initial_rank >= 1")

    if not 0.0 < args.initial_fraction < 1.0:
        parser.error("initial_fraction must be between 0 and 1")

    t, S, model = load_snapshot_matrix(args.data, as_jax=True)

    split_index = int(args.initial_fraction * S.shape[1])

    if split_index <= 0 or split_index >= S.shape[1]:
        parser.error("Invalid snapshot split")

    S_initial = S[:, :split_index]
    S_enrichment = S[:, split_index:]

    t_initial = t[:split_index]
    t_enrichment = t[split_index:]

    if args.initial_rank > S_initial.shape[1]:
        parser.error("Not enough initial snapshots for the requested rank")

    print("Initial snapshots:", S_initial.shape[1])
    print("Enrichment snapshots:", S_enrichment.shape[1])

    alpha = get_alpha(model) if args.alpha is None else args.alpha
    print("Energy-POD alpha:", alpha)

    V_all, initial_s = compute_pod_basis(S_initial, model)
    V_r = V_all[:, :args.initial_rank]

    results = {}

    for method, use_energy_pod in [("mass_pod", False),("energy_pod", True),]:

        V_new, W, s = perform_enrichment(
            V_r,
            S_enrichment,
            model,
            tolerance=args.tolerance,
            max_modes=args.max_modes,
            max_rank=args.max_rank,
            use_energy_pod=use_energy_pod,
            alpha=alpha,
        )

        results[method] = {
            "basis": np.asarray(V_new),
            "modes": np.asarray(W),
            "eigenvalues": np.asarray(s),
        }

        print(f"\n{method}:")
        print("  Initial rank:", V_r.shape[1])
        print("  Added modes:", W.shape[1])
        print("  Final rank:", V_new.shape[1])


    args.out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out,
        t_initial=np.asarray(t_initial),
        t_enrichment=np.asarray(t_enrichment),
        initial_basis=np.asarray(V_r),
        initial_eigenvalues=np.asarray(initial_s),
        mass_basis=results["mass_pod"]["basis"],
        mass_modes=results["mass_pod"]["modes"],
        mass_eigenvalues=results["mass_pod"]["eigenvalues"],
        energy_basis=results["energy_pod"]["basis"],
        energy_modes=results["energy_pod"]["modes"],
        energy_eigenvalues=results["energy_pod"]["eigenvalues"],
        alpha=np.asarray(alpha),
    )

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()