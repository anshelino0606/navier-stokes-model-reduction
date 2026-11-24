import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maths.ns2d_io import load_snapshot_matrix
from maths.sp_rom import (build_rom, ops_from_ns2d, reduce_state, rom_energy,
                          rom_energy_terms, simulate_rom, weighted_pod)


def m_norm(X, h2):
    return np.sqrt(h2 * np.sum(np.asarray(X) ** 2, axis=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=REPO / "data/raw/ns2d/training.npz")
    p.add_argument("--reference", type=Path, default=REPO / "data/raw/ns2d/reference.npz")
    p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=REPO / "figures/sp_rom")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_tr, S, model = load_snapshot_matrix(args.train, as_jax=True)
    t_ref, U, _ = load_snapshot_matrix(args.reference, as_jax=True)
    ops = ops_from_ns2d(model)
    h2 = model.dx * model.dy
    dt = 0.01
    save_every = int(round((t_ref[1] - t_ref[0]) / dt))
    steps = save_every * (len(t_ref) - 1)
    ref_norm = m_norm(U, h2)
    E_ref = 0.5 * ref_norm**2

    V_all, sigma = weighted_pod(S, ops, max(args.ranks), alpha=args.alpha)
    results = {"sigma_over_sigma1": (np.asarray(sigma) / float(sigma[0])).tolist()[:16],
               "t0": float(t_tr[-1]), "runs": {}}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for r in args.ranks:
        V = V_all[:, :r]
        tic = time.perf_counter()
        rom, info = build_rom(V, ops, return_info=True)
        t_build = time.perf_counter() - tic
        a0 = reduce_state(U[:, 0], rom)
        simulate_rom(a0, dt, steps, rom, save_every)[1].block_until_ready()  # compile
        tic = time.perf_counter()
        _, A, worst = simulate_rom(a0, dt, steps, rom, save_every)
        A.block_until_ready()
        t_online = time.perf_counter() - tic

        err = m_norm(U - V @ A.T, h2) / ref_norm
        proj = m_norm(U - V @ (rom.MV.T @ U), h2) / ref_norm
        E_rom = np.asarray(0.5 * jnp.sum(A**2, axis=1))
        results["runs"][r] = dict(
            build_seconds=t_build, online_seconds=t_online, skew_defect=info["skew_defect"],
            max_step_energy_defect=float(worst),
            rel_error_at_T0=float(np.interp(float(t_tr[-1]), t_ref, err)),
            rel_error_at_T=float(err[-1]), max_rel_error=float(err.max()),
            projection_error_at_T=float(proj[-1]),
        )
        ax[0].semilogy(t_ref, err, label=f"r={r}")
        ax[1].semilogy(t_ref, proj, label=f"r={r}")
        ax[2].plot(t_ref, E_rom, label=f"r={r}")
        print(r, json.dumps(results["runs"][r]))
    ax[2].plot(t_ref, E_ref, "k--", label="FOM")
    for a_, title in zip(ax, ["ROM error $\\|u-Va\\|_M/\\|u\\|_M$",
                              "best-approximation error (projection)", "kinetic energy"]):
        a_.axvline(float(t_tr[-1]), color="gray", ls=":")
        a_.set_title(title); a_.set_xlabel("t")
    ax[0].legend(fontsize=8); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / f"baseline_alpha{args.alpha:g}.png", dpi=150)
    (args.out / f"baseline_alpha{args.alpha:g}.json").write_text(json.dumps(results, indent=2))
    print("Saved:", (args.out / f"baseline_alpha{args.alpha:g}.png").resolve())
    print("Saved:", (args.out / f"baseline_alpha{args.alpha:g}.json").resolve())


if __name__ == "__main__":
    main()
