import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from maths.ns2d import (
    divergence, energy, energy_terms, initial_velocity, make_model,
    relative_high_mode_energy, simulate_audited,
)


def _steps(time, dt, save_every):
    steps = int(round(time / dt))
    if steps < 1 or not np.isclose(steps * dt, time, rtol=0, atol=1e-11):
        raise ValueError("Time must be a positive integer multiple of dt")
    if steps % save_every:
        raise ValueError("Both training and final times must be multiples of dt * save_every")
    return steps


def _save(path, times, velocity, fields, args, model, train_end_index):
    np.savez_compressed(
        path,
        t=times,
        velocity=velocity,
        **fields,
        nx=model.nx, ny=model.ny, lx=model.lx, ly=model.ly,
        nu=model.nu, forcing_mode=model.forcing_mode,
        forcing_amplitude=model.amplitude,
        initial_amplitude=args.initial_amplitude,
        dt=args.dt, save_every=args.save_every,
        train_end_time=args.train_end, final_time=args.final_time,
        train_end_index=train_end_index,
        mass_weight=model.dx * model.dy,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=72)
    parser.add_argument("--lx", type=float, default=4 * np.pi)
    parser.add_argument("--ly", type=float, default=2 * np.pi)
    parser.add_argument("--nu", type=float, default=0.02)
    parser.add_argument("--forcing-mode", type=int, default=4)
    parser.add_argument("--forcing-amplitude", type=float, default=1.0)
    parser.add_argument("--initial-amplitude", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--train-end", type=float, default=2.0)
    parser.add_argument("--final-time", type=float, default=6.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/ns2d"))
    args = parser.parse_args()

    if args.dt <= 0 or args.save_every < 1 or args.final_time < args.train_end:
        parser.error("Require dt > 0, save_every >= 1, and final_time >= train_end")

    steps = _steps(args.final_time, args.dt, args.save_every)
    train_steps = _steps(args.train_end, args.dt, args.save_every)
    train_idx = train_steps // args.save_every
    model = make_model(
        nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
        nu=args.nu, forcing_mode=args.forcing_mode,
        amplitude=args.forcing_amplitude,
        tol=args.tol, max_iter=args.max_iter,
    )
    u0 = initial_velocity(model, amplitude=args.initial_amplitude)

    start = perf_counter()
    times, states, audit = simulate_audited(u0, args.dt, steps, model, args.save_every)
    states.block_until_ready()
    elapsed = perf_counter() - start

    instantaneous = jax.vmap(lambda u: (
        energy(u, model), *energy_terms(u, model),
        jnp.linalg.norm(divergence(u, model)),
        relative_high_mode_energy(u, model),
    ))(states)

    t = np.asarray(times)
    U = np.asarray(states)
    E, P, D, C, div, high_k = map(np.asarray, instantaneous)
    inj = np.asarray(audit["cumulative_injection"])
    diss = np.asarray(audit["cumulative_dissipation"])
    diagnostics = dict(
        energy=E, injection=P, dissipation=D, convection_work=C,
        divergence=div, high_mode_energy_fraction=high_k,
        cumulative_injection=inj, cumulative_dissipation=diss,
    )

    audit_values = {k: float(np.asarray(v)) for k, v in audit.items() if np.ndim(v) == 0}
    integrated_defect = float(E[-1] - E[0] - inj[-1] + diss[-1])
    audit_values.update(
        integral_energy_defect=integrated_defect,
        max_saved_high_mode_energy_fraction=float(np.max(high_k)),
        max_saved_divergence=float(np.max(div)),
        model_seconds=elapsed,
        steps=steps,
        train_end_index=train_idx,
        num_reference_frames=len(t),
        num_training_frames=train_idx + 1,
    )

    if int(audit_values["failed_steps"]) > 0:
        raise RuntimeError(
            f"{int(audit_values['failed_steps'])} midpoint solves did not converge; "
            "lower dt or raise max_iter"
        )
    if audit_values["maximum_energy_defect"] > 1e-8 or audit_values["maximum_divergence"] > 1e-8:
        raise RuntimeError("Energy balance or incompressibility audit failed")

    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "reference.npz", t, U, diagnostics, args, model, train_idx)
    _save(output / "training.npz", t[:train_idx + 1], U[:train_idx + 1],
          {k: v[:train_idx + 1] for k, v in diagnostics.items()}, args, model, train_idx)

    with (output / "diagnostics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", *diagnostics])
        for i, time in enumerate(t):
            writer.writerow([float(time), *(float(data[i]) for data in diagnostics.values())])

    with (output / "validation.json").open("w") as file:
        json.dump(audit_values, file, indent=2)

    print(f"Saved {len(t)} reference frames and {train_idx + 1} training frames to {output}")
    print(f"Runtime including JIT: {elapsed:.3f}s; final energy: {E[-1]:.6g}")
    print(f"Max divergence: {audit_values['maximum_divergence']:.3e}; "
          f"midpoint defect: {audit_values['maximum_energy_defect']:.3e}")
    print(f"High-mode energy fraction (max over saved frames): {np.max(high_k):.3e}")


if __name__ == "__main__":
    main()
