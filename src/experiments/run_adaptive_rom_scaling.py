from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maths.ns2d import make_model, initial_velocity, fom_step_info, fom_step_flat, apply_M_flat, apply_K_flat, flatten_velocity, unflatten_velocity
from maths.sp_rom import ops_from_ns2d, weighted_pod, m_orthonormalize, build_rom, extend_rom, reduce_state, lift, rom_step, simulate_rom
from maths.ns2d_ae import get_alpha, choose_number_of_modes, project_snapshots, residual_error_indicator

jax.config.update("jax_enable_x64", True)


LX = 4.0 * np.pi
LY = 2.0 * np.pi

NU = 0.02

FORCING_MODE = 4
FORCING_AMPLITUDE = 1.0
INITIAL_AMPLITUDE = 0.1

FOM_TOL = 1e-12
FOM_MAX_ITER = 40


@jax.jit
def _fom_step_flat(u, dt, model):
    u_next, _, _, _ = fom_step_info(u.reshape(2, model.ny, model.nx), dt, model)
    return u_next.reshape(-1)


def make_fom_simulator(model, dt, steps):
    @jax.jit
    def simulate(u0):
        def one_step(u, _):
            u_next = _fom_step_flat(u, dt, model)
            return u_next, u_next

        _, trajectory = jax.lax.scan(one_step, u0, None, length=steps)
        return jnp.concatenate([u0[None, :], trajectory], axis=0)

    return simulate


def generate_fom_trajectory(u0, model, dt, steps, label):
    simulate = make_fom_simulator(model, dt, steps)

    print(f"\nPreparing {label} FOM trajectory...")
    print(f"  Grid:  {model.nx} x {model.ny}")
    print(f"  dt:    {dt}")
    print(f"  Steps: {steps}")
    print(f"  Time:  {steps * dt:.3f} seconds")

    # Compilation and warm-up
    warm = simulate(u0)
    jax.block_until_ready(warm)

    # Timed execution
    start = time.perf_counter()

    trajectory = simulate(u0)
    jax.block_until_ready(trajectory)

    runtime = time.perf_counter() - start
    print(f"  Runtime: {runtime:.6f} s")

    if not bool(jnp.all(jnp.isfinite(trajectory))):
        raise RuntimeError(f"{label} FOM produced non-finite values.")

    return trajectory, runtime


def build_fine_model(nx, ny):
    return make_model(
        nx=nx, ny=ny, lx=LX, ly=LY, nu=NU, forcing_mode=FORCING_MODE,
        amplitude=FORCING_AMPLITUDE, tol=FOM_TOL, max_iter=FOM_MAX_ITER,
    )


def build_initial_condition(model):
    u0_field = initial_velocity(model, amplitude=INITIAL_AMPLITUDE)
    return flatten_velocity(u0_field, model)


def m_norm(u, model):
    Mu = apply_M_flat(u, model)
    return jnp.sqrt(jnp.maximum(jnp.sum(u * Mu), 0.0))


def relative_error(u, u_ref, model):
    return float(m_norm(u - u_ref, model) / (m_norm(u_ref, model) + 1e-12))


def projection_error(u, rom, model):
    return relative_error(lift(reduce_state(u, rom), rom), u, model)


def rho_value(rom, a, model):
    _, rho = residual_error_indicator(rom, a, model)
    return float(rho)


def pod_energy_basis(S, ops, alpha, tolerance):
    r_try = min(S.shape)

    while r_try > 0:
        try:
            V, sigma = weighted_pod(S, ops, r=r_try, alpha=alpha)
            break
        except ValueError:
            r_try -= 1

    if r_try == 0:
        raise ValueError("No numerically independent POD modes found.")

    r = int(choose_number_of_modes(sigma, tolerance))
    r = max(1, min(r, V.shape[1]))

    return V[:, :r], sigma[:r], sigma


def enrich_rom(rom, S, ops, alpha, tolerance):
    S_perp = project_snapshots(rom, S)

    if float(jnp.linalg.norm(S_perp)) <= 1e-14:
        return rom, rom.V[:, :0], jnp.array([])

    r_try = min(S_perp.shape)

    while r_try > 0:
        try:
            W, sigma = weighted_pod(S_perp, ops, r=r_try, alpha=alpha)
            break
        except ValueError:
            r_try -= 1

    if r_try == 0:
        return rom, rom.V[:, :0], jnp.array([])

    p = int(choose_number_of_modes(sigma, tolerance))
    p = max(1, min(p, W.shape[1]))

    W = m_orthonormalize(W[:, :p], ops, against=rom.V)
    rom_new = extend_rom(rom, W, ops)

    return rom_new, W, sigma[:p]


def diagnostics(U, model):
    M_U = jax.vmap(lambda u: apply_M_flat(u, model), in_axes=1, out_axes=1)(U)
    K_U = jax.vmap(lambda u: apply_K_flat(u, model), in_axes=1, out_axes=1)(U)

    force = model.force.reshape(-1)
    M_force = apply_M_flat(force, model)

    energy = 0.5 * jnp.sum(U * M_U, axis=0)
    injection = jnp.sum(U * M_force[:, None], axis=0)
    dissipation = model.nu * jnp.sum(U * K_U, axis=0)

    return tuple(np.asarray(x) for x in (energy, injection, dissipation))


def save_field_data(path, times, U, model):
    velocity = jax.vmap(lambda u: unflatten_velocity(u, model), in_axes=1, out_axes=0)(U)
    energy, injection, dissipation = diagnostics(U, model)
    dt_values = np.diff(times)

    cumulative_injection = np.concatenate([[0.0], np.cumsum(0.5 - (injection[1:] + injection[:-1]) - dt_values)])
    cumulative_dissipation = np.concatenate([[0.0], np.cumsum(0.5 - (dissipation[1:] + dissipation[:-1]) - dt_values)])

    # np.savez_compressed(
    #     path,
    #     t=times,
    #     velocity=np.asarray(velocity),
    #     nx=model.nx,
    #     ny=model.ny,
    #     lx=model.lx,
    #     ly=model.ly,
    #     energy=energy,
    #     injection=injection,
    #     dissipation=dissipation,
    #     cumulative_injection=cumulative_injection,
    #     cumulative_dissipation=cumulative_dissipation,
    # )

def save_plots(
    out, times, U_ref, U_fixed, U_adapt, errors, projections,
    fixed_rho, adapt_rho, ranks, model,
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].semilogy(times, errors["fixed"], label="Fixed POD")
    ax[0].semilogy(times, errors["adaptive"], label="Adaptive")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Relative velocity error")
    ax[0].legend()

    ax[1].semilogy(times, projections["fixed"], label="Fixed POD")
    ax[1].semilogy(times, projections["adaptive"], label="Adaptive")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Projection error")
    ax[1].legend()

    ax[2].plot(times, ranks)
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Adaptive rank")

    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    if len(fixed_rho):
        ax.plot(*np.asarray(fixed_rho).T, label="Fixed POD")

    if len(adapt_rho):
        ax.plot(*np.asarray(adapt_rho).T, label="Adaptive")

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Residual indicator $\rho$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "residual_indicators.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(times, ranks)
    ax.set_xlabel("Time")
    ax.set_ylabel("Adaptive rank")

    fig.tight_layout()
    fig.savefig(out / "rank_evolution.png", dpi=180)
    plt.close(fig)

    E_ref, I_ref, D_ref = diagnostics(U_ref, model)
    E_fixed, I_fixed, D_fixed = diagnostics(U_fixed, model)
    E_adapt, I_adapt, D_adapt = diagnostics(U_adapt, model)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(times, E_ref, label="Reference")
    ax[0].plot(times, E_fixed, "--", label="Fixed POD")
    ax[0].plot(times, E_adapt, ":", label="Adaptive")
    ax[0].set_ylabel("Kinetic energy")

    ax[1].plot(times, I_ref, label="Reference")
    ax[1].plot(times, I_adapt, label="Adaptive")
    ax[1].set_ylabel("Injected power")

    ax[2].plot(times, D_ref, label="Reference")
    ax[2].plot(times, D_adapt, label="Adaptive")
    ax[2].set_ylabel("Dissipation")

    for a in ax:
        a.set_xlabel("Time")
        a.legend()

    fig.tight_layout()
    fig.savefig(out / "energy_diagnostics.png", dpi=180)
    plt.close(fig)


def run_adaptive(t_train, S, t_ref, U_ref, model, args):
    ops = ops_from_ns2d(model)

    dt = args.dt
    T0 = float(t_train[-1])
    alpha = float(get_alpha(model))

    V, sigma_initial, sigma_full = pod_energy_basis(S, ops, alpha, args.pod_tolerance)
    initial_rank = V.shape[1]
    rom = build_rom(V, ops)

    print(f"\nUsing alpha = {alpha:.8e}")
    print(f"Initial POD rank = {initial_rank}")

    train_steps = int(round((T0 - float(t_train[0])) / dt))
    a_train = reduce_state(S[:, 0], rom)
    _, A_train, _ = simulate_rom(a_train, dt, train_steps, rom, save_every=1)

    rho_train = [rho_value(rom, a, model) for a in A_train]
    tau = args.tau if args.tau is not None else args.threshold_factor * max(rho_train)

    print(f"Residual threshold tau = {tau:.8e}")

    i0 = int(np.argmin(np.abs(t_ref - T0)))

    if abs(float(t_ref[i0]) - T0) > 1e-7:
        raise ValueError("The reference trajectory must contain a frame at T0.")

    times = np.asarray(t_ref[i0:], dtype=float)
    U_eval = U_ref[:, i0:]
    total_steps = int(round((times[-1] - T0) / dt))

    save_steps = {}

    for i, t in enumerate(times):
        step_i = int(round((float(t) - T0) / dt))
        save_steps.setdefault(step_i, []).append(i)

    states = {}
    ranks = {}
    adaptive_projections = {}

    adapt_rho = []
    events = []

    u0 = U_eval[:, 0]
    a = reduce_state(u0, rom)

    states[0] = np.asarray(u0)
    ranks[0] = rom.r
    adaptive_projections[0] = projection_error(u0, rom, model)


    start = time.perf_counter()

    step = 0
    cooldown_until = 0

    while step < total_steps:
        a = rom_step(a, dt, rom)
        step += 1
        u = lift(a, rom)

        if step in save_steps:
            for idx in save_steps[step]:
                states[idx] = np.asarray(u)
                ranks[idx] = rom.r
                adaptive_projections[idx] = projection_error(U_eval[:, idx], rom, model)

        if step % args.check_every != 0:
            continue

        rho = rho_value(rom, a, model)
        adapt_rho.append((T0 + step * dt, rho))

        if rho <= tau or step < cooldown_until:
            continue

        trigger_time = T0 + step * dt
        window_states = [u]
        u_fom = u
        window_end = min(step + args.window_steps, total_steps)

        # FOM enrichment window
        while step < window_end:
            u_fom = fom_step_flat(u_fom, dt, model)
            step += 1
            window_states.append(u_fom)

            if step in save_steps:
                for idx in save_steps[step]:
                    states[idx] = np.asarray(u_fom)
                    ranks[idx] = rom.r
                    adaptive_projections[idx] = projection_error(U_eval[:, idx], rom, model)

        S_window = jnp.stack(window_states, axis=1)
        old_rank = rom.r

        rom, W_new, sigma_new = enrich_rom(rom, S_window, ops, alpha, args.pod_tolerance)
        a = reduce_state(u_fom, rom)
        cooldown_until = step + args.cooldown_steps

        events.append({
            "trigger_time": trigger_time,
            "window_end_time": T0 + step * dt,
            "rank_before": old_rank,
            "rank_after": rom.r,
            "modes_added": int(rom.r - old_rank),
            "singular_values": np.asarray(sigma_new).tolist(),
        })

        if step in save_steps:
            for idx in save_steps[step]:
                states[idx] = np.asarray(u_fom)
                ranks[idx] = rom.r
                adaptive_projections[idx] = projection_error(U_eval[:, idx], rom, model)

    adaptive_runtime = time.perf_counter() - start

    if len(states) != len(times):
        missing = sorted(set(range(len(times))) - set(states))
        raise RuntimeError(f"Missing saved reference frames: {missing[:10]}")

    U_adapt = jnp.stack([jnp.asarray(states[i]) for i in range(len(times))], axis=1)
    rank_history = np.asarray([ranks[i] for i in range(len(times))])

    fixed_V, _, _ = pod_energy_basis(S, ops, alpha, args.pod_tolerance)
    fixed_V = fixed_V[:, :initial_rank]
    fixed_rom = build_rom(fixed_V, ops)
    a_fixed0 = reduce_state(u0, fixed_rom)

    fixed_start = time.perf_counter()
    _, A_fixed, _ = simulate_rom(a_fixed0, dt, total_steps, fixed_rom, save_every=1)
    jax.block_until_ready(A_fixed)
    fixed_runtime = time.perf_counter() - fixed_start

    U_fixed_all = jax.vmap(lambda coeff: lift(coeff, fixed_rom), in_axes=0, out_axes=1)(A_fixed)
    sample_steps = [int(round((float(t) - T0) / dt)) for t in times]
    U_fixed = U_fixed_all[:, jnp.asarray(sample_steps)]

    errors = {
        "fixed": [relative_error(U_fixed[:, i], U_eval[:, i], model) for i in range(len(times))],
        "adaptive": [relative_error(U_adapt[:, i], U_eval[:, i], model) for i in range(len(times))],
    }

    projections = {
        "fixed": [projection_error(U_eval[:, i], fixed_rom, model) for i in range(len(times))],
        "adaptive": [adaptive_projections[i] for i in range(len(times))],
    }

    fixed_rho = [(T0 + i * dt, rho_value(fixed_rom, A_fixed[i], model)) for i in range(0, len(A_fixed), args.check_every)]


    args.out.mkdir(parents=True, exist_ok=True)

    save_field_data(args.out / "adaptive_fields.npz", times, U_adapt, model)
    save_plots(
        args.out, times, U_eval, U_fixed, U_adapt, errors, projections,
        fixed_rho, adapt_rho, rank_history, model,
    )

    pd.DataFrame({
        "time": times,
        "fixed_error": errors["fixed"],
        "adaptive_error": errors["adaptive"],
        "fixed_projection_error": projections["fixed"],
        "adaptive_projection_error": projections["adaptive"],
        "adaptive_rank": rank_history,
    }).to_csv(args.out / "time_history.csv", index=False)

    result = {
        "grid_nx": int(model.nx),
        "grid_ny": int(model.ny),
        "velocity_dofs": int(2 * model.nx * model.ny),
        "initial_rank": int(initial_rank),
        "final_rank": int(rom.r),
        "alpha": float(alpha),
        "pod_energy_tail_tolerance": float(args.pod_tolerance),
        "threshold": float(tau),
        "max_fixed_error": float(max(errors["fixed"])),
        "final_fixed_error": float(errors["fixed"][-1]),
        "max_adaptive_error": float(max(errors["adaptive"])),
        "final_adaptive_error": float(errors["adaptive"][-1]),
        "adaptive_runtime_seconds": float(adaptive_runtime),
        "fixed_runtime_seconds": float(fixed_runtime),
        "initial_singular_values": np.asarray(sigma_full).tolist(),
        "initial_selected_singular_values": np.asarray(sigma_initial).tolist(),
        "enrichment_events": events,
    }

    with open(args.out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nADAPTIVE ROM RESULTS")
    print("--------------------")
    print(f"Initial rank:       {initial_rank}")
    print(f"Final rank:         {rom.r}")
    print(f"Enrichments:        {len(events)}")
    print(f"Fixed final error:  {errors['fixed'][-1]:.6e}")
    print(f"Adaptive final error: {errors['adaptive'][-1]:.6e}")
    print(f"Adaptive runtime:   {adaptive_runtime:.3f} s")
    print(f"Fixed runtime:      {fixed_runtime:.3f} s")
    print(f"\nResults saved in: {args.out}")

    return result


def run_benchmark(args):
    dt = args.dt

    training_duration = args.training_duration
    prediction_duration = args.prediction_duration

    training_steps = int(round(training_duration / dt))
    prediction_steps = int(round(prediction_duration / dt))
    training_save_every = args.training_save_every

    if not np.isclose(training_steps * dt, training_duration):
        raise ValueError("Training duration must be an integer multiple of dt.")

    if not np.isclose(prediction_steps * dt, prediction_duration):
        raise ValueError("Prediction duration must be an integer multiple of dt.")

    if training_steps % training_save_every != 0:
        raise ValueError("Training steps must be divisible by training_save_every.")

    if args.check_every < 1:
        raise ValueError("check_every must be positive.")

    if args.window_steps < 1:
        raise ValueError("window_steps must be positive.")

    if args.cooldown_steps < 0:
        raise ValueError("cooldown_steps cannot be negative.")

    # Build fine-grid model
    fine_model = build_fine_model(args.nx, args.ny)

    print("\n================================================")
    print("STANDALONE ADAPTIVE POD-ROM GRID-SCALING BENCHMARK")
    print("================================================")

    print("\nPHYSICAL AND GRID CONFIGURATION")
    print("--------------------------------")
    print(f"Fine grid:             {args.nx} x {args.ny}")
    print(f"Fine velocity DOFs:    {2 * args.nx * args.ny}")
    print(f"Domain:                {LX} x {LY}")
    print(f"Viscosity:             {NU}")
    print(f"Training duration:     {training_duration:.3f} s")
    print(f"Prediction duration:   {prediction_duration:.3f} s")
    print(f"Fine timestep:         {dt}")
    print(f"Training FOM steps:    {training_steps}")
    print(f"Prediction FOM steps:  {prediction_steps}")
    print(f"Training save every:   {training_save_every}")
    print(f"Adaptive check every:  {args.check_every}")
    print(f"Enrichment window:     {args.window_steps}")


    u0 = build_initial_condition(fine_model)
    jax.block_until_ready(u0)


    print("\n================================================")
    print("1. FINE FOM TRAINING TRAJECTORY")
    print("================================================")

    training_trajectory, training_fom_runtime = generate_fom_trajectory(u0=u0, model=fine_model, dt=dt, steps=training_steps, label="Training")

    training_indices = np.arange(0, training_steps + 1, training_save_every)
    S = training_trajectory[jnp.asarray(training_indices)].T
    t_train = training_indices.astype(float) * dt

    print("\nTraining snapshot information")
    print("-----------------------------")
    print(f"Training duration:    {t_train[-1]:.3f} s")
    print(f"Training snapshots:   {S.shape[1]}")
    print(f"Snapshot dimension:   {S.shape[0]}")
    print(f"Snapshot interval:    {training_save_every * dt:.3f} s")
    print(f"Training FOM runtime: {training_fom_runtime:.6f} s")

    # Final training state is the initial state for prediction
    u_train_end = training_trajectory[-1]


    print("\n================================================")
    print("2. FINE FOM PREDICTION REFERENCE")
    print("================================================")

    reference_trajectory, reference_fom_runtime = generate_fom_trajectory(
        u0=u_train_end, model=fine_model, dt=dt, steps=prediction_steps, label="Reference"
    )

    t_ref = training_duration + np.arange(prediction_steps + 1, dtype=float) * dt
    U_ref = reference_trajectory.T

    print("\nReference information")
    print("---------------------")
    print(f"Reference start:      {t_ref[0]:.3f} s")
    print(f"Reference end:        {t_ref[-1]:.3f} s")
    print(f"Reference snapshots:  {U_ref.shape[1]}")
    print(f"Reference FOM runtime: {reference_fom_runtime:.6f} s")


    out = args.out / f"{args.nx}x{args.ny}"

    experiment_args = argparse.Namespace(
        dt=dt, window_steps=args.window_steps, check_every=args.check_every,
        cooldown_steps=args.cooldown_steps, threshold_factor=args.threshold_factor,
        tau=args.tau, pod_tolerance=args.pod_tolerance, out=out,
    )

    print("\n================================================")
    print("3. ADAPTIVE ROM EXPERIMENT")
    print("================================================")

    result = run_adaptive(t_train=t_train, S=S, t_ref=t_ref, U_ref=U_ref, model=fine_model, args=experiment_args)


    result.update({
        "training_duration_seconds": float(training_duration),
        "prediction_duration_seconds": float(prediction_duration),
        "training_fom_steps": int(training_steps),
        "prediction_fom_steps": int(prediction_steps),
        "training_snapshot_count": int(S.shape[1]),
        "training_snapshot_interval_seconds": float(training_save_every * dt),
        "training_fom_runtime_seconds": float(training_fom_runtime),
        "reference_fom_runtime_seconds": float(reference_fom_runtime),
        "reference_runtime_excluded_from_method_comparison": True,
    })

    out.mkdir(parents=True, exist_ok=True)

    with open(out / "scaling_metrics.json", "w") as f:
        json.dump(result, f, indent=2)


    print("\n================================================")
    print("GRID-SCALING BENCHMARK SUMMARY")
    print("================================================")

    print(f"\nGrid: {args.nx} x {args.ny}")
    print(f"Fine DOFs: {2 * args.nx * args.ny}")
    print(f"Training interval: {training_duration:.2f} s")
    print(f"Prediction interval: {prediction_duration:.2f} s")

    print("\nFOM GENERATION")
    print("--------------")
    print(f"Training FOM:       {training_fom_runtime:.6f} s")
    print(f"Reference FOM:      {reference_fom_runtime:.6f} s")

    print("\nROM PREDICTION COST")
    print("-------------------")
    print(f"Adaptive ROM:       {result['adaptive_runtime_seconds']:.6f} s")
    print(f"Fixed ROM:          {result['fixed_runtime_seconds']:.6f} s")

    print("\nACCURACY")
    print("--------")
    print(f"Initial rank:       {result['initial_rank']}")
    print(f"Final rank:         {result['final_rank']}")
    print(f"Enrichment events:  {len(result['enrichment_events'])}")
    print(f"Adaptive final error: {result['final_adaptive_error']:.6e}")
    print(f"Fixed final error:    {result['final_fixed_error']:.6e}")

    print("\nSaved results to:")
    print(out)

    return result


def main():
    parser = argparse.ArgumentParser(description="Standalone grid-scaling benchmark for adaptive POD-ROM.")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=72)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--training-duration", type=float, default=2.0)
    parser.add_argument("--prediction-duration", type=float, default=6.0)
    parser.add_argument("--training-save-every", type=int, default=5)
    parser.add_argument("--window-steps", type=int, default=20)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--cooldown-steps", type=int, default=20)
    parser.add_argument("--threshold-factor", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--pod-tolerance", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/adaptive_rom")

    args = parser.parse_args()

    if args.nx < 24 or args.ny < 24:
        parser.error("Grid dimensions must be at least 24.")

    if args.dt <= 0:
        parser.error("dt must be positive.")

    if args.training_duration <= 0:
        parser.error("Training duration must be positive.")

    if args.prediction_duration <= 0:
        parser.error("Prediction duration must be positive.")

    if args.training_save_every < 1:
        parser.error("training-save-every must be positive.")

    if args.check_every < 1:
        parser.error("check-every must be positive.")

    if args.window_steps < 1:
        parser.error("window-steps must be positive.")

    if args.cooldown_steps < 0:
        parser.error("cooldown-steps cannot be negative.")

    if args.pod_tolerance <= 0:
        parser.error("pod-tolerance must be positive.")

    run_benchmark(args)


if __name__ == "__main__":
    main()