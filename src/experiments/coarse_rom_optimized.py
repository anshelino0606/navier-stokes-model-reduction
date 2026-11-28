from pathlib import Path
from functools import partial
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

from maths.ns2d import make_model, fom_step_info, apply_M_flat, apply_K_flat, unflatten_velocity, flatten_velocity, project
from maths.coarse_transfer_patch import restrict_to_coarse, prolong_to_fine, restrict_basis_to_coarse, correction_projection_rows, prolong_columns_to_fine, least_squares_target, nudge
from maths.ns2d_io import load_snapshot_matrix
from maths.sp_rom_optimized import ops_from_ns2d, weighted_pod, m_orthonormalize, build_rom, pad_rom, extend_padded_rom, reduce_state, lift, simulate_rom_fast
from maths.ns2d_ae import get_alpha, choose_number_of_modes, project_snapshots

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("steps",))
def simulate_coarse(u0, dt, model, steps):
    update_tol = 1e-4
    residual_tol = 2e-4

    def one_step(u, _):
        u_next, iterations, update, residual = fom_step_info(u.reshape((2, model.ny, model.nx)), dt, model)
        u_next = u_next.reshape(-1)

        failed = (
            (update > update_tol)
            | (residual > residual_tol)
            | (~jnp.isfinite(update))
            | (~jnp.isfinite(residual))
            | (~jnp.all(jnp.isfinite(u_next)))
        )

        audit = (iterations, update, residual, failed.astype(jnp.int32))
        return u_next, (u_next, audit)

    _, (trajectory, metrics) = jax.lax.scan(one_step, u0, None, length=steps)

    states = jnp.concatenate([u0[None, :], trajectory], axis=0)
    iterations, updates, residuals, failures = metrics
    audit = (jnp.max(iterations), jnp.max(updates), jnp.max(residuals), jnp.sum(failures))

    return states, audit


def run_coarse_fom(u0, dt, model, steps, label):
    warm = simulate_coarse(u0, dt, model, steps)
    jax.block_until_ready(warm)
    start = time.perf_counter()

    states, audit_values = simulate_coarse(u0, dt, model, steps)
    jax.block_until_ready((states, audit_values))

    runtime = time.perf_counter() - start
    max_iter, max_update, max_residual, failures = audit_values

    audit = {
        "max_picard_iterations": int(max_iter),
        "max_relative_picard_update": float(max_update),
        "max_relative_midpoint_residual": float(max_residual),
        "failed_steps": int(failures),
    }

    print(f"\n{label} coarse-FOM audit")
    print(f"  Steps:              {steps}")
    print(f"  Timestep:           {dt:g}")
    print(f"  Maximum iterations: {audit['max_picard_iterations']}")
    print(f"  Maximum update:     {audit['max_relative_picard_update']:.4e}")
    print(f"  Maximum residual:   {audit['max_relative_midpoint_residual']:.4e}")
    print(f"  Failed steps:       {audit['failed_steps']}")
    print(f"  Runtime:            {runtime:.3f} s")

    if audit["failed_steps"]:
        raise RuntimeError(f"{label}: coarse FOM failed its convergence audit. Reduce --dt-coarse.")

    return states, runtime, audit


@jax.jit
def _m_norm_from_M(u, Mu):
    return jnp.sqrt(jnp.maximum(jnp.sum(u * Mu), 0.0))


def m_norm(u, model):
    return _m_norm_from_M(u, apply_M_flat(u, model))


def coarse_discrepancy_from_coeff(a, V_coarse, u_coarse, coarse_model):
    u_rom_coarse = V_coarse @ a
    difference = u_coarse - u_rom_coarse
    value = m_norm(difference, coarse_model) / (m_norm(u_coarse, coarse_model) + 1e-12)
    return float(value), difference


def coarse_discrepancy(u_rom, u_coarse, fine_model, coarse_model):
    u_rom_coarse = restrict_to_coarse(u_rom, fine_model, coarse_model)
    difference = u_coarse - u_rom_coarse
    value = m_norm(difference, coarse_model) / (m_norm(u_coarse, coarse_model) + 1e-12)
    return float(value), difference


def make_coarse_model(fine_model, factor):
    if factor < 2:
        raise ValueError("Coarse factor must be at least 2.")

    if fine_model.nx % factor or fine_model.ny % factor:
        raise ValueError("Fine grid dimensions must be divisible by coarse factor.")

    nx = fine_model.nx // factor
    ny = fine_model.ny // factor

    if nx < 12 or ny < 12:
        raise ValueError(f"Coarse grid {nx} x {ny} is too small. Choose a smaller factor.")

    return make_model(
        nx=nx, ny=ny, lx=fine_model.lx, ly=fine_model.ly, nu=fine_model.nu,
        forcing_mode=fine_model.forcing_mode, amplitude=fine_model.amplitude, tol=1e-4, max_iter=100,
    )


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

    rank = int(choose_number_of_modes(sigma, tolerance))
    rank = max(1, min(rank, V.shape[1]))

    return V[:, :rank], sigma[:rank], sigma


def enrich_from_coarse_differences(rom, difference_snapshots, fine_model, coarse_model, ops, alpha, tolerance, max_rank):
    """Enrich a fixed-capacity ROM using projected coarse discrepancies."""
    remaining = min(max_rank, rom.capacity) - rom.r

    if remaining <= 0 or difference_snapshots.shape[1] == 0:
        return rom, rom.V[:, :0], jnp.empty((0,), dtype=rom.V.dtype)

    fine_snapshots = prolong_columns_to_fine(difference_snapshots, coarse_model, fine_model)
    residual_snapshots = project_snapshots(rom, fine_snapshots)
    residual_norm = float(jnp.linalg.norm(residual_snapshots))

    if not np.isfinite(residual_norm) or residual_norm < 1e-12:
        return rom, rom.V[:, :0], jnp.empty((0,), dtype=rom.V.dtype)

    r_try = min(residual_snapshots.shape[0], residual_snapshots.shape[1], remaining)

    while r_try > 0:
        try:
            W, sigma = weighted_pod(residual_snapshots, ops, r=r_try, alpha=alpha)
            break
        except ValueError:
            r_try -= 1

    if r_try == 0:
        return rom, rom.V[:, :0], jnp.empty((0,), dtype=rom.V.dtype)

    number = int(choose_number_of_modes(sigma, tolerance))
    number = max(1, min(number, W.shape[1], remaining))

    active_basis = rom.V[:, :rom.r]
    W = m_orthonormalize(W[:, :number], ops, against=active_basis)

    if W.shape[1] == 0:
        return rom, rom.V[:, :0], jnp.empty((0,), dtype=rom.V.dtype)

    rom_new = extend_padded_rom(rom, W, ops)

    if rom_new.capacity != rom.capacity:
        raise RuntimeError("ROM enrichment changed matrix capacity. Expected fixed capacity.")

    return rom_new, W, sigma[:W.shape[1]]


def integer_steps(duration, dt, description):
    value = duration / dt
    steps = int(round(value))

    if steps < 1 or not np.isclose(value, steps, atol=1e-8):
        raise ValueError(f"{description} must be an integer multiple of the timestep.")

    return steps


def find_reference_start(t_ref, t_train_end):
    i0 = int(np.argmin(np.abs(t_ref - t_train_end)))

    if abs(float(t_ref[i0]) - t_train_end) > 1e-7:
        raise ValueError("Reference trajectory must contain the training endpoint.")

    return i0


def make_save_step_map(times, t0, dt):
    save_steps = {}

    for i, t in enumerate(times):
        step = int(round((float(t) - t0) / dt))

        if abs(t0 + step * dt - float(t)) > 1e-7:
            raise ValueError("Reference times must align with dt_rom.")

        save_steps.setdefault(step, []).append(i)

    return save_steps


def calibrate_coarse_threshold(S, t_train, rom, fine_model, coarse_model, dt_rom, dt_coarse, threshold_factor):
    duration = float(t_train[-1] - t_train[0])
    coarse_steps = integer_steps(duration, dt_coarse, "Training duration")
    ratio = integer_steps(dt_coarse, dt_rom, "dt_coarse / dt_rom")

    rom_steps = coarse_steps * ratio
    a0 = reduce_state(S[:, 0], rom)

    warm = simulate_rom_fast(a0, dt_rom, rom_steps, rom, save_every=ratio)
    jax.block_until_ready(warm)

    start = time.perf_counter()
    _, A_train = simulate_rom_fast(a0, dt_rom, rom_steps, rom, save_every=ratio)
    jax.block_until_ready(A_train)
    rom_runtime = time.perf_counter() - start

    u_coarse0 = restrict_to_coarse(S[:, 0], fine_model, coarse_model)
    coarse_states, coarse_runtime, audit = run_coarse_fom(u_coarse0, dt_coarse, coarse_model, coarse_steps, "Calibration")

    V_coarse = restrict_basis_to_coarse(rom.V, fine_model, coarse_model)
    U_rom_coarse = V_coarse @ A_train.T
    differences = coarse_states.T - U_rom_coarse

    M_differences = jax.vmap(lambda u: apply_M_flat(u, coarse_model), in_axes=1, out_axes=1)(differences)
    M_coarse_states = jax.vmap(lambda u: apply_M_flat(u, coarse_model), in_axes=1, out_axes=1)(coarse_states.T)

    difference_norms = jnp.sqrt(jnp.maximum(jnp.sum(differences * M_differences, axis=0), 0.0))
    reference_norms = jnp.sqrt(jnp.maximum(jnp.sum(coarse_states.T * M_coarse_states, axis=0), 0.0))

    discrepancies = np.asarray(difference_norms / (reference_norms + 1e-12))
    threshold = max(threshold_factor * float(np.max(discrepancies)), 1e-12)
    check_times = float(t_train[0]) + dt_coarse * np.arange(coarse_steps + 1)

    return threshold, discrepancies, check_times, rom_runtime, coarse_runtime, audit

def relative_error(u, u_ref, model):
    return float(m_norm(u - u_ref, model) / (m_norm(u_ref, model) + 1e-12))


def projection_error(u, rom, model):
    residual = project_snapshots(rom, jnp.asarray(u)[:, None])[:, 0]
    return float(m_norm(residual, model) / (m_norm(u, model) + 1e-12))


def diagnostics(U, model):
    """Post-processing only: energy, injection and dissipation."""
    force = jnp.asarray(model.force).reshape(-1)
    M_force = apply_M_flat(force, model)

    MU = jax.vmap(lambda u: apply_M_flat(u, model), in_axes=1, out_axes=1)(U)
    KU = jax.vmap(lambda u: apply_K_flat(u, model), in_axes=1, out_axes=1)(U)

    energy = 0.5 * jnp.sum(U * MU, axis=0)
    injection = U.T @ M_force
    dissipation = model.nu * jnp.sum(U * KU, axis=0)

    return np.asarray(energy), np.asarray(injection), np.asarray(dissipation)


def save_field_data(path, times, U, model):
    fields = jax.vmap(lambda u: unflatten_velocity(u, model), in_axes=1, out_axes=0)(U)
    energy, injection, dissipation = diagnostics(U, model)

    # np.savez_compressed(
    #     path,
    #     t=np.asarray(times),
    #     U=np.asarray(U),
    #     velocity_fields=np.asarray(fields),
    #     kinetic_energy=energy,
    #     forcing_injection=injection,
    #     viscous_dissipation=dissipation,
    # )


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(
    out, times, adaptive_errors, fixed_errors, adaptive_projections, fixed_projections,
    ranks, discrepancy_history, threshold, train_discrepancies, train_check_times,
    U_ref, U_fixed, U_adaptive, model,
):
    out.mkdir(parents=True, exist_ok=True)

    E_ref, I_ref, D_ref = diagnostics(U_ref, model)
    E_fixed, I_fixed, D_fixed = diagnostics(U_fixed, model)
    E_adapt, I_adapt, D_adapt = diagnostics(U_adaptive, model)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].semilogy(times, np.maximum(fixed_errors, 1e-16), label="Fixed POD")
    ax[0].semilogy(times, np.maximum(adaptive_errors, 1e-16), label="Coarse-adaptive ROM")
    ax[0].set_ylabel("Relative velocity error")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].semilogy(times, np.maximum(fixed_projections, 1e-16), label="Fixed POD")
    ax[1].semilogy(times, np.maximum(adaptive_projections, 1e-16), label="Coarse-adaptive ROM")
    ax[1].set_ylabel("Projection error")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    ax[2].step(times, ranks, where="post")
    ax[2].set_ylabel("Adaptive ROM rank")
    ax[2].grid(True, alpha=0.3)

    for axis in ax:
        axis.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(train_check_times, train_discrepancies, "--o", markersize=3, label="Training discrepancy")

    if len(discrepancy_history):
        dh = np.asarray(discrepancy_history)
        ax.plot(dh[:, 0], dh[:, 1], "-o", markersize=3, label="Online coarse discrepancy")

    ax.axhline(threshold, color="black", linestyle=":", label="Trigger threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative coarse-grid discrepancy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "coarse_discrepancy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    series = [
        (E_ref, E_fixed, E_adapt, "Kinetic energy"),
        (I_ref, I_fixed, I_adapt, "Forcing injection"),
        (D_ref, D_fixed, D_adapt, "Viscous dissipation"),
    ]

    for axis, (ref, fixed, adaptive, ylabel) in zip(ax, series):
        axis.plot(times, ref, label="Reference")
        axis.plot(times, fixed, "--", label="Fixed POD")
        axis.plot(times, adaptive, ":", label="Coarse-adaptive")
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "energy_diagnostics.png", dpi=180)
    plt.close(fig)

    dt_values = np.diff(times)

    def cumulative_trapezoid(values):
        return np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * dt_values)])

    cumulative = {
        "injection_reference": cumulative_trapezoid(I_ref),
        "injection_fixed": cumulative_trapezoid(I_fixed),
        "injection_adaptive": cumulative_trapezoid(I_adapt),
        "dissipation_reference": cumulative_trapezoid(D_ref),
        "dissipation_fixed": cumulative_trapezoid(D_fixed),
        "dissipation_adaptive": cumulative_trapezoid(D_adapt),
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for key, label in [("injection_reference", "Reference"), ("injection_fixed", "Fixed POD"), ("injection_adaptive", "Adaptive")]:
        ax[0].plot(times, cumulative[key], label=label)

    for key, label in [("dissipation_reference", "Reference"), ("dissipation_fixed", "Fixed POD"), ("dissipation_adaptive", "Adaptive")]:
        ax[1].plot(times, cumulative[key], label=label)

    ax[0].set_ylabel("Cumulative energy injection")
    ax[1].set_ylabel("Cumulative viscous dissipation")

    for axis in ax:
        axis.set_xlabel("Time")
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "cumulative_energy_budget.png", dpi=180)
    plt.close(fig)

    pd.DataFrame({
        "time": times,
        "adaptive_error": adaptive_errors,
        "fixed_error": fixed_errors,
        "adaptive_projection_error": adaptive_projections,
        "fixed_projection_error": fixed_projections,
        "adaptive_rank": ranks,
    }).to_csv(out / "time_history.csv", index=False)

    dh = np.asarray(discrepancy_history).reshape(-1, 2)
    pd.DataFrame(dh, columns=["time", "coarse_discrepancy"]).to_csv(out / "coarse_discrepancy_history.csv", index=False)

    pd.DataFrame({"time": train_check_times, "training_coarse_discrepancy": train_discrepancies}).to_csv(
        out / "training_coarse_discrepancy.csv", index=False
    )

    pd.DataFrame({
        "time": times,
        "reference_energy": E_ref,
        "fixed_energy": E_fixed,
        "adaptive_energy": E_adapt,
        "reference_injection": I_ref,
        "fixed_injection": I_fixed,
        "adaptive_injection": I_adapt,
        "reference_dissipation": D_ref,
        "fixed_dissipation": D_fixed,
        "adaptive_dissipation": D_adapt,
        **cumulative,
    }).to_csv(out / "energy_history.csv", index=False)

    return {
        "reference_energy": E_ref,
        "fixed_energy": E_fixed,
        "adaptive_energy": E_adapt,
        "reference_injection": I_ref,
        "fixed_injection": I_fixed,
        "adaptive_injection": I_adapt,
        "reference_dissipation": D_ref,
        "fixed_dissipation": D_fixed,
        "adaptive_dissipation": D_adapt,
    }

def run_experiment(t_train, S, t_ref, U_ref, fine_model, args):
    dt = args.dt
    dt_coarse = args.dt_coarse
    ratio = integer_steps(dt_coarse, dt, "dt_coarse / dt")

    if args.check_every != ratio:
        raise ValueError(f"check_every must equal dt_coarse / dt ({ratio}).")

    t0 = float(t_train[-1])
    coarse_model = make_coarse_model(fine_model, args.coarse_factor)
    ops = ops_from_ns2d(fine_model)
    alpha = float(get_alpha(fine_model))

    V, _, _ = pod_energy_basis(S, ops, alpha, args.pod_tolerance)
    initial_rom = build_rom(V, ops)

    if args.max_rank < initial_rom.r:
        raise ValueError(f"Maximum rank {args.max_rank} is smaller than initial rank {initial_rom.r}.")

    rom = pad_rom(initial_rom, args.max_rank)

    if rom.capacity != args.max_rank:
        raise RuntimeError("pad_rom did not create the requested capacity.")

    initial_rank = rom.r

    print("\nINITIAL ROM")
    print("-----------")
    print(f"Fine grid:       {fine_model.nx} x {fine_model.ny}")
    print(f"Coarse grid:     {coarse_model.nx} x {coarse_model.ny}")
    print(f"Initial rank:    {initial_rank}")
    print(f"Maximum capacity:{rom.capacity}")
    print(f"Training end:    {t0:.4f}")
    print(f"Alpha:           {alpha:.6e}")
    print(f"ROM dt:          {dt:g}")
    print(f"Coarse FOM dt:   {dt_coarse:g}")
    print(f"Check interval:  {args.check_every} ROM steps")
    print("\nCALIBRATING COARSE INDICATOR")
    print("----------------------------")

    (
        threshold, train_discrepancies, train_check_times,
        calibration_rom_runtime, calibration_coarse_runtime, calibration_audit,
    ) = calibrate_coarse_threshold(S, t_train, rom, fine_model, coarse_model, dt, dt_coarse, args.threshold_factor)

    print(f"Training discrepancy max: {np.max(train_discrepancies):.6e}")
    print(f"Online threshold:          {threshold:.6e}")

    i0 = find_reference_start(t_ref, t0)
    times = np.asarray(t_ref[i0:], dtype=float)
    U_eval = U_ref[:, i0:]

    duration = float(times[-1] - t0)
    total_steps = integer_steps(duration, dt, "Prediction duration")
    total_coarse_steps = integer_steps(duration, dt_coarse, "Prediction duration / dt_coarse")

    if total_coarse_steps * ratio != total_steps:
        raise ValueError("Prediction duration must contain an integer number of coarse intervals.")

    if total_steps % ratio != 0:
        raise ValueError("Prediction steps must be divisible by check_every.")

    save_steps = make_save_step_map(times, t0, dt)

    u0 = U_eval[:, 0]
    a0 = reduce_state(u0, rom)
    a = a0
    u_coarse0 = restrict_to_coarse(u0, fine_model, coarse_model)

    print("\nOFFLINE PRECOMPUTATION")
    print("----------------------")

    precompute_start = time.perf_counter()
    V_coarse = restrict_basis_to_coarse(rom.V, fine_model, coarse_model)
    B_correction = correction_projection_rows(rom.V, fine_model, coarse_model)
    jax.block_until_ready((V_coarse, B_correction))
    precomputation_runtime = time.perf_counter() - precompute_start

    print(f"Initial correction matrix: {B_correction.shape}")
    print(f"Precomputation runtime:    {precomputation_runtime:.6f} s")

    initial_discrepancy, _ = coarse_discrepancy_from_coeff(a, V_coarse, u_coarse0, coarse_model)

    print("\nPREDICTION SETUP")
    print("----------------")
    print(f"Prediction end:         {times[-1]:.4f}")
    print(f"ROM steps:              {total_steps}")
    print(f"Coarse FOM steps:       {total_coarse_steps}")
    print(f"Check interval:         {dt_coarse:g}")
    print(f"Initial discrepancy:    {initial_discrepancy:.6e}")

    coarse_states, coarse_runtime, prediction_audit = run_coarse_fom(u_coarse0, dt_coarse, coarse_model, total_coarse_steps, "Prediction")
    coarse_states = jax.block_until_ready(coarse_states)

    states = {0: lift(a, rom)}
    discrepancy_history = []
    events = []
    difference_buffer = []

    cooldown_until = 0
    step = 0

    print("\nWARMING UP FIXED-SHAPE ROM")
    print("-------------------------")

    warm = simulate_rom_fast(a0, dt, ratio, rom, save_every=1)
    jax.block_until_ready(warm)

    print("ROM block compiled and warmed up.")

    adaptive_start = time.perf_counter()

    while step < total_steps:
        _, A_segment = simulate_rom_fast(a, dt, ratio, rom, save_every=1)
        jax.block_until_ready(A_segment)

        for local_step in range(1, ratio + 1):
            global_step = step + local_step

            if global_step in save_steps:
                state = lift(A_segment[local_step], rom)
                for idx in save_steps[global_step]:
                    states[idx] = state

        step += ratio
        a = A_segment[-1]

        coarse_index = step // ratio
        u_coarse = coarse_states[coarse_index]

        discrepancy, difference = coarse_discrepancy_from_coeff(a, V_coarse, u_coarse, coarse_model)
        current_time = t0 + step * dt

        discrepancy_history.append((current_time, discrepancy))
        difference_buffer.append(difference)

        if len(difference_buffer) > args.difference_window:
            difference_buffer.pop(0)

        print(f"t={current_time:.3f} | discrepancy={discrepancy:.4e} | threshold={threshold:.4e} | rank={rom.r}/{rom.capacity}")

        if discrepancy > threshold and step >= cooldown_until and rom.r < args.max_rank:
            old_rank = rom.r
            print(f"\nCOARSE INDICATOR TRIGGERED at t={current_time:.4f}")

            difference_snapshots = jnp.stack(difference_buffer, axis=1)
            rom_new, new_modes, singular_values = enrich_from_coarse_differences(
                rom, difference_snapshots, fine_model, coarse_model, ops, alpha, args.enrichment_tolerance, args.max_rank
            )

            added = rom_new.r - old_rank

            if added > 0:
                # -------------------------------------------------------------
                # PRESERVED COARSE-FOM STATE CORRECTION
                #
                # Target fine-grid state:
                #
                # u_target = V_r @ a + Pi @ P @ correction_coarse
                #
                # Reduced-coordinate equivalent:
                #
                # a_new = [a; 0] + B_correction @ correction_coarse
                #
                # B_correction = V.T @ M @ Pi @ P
                # -------------------------------------------------------------

                u_rom_coarse = V_coarse @ a
                correction_coarse = u_coarse - u_rom_coarse

                # Compute projected correction rows for newly added modes.
                B_new = correction_projection_rows(rom_new.V[:, old_rank:rom_new.r], fine_model, coarse_model)
                B_correction = B_correction.at[old_rank:rom_new.r, :].set(B_new)

                # Preserve old coefficients and initialize added modes.
                a_new = jnp.zeros_like(a)
                a_new = a_new.at[:old_rank].set(a[:old_rank])

                # Apply the projected coarse-FOM state correction.
                a_new = a_new.at[:rom_new.r].add(B_correction[:rom_new.r, :] @ correction_coarse)

                # Switch to the enriched ROM.
                rom = rom_new
                a = a_new

                if rom.capacity != args.max_rank:
                    raise RuntimeError("ROM capacity changed during enrichment.")

                V_coarse = restrict_basis_to_coarse(rom.V, fine_model, coarse_model)
                cooldown_until = step + args.cooldown_steps

                events.append({
                    "time": float(current_time),
                    "discrepancy": float(discrepancy),
                    "threshold": float(threshold),
                    "rank_before": int(old_rank),
                    "rank_after": int(rom.r),
                    "modes_added": int(added),
                    "singular_values": np.asarray(singular_values).tolist(),
                })

                print(f"Added {added} modes: rank {old_rank} -> {rom.r}; capacity remains {rom.capacity}")
                difference_buffer.clear()

                if step in save_steps:
                    corrected_state = lift(a, rom)
                    for idx in save_steps[step]:
                        states[idx] = corrected_state

            else:
                print("No independent enrichment modes found.")
                cooldown_until = step + args.cooldown_steps

    jax.block_until_ready(a)
    adaptive_runtime = time.perf_counter() - adaptive_start

    if len(states) != len(times):
        missing = sorted(set(range(len(times))) - set(states))
        raise RuntimeError(f"Missing adaptive output frames: {missing[:10]}")

    U_adaptive = jnp.stack([states[i] for i in range(len(times))], axis=1)
    rank_history = np.full(len(times), initial_rank, dtype=int)

    for event in events:
        event_step = int(round((event["time"] - t0) / dt))
        for idx, t in enumerate(times):
            output_step = int(round((float(t) - t0) / dt))
            if output_step >= event_step:
                rank_history[idx] = event["rank_after"]

    fixed_rom = build_rom(V, ops)
    a_fixed0 = reduce_state(u0, fixed_rom)

    warm = simulate_rom_fast(a_fixed0, dt, total_steps, fixed_rom, save_every=1)
    jax.block_until_ready(warm)

    fixed_start = time.perf_counter()
    _, A_fixed = simulate_rom_fast(a_fixed0, dt, total_steps, fixed_rom, save_every=1)
    jax.block_until_ready(A_fixed)
    fixed_runtime = time.perf_counter() - fixed_start

    fixed_indices = np.asarray([int(round((float(t) - t0) / dt)) for t in times])
    A_fixed_saved = A_fixed[jnp.asarray(fixed_indices)]
    U_fixed = jax.vmap(lambda coeff: lift(coeff, fixed_rom), in_axes=0, out_axes=1)(A_fixed_saved)

    print("\nPOST-PROCESSING")
    print("---------------")
    print("Calculating reference errors and projection errors...")

    adaptive_errors = np.asarray([relative_error(U_adaptive[:, i], U_eval[:, i], fine_model) for i in range(len(times))])
    fixed_errors = np.asarray([relative_error(U_fixed[:, i], U_eval[:, i], fine_model) for i in range(len(times))])

    adaptive_projection_errors = np.full(len(times), np.nan)
    fixed_projection_errors = np.asarray([projection_error(U_eval[:, i], fixed_rom, fine_model) for i in range(len(times))])


    args.out.mkdir(parents=True, exist_ok=True)

    save_field_data(args.out / "adaptive_fields.npz", times, U_adaptive, fine_model)
    save_field_data(args.out / "fixed_fields.npz", times, U_fixed, fine_model)
    save_field_data(args.out / "reference_fields.npz", times, U_eval, fine_model)

    energy_data = save_results(
        out=args.out, times=times, adaptive_errors=adaptive_errors, fixed_errors=fixed_errors,
        adaptive_projections=adaptive_projection_errors, fixed_projections=fixed_projection_errors,
        ranks=rank_history, discrepancy_history=discrepancy_history, threshold=threshold,
        train_discrepancies=train_discrepancies, train_check_times=train_check_times,
        U_ref=U_eval, U_fixed=U_fixed, U_adaptive=U_adaptive, model=fine_model,
    )

    with open(args.out / "enrichment_events.json", "w") as f:
        json.dump(events, f, indent=2)

    discrepancy_values = np.asarray(discrepancy_history, dtype=float).reshape(-1, 2)

    result = {
        "fine_grid": [int(fine_model.nx), int(fine_model.ny)],
        "coarse_grid": [int(coarse_model.nx), int(coarse_model.ny)],
        "coarse_factor": int(args.coarse_factor),
        "dt_rom": float(dt),
        "dt_coarse": float(dt_coarse),
        "rom_steps": int(total_steps),
        "coarse_steps": int(total_coarse_steps),
        "check_every_steps": int(args.check_every),
        "initial_rank": int(initial_rank),
        "final_rank": int(rom.r),
        "max_rank": int(args.max_rank),
        "fixed_rom_capacity": int(fixed_rom.capacity),
        "alpha": float(alpha),
        "threshold": float(threshold),
        "training_discrepancy_max": float(np.max(train_discrepancies)),
        "initial_prediction_discrepancy": float(initial_discrepancy),
        "max_prediction_discrepancy": float(np.max(discrepancy_values[:, 1])) if len(discrepancy_values) else float(initial_discrepancy),
        "max_adaptive_error": float(np.max(adaptive_errors)),
        "final_adaptive_error": float(adaptive_errors[-1]),
        "max_fixed_error": float(np.max(fixed_errors)),
        "final_fixed_error": float(fixed_errors[-1]),
        "max_fixed_projection_error": float(np.max(fixed_projection_errors)),
        "final_fixed_projection_error": float(fixed_projection_errors[-1]),
        "enrichment_events": events,
        "calibration_rom_runtime_seconds": float(calibration_rom_runtime),
        "calibration_coarse_runtime_seconds": float(calibration_coarse_runtime),
        "offline_precomputation_runtime_seconds": float(precomputation_runtime),
        "coarse_fom_runtime_seconds": float(coarse_runtime),
        "adaptive_rom_runtime_seconds": float(adaptive_runtime),
        "adaptive_total_runtime_seconds": float(coarse_runtime + adaptive_runtime),
        "fixed_runtime_seconds": float(fixed_runtime),
        "adaptive_total_with_precomputation_seconds": float(precomputation_runtime + coarse_runtime + adaptive_runtime),
        "calibration_coarse_audit": calibration_audit,
        "prediction_coarse_audit": prediction_audit,
    }

    for key, values in energy_data.items():
        result[f"final_{key}"] = float(values[-1])

    with open(args.out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n============== RUNTIME RESULTS ==============")
    print(f"Offline precomputation:          {precomputation_runtime:.6f} s")
    print(f"Coarse FOM prediction:           {coarse_runtime:.6f} s")
    print(f"Adaptive ROM prediction:         {adaptive_runtime:.6f} s")
    print(f"Adaptive total (without setup):  {coarse_runtime + adaptive_runtime:.6f} s")
    print(f"Adaptive total (with setup):     {precomputation_runtime + coarse_runtime + adaptive_runtime:.6f} s")
    print(f"Fixed ROM prediction:            {fixed_runtime:.6f} s")

    print("\n============== ACCURACY RESULTS ==============")
    print(f"Initial rank:              {initial_rank}")
    print(f"Final rank:                {rom.r}")
    print(f"Fixed matrix capacity:     {rom.capacity}")
    print(f"Enrichment events:         {len(events)}")
    print(f"Coarse threshold:          {threshold:.6e}")
    print(f"Adaptive final error:      {adaptive_errors[-1]:.6e}")
    print(f"Fixed ROM final error:     {fixed_errors[-1]:.6e}")
    print(f"Results saved to:          {args.out}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Optimized coarse-FOM-guided adaptive POD-ROM.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/ns2d/training.npz")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/raw/ns2d/reference.npz")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dt-coarse", type=float, default=0.5)
    parser.add_argument("--coarse-factor", type=int, default=2)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--threshold-factor", type=float, default=1.5)
    parser.add_argument("--difference-window", type=int, default=10)
    parser.add_argument("--cooldown-steps", type=int, default=50)
    parser.add_argument("--pod-tolerance", type=float, default=1e-4)
    parser.add_argument("--enrichment-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-rank", type=int, default=25)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/coarse_rom_optimized")

    args = parser.parse_args()

    if args.dt <= 0:
        parser.error("--dt must be positive")
    if args.dt_coarse <= 0:
        parser.error("--dt-coarse must be positive")
    if args.coarse_factor < 2:
        parser.error("--coarse-factor must be at least 2")
    if args.check_every < 1:
        parser.error("--check-every must be at least 1")
    if args.difference_window < 1:
        parser.error("--difference-window must be at least 1")
    if args.cooldown_steps < 0:
        parser.error("--cooldown-steps cannot be negative")
    if args.threshold_factor <= 0:
        parser.error("--threshold-factor must be positive")
    if args.max_rank < 1:
        parser.error("--max-rank must be positive")
    if args.pod_tolerance <= 0 or args.enrichment_tolerance <= 0:
        parser.error("POD tolerances must be positive")

    ratio = args.dt_coarse / args.dt
    ratio_int = int(round(ratio))

    if ratio_int < 1 or not np.isclose(ratio, ratio_int, atol=1e-8):
        parser.error("--dt-coarse must be an integer multiple of --dt")

    if args.check_every != ratio_int:
        parser.error(f"--check-every must equal --dt-coarse / --dt (expected {ratio_int})")

    if not args.train.exists():
        raise FileNotFoundError(f"Training file not found: {args.train}")

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")

    t_train, S, fine_model = load_snapshot_matrix(args.train, as_jax=True)
    t_ref, U_ref, _ = load_snapshot_matrix(args.reference, as_jax=True)

    t_train = np.asarray(t_train)
    t_ref = np.asarray(t_ref)

    expected_dim = 2 * fine_model.nx * fine_model.ny

    if S.shape[0] != expected_dim or U_ref.shape[0] != expected_dim:
        raise ValueError("Snapshot dimensions do not match the fine model.")

    if len(t_train) != S.shape[1]:
        raise ValueError("Training times and snapshots have different lengths.")

    if len(t_ref) != U_ref.shape[1]:
        raise ValueError("Reference times and snapshots have different lengths.")

    print("OPTIMIZED COARSE-FOM-GUIDED ADAPTIVE ROM")
    print("========================================")
    print(f"Training snapshots: {S.shape[1]}")
    print(f"Reference snapshots: {U_ref.shape[1]}")
    print(f"Fine grid: {fine_model.nx} x {fine_model.ny}")
    print(f"ROM dt: {args.dt}")
    print(f"Coarse FOM dt: {args.dt_coarse}")
    print(f"Coarse steps per check: {args.check_every}")
    print(f"Maximum rank: {args.max_rank}")

    run_experiment(t_train, S, t_ref, U_ref, fine_model, args)


if __name__ == "__main__":
    main()