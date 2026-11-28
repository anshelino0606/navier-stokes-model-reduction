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

from maths.ns2d import (
    make_model,
    fom_step_info,
    apply_M_flat,
    apply_K_flat,
    unflatten_velocity,
    flatten_velocity,
)
from maths.ns2d_io import load_snapshot_matrix
from maths.sp_rom import (
    ops_from_ns2d,
    weighted_pod,
    m_orthonormalize,
    build_rom,
    extend_rom,
    reduce_state,
    lift,
    simulate_rom,
)
from maths.ns2d_ae import get_alpha, choose_number_of_modes, project_snapshots

jax.config.update("jax_enable_x64", True)


# =============================================================================
# COARSE FOM
# =============================================================================

@partial(jax.jit, static_argnames=("steps",))

def simulate_coarse_audited(u0, dt, model, steps):
    """Run the coarse FOM with a relaxed convergence audit."""

    COARSE_UPDATE_TOL = 1e-5
    COARSE_RESIDUAL_TOL = 2e-4

    def one_step(u, _):
        u_next, iterations, update, residual = fom_step_info(
            u.reshape((2, model.ny, model.nx)), dt, model
        )
        u_next = u_next.reshape(-1)

        failed = (
            (update > COARSE_UPDATE_TOL)
            | (residual > COARSE_RESIDUAL_TOL)
            | (~jnp.isfinite(residual))
            | (~jnp.all(jnp.isfinite(u_next)))
        )

        metrics = (
            iterations,
            update,
            residual,
            failed.astype(jnp.int32),
        )
        return u_next, (u_next, metrics)

    _, (trajectory, metrics) = jax.lax.scan(
        one_step, u0, xs=None, length=steps
    )

    states = jnp.concatenate((u0[None, :], trajectory), axis=0)

    iterations, updates, residuals, failures = metrics

    audit = (
        jnp.max(iterations),
        jnp.max(updates),
        jnp.max(residuals),
        jnp.sum(failures),
    )

    return states, audit

def run_coarse_fom(u0, dt, model, steps, label):
    """Warm up and time the compiled coarse FOM."""

    warm = simulate_coarse_audited(u0, dt, model, steps)
    jax.block_until_ready(warm)
    del warm

    start = time.perf_counter()

    result = simulate_coarse_audited(u0, dt, model, steps)
    jax.block_until_ready(result)

    runtime = time.perf_counter() - start

    states, audit_values = result

    max_iter, max_update, max_residual, failed_steps = audit_values

    audit = {
        "max_picard_iterations": int(max_iter),
        "max_relative_picard_update": float(max_update),
        "max_relative_midpoint_residual": float(max_residual),
        "failed_steps": int(failed_steps),
    }

    print(f"\n{label} coarse-FOM audit")
    print(f"  Steps:              {steps}")
    print(f"  Timestep:           {dt:g}")
    print(f"  Maximum iterations: {audit['max_picard_iterations']}")
    print(f"  Maximum update:     {audit['max_relative_picard_update']:.4e}")
    print(f"  Maximum residual:   {audit['max_relative_midpoint_residual']:.4e}")
    print(f"  Failed steps:       {audit['failed_steps']}")
    print(f"  Runtime:            {runtime:.3f} s")

    if audit["failed_steps"] > 0:
        raise RuntimeError(
            f"{label}: coarse FOM did not converge at dt_coarse={dt:g}. "
            "Reduce --dt-coarse before using this trajectory."
        )

    return states, runtime, audit


# =============================================================================
# METRICS AND GRID TRANSFER
# =============================================================================

def m_norm(u, model):
    Mu = apply_M_flat(u, model)
    return jnp.sqrt(jnp.maximum(jnp.sum(u * Mu), 0.0))


def relative_error(u, u_ref, model):
    return float(
        m_norm(u - u_ref, model) /
        (m_norm(u_ref, model) + 1e-12)
    )


def projection_error(u, rom, model):
    residual = project_snapshots(rom, jnp.asarray(u)[:, None])[:, 0]
    return float(
        m_norm(residual, model) /
        (m_norm(u, model) + 1e-12)
    )


def diagnostics(U, model):
    """Kinetic energy, forcing injection, and viscous dissipation."""

    energies = []
    injections = []
    dissipations = []

    force = jnp.asarray(model.force).reshape(-1)
    M_force = apply_M_flat(force, model)

    for i in range(U.shape[1]):
        u = U[:, i]

        Mu = apply_M_flat(u, model)
        Ku = apply_K_flat(u, model)

        energies.append(float(0.5 * jnp.dot(u, Mu)))
        injections.append(float(jnp.dot(u, M_force)))
        dissipations.append(float(model.nu * jnp.dot(u, Ku)))

    return (
        np.asarray(energies),
        np.asarray(injections),
        np.asarray(dissipations),
    )


def save_field_data(path, times, U, model):
    fields = jax.vmap(
        lambda u: unflatten_velocity(u, model),
        in_axes=1,
        out_axes=0,
    )(jnp.asarray(U))

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


def restrict_to_coarse(u_fine, fine_model, coarse_model):

    if fine_model.nx % coarse_model.nx or fine_model.ny % coarse_model.ny:
        raise ValueError(
            "Fine grid dimensions must be divisible by coarse dimensions."
        )

    fx = fine_model.nx // coarse_model.nx
    fy = fine_model.ny // coarse_model.ny

    field = unflatten_velocity(u_fine, fine_model)

    coarse = field.reshape(
        2, coarse_model.ny, fy, coarse_model.nx, fx
    ).mean(axis=(2, 4))

    return flatten_velocity(coarse, coarse_model)


def prolong_to_fine(u_coarse, coarse_model, fine_model):
    fx = fine_model.nx // coarse_model.nx
    fy = fine_model.ny // coarse_model.ny

    field = unflatten_velocity(u_coarse, coarse_model)

    fine = jnp.repeat(
        jnp.repeat(field, fy, axis=1),
        fx,
        axis=2,
    )

    return flatten_velocity(fine, fine_model)


def coarse_discrepancy(u_rom, u_coarse, fine_model, coarse_model):
    u_rom_coarse = restrict_to_coarse(
        u_rom, fine_model, coarse_model
    )

    difference = u_coarse - u_rom_coarse

    value = m_norm(difference, coarse_model) / (
        m_norm(u_coarse, coarse_model) + 1e-12
    )

    return float(value), difference


def make_coarse_model(fine_model, factor):
    if factor < 2:
        raise ValueError("Coarse factor must be at least 2.")

    if fine_model.nx % factor or fine_model.ny % factor:
        raise ValueError(
            "Fine grid dimensions must be divisible by coarse factor."
        )

    nx = fine_model.nx // factor
    ny = fine_model.ny // factor

    if nx < 12 or ny < 12:
        raise ValueError(
            f"Coarse grid {nx} x {ny} is too small. "
            "Choose a smaller factor."
        )

    return make_model(
        nx=nx,
        ny=ny,
        lx=fine_model.lx,
        ly=fine_model.ly,
        nu=fine_model.nu,
        forcing_mode=fine_model.forcing_mode,
        amplitude=fine_model.amplitude,
        tol=1e-5,
max_iter=100,
    )


# =============================================================================
# POD AND ENRICHMENT
# =============================================================================

def pod_energy_basis(S, ops, alpha, tolerance):
    r_try = min(S.shape)

    while r_try > 0:
        try:
            V, sigma = weighted_pod(
                S, ops, r=r_try, alpha=alpha
            )
            break
        except ValueError:
            r_try -= 1

    if r_try == 0:
        raise ValueError("No numerically independent POD modes found.")

    rank = int(choose_number_of_modes(sigma, tolerance))
    rank = max(1, min(rank, V.shape[1]))

    return V[:, :rank], sigma[:rank], sigma


def enrich_from_coarse_differences(
    rom,
    difference_snapshots,
    fine_model,
    coarse_model,
    ops,
    alpha,
    tolerance,
    max_rank,
):
    if rom.r >= max_rank:
        return rom, rom.V[:, :0], jnp.array([])

    fine_snapshots = jnp.stack(
        [
            prolong_to_fine(
                difference_snapshots[:, i],
                coarse_model,
                fine_model,
            )
            for i in range(difference_snapshots.shape[1])
        ],
        axis=1,
    )

    residual_snapshots = project_snapshots(rom, fine_snapshots)

    if float(jnp.linalg.norm(residual_snapshots)) < 1e-12:
        return rom, rom.V[:, :0], jnp.array([])

    r_try = min(
        residual_snapshots.shape[0],
        residual_snapshots.shape[1],
        max_rank - rom.r,
    )

    while r_try > 0:
        try:
            W, sigma = weighted_pod(
                residual_snapshots,
                ops,
                r=r_try,
                alpha=alpha,
            )
            break
        except ValueError:
            r_try -= 1

    if r_try == 0:
        return rom, rom.V[:, :0], jnp.array([])

    number = int(choose_number_of_modes(sigma, tolerance))
    number = max(1, min(number, W.shape[1], max_rank - rom.r))

    W = m_orthonormalize(
        W[:, :number],
        ops,
        against=rom.V,
    )

    if W.shape[1] == 0:
        return rom, rom.V[:, :0], jnp.array([])

    rom_new = extend_rom(rom, W, ops)

    return rom_new, W, sigma[:W.shape[1]]


# =============================================================================
# TIME ALIGNMENT AND CALIBRATION
# =============================================================================

def find_reference_start(t_ref, t_train_end):
    i0 = int(np.argmin(np.abs(t_ref - t_train_end)))

    if abs(float(t_ref[i0]) - t_train_end) > 1e-7:
        raise ValueError(
            "Reference trajectory must contain the training endpoint."
        )

    return i0


def make_save_step_map(times, t0, dt):
    save_steps = {}

    for i, t in enumerate(times):
        step = int(round((float(t) - t0) / dt))

        if abs(t0 + step * dt - float(t)) > 1e-7:
            raise ValueError(
                "Reference times must align with dt_rom."
            )

        save_steps.setdefault(step, []).append(i)

    return save_steps


def integer_steps(duration, dt, description):
    value = duration / dt
    steps = int(round(value))

    if steps < 1 or not np.isclose(value, steps, atol=1e-8):
        raise ValueError(
            f"{description} must be an integer multiple of the timestep."
        )

    return steps


def calibrate_coarse_threshold(
    S,
    t_train,
    rom,
    fine_model,
    coarse_model,
    dt_rom,
    dt_coarse,
    threshold_factor,
):
    duration = float(t_train[-1] - t_train[0])

    coarse_steps = integer_steps(
        duration, dt_coarse, "Training duration"
    )

    ratio = integer_steps(
        dt_coarse, dt_rom, "dt_coarse / dt_rom"
    )

    # Integrate the training ROM at dt_rom.
    a0 = reduce_state(S[:, 0], rom)

    warm = simulate_rom(
        a0,
        dt_rom,
        coarse_steps * ratio,
        rom,
        save_every=1,
    )

    jax.block_until_ready(warm)
    del warm

    start = time.perf_counter()

    _, A_train, _ = simulate_rom(
        a0,
        dt_rom,
        coarse_steps * ratio,
        rom,
        save_every=1,
    )

    jax.block_until_ready(A_train)

    rom_runtime = time.perf_counter() - start

    # Integrate coarse FOM using dt_coarse.
    u_coarse0 = restrict_to_coarse(
        S[:, 0],
        fine_model,
        coarse_model,
    )

    coarse_states, coarse_runtime, audit = run_coarse_fom(
        u_coarse0,
        dt_coarse,
        coarse_model,
        coarse_steps,
        "Calibration",
    )

    discrepancies = []
    check_times = []

    for k in range(coarse_steps + 1):
        rom_index = k * ratio

        u_rom = lift(A_train[rom_index], rom)

        discrepancy, _ = coarse_discrepancy(
            u_rom,
            coarse_states[k],
            fine_model,
            coarse_model,
        )

        discrepancies.append(discrepancy)
        check_times.append(float(t_train[0]) + k * dt_coarse)

    discrepancies = np.asarray(discrepancies)

    threshold = max(
        threshold_factor * float(np.max(discrepancies)),
        1e-12,
    )

    return (
        threshold,
        discrepancies,
        np.asarray(check_times),
        rom_runtime,
        coarse_runtime,
        audit,
    )


# =============================================================================
# PLOTS AND CSV OUTPUT
# =============================================================================

def save_results(
    out,
    times,
    adaptive_errors,
    fixed_errors,
    adaptive_projections,
    fixed_projections,
    ranks,
    discrepancy_history,
    threshold,
    train_discrepancies,
    train_check_times,
    U_ref,
    U_fixed,
    U_adaptive,
    model,
):
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].semilogy(
        times,
        np.maximum(fixed_errors, 1e-16),
        label="Fixed POD",
    )

    ax[0].semilogy(
        times,
        np.maximum(adaptive_errors, 1e-16),
        label="Coarse-adaptive ROM",
    )

    ax[0].set_ylabel("Relative velocity error")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].semilogy(
        times,
        np.maximum(fixed_projections, 1e-16),
        label="Fixed POD",
    )

    ax[1].semilogy(
        times,
        np.maximum(adaptive_projections, 1e-16),
        label="Coarse-adaptive ROM",
    )

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

    # Coarse discrepancy history.
    fig, ax = plt.subplots(figsize=(7, 4))

    if len(train_discrepancies):
        ax.plot(
            train_check_times,
            train_discrepancies,
            "--o",
            markersize=3,
            label="Training discrepancy",
        )

    if len(discrepancy_history):
        dh = np.asarray(discrepancy_history).reshape(-1, 2)

        ax.plot(
            dh[:, 0],
            dh[:, 1],
            "-o",
            markersize=3,
            label="Online coarse discrepancy",
        )

    ax.axhline(
        threshold,
        color="black",
        linestyle=":",
        label="Trigger threshold",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Relative coarse-grid discrepancy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "coarse_discrepancy.png", dpi=180)
    plt.close(fig)

    # Energy diagnostics.
    E_ref, I_ref, D_ref = diagnostics(U_ref, model)
    E_fixed, I_fixed, D_fixed = diagnostics(U_fixed, model)
    E_adapt, I_adapt, D_adapt = diagnostics(U_adaptive, model)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    for axis, series, ylabel in zip(
        ax,
        [
            (E_ref, E_fixed, E_adapt),
            (I_ref, I_fixed, I_adapt),
            (D_ref, D_fixed, D_adapt),
        ],
        [
            "Kinetic energy",
            "Forcing injection",
            "Viscous dissipation",
        ],
    ):
        axis.plot(times, series[0], label="Reference")
        axis.plot(times, series[1], "--", label="Fixed POD")
        axis.plot(times, series[2], ":", label="Coarse-adaptive")

        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "energy_diagnostics.png", dpi=180)
    plt.close(fig)

    dt_values = np.diff(times)

    def cumulative_trapezoid(values):
        values = np.asarray(values)

        return np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5 * (values[1:] + values[:-1]) * dt_values
                ),
            ]
        )

    cumulative = {
        "injection_reference": cumulative_trapezoid(I_ref),
        "injection_fixed": cumulative_trapezoid(I_fixed),
        "injection_adaptive": cumulative_trapezoid(I_adapt),
        "dissipation_reference": cumulative_trapezoid(D_ref),
        "dissipation_fixed": cumulative_trapezoid(D_fixed),
        "dissipation_adaptive": cumulative_trapezoid(D_adapt),
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(
        times,
        cumulative["injection_reference"],
        label="Reference",
    )
    ax[0].plot(
        times,
        cumulative["injection_fixed"],
        "--",
        label="Fixed POD",
    )
    ax[0].plot(
        times,
        cumulative["injection_adaptive"],
        ":",
        label="Adaptive",
    )

    ax[0].set_ylabel("Cumulative energy injection")

    ax[1].plot(
        times,
        cumulative["dissipation_reference"],
        label="Reference",
    )
    ax[1].plot(
        times,
        cumulative["dissipation_fixed"],
        "--",
        label="Fixed POD",
    )
    ax[1].plot(
        times,
        cumulative["dissipation_adaptive"],
        ":",
        label="Adaptive",
    )

    ax[1].set_ylabel("Cumulative viscous dissipation")

    for axis in ax:
        axis.set_xlabel("Time")
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "cumulative_energy_budget.png", dpi=180)
    plt.close(fig)

    # Time-history CSV.
    pd.DataFrame(
        {
            "time": times,
            "adaptive_error": adaptive_errors,
            "fixed_error": fixed_errors,
            "adaptive_projection_error": adaptive_projections,
            "fixed_projection_error": fixed_projections,
            "adaptive_rank": ranks,
        }
    ).to_csv(out / "time_history.csv", index=False)

    pd.DataFrame(
        np.asarray(discrepancy_history).reshape(-1, 2),
        columns=["time", "coarse_discrepancy"],
    ).to_csv(
        out / "coarse_discrepancy_history.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "time": train_check_times,
            "training_coarse_discrepancy": train_discrepancies,
        }
    ).to_csv(
        out / "training_coarse_discrepancy.csv",
        index=False,
    )

    pd.DataFrame(
        {
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
        }
    ).to_csv(out / "energy_history.csv", index=False)

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


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(t_train, S, t_ref, U_ref, fine_model, args):
    dt_rom = args.dt
    dt_coarse = args.dt_coarse

    ratio = integer_steps(
        dt_coarse,
        dt_rom,
        "dt_coarse / dt_rom",
    )

    if args.check_every != ratio:
        raise ValueError(
            f"check_every must equal dt_coarse / dt_rom ({ratio})."
        )

    t0 = float(t_train[-1])

    coarse_model = make_coarse_model(
        fine_model,
        args.coarse_factor,
    )

    ops = ops_from_ns2d(fine_model)
    alpha = float(get_alpha(fine_model))

    # Initial POD basis and ROM.
    V, _, _ = pod_energy_basis(
        S,
        ops,
        alpha,
        args.pod_tolerance,
    )

    rom = build_rom(V, ops)
    initial_rank = rom.r

    print("\nINITIAL ROM")
    print("-----------")
    print(f"Fine grid:       {fine_model.nx} x {fine_model.ny}")
    print(f"Coarse grid:     {coarse_model.nx} x {coarse_model.ny}")
    print(f"Initial rank:    {initial_rank}")
    print(f"Training end:    {t0:.4f}")
    print(f"Alpha:           {alpha:.6e}")
    print(f"ROM dt:          {dt_rom:g}")
    print(f"Coarse FOM dt:   {dt_coarse:g}")

    # Calibrate coarse discrepancy threshold.
    print("\nCALIBRATING COARSE INDICATOR")
    print("----------------------------")

    (
        threshold,
        train_discrepancies,
        train_check_times,
        calibration_rom_runtime,
        calibration_coarse_runtime,
        calibration_audit,
    ) = calibrate_coarse_threshold(
        S,
        t_train,
        rom,
        fine_model,
        coarse_model,
        dt_rom,
        dt_coarse,
        args.threshold_factor,
    )

    print(
        "Training discrepancy max: "
        f"{np.max(train_discrepancies):.6e}"
    )
    print(f"Online threshold:          {threshold:.6e}")

    # Prediction interval.
    i0 = find_reference_start(t_ref, t0)

    times = np.asarray(t_ref[i0:], dtype=float)
    U_eval = U_ref[:, i0:]

    total_duration = float(times[-1] - t0)

    total_steps = integer_steps(
        total_duration,
        dt_rom,
        "Prediction duration",
    )

    total_coarse_steps = integer_steps(
        total_duration,
        dt_coarse,
        "Prediction duration / dt_coarse",
    )

    if total_coarse_steps * ratio != total_steps:
        raise ValueError(
            "ROM and coarse-FOM step counts do not align."
        )

    save_steps = make_save_step_map(times, t0, dt_rom)

    u0 = U_eval[:, 0]

    a = reduce_state(u0, rom)

    u_coarse0 = restrict_to_coarse(
        u0,
        fine_model,
        coarse_model,
    )

    initial_discrepancy, _ = coarse_discrepancy(
        lift(a, rom),
        u_coarse0,
        fine_model,
        coarse_model,
    )

    print("\nPREDICTION SETUP")
    print("----------------")
    print(f"Prediction end:         {times[-1]:.4f}")
    print(f"ROM steps:              {total_steps}")
    print(f"Coarse FOM steps:       {total_coarse_steps}")
    print(f"Check interval:         {dt_coarse:g}")
    print(f"Initial discrepancy:    {initial_discrepancy:.6e}")

    # Precompute coarse FOM prediction trajectory.
    coarse_states, coarse_runtime, prediction_audit = run_coarse_fom(
        u_coarse0,
        dt_coarse,
        coarse_model,
        total_coarse_steps,
        "Prediction",
    )

    # -------------------------------------------------------------------------
    # Adaptive ROM
    # -------------------------------------------------------------------------

    states = {0: np.asarray(u0)}
    ranks = {0: rom.r}

    adaptive_projections = {
        0: projection_error(U_eval[:, 0], rom, fine_model)
    }

    discrepancy_history = []
    events = []
    difference_buffer = []

    cooldown_until = 0
    step = 0

    adaptive_rom_start = time.perf_counter()

    while step < total_steps:
        segment_steps = min(
            ratio,
            total_steps - step,
        )

        # Integrate one coarse-time interval with the current ROM.
        _, A_segment, _ = simulate_rom(
            a,
            dt_rom,
            segment_steps,
            rom,
            save_every=1,
        )

        jax.block_until_ready(A_segment)

        # Save requested reference times inside this segment.
        for local_step in range(1, segment_steps + 1):
            global_step = step + local_step

            if global_step in save_steps:
                u_current = lift(
                    A_segment[local_step],
                    rom,
                )

                for idx in save_steps[global_step]:
                    states[idx] = np.asarray(u_current)
                    ranks[idx] = rom.r

                    adaptive_projections[idx] = projection_error(
                        U_eval[:, idx],
                        rom,
                        fine_model,
                    )

        step += segment_steps
        a = A_segment[-1]

        # Coarse FOM state at the same physical time.
        coarse_index = step // ratio
        u_coarse = coarse_states[coarse_index]

        u_rom = lift(a, rom)

        discrepancy, difference = coarse_discrepancy(
            u_rom,
            u_coarse,
            fine_model,
            coarse_model,
        )

        current_time = t0 + step * dt_rom

        discrepancy_history.append(
            (current_time, discrepancy)
        )

        difference_buffer.append(
            np.asarray(difference)
        )

        difference_buffer = difference_buffer[
            -args.difference_window:
        ]

        print(
            f"t={current_time:.3f} | "
            f"discrepancy={discrepancy:.4e} | "
            f"threshold={threshold:.4e} | "
            f"rank={rom.r}"
        )

        # Enrich when the coarse discrepancy exceeds the threshold.
        if (
            discrepancy > threshold
            and step >= cooldown_until
            and rom.r < args.max_rank
        ):
            old_rank = rom.r

            print(
                f"\nCOARSE INDICATOR TRIGGERED "
                f"at t={current_time:.4f}"
            )

            difference_snapshots = jnp.stack(
                [
                    jnp.asarray(x)
                    for x in difference_buffer
                ],
                axis=1,
            )

            rom_new, new_modes, singular_values = (
                enrich_from_coarse_differences(
                    rom,
                    difference_snapshots,
                    fine_model,
                    coarse_model,
                    ops,
                    alpha,
                    args.enrichment_tolerance,
                    args.max_rank,
                )
            )

            added = rom_new.r - old_rank

            if added > 0:
                # Correct the current ROM state with the coarse discrepancy.
                u_rom_coarse = restrict_to_coarse(
                    u_rom,
                    fine_model,
                    coarse_model,
                )

                coarse_correction = (
                    u_coarse - u_rom_coarse
                )

                correction_fine = prolong_to_fine(
                    coarse_correction,
                    coarse_model,
                    fine_model,
                )

                u_corrected = u_rom + correction_fine

                rom = rom_new
                a = reduce_state(u_corrected, rom)

                cooldown_until = (
                    step + args.cooldown_steps
                )

                events.append(
                    {
                        "time": float(current_time),
                        "discrepancy": float(discrepancy),
                        "threshold": float(threshold),
                        "rank_before": int(old_rank),
                        "rank_after": int(rom.r),
                        "modes_added": int(added),
                        "singular_values": np.asarray(
                            singular_values
                        ).tolist(),
                    }
                )

                print(
                    f"Added {added} modes: "
                    f"rank {old_rank} -> {rom.r}"
                )

                difference_buffer = []

                # Save the corrected state at the enrichment time.
                if step in save_steps:
                    corrected_state = lift(a, rom)

                    for idx in save_steps[step]:
                        states[idx] = np.asarray(corrected_state)
                        ranks[idx] = rom.r

                        adaptive_projections[idx] = projection_error(
                            U_eval[:, idx],
                            rom,
                            fine_model,
                        )

            else:
                print("No independent enrichment modes found.")

                cooldown_until = (
                    step + args.cooldown_steps
                )

    jax.block_until_ready(a)

    adaptive_rom_runtime = (
        time.perf_counter() - adaptive_rom_start
    )

    if len(states) != len(times):
        missing = sorted(
            set(range(len(times))) - set(states)
        )

        raise RuntimeError(
            f"Missing adaptive reference frames: {missing[:10]}"
        )

    U_adaptive = jnp.stack(
        [
            jnp.asarray(states[i])
            for i in range(len(times))
        ],
        axis=1,
    )

    rank_history = np.asarray(
        [ranks[i] for i in range(len(times))]
    )

    adaptive_projection_errors = np.asarray(
        [
            adaptive_projections[i]
            for i in range(len(times))
        ]
    )

    # -------------------------------------------------------------------------
    # Fixed-ROM baseline
    # -------------------------------------------------------------------------

    fixed_rom = build_rom(V, ops)

    a_fixed0 = reduce_state(
        u0,
        fixed_rom,
    )

    warm = simulate_rom(
        a_fixed0,
        dt_rom,
        total_steps,
        fixed_rom,
        save_every=1,
    )

    jax.block_until_ready(warm)
    del warm

    fixed_start = time.perf_counter()

    _, A_fixed, _ = simulate_rom(
        a_fixed0,
        dt_rom,
        total_steps,
        fixed_rom,
        save_every=1,
    )

    jax.block_until_ready(A_fixed)

    fixed_runtime = time.perf_counter() - fixed_start

    fixed_indices = [
        int(round((float(t) - t0) / dt_rom))
        for t in times
    ]

    U_fixed_all = jax.vmap(
        lambda coeff: lift(coeff, fixed_rom),
        in_axes=0,
        out_axes=1,
    )(A_fixed)

    U_fixed = U_fixed_all[
        :,
        jnp.asarray(fixed_indices),
    ]

    fixed_projection_errors = np.asarray(
        [
            projection_error(
                U_eval[:, i],
                fixed_rom,
                fine_model,
            )
            for i in range(len(times))
        ]
    )

    # -------------------------------------------------------------------------
    # Errors against fine reference
    # -------------------------------------------------------------------------

    adaptive_errors = np.asarray(
        [
            relative_error(
                U_adaptive[:, i],
                U_eval[:, i],
                fine_model,
            )
            for i in range(len(times))
        ]
    )

    fixed_errors = np.asarray(
        [
            relative_error(
                U_fixed[:, i],
                U_eval[:, i],
                fine_model,
            )
            for i in range(len(times))
        ]
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_field_data(
        args.out / "adaptive_fields.npz",
        times,
        U_adaptive,
        fine_model,
    )

    save_field_data(
        args.out / "fixed_fields.npz",
        times,
        U_fixed,
        fine_model,
    )

    save_field_data(
        args.out / "reference_fields.npz",
        times,
        U_eval,
        fine_model,
    )

    energy_data = save_results(
        out=args.out,
        times=times,
        adaptive_errors=adaptive_errors,
        fixed_errors=fixed_errors,
        adaptive_projections=adaptive_projection_errors,
        fixed_projections=fixed_projection_errors,
        ranks=rank_history,
        discrepancy_history=discrepancy_history,
        threshold=threshold,
        train_discrepancies=train_discrepancies,
        train_check_times=train_check_times,
        U_ref=U_eval,
        U_fixed=U_fixed,
        U_adaptive=U_adaptive,
        model=fine_model,
    )

    with open(
        args.out / "enrichment_events.json",
        "w",
    ) as f:
        json.dump(events, f, indent=2)

    result = {
        "fine_grid": [
            int(fine_model.nx),
            int(fine_model.ny),
        ],
        "coarse_grid": [
            int(coarse_model.nx),
            int(coarse_model.ny),
        ],
        "coarse_factor": int(args.coarse_factor),
        "dt_rom": float(dt_rom),
        "dt_coarse": float(dt_coarse),
        "dt": float(dt_rom),
        "rom_steps": int(total_steps),
        "coarse_steps": int(total_coarse_steps),
        "check_every_steps": int(args.check_every),
        "check_interval_time": float(dt_coarse),
        "initial_rank": int(initial_rank),
        "final_rank": int(rom.r),
        "max_rank": int(args.max_rank),
        "alpha": float(alpha),
        "threshold": float(threshold),
        "training_discrepancy_max": float(
            np.max(train_discrepancies)
        ),
        "initial_prediction_discrepancy": float(
            initial_discrepancy
        ),
        "max_prediction_discrepancy": float(
            np.max(
                np.asarray(discrepancy_history)[:, 1]
            )
        ),
        "max_adaptive_error": float(
            np.max(adaptive_errors)
        ),
        "final_adaptive_error": float(
            adaptive_errors[-1]
        ),
        "max_fixed_error": float(
            np.max(fixed_errors)
        ),
        "final_fixed_error": float(
            fixed_errors[-1]
        ),
        "max_adaptive_projection_error": float(
            np.max(adaptive_projection_errors)
        ),
        "final_adaptive_projection_error": float(
            adaptive_projection_errors[-1]
        ),
        "max_fixed_projection_error": float(
            np.max(fixed_projection_errors)
        ),
        "final_fixed_projection_error": float(
            fixed_projection_errors[-1]
        ),
        "enrichment_events": events,
        "calibration_rom_runtime_seconds": float(
            calibration_rom_runtime
        ),
        "calibration_coarse_runtime_seconds": float(
            calibration_coarse_runtime
        ),
        "coarse_fom_runtime_seconds": float(
            coarse_runtime
        ),
        "adaptive_rom_runtime_seconds": float(
            adaptive_rom_runtime
        ),
        "adaptive_total_runtime_seconds": float(
            coarse_runtime + adaptive_rom_runtime
        ),
        "adaptive_runtime_seconds": float(
            coarse_runtime + adaptive_rom_runtime
        ),
        "fixed_runtime_seconds": float(
            fixed_runtime
        ),
        "calibration_coarse_audit": calibration_audit,
        "prediction_coarse_audit": prediction_audit,
    }

    for key, values in energy_data.items():
        result[f"final_{key}"] = float(values[-1])

    with open(
        args.out / "metrics.json",
        "w",
    ) as f:
        json.dump(result, f, indent=2)

    print("\n============== RESULTS ==============")
    print(f"Initial rank:              {initial_rank}")
    print(f"Final rank:                {rom.r}")
    print(f"Enrichment events:         {len(events)}")
    print(f"Coarse threshold:          {threshold:.6e}")
    print(f"Adaptive final error:      {adaptive_errors[-1]:.6e}")
    print(f"Fixed ROM final error:     {fixed_errors[-1]:.6e}")
    print(
        "Adaptive projection error: "
        f"{adaptive_projection_errors[-1]:.6e}"
    )
    print(
        "Fixed projection error:    "
        f"{fixed_projection_errors[-1]:.6e}"
    )
    print(f"Coarse FOM runtime:        {coarse_runtime:.3f} s")
    print(f"Adaptive ROM runtime:     {adaptive_rom_runtime:.3f} s")
    print(
        "Adaptive total runtime:   "
        f"{coarse_runtime + adaptive_rom_runtime:.3f} s"
    )
    print(f"Fixed ROM runtime:        {fixed_runtime:.3f} s")
    print(f"Results saved to:         {args.out}")

    return result


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "JIT-compiled coarse-FOM-guided adaptive POD-ROM."
        )
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data/raw/ns2d/training.npz",
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/raw/ns2d/reference.npz",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--dt-coarse",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--coarse-factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--check-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--difference-window",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--pod-tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--enrichment-tolerance",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--max-rank",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures/coarse_rom",
    )

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
        parser.error(
            "POD and enrichment tolerances must be positive"
        )

    ratio = args.dt_coarse / args.dt
    ratio_int = int(round(ratio))

    if ratio_int < 1 or not np.isclose(
        ratio,
        ratio_int,
        atol=1e-8,
    ):
        parser.error(
            "--dt-coarse must be an integer multiple of --dt"
        )

    if args.check_every != ratio_int:
        parser.error(
            "--check-every must equal --dt-coarse / --dt "
            f"(expected {ratio_int})"
        )

    if not args.train.exists():
        raise FileNotFoundError(
            f"Training file not found: {args.train}"
        )

    if not args.reference.exists():
        raise FileNotFoundError(
            f"Reference file not found: {args.reference}"
        )

    t_train, S, fine_model = load_snapshot_matrix(
        args.train,
        as_jax=True,
    )

    t_ref, U_ref, _ = load_snapshot_matrix(
        args.reference,
        as_jax=True,
    )

    t_train = np.asarray(t_train)
    t_ref = np.asarray(t_ref)

    expected_dim = (
        2 * fine_model.nx * fine_model.ny
    )

    if S.shape[0] != expected_dim or U_ref.shape[0] != expected_dim:
        raise ValueError(
            "Snapshot dimensions do not match the fine model."
        )

    if len(t_train) != S.shape[1]:
        raise ValueError(
            "Training times and snapshots have different lengths."
        )

    if len(t_ref) != U_ref.shape[1]:
        raise ValueError(
            "Reference times and snapshots have different lengths."
        )

    print("COARSE-FOM-GUIDED ADAPTIVE ROM")
    print("==============================")
    print(f"Training snapshots: {S.shape[1]}")
    print(f"Reference snapshots: {U_ref.shape[1]}")
    print(f"Fine grid: {fine_model.nx} x {fine_model.ny}")
    print(f"ROM dt: {args.dt}")
    print(f"Coarse FOM dt: {args.dt_coarse}")
    print(f"Coarse steps per check: {args.check_every}")
    print(
        f"Check interval: "
        f"{args.check_every * args.dt}"
    )

    run_experiment(
        t_train,
        S,
        t_ref,
        U_ref,
        fine_model,
        args,
    )


if __name__ == "__main__":
    main()