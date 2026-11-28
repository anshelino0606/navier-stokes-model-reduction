
from pathlib import Path
import argparse
import json
import time
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maths.ns2d import fom_step_flat, apply_M_flat, apply_K_flat, unflatten_velocity
from maths.ns2d_io import load_snapshot_matrix
from maths.sp_rom import ops_from_ns2d, weighted_pod, m_orthonormalize, build_rom, extend_rom, reduce_state, lift, rom_step, simulate_rom
from maths.ns2d_ae import get_alpha,choose_number_of_modes,project_snapshots,residual_error_indicator

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]


def m_norm(u, model):
    Mu = apply_M_flat(u, model)
    return jnp.sqrt(jnp.maximum(jnp.sum(u * Mu), 0.0))


def relative_error(u, u_ref, model):
    return float( m_norm(u - u_ref, model) /(m_norm(u_ref, model) + 1e-12))


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
            W, sigma = weighted_pod(S_perp,ops,r=r_try,alpha=alpha,)
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
    M_U = jax.vmap(lambda u: apply_M_flat(u, model),in_axes=1,out_axes=1,)(U)

    K_U = jax.vmap(lambda u: apply_K_flat(u, model),in_axes=1,out_axes=1,)(U)

    force = model.force.reshape(-1)
    M_force = apply_M_flat(force, model)

    energy = 0.5 * jnp.sum(U * M_U, axis=0)
    injection = jnp.sum(U * M_force[:, None], axis=0)
    dissipation = model.nu * jnp.sum(U * K_U, axis=0)

    return tuple( np.asarray(x) for x in (energy, injection, dissipation))


def save_field_data(path, times, U, model):
    velocity = jax.vmap( lambda u: unflatten_velocity(u, model), in_axes=1, out_axes=0,)(U)

    energy, injection, dissipation = diagnostics(U, model)
    dt = np.diff(times)

    cumulative_injection = np.concatenate([[0.0], np.cumsum( 0.5 * (injection[1:] + injection[:-1]) * dt),])

    cumulative_dissipation = np.concatenate([ [0.0], np.cumsum( 0.5 * (dissipation[1:] + dissipation[:-1]) * dt),])

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


def save_plots(out,times,U_ref,U_fixed,U_adapt,errors,projections,fixed_rho,adapt_rho,ranks,model,):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(times, errors["fixed"], label="Fixed POD")
    ax[0].semilogy(times, errors["adaptive"], label="Adaptive")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Relative velocity error")
    ax[0].legend()
    ax[1].semilogy( times, projections["fixed"], label="Fixed POD",)
    ax[1].semilogy( times, projections["adaptive"], label="Adaptive",)
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
        ax.plot(*np.asarray(fixed_rho).T, label="Fixed POD",)

    if len(adapt_rho):
        ax.plot( *np.asarray(adapt_rho).T, label="Adaptive",)

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

    V, sigma_initial, sigma_full = pod_energy_basis( S, ops, alpha, args.pod_tolerance,)

    initial_rank = V.shape[1]
    rom = build_rom(V, ops)

    print(f"Using alpha = {alpha:.8e}")
    print(f"Initial POD rank = {initial_rank}")

    train_steps = int(round( (T0 - float(t_train[0])) / dt))
    a_train = reduce_state(S[:, 0], rom)

    _, A_train, _ = simulate_rom( a_train, dt, train_steps, rom, save_every=1,)

    rho_train = [ rho_value(rom, a, model) for a in A_train]

    tau = ( args.tau if args.tau is not None else args.threshold_factor * max(rho_train))
    i0 = int(np.argmin(np.abs(t_ref - T0)))
    if abs(float(t_ref[i0]) - T0) > 1e-7:
        raise ValueError( "The reference trajectory must contain a frame at T0.")

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
                adaptive_projections[idx] = projection_error(U_eval[:, idx],rom,model,)

        if step % args.check_every != 0:
            continue

        rho = rho_value(rom, a, model)
        adapt_rho.append((T0 + step * dt, rho))

        if rho <= tau or step < cooldown_until:
            continue

        trigger_time = T0 + step * dt
        window_states = [u]

        u_fom = u
        window_end = min(step + args.window_steps, total_steps,)

        while step < window_end:
            u_fom = fom_step_flat(u_fom, dt, model)
            step += 1
            window_states.append(u_fom)

            if step in save_steps:
                for idx in save_steps[step]:
                    states[idx] = np.asarray(u_fom)
                    ranks[idx] = rom.r
                    adaptive_projections[idx] = projection_error(U_eval[:, idx], rom, model,)

        S_window = jnp.stack(window_states, axis=1)
        old_rank = rom.r
        rom, W_new, sigma_new = enrich_rom( rom, S_window, ops, alpha, args.pod_tolerance,)
        a = reduce_state(u_fom, rom)
        cooldown_until = step + args.cooldown_steps
        events.append({ "trigger_time": trigger_time, "window_end_time": T0 + step * dt, "rank_before": old_rank, "rank_after": rom.r, "modes_added": int(rom.r - old_rank), "singular_values": np.asarray(sigma_new).tolist(),})

        if step in save_steps:
            for idx in save_steps[step]:
                states[idx] = np.asarray(u_fom)
                ranks[idx] = rom.r
                adaptive_projections[idx] = projection_error( U_eval[:, idx], rom, model, )

    adaptive_runtime = time.perf_counter() - start

    if len(states) != len(times):
        missing = sorted(set(range(len(times))) - set(states))
        raise RuntimeError( f"Missing saved reference frames: {missing[:10]}")

    U_adapt = jnp.stack( [jnp.asarray(states[i]) for i in range(len(times))], axis=1,)

    rank_history = np.asarray([ranks[i] for i in range(len(times))])

    # Fixed-rank comparison using the same initial rank and alpha.
    fixed_V, _, _ = pod_energy_basis( S, ops, alpha, args.pod_tolerance,)
    fixed_V = fixed_V[:, :initial_rank]
    fixed_rom = build_rom(fixed_V, ops)
    a_fixed0 = reduce_state(u0, fixed_rom)
    fixed_start = time.perf_counter()
    _, A_fixed, _ = simulate_rom( a_fixed0, dt, total_steps, fixed_rom, save_every=1,)
    fixed_runtime = time.perf_counter() - fixed_start
    U_fixed_all = jax.vmap( lambda coeff: lift(coeff, fixed_rom), in_axes=0, out_axes=1, )(A_fixed)
    sample_steps = [int(round((float(t) - T0) / dt))for t in times]
    U_fixed = U_fixed_all[:, jnp.asarray(sample_steps)]
    errors = {
        "fixed": [ relative_error(U_fixed[:, i], U_eval[:, i], model) for i in range(len(times))],
        "adaptive": [ relative_error(U_adapt[:, i], U_eval[:, i], model) for i in range(len(times)) ],
    }

    projections = {
        "fixed": [ projection_error(U_eval[:, i], fixed_rom, model) for i in range(len(times))],
        "adaptive": [ adaptive_projections[i] for i in range(len(times))],
    }

    fixed_rho = [(T0 + i * dt, rho_value(fixed_rom, A_fixed[i], model), ) for i in range(0, len(A_fixed), args.check_every)]

    args.out.mkdir(parents=True, exist_ok=True)

    save_field_data( args.out / "adaptive_fields.npz", times, U_adapt, model,)

    save_plots( args.out, times, U_eval, U_fixed, U_adapt, errors, projections, fixed_rho, adapt_rho, rank_history, model,)

    pd.DataFrame({
        "time": times,
        "fixed_error": errors["fixed"],
        "adaptive_error": errors["adaptive"],
        "fixed_projection_error": projections["fixed"],
        "adaptive_projection_error": projections["adaptive"],
        "adaptive_rank": rank_history,}).to_csv( args.out / "time_history.csv", index=False,)

    result = {
        "initial_rank": int(initial_rank),
        "final_rank": int(rom.r),
        "alpha": alpha,
        "pod_energy_tail_tolerance": args.pod_tolerance,
        "threshold": tau,
        "max_fixed_error": float(max(errors["fixed"])),
        "final_fixed_error": float(errors["fixed"][-1]),
        "max_adaptive_error": float(max(errors["adaptive"])),
        "final_adaptive_error": float(errors["adaptive"][-1]),
        "adaptive_runtime_seconds": adaptive_runtime,
        "fixed_runtime_seconds": fixed_runtime,
        "initial_singular_values": np.asarray(sigma_full).tolist(),
        "initial_selected_singular_values": np.asarray(sigma_initial).tolist(),
        "enrichment_events": events,
    }

    with open(args.out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Initial rank: {initial_rank}")
    print(f"Final rank:   {rom.r}")
    print(f"Enrichments:  {len(events)}")
    print(f"Fixed error:  {errors['fixed'][-1]:.6e}")
    print(f"Adaptive error: {errors['adaptive'][-1]:.6e}")
    print(f"Adaptive runtime: {adaptive_runtime:.3f} s")
    print(f"Fixed runtime:    {fixed_runtime:.3f} s")
    print(f"\nResults saved in: {args.out}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--train", type=Path, default=ROOT / "data/raw/ns2d/training.npz",)

    parser.add_argument( "--reference", type=Path, default=ROOT / "data/raw/ns2d/reference.npz",)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--window-steps", type=int, default=20)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--cooldown-steps", type=int, default=20)
    parser.add_argument("--threshold-factor", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--pod-tolerance", type=float, default=1e-4)

    parser.add_argument( "--out", type=Path, default=ROOT / "figures/adaptive_rom",)

    args = parser.parse_args()

    t_train, S, model = load_snapshot_matrix( args.train, as_jax=True,)

    t_ref, U_ref, _ = load_snapshot_matrix( args.reference, as_jax=True,)

    t_train = np.asarray(t_train)
    t_ref = np.asarray(t_ref)

    run_adaptive( t_train, S, t_ref, U_ref, model, args,)


if __name__ == "__main__":
    main()