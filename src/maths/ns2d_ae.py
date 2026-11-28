
import jax
import os
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
import numpy as np
from maths.ns2d import F_h_flat, apply_M_flat, apply_K_flat
from maths.sp_rom import weighted_pod, m_orthonormalize, extend_rom

jax.config.update("jax_enable_x64", True)

@jax.jit
def m_norm(X, model):
    return jnp.sqrt(model.dx * model.dy * jnp.sum(X**2, axis=0))

@jax.jit
def compute_error_metrics(U, A, rom, model):
    U_rom = rom.V @ A.T
    U_proj = rom.V @ (rom.MV.T @ U)
    ref_norm = m_norm(U, model)
    velocity_error = m_norm(U - U_rom, model) / (ref_norm + 1e-12)
    projection_error = m_norm(U - U_proj, model) / (ref_norm + 1e-12)

    E_ref = 0.5 * jnp.sum(U * apply_M_flat(U, model), axis=0)
    E_rom = 0.5 * jnp.sum(A**2, axis=1)
    K_ref = jax.vmap(lambda u: apply_K_flat(u, model), in_axes=1, out_axes=1)(U)
    K_rom = jax.vmap(lambda u: apply_K_flat(u, model), in_axes=1, out_axes=1)(U_rom)
    D_ref = model.nu * jnp.sum(U * K_ref, axis=0)
    D_rom = model.nu * jnp.sum(U_rom * K_rom, axis=0)
    energy_error = jnp.abs(E_ref - E_rom) / (jnp.abs(E_ref) + 1e-12)
    dissipation_error = jnp.abs(D_ref - D_rom) / (jnp.abs(D_ref) + 1e-12)
    return {
        "velocity_error": velocity_error,
        "projection_error": projection_error,
        "energy_error": energy_error,
        "dissipation_error": dissipation_error,
        "E_ref": E_ref,
        "E_rom": E_rom,
        "D_ref": D_ref,
        "D_rom": D_rom,
    }


@jax.jit
def get_alpha(model, scale=1.0):
    k_f = 2.0 * jnp.pi * model.forcing_mode / model.ly
    return scale / (k_f ** 2)


@jax.jit
def residual_error_indicator(rom, a, model):
    u_r = rom.V @ a
    F = F_h_flat(u_r, model)
    MF = apply_M_flat(F, model)
    P_r_F = rom.MV.T @ F
    eta_sq = F @ MF - P_r_F @ P_r_F
    F_norm_sq = F @ MF
    eta = jnp.sqrt(jnp.maximum(eta_sq, 0.0))
    F_norm = jnp.sqrt(jnp.maximum(F_norm_sq, 0.0))
    rho = eta / (F_norm + 1e-12)
    return eta, rho

@jax.jit
def project_snapshots(rom, S):
    P_S = rom.V @ (rom.MV.T @ S)
    S_perp = S - P_S
    return S_perp


def compute_rolling_zscore(rho_history, window=50, min_samples=10, eps=1e-12):
    if len(rho_history) <= min_samples:
        return 0.0, 0.0, 0.0
    recent = jnp.array(rho_history[-window-1:-1])
    mu = jnp.mean(recent)
    sigma = jnp.std(recent)
    rho_current = rho_history[-1]
    z = (rho_current - mu) / (sigma + eps)
    return z, mu, sigma

@jax.jit
def check_residual_trigger(rho, z, z_threshold, rho_min):
    z_trigger = z > z_threshold
    residual_trigger = rho > rho_min
    return z_trigger & residual_trigger


def should_enrich(rho, z, step, last_enrichment_step, current_rank, max_rank,
                  z_threshold, rho_min, cooldown):
    trigger = check_residual_trigger(rho, z, z_threshold, rho_min)
    cooldown_finished = (step - last_enrichment_step >= cooldown)
    rank_available = current_rank < max_rank
    return trigger & cooldown_finished & rank_available


def choose_number_of_modes(sigma, tolerance=1e-4):
    energy = sigma ** 2
    total_energy = jnp.sum(energy)
    if float(total_energy) <= 1e-14:
        return 0
    relative_tail = (total_energy - jnp.cumsum(energy)) / total_energy
    valid = relative_tail <= tolerance
    if not bool(jnp.any(valid)):
        return len(sigma)
    return int(jnp.argmax(valid)) + 1



def perform_enrichment(rom, S, ops, model, tolerance=1e-4, max_modes=10,max_rank=50, use_energy_pod=False, alpha=None):
    S_perp = project_snapshots(rom, S)

    if alpha is None:
        alpha = get_alpha(model)

    if S_perp.shape[1] == 0 or float(jnp.linalg.norm(S_perp)) <= 1e-14:
        return rom, rom.V[:, :0], jnp.array([])

    remaining = max_rank - rom.r
    r_try = min(remaining, S_perp.shape[0], S_perp.shape[1])

    if r_try <= 0:
        return rom, rom.V[:, :0], jnp.array([])

    pod_alpha = alpha if use_energy_pod else 0.0

    while r_try > 0:
        try:
            W, sigma = weighted_pod(S_perp, ops, r=r_try, alpha=pod_alpha)
            break
        except ValueError:
            r_try -= 1

    if r_try == 0:
        return rom, rom.V[:, :0], jnp.array([])

    k = choose_number_of_modes(sigma, tolerance)
    k = min(k, max_modes, remaining, W.shape[1])

    if k <= 0:
        return rom, rom.V[:, :0], jnp.array([])

    W = m_orthonormalize(W[:, :k], ops, against=rom.V)
    sigma = sigma[:k]
    rom_new = extend_rom(rom, W, ops)

    return rom_new, W, sigma


@jax.jit
def restart_rom(rom, u_fom):
    return rom.MV.T @ u_fom

def record_enrichment_event(history, time, eta, rho, old_rank, new_rank,sigma, alpha, use_energy_pod):
    history.append({
        "time": float(time),
        "eta": float(eta),
        "rho": float(rho),
        "old_rank": int(old_rank),
        "new_rank": int(new_rank),
        "added_modes": int(new_rank - old_rank),
        "sigma": np.asarray(sigma).tolist(),
        "alpha": float(alpha),
        "energy_pod": bool(use_energy_pod),
    })
    return history