
import jax
import os
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
import numpy as np
from maths.ns2d import F_h_flat, apply_M_flat, apply_K_flat

jax.config.update("jax_enable_x64", True)


def get_alpha(model, scale=1.0):
    k_f = 2.0 * jnp.pi * model.forcing_mode / model.ly
    return scale / (k_f ** 2)


@jax.jit
def apply_M_matrix(S, model):
    MS = jax.vmap(lambda s: apply_M_flat(s, model), in_axes=1, out_axes=1)(S)
    return MS


@jax.jit
def apply_K_matrix(S, model):
    KS = jax.vmap(lambda s: apply_K_flat(s, model), in_axes=1, out_axes=1)(S)
    return KS


@jax.jit
def residual_error_indicator(V_r, a, model):
    u_r = V_r @ a
    F = F_h_flat(u_r, model)
    MF = apply_M_flat(F, model)
    P_r_F = V_r.T @ MF
    eta_sq = F @ MF - P_r_F @ P_r_F
    F_norm_sq = F @ MF
    eta = jnp.sqrt(jnp.maximum(eta_sq, 0.0))
    F_norm = jnp.sqrt(jnp.maximum(F_norm_sq, 0.0))
    rho = eta / (F_norm + 1e-12)
    return eta, rho


@jax.jit
def project_snapshots(V_r, S, model):
    MS = apply_M_matrix(S, model)
    P_S = V_r @ (V_r.T @ MS)
    S_perp = S - P_S
    return S_perp


@jax.jit
def compute_pod_basis(S_perp, model):
    MS = apply_M_matrix(S_perp, model)
    C = S_perp.T @ MS
    s, Z = jnp.linalg.eigh(C)
    s = jnp.maximum(s[::-1], 0.0)
    Z = Z[:, ::-1]
    W = S_perp @ Z
    W = W / jnp.sqrt(jnp.maximum(s, 1e-30))[None, :]
    return W, s


@jax.jit
def compute_energy_pod_basis(S_perp, model, alpha):
    MS = apply_M_matrix(S_perp, model)
    KS = apply_K_matrix(S_perp, model)
    GS = MS + alpha * KS
    C = S_perp.T @ GS
    s, Z = jnp.linalg.eigh(C)
    s = jnp.maximum(s[::-1], 0.0)
    Z = Z[:, ::-1]
    W = S_perp @ Z
    W = W / jnp.sqrt(jnp.maximum(s, 1e-30))[None, :]
    return W, s


@jax.jit
def enrich_basis(V_r, W):
    V_new = jnp.concatenate([V_r, W], axis=1)
    return V_new


def compute_rolling_zscore(rho_history, window=50, min_samples=10, eps=1e-12):
    if len(rho_history) <= min_samples:
        return 0.0, 0.0, 0.0
    recent = jnp.array(rho_history[-window-1:-1])
    mu = jnp.mean(recent)
    sigma = jnp.std(recent)
    rho_current = rho_history[-1]
    z = (rho_current - mu) / (sigma + eps)
    return z, mu, sigma


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


def choose_number_of_modes(s, tolerance=1e-4):
    total_energy = jnp.sum(s)
    if float(total_energy) <= 1e-14:
        return 0
    relative_tail = (total_energy - jnp.cumsum(s)) / total_energy
    valid = relative_tail <= tolerance
    if not bool(jnp.any(valid)):
        return len(s)
    return int(jnp.argmax(valid)) + 1


@jax.jit
def M_orthonormalize(W, V_r, model):
    MW = apply_M_matrix(W, model)
    W = W - V_r @ (V_r.T @ MW)
    MW = apply_M_matrix(W, model)
    H = W.T @ MW
    L = jnp.linalg.cholesky(H)
    W = W @ jnp.linalg.inv(L.T)
    return W


def perform_enrichment(V_r, S, model, tolerance=1e-4, max_modes=10,max_rank=50, use_energy_pod=False, alpha=None):
    S_perp = project_snapshots(V_r, S, model)

    if alpha is None:
        alpha = get_alpha(model)

    if use_energy_pod:
        W_all, s = compute_energy_pod_basis(S_perp, model, alpha)
    else:
        W_all, s = compute_pod_basis(S_perp, model)

    if len(s) == 0 or float(s[0]) <= 1e-14:
        return V_r, W_all[:, :0], s

    valid_modes = int(jnp.sum(s > 1e-12 * s[0]))
    if valid_modes == 0:
        return V_r, W_all[:, :0], s

    k = choose_number_of_modes(s[:valid_modes], tolerance)
    available_rank = max_rank - V_r.shape[1]
    k = min(k, max_modes, available_rank, valid_modes)

    if k <= 0:
        return V_r, W_all[:, :0], s

    W = W_all[:, :k]

    if use_energy_pod:
        W = M_orthonormalize(W, V_r, model)

    V_new = enrich_basis(V_r, W)
    return V_new, W, s
