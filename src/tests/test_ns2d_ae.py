
import jax
import jax.numpy as jnp

from maths.ns2d import (
    apply_M_flat, make_model, initial_velocity, flatten_velocity,
)
from maths.ns2d_ae import (
    residual_error_indicator, project_snapshots,
    compute_pod_basis, compute_energy_pod_basis,
    apply_M_matrix, M_orthonormalize, perform_enrichment,
    choose_number_of_modes, check_residual_trigger, get_alpha,
)

jax.config.update("jax_enable_x64", True)


def test_residual_indicator():
    model = make_model(nx=16, ny=16)
    u = initial_velocity(model, 0.2)
    a = flatten_velocity(u, model)

    V_r = a[:, None]
    V_r = V_r / jnp.sqrt(a @ apply_M_flat(a, model))

    eta, rho = residual_error_indicator(V_r, jnp.ones(1), model)

    assert float(eta) >= 0.0
    assert float(rho) >= 0.0
    assert float(rho) <= 1.0 + 1e-10


def test_snapshot_projection():
    model = make_model(nx=16, ny=16)
    u = initial_velocity(model, 0.2)
    a = flatten_velocity(u, model)

    V_r = a[:, None]
    V_r = V_r / jnp.sqrt(a @ apply_M_flat(a, model))

    S = jnp.stack([a, 2.0 * a], axis=1)
    S_perp = project_snapshots(V_r, S, model)

    MS = apply_M_matrix(S_perp, model)

    assert jnp.allclose(V_r.T @ MS, 0.0, atol=1e-10)

def test_alpha_from_forcing_scale():
    model = make_model(nx=16, ny=16)

    alpha = get_alpha(model)
    expected = (model.ly / (2.0 * jnp.pi * model.forcing_mode)) ** 2

    assert jnp.isclose(alpha, expected)
    assert float(alpha) > 0.0


def test_pod_basis_orthonormality():
    model = make_model(nx=16, ny=16)
    key = jax.random.key(10)
    S = jax.random.normal(key, (2 * 16 * 16, 8))

    W, s = compute_pod_basis(S, model)
    MW = apply_M_matrix(W, model)

    assert jnp.allclose(W.T @ MW, jnp.eye(W.shape[1]), atol=1e-8)


def test_energy_pod_orthonormality_after_conversion():
    model = make_model(nx=16, ny=16)
    key = jax.random.key(11)
    S = jax.random.normal(key, (2 * 16 * 16, 8))

    W, s = compute_energy_pod_basis(S, model, get_alpha(model))
    W = M_orthonormalize(W, jnp.zeros((W.shape[0], 0)), model)

    MW = apply_M_matrix(W, model)

    assert jnp.allclose(W.T @ MW, jnp.eye(W.shape[1]), atol=1e-8)


def test_enrichment_increases_rank():
    model = make_model(nx=16, ny=16)
    key = jax.random.key(12)
    S = jax.random.normal(key, (2 * 16 * 16, 12))

    V_all, _ = compute_pod_basis(S, model)
    V_r = V_all[:, :2]

    V_new, W, s = perform_enrichment(
        V_r, S, model,
        max_modes=3,
        max_rank=5,
        use_energy_pod=False,
    )

    assert V_new.shape[1] <= 5
    assert V_new.shape[1] >= V_r.shape[1]
    assert W.shape[1] == V_new.shape[1] - V_r.shape[1]


def test_choose_number_of_modes():
    s = jnp.array([10.0, 1.0, 0.01, 0.0001])
    k = choose_number_of_modes(s, tolerance=1e-2)

    assert 1 <= k <= len(s)


def test_residual_trigger():
    assert bool(check_residual_trigger(0.5, 3.0, 2.0, 0.1))
    assert not bool(check_residual_trigger(0.05, 3.0, 2.0, 0.1))
    assert not bool(check_residual_trigger(0.5, 1.0, 2.0, 0.1))
