import jax
import jax.numpy as jnp
import numpy as np

from maths.ns2d import (
    F_h, apply_K, conv, divergence, energy, energy_terms, fom_step,
    initial_velocity, make_model, mass_dot, midpoint_residual, project, simulate,
)

jax.config.update("jax_enable_x64", True)


def test_projection_and_operator_symmetry():
    model = make_model(nx=32, ny=24, forcing_mode=3)
    key = jax.random.key(17)
    raw = jax.random.normal(key, (2, model.ny, model.nx))
    u = project(raw, model)
    v = project(jax.random.normal(jax.random.key(19), raw.shape), model)
    assert float(jnp.linalg.norm(divergence(u, model))) < 1e-10
    assert float(jnp.linalg.norm(project(u, model) - u)) < 1e-10
    assert np.isclose(float(mass_dot(u, apply_K(v, model), model)),
                      float(mass_dot(apply_K(u, model), v, model)), rtol=1e-11, atol=1e-11)
    assert float(jnp.sum(u * apply_K(u, model))) >= -1e-11


def test_skew_convection_does_no_work():
    model = make_model(nx=32, ny=24, forcing_mode=3)
    u = initial_velocity(model, 0.4)
    v = project(jax.random.normal(jax.random.key(5), u.shape), model)
    assert abs(float(mass_dot(u, conv(v, u, model), model))) < 1e-11
    assert float(jnp.linalg.norm(conv(u, u, model))) > 1e-5
    injection, dissipation, conv_work = energy_terms(u, model)
    rate = mass_dot(u, F_h(u, model), model)
    assert abs(float(rate - (injection - dissipation))) < 1e-10
    assert abs(float(conv_work)) < 1e-11


def test_midpoint_discrete_energy_balance():
    model = make_model(nx=32, ny=24, forcing_mode=3, amplitude=0.15, tol=1e-13)
    u = initial_velocity(model, 0.25)
    dt = 0.01
    for _ in range(6):
        v = fom_step(u, dt, model)
        mid = 0.5 * (u + v)
        injection, dissipation, _ = energy_terms(mid, model)
        defect = (energy(v, model) - energy(u, model)) - dt * (injection - dissipation)
        assert abs(float(defect)) < 1e-10
        assert float(jnp.linalg.norm(divergence(v, model))) < 1e-10
        assert float(jnp.linalg.norm(midpoint_residual(u, v, dt, model))) < 1e-10
        u = v


def test_simulation_snapshots():
    model = make_model(nx=24, ny=18, forcing_mode=2)
    u = initial_velocity(model)
    t, states = simulate(u, 0.01, 10, model, save_every=2)
    assert t.shape == (6,)
    assert states.shape == (6, 2, 18, 24)
    assert np.isclose(float(t[-1]), 0.1)
    assert np.allclose(np.asarray(states[0]), np.asarray(u))


def test_kolmogorov_shear_midpoint_solution():
    model = make_model(nx=24, ny=24, lx=4 * np.pi, ly=2 * np.pi,
                       nu=0.04, forcing_mode=3, amplitude=1.0)
    u0 = jnp.zeros((2, model.ny, model.nx))
    dt = 0.02
    t, U = simulate(u0, dt, 12, model, save_every=1)
    ky = 2 * np.pi * model.forcing_mode / model.ly
    lam = model.nu * ky**2
    q = (1 - 0.5 * dt * lam) / (1 + 0.5 * dt * lam)
    expected_amplitude = (1 - q**12) / lam
    y = np.arange(model.ny) * model.dy
    expected = expected_amplitude * np.sin(ky * y)
    assert np.max(np.abs(np.asarray(U[-1, 0]) - expected[:, None])) < 1e-11
    assert np.max(np.abs(np.asarray(U[-1, 1]))) < 1e-11
