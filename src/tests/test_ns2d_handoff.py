import jax
import jax.numpy as jnp
import numpy as np

from maths.ns2d import (
    F_h, F_h_flat, apply_K, apply_K_flat, apply_M, apply_M_flat,
    conv, conv_flat, flatten_velocity, fom_step, fom_step_flat,
    initial_velocity, make_model, mass_dot, project_flat,
    simulate, simulate_audited, snapshot_matrix, unflatten_velocity,
)

jax.config.update("jax_enable_x64", True)


def test_flat_contract_and_pod_inner_product():
    model = make_model(nx=24, ny=24, forcing_mode=2)
    u = initial_velocity(model, 0.3)
    v = initial_velocity(model, 0.2)
    a = flatten_velocity(u, model)
    b = flatten_velocity(v, model)
    assert a.shape == (2 * model.ny * model.nx,)
    assert jnp.allclose(unflatten_velocity(a, model), u)
    assert jnp.allclose(F_h_flat(a, model), F_h(u, model).reshape(-1))
    assert jnp.allclose(apply_M_flat(a, model), apply_M(u, model).reshape(-1))
    assert jnp.allclose(apply_K_flat(a, model), apply_K(u, model).reshape(-1))
    assert jnp.allclose(conv_flat(a, b, model), conv(u, v, model).reshape(-1))
    assert jnp.allclose(project_flat(a, model), a)
    assert jnp.allclose(fom_step_flat(a, 0.01, model), fom_step(u, 0.01, model).reshape(-1))
    assert np.isclose(float(a @ apply_M_flat(a, model)), float(mass_dot(u, u, model)))

    times, snapshots = simulate(u, 0.01, 3, model)
    S = snapshot_matrix(snapshots, model)
    assert S.shape == (2 * model.ny * model.nx, len(times))
    assert np.allclose(np.asarray(S[:, 0]), np.asarray(a))


def test_continuity_and_audit():
    model = make_model(nx=24, ny=24, forcing_mode=2, amplitude=0.4)
    initial = initial_velocity(model, 0.1)
    times, states, info = simulate_audited(initial, 0.01, 12, model, save_every=3)
    _, baseline = simulate(initial, 0.01, 12, model, save_every=3)
    assert times.shape == (5,)
    assert jnp.max(jnp.abs(states - baseline)) < 1e-11
    assert int(info["failed_steps"]) == 0
    assert float(info["maximum_energy_defect"]) < 1e-10
    assert float(info["maximum_divergence"]) < 1e-10
    assert float(info["maximum_midpoint_residual"]) < 1e-10
    from maths.ns2d import energy
    e0, eT = float(energy(states[0], model)), float(energy(states[-1], model))
    defect = eT - e0 - float(info["cumulative_injection"][-1] - info["cumulative_dissipation"][-1])
    assert abs(defect) < 1e-10


def test_saved_snapshot_contract(tmp_path):
    from maths.ns2d_io import load_snapshot_matrix, load_velocity
    from maths.ns2d import weighted_dot_flat

    model = make_model(nx=24, ny=24, forcing_mode=2)
    u = initial_velocity(model)
    times, U = simulate(u, 0.01, 6, model, save_every=2)
    filename = tmp_path / "test.npz"
    np.savez(filename, t=np.asarray(times), velocity=np.asarray(U),
             nx=model.nx, ny=model.ny, lx=model.lx, ly=model.ly,
             nu=model.nu, forcing_mode=model.forcing_mode,
             forcing_amplitude=model.amplitude)
    t, V, loaded = load_velocity(filename)
    _, S, _ = load_snapshot_matrix(filename)
    assert V.shape == (4, 2, 24, 24)
    assert S.shape == (1152, 4)
    assert np.allclose(S[:, 0], np.asarray(flatten_velocity(u, model)))
    assert np.allclose(t, np.asarray(times))
    a = jnp.asarray(S[:, 0])
    assert jnp.allclose(weighted_dot_flat(a, a, 0.0, loaded), mass_dot(u, u, model))
    assert float(weighted_dot_flat(a, a, 0.1, loaded)) > float(weighted_dot_flat(a, a, 0.0, loaded))
