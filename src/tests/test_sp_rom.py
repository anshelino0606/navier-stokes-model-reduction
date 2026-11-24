import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maths import ns2d
from maths.sp_rom import (build_rom, extend_rom, m_orthonormalize, ops_from_ns2d,
                          reduce_state, rom_energy, rom_energy_terms, rom_rhs,
                          simulate_rom, weighted_pod, check_basis)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def setup():
    model = ns2d.make_model(nx=32, ny=24, forcing_mode=3, nu=0.02)
    ops = ops_from_ns2d(model)
    raw = jax.random.normal(jax.random.key(0), (2, model.ny, model.nx))
    smooth = ns2d.filter_modes(raw, model)
    for _ in range(3):
        smooth = smooth + 0.3 * ns2d.laplace(smooth, model) / float(jnp.max(model.k2))
    u0 = ns2d.project(smooth, model)
    u0 = u0 / jnp.sqrt(2 * ns2d.energy(u0, model)) * 2.0
    _, U = ns2d.simulate(u0, 0.01, 60, model, save_every=3)
    S = U.reshape(U.shape[0], -1).T
    return model, ops, S


def random_basis(ops, D, r, seed):
    X = jax.random.normal(jax.random.key(seed), (D, r))
    X = jax.vmap(ops.project, in_axes=1, out_axes=1)(X)
    return m_orthonormalize(X, ops)


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_pod_basis_properties(setup, alpha):
    _, ops, S = setup
    V, sig = weighted_pod(S, ops, 6, alpha=alpha)
    info = check_basis(V, ops)
    assert info["orthonormality"] < 1e-12 and info["projection"] < 1e-12
    assert np.all(np.diff(np.asarray(sig[:6])) <= 0)


def test_rhs_matches_galerkin_projection_of_fom(setup):
    model, ops, S = setup
    V, _ = weighted_pod(S, ops, 8)
    rom = build_rom(V, ops)
    a = reduce_state(S[:, -1], rom) + 0.1 * jax.random.normal(jax.random.key(1), (8,))
    direct = rom.MV.T @ ns2d.F_h_flat(V @ a, model)
    assert np.allclose(np.asarray(rom_rhs(a, rom)), np.asarray(direct), rtol=1e-10, atol=1e-10)


def test_energy_conservation_any_basis(setup):
    _, ops, S = setup
    V = random_basis(ops, S.shape[0], 7, seed=3)
    rom, info = build_rom(V, ops, return_info=True)
    assert info["skew_defect"] < 1e-10           # skew-symmetry holds before symmetrising
    a = jax.random.normal(jax.random.key(4), (7,))
    inj, dis, work = rom_energy_terms(a, rom)
    assert abs(float(work)) < 1e-12 * float(jnp.abs(rom.N).max() * jnp.linalg.norm(a) ** 3)
    assert np.isclose(float(a @ rom_rhs(a, rom)), float(inj - dis), rtol=1e-12, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(np.asarray(rom.A)) > -1e-12)


def test_discrete_energy_balance_midpoint(setup):
    _, ops, S = setup
    V, _ = weighted_pod(S, ops, 6)
    rom = build_rom(V, ops)
    _, coeffs, worst = simulate_rom(reduce_state(S[:, 0], rom), 0.01, 200, rom, save_every=10)
    assert float(worst) < 1e-11 * max(1.0, float(rom_energy(coeffs[-1])))
    assert np.all(np.isfinite(np.asarray(coeffs)))


def test_extend_equals_rebuild(setup):
    _, ops, S = setup
    V, _ = weighted_pod(S, ops, 9)
    full = build_rom(V, ops)
    ext = extend_rom(build_rom(V[:, :6], ops), V[:, 6:], ops)
    for name in ("A", "N", "f"):
        assert np.allclose(np.asarray(getattr(ext, name)), np.asarray(getattr(full, name)),
                           rtol=1e-11, atol=1e-11), name


def test_extend_rejects_non_orthogonal(setup):
    _, ops, S = setup
    V, _ = weighted_pod(S, ops, 5)
    with pytest.raises(ValueError):
        extend_rom(build_rom(V[:, :3], ops), V[:, 2:4], ops)
