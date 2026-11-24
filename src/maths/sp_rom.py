from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Interface to the full model
# ---------------------------------------------------------------------------
class FOMOps(NamedTuple):
    """Flat-vector operations of the full model, all acting on length-D vectors."""
    conv: Callable[[Array, Array], Array]   # C(w, u), positive sign, no projection
    apply_M: Callable[[Array], Array]       # M u
    apply_K: Callable[[Array], Array]       # K u (without nu), K symmetric PSD
    project: Callable[[Array], Array]       # onto filtered divergence-free space
    force: Array                            # f, flat
    nu: float


def ops_from_ns2d(model) -> FOMOps:
    """Wrap A's `maths.ns2d` interface."""
    from maths import ns2d
    return FOMOps(
        conv=lambda w, u: ns2d.conv_flat(w, u, model),
        apply_M=lambda u: ns2d.apply_M_flat(u, model),
        apply_K=lambda u: ns2d.apply_K_flat(u, model),
        project=lambda u: ns2d.project_flat(u, model),
        force=model.force.reshape(-1),
        nu=model.nu,
    )


def _cols(fn, V):
    return jax.vmap(fn, in_axes=1, out_axes=1)(V)


# ---------------------------------------------------------------------------
# Weighted POD and M-orthonormalisation
# ---------------------------------------------------------------------------
def m_orthonormalize(V: Array, ops: FOMOps, against: Array | None = None) -> Array:
    for _ in range(2):
        if against is not None:
            V = V - against @ (against.T @ _cols(ops.apply_M, V))
        G = V.T @ _cols(ops.apply_M, V)
        L = jnp.linalg.cholesky(0.5 * (G + G.T))
        V = jax.scipy.linalg.solve_triangular(L, V.T, lower=True).T
    return V


def weighted_pod(S: Array, ops: FOMOps, r: int, alpha: float = 0.0):
    S = jnp.asarray(S)
    Q, R = jnp.linalg.qr(S)
    WQ = _cols(ops.apply_M, Q)
    if alpha:
        WQ = WQ + alpha * _cols(ops.apply_K, Q)
    G = Q.T @ WQ
    L = jnp.linalg.cholesky(0.5 * (G + G.T))
    U, sigma, _ = jnp.linalg.svd(L.T @ R)
    if r > int(jnp.sum(sigma > 1e-10 * sigma[0])):
        raise ValueError(f"r={r} exceeds the numerical rank of the snapshot set "
                         f"(sigma/sigma_1 = {sigma / sigma[0]})")
    V = Q @ jax.scipy.linalg.solve_triangular(L.T, U[:, :r], lower=False)
    V = _cols(ops.project, V)  # remove round-off outside the divergence-free space
    V = m_orthonormalize(V, ops)
    return V, sigma


# ---------------------------------------------------------------------------
# Reduced operators
# ---------------------------------------------------------------------------
@partial(jax.tree_util.register_dataclass,
         data_fields=["V", "MV", "A", "N", "f"], meta_fields=["nu"])
@dataclass(frozen=True)
class ROM:
    V: Array    # (D, r)  M-orthonormal, divergence-free basis
    MV: Array   # (D, r)  M V (for projections a = MV^T u)
    A: Array    # (r, r)  V^T K V            (symmetric PSD, without nu)
    N: Array    # (r, r, r) N[i,j,k] = phi_i^T M C(phi_j, phi_k), skew in (i, k)
    f: Array    # (r,)    V^T M f
    nu: float

    @property
    def r(self):
        return self.V.shape[1]


def _conv_block(ops, W, U):
    inner = jax.vmap(lambda w, u: ops.conv(w, u), in_axes=(None, 1), out_axes=1)
    return jax.vmap(lambda w: inner(w, U), in_axes=1, out_axes=1)(W)


def _skew(N):
    return 0.5 * (N - jnp.transpose(N, (2, 1, 0)))


def check_basis(V: Array, ops: FOMOps) -> dict:
    """Orthonormality and divergence-free defects of a basis."""
    G = V.T @ _cols(ops.apply_M, V)
    return dict(
        orthonormality=float(jnp.max(jnp.abs(G - jnp.eye(V.shape[1])))),
        projection=float(jnp.linalg.norm(_cols(ops.project, V) - V) / jnp.linalg.norm(V)),
    )


def build_rom(V: Array, ops: FOMOps, *, tol: float = 1e-10, return_info: bool = False):
    info = check_basis(V, ops)
    if info["orthonormality"] > tol or info["projection"] > tol:
        raise ValueError(f"basis is not M-orthonormal and divergence-free: {info}")
    MV = _cols(ops.apply_M, V)
    A = V.T @ _cols(ops.apply_K, V)
    f = MV.T @ ops.force
    # N[i, j, k]; loop over j to keep memory at O(D r) instead of O(D r^2)
    rows = [jnp.einsum("di,dk->ik", MV, _conv_block(ops, V[:, j:j + 1], V)[:, 0, :])
            for j in range(V.shape[1])]
    N_raw = jnp.stack(rows, axis=1)
    info["skew_defect"] = float(jnp.max(jnp.abs(N_raw + jnp.transpose(N_raw, (2, 1, 0))))
                                / jnp.max(jnp.abs(N_raw)))
    rom = ROM(V=V, MV=MV, A=0.5 * (A + A.T), N=_skew(N_raw), f=f, nu=float(ops.nu))
    return (rom, info) if return_info else rom


def extend_rom(rom: ROM, Phi_new: Array, ops: FOMOps, *, tol: float = 1e-10) -> ROM:
    r, p = rom.r, Phi_new.shape[1]
    Vn = jnp.concatenate([rom.V, Phi_new], axis=1)
    info = check_basis(Vn, ops)
    if info["orthonormality"] > tol or info["projection"] > tol:
        raise ValueError(f"extended basis is not M-orthonormal / divergence-free: {info}")
    MPhi = _cols(ops.apply_M, Phi_new)
    MVn = jnp.concatenate([rom.MV, MPhi], axis=1)
    n = r + p

    N = jnp.zeros((n, n, n)).at[:r, :r, :r].set(rom.N)
    # j in all, k new
    blk = jnp.stack([jnp.einsum("di,dk->ik", MVn, _conv_block(ops, Vn[:, j:j + 1], Phi_new)[:, 0, :])
                     for j in range(n)], axis=1)                    # (n, n, p)
    N = N.at[:, :, r:].set(blk)
    # j new, k old
    blk = jnp.stack([jnp.einsum("di,dk->ik", MVn, _conv_block(ops, Phi_new[:, j:j + 1], rom.V)[:, 0, :])
                     for j in range(p)], axis=1)                    # (n, p, r)
    N = N.at[:, r:, :r].set(blk)
    # i new, j and k old: N[i,j,k] = -N[k,j,i]
    N = N.at[r:, :r, :r].set(-jnp.transpose(N[:r, :r, r:], (2, 1, 0)))

    KPhi = _cols(ops.apply_K, Phi_new)
    A = jnp.block([[rom.A, rom.V.T @ KPhi], [KPhi.T @ rom.V, Phi_new.T @ KPhi]])
    f = jnp.concatenate([rom.f, MPhi.T @ ops.force])
    return ROM(V=Vn, MV=MVn, A=0.5 * (A + A.T), N=_skew(N), f=f, nu=rom.nu)


# ---------------------------------------------------------------------------
# Online phase
# ---------------------------------------------------------------------------
@jax.jit
def rom_rhs(a: Array, rom: ROM) -> Array:
    return -jnp.einsum("ijk,j,k->i", rom.N, a, a) - rom.nu * (rom.A @ a) + rom.f


@jax.jit
def rom_energy(a: Array) -> Array:
    return 0.5 * jnp.dot(a, a)


@jax.jit
def rom_energy_terms(a: Array, rom: ROM):
    """(injection f_r^T a, dissipation nu a^T A a, convective work a^T N(a, a))."""
    return (jnp.dot(rom.f, a), rom.nu * a @ (rom.A @ a),
            jnp.einsum("i,ijk,j,k->", a, rom.N, a, a))


@jax.jit
def rom_step_info(a: Array, dt: float, rom: ROM, tol: float = 1e-13, max_iter: int = 20):
    """Implicit midpoint, b - a = dt g((a+b)/2), solved by Newton on the r x r system."""
    def G(b):
        return b - a - dt * rom_rhs(0.5 * (a + b), rom)

    def body(state):
        i, b, _ = state
        db = jnp.linalg.solve(jax.jacfwd(G)(b), -G(b))
        return i + 1, b + db, jnp.linalg.norm(db) / jnp.maximum(1.0, jnp.linalg.norm(b))

    b0 = a + dt * rom_rhs(a, rom)   # explicit Euler predictor
    it, b, change = jax.lax.while_loop(lambda s: (s[0] < max_iter) & (s[2] > tol),
                                       body, (0, b0, jnp.inf))
    return b, it, jnp.linalg.norm(G(b))


@jax.jit
def rom_step(a: Array, dt: float, rom: ROM) -> Array:
    return rom_step_info(a, dt, rom)[0]


@partial(jax.jit, static_argnames=("steps", "save_every"))
def simulate_rom(a0: Array, dt: float, steps: int, rom: ROM, save_every: int = 1):
    def one(carry, _):
        a, worst = carry
        b = rom_step(a, dt, rom)
        m = 0.5 * (a + b)
        inj, dis, _ = rom_energy_terms(m, rom)
        defect = rom_energy(b) - rom_energy(a) - dt * (inj - dis)
        return (b, jnp.maximum(worst, jnp.abs(defect))), None

    def block(carry, _):
        carry, _ = jax.lax.scan(one, carry, None, length=save_every)
        return carry, carry[0]

    (_, worst), hist = jax.lax.scan(block, (a0, 0.0), None, length=steps // save_every)
    t = dt * save_every * jnp.arange(steps // save_every + 1)
    return t, jnp.concatenate([a0[None], hist]), worst


def reduce_state(u_flat: Array, rom: ROM) -> Array:
    return rom.MV.T @ u_flat


def lift(a: Array, rom: ROM) -> Array:
    return rom.V @ a
