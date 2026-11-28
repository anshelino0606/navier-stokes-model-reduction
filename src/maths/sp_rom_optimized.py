
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


class FOMOps(NamedTuple):
    """Flat-vector operations of the full model."""
    conv: Callable[[Array, Array], Array]
    apply_M: Callable[[Array], Array]
    apply_K: Callable[[Array], Array]
    project: Callable[[Array], Array]
    force: Array
    nu: float


def ops_from_ns2d(model) -> FOMOps:
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


def m_orthonormalize(V, ops, against=None):
    """Two-pass M-orthonormalisation."""
    for _ in range(2):
        if against is not None:
            MV = _cols(ops.apply_M, V)
            V = V - against @ (against.T @ MV)

        MV = _cols(ops.apply_M, V)
        G = V.T @ MV
        G = 0.5 * (G + G.T)

        L = jnp.linalg.cholesky(G)
        V = jax.scipy.linalg.solve_triangular(
            L, V.T, lower=True
        ).T

    return V


def weighted_pod(S, ops, r, alpha=0.0):
    """Weighted POD with the M + alpha*K inner product."""
    S = jnp.asarray(S)

    Q, R = jnp.linalg.qr(S, mode="reduced")
    WQ = _cols(ops.apply_M, Q)

    if alpha:
        WQ = WQ + alpha * _cols(ops.apply_K, Q)

    G = Q.T @ WQ
    G = 0.5 * (G + G.T)
    L = jnp.linalg.cholesky(G)

    U, sigma, _ = jnp.linalg.svd(
        L.T @ R,
        full_matrices=False,
    )

    numerical_rank = int(
        jnp.sum(sigma > 1e-10 * sigma[0])
    )

    if r > numerical_rank:
        raise ValueError(
            f"r={r} exceeds numerical snapshot rank {numerical_rank}"
        )

    V = Q @ jax.scipy.linalg.solve_triangular(
        L.T, U[:, :r], lower=False
    )

    V = _cols(ops.project, V)
    V = m_orthonormalize(V, ops)

    return V, sigma


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["V", "MV", "A", "N", "f", "active_rank"],
    meta_fields=["nu"],
)
@dataclass(frozen=True)
class ROM:
    V: Array
    MV: Array
    A: Array
    N: Array
    f: Array
    active_rank: Array
    nu: float

    @property
    def r(self):
        """Host-side active rank. Do not use inside a jitted function."""
        return int(self.active_rank)

    @property
    def capacity(self):
        return self.V.shape[1]


def _skew(N):
    return 0.5 * (
        N - jnp.transpose(N, (2, 1, 0))
    )


def _projected_conv_tensor(MV, W, U, ops):
    """
    T[i,j,k] = MV[:,i]^T conv(W[:,j], U[:,k]).
    """
    def one(w):
        C = jax.vmap(
            lambda u: ops.conv(w, u),
            in_axes=1,
            out_axes=1,
        )(U)
        return MV.T @ C

    T = jax.lax.map(one, W.T)
    return jnp.transpose(T, (1, 0, 2))


def check_basis(V, ops):
    MV = _cols(ops.apply_M, V)
    G = V.T @ MV
    projection_error = _cols(ops.project, V) - V

    return {
        "orthonormality": float(
            jnp.max(jnp.abs(G - jnp.eye(V.shape[1])))
        ),
        "projection": float(
            jnp.linalg.norm(projection_error)
            / jnp.maximum(jnp.linalg.norm(V), 1e-30)
        ),
    }


def build_rom(V, ops, *, tol=1e-10, return_info=False):
    """Build a conventional, variable-rank ROM."""
    info = check_basis(V, ops)

    if (
        info["orthonormality"] > tol
        or info["projection"] > tol
    ):
        raise ValueError(
            f"Basis is not M-orthonormal/divergence-free: {info}"
        )

    MV = _cols(ops.apply_M, V)
    KV = _cols(ops.apply_K, V)

    A = V.T @ KV
    A = 0.5 * (A + A.T)

    f = MV.T @ ops.force
    N_raw = _projected_conv_tensor(MV, V, V, ops)

    scale = jnp.maximum(jnp.max(jnp.abs(N_raw)), 1e-30)

    info["skew_defect"] = float(
        jnp.max(
            jnp.abs(
                N_raw + jnp.transpose(N_raw, (2, 1, 0))
            )
        ) / scale
    )

    rom = ROM(
        V=V,
        MV=MV,
        A=A,
        N=_skew(N_raw),
        f=f,
        active_rank=jnp.asarray(V.shape[1], dtype=jnp.int32),
        nu=float(ops.nu),
    )

    return (rom, info) if return_info else rom


def pad_rom(rom, max_rank):
    """
    Allocate a fixed maximum-rank ROM.

    The array shapes remain constant when active_rank increases.
    """
    r = rom.r

    if max_rank < r:
        raise ValueError(
            f"max_rank={max_rank} is smaller than initial rank={r}"
        )

    if max_rank == r:
        return rom

    D = rom.V.shape[0]
    dtype = rom.V.dtype

    V = jnp.zeros((D, max_rank), dtype=dtype)
    MV = jnp.zeros((D, max_rank), dtype=dtype)
    A = jnp.zeros((max_rank, max_rank), dtype=dtype)
    N = jnp.zeros(
        (max_rank, max_rank, max_rank), dtype=dtype
    )
    f = jnp.zeros((max_rank,), dtype=dtype)

    V = V.at[:, :r].set(rom.V)
    MV = MV.at[:, :r].set(rom.MV)
    A = A.at[:r, :r].set(rom.A)
    N = N.at[:r, :r, :r].set(rom.N)
    f = f.at[:r].set(rom.f)

    return ROM(
        V=V,
        MV=MV,
        A=A,
        N=N,
        f=f,
        active_rank=jnp.asarray(r, dtype=jnp.int32),
        nu=rom.nu,
    )


def extend_padded_rom(rom, Phi_new, ops, *, tol=1e-10):
    """
    Enrich a fixed-capacity ROM without changing any array shape.

    Only the unused columns/rows/tensor entries are updated.
    """
    r = rom.r
    p_requested = Phi_new.shape[1]
    remaining = rom.capacity - r
    p = min(p_requested, remaining)

    if p <= 0:
        return rom

    Phi_new = Phi_new[:, :p]

    V_active = rom.V[:, :r]
    Vn = jnp.concatenate([V_active, Phi_new], axis=1)

    info = check_basis(Vn, ops)

    if (
        info["orthonormality"] > tol
        or info["projection"] > tol
    ):
        raise ValueError(
            f"Enriched basis is invalid: {info}"
        )

    MPhi = _cols(ops.apply_M, Phi_new)
    MVn = jnp.concatenate([rom.MV[:, :r], MPhi], axis=1)

    # Insert new basis vectors into the unused capacity.
    V = rom.V.at[:, r:r + p].set(Phi_new)
    MV = rom.MV.at[:, r:r + p].set(MPhi)

    # Diffusion matrix.
    KPhi = _cols(ops.apply_K, Phi_new)

    A_cross = V_active.T @ KPhi
    A_new = Phi_new.T @ KPhi

    A = rom.A.at[:r, r:r + p].set(A_cross)
    A = A.at[r:r + p, :r].set(A_cross.T)
    A = A.at[r:r + p, r:r + p].set(A_new)

    # Forcing vector.
    f_new = MPhi.T @ ops.force
    f = rom.f.at[r:r + p].set(f_new)

    # Convection tensor entries involving new modes.
    N = rom.N

    blk = _projected_conv_tensor(
        MVn,
        jnp.concatenate(
            [rom.V[:, :r], Phi_new], axis=1
        ),
        Phi_new,
        ops,
    )

    N = N.at[:r + p, :r + p, r:r + p].set(blk)

    blk = _projected_conv_tensor(
        MVn,
        Phi_new,
        rom.V[:, :r],
        ops,
    )

    N = N.at[:r + p, r:r + p, :r].set(blk)

    # Complete the tensor using its skew symmetry.
    N = N.at[r:r + p, :r, :r].set(
        -jnp.transpose(
            N[:r, :r, r:r + p],
            (2, 1, 0),
        )
    )

    N = _skew(N)

    return ROM(
        V=V,
        MV=MV,
        A=A,
        N=N,
        f=f,
        active_rank=jnp.asarray(r + p, dtype=jnp.int32),
        nu=rom.nu,
    )


def extend_rom(rom, Phi_new, ops, *, tol=1e-10):
    """Compatibility function for variable-rank ROMs."""
    r = rom.r
    p = Phi_new.shape[1]

    if p == 0:
        return rom
    
    if rom.capacity != r:
        raise ValueError(
            "Use extend_padded_rom() for a padded ROM."
        )

    Vn = jnp.concatenate([rom.V, Phi_new], axis=1)
    info = check_basis(Vn, ops)

    if (
        info["orthonormality"] > tol
        or info["projection"] > tol
    ):
        raise ValueError(
            f"Extended basis is invalid: {info}"
        )

    MPhi = _cols(ops.apply_M, Phi_new)
    MVn = jnp.concatenate([rom.MV, MPhi], axis=1)
    n = r + p

    N = jnp.zeros((n, n, n), dtype=rom.N.dtype)
    N = N.at[:r, :r, :r].set(rom.N)

    blk = _projected_conv_tensor(MVn, Vn, Phi_new, ops)
    N = N.at[:, :, r:].set(blk)

    blk = _projected_conv_tensor(MVn, Phi_new, rom.V, ops)
    N = N.at[:, r:, :r].set(blk)

    N = N.at[r:, :r, :r].set(
        -jnp.transpose(N[:r, :r, r:], (2, 1, 0))
    )

    KPhi = _cols(ops.apply_K, Phi_new)
    A_cross = rom.V.T @ KPhi
    A_new = Phi_new.T @ KPhi

    A = jnp.block([
        [rom.A, A_cross],
        [A_cross.T, A_new],
    ])
    A = 0.5 * (A + A.T)

    f_new = MPhi.T @ ops.force
    f = jnp.concatenate([rom.f, f_new])

    return ROM(
        V=Vn,
        MV=MVn,
        A=A,
        N=_skew(N),
        f=f,
        active_rank=jnp.asarray(n, dtype=jnp.int32),
        nu=rom.nu,
    )


@jax.jit
def _active_mask(rom):
    return (
        jnp.arange(rom.V.shape[1]) < rom.active_rank
    )


@jax.jit
def rom_rhs(a, rom):

    mask = _active_mask(rom)
    a_active = a * mask

    rhs = (
        -jnp.einsum("ijk,j,k->i", rom.N, a_active, a_active)
        - rom.nu * (rom.A @ a_active)
        + rom.f
    )

    return rhs * mask


@jax.jit
def rom_energy(a):
    return 0.5 * jnp.dot(a, a)


@jax.jit
def rom_energy_terms(a, rom):
    mask = _active_mask(rom)
    a = a * mask

    return (
        jnp.dot(rom.f, a),
        rom.nu * a @ (rom.A @ a),
        jnp.einsum("i,ijk,j,k->", a, rom.N, a, a),
    )


@jax.jit
def rom_step_info(a,dt,rom,tol=1e-13,max_iter=20,):
    """Implicit midpoint step with a masked Newton system."""
    mask = _active_mask(rom)
    mask2 = mask[:, None] * mask[None, :]

    a = a * mask

    def residual(b):
        midpoint = 0.5 * (a + b)
        return (b - a - dt * rom_rhs(midpoint, rom)) * mask

    def jacobian(b):
        midpoint = 0.5 * (a + b)

        dconv_1 = jnp.einsum("ilk,k->il", rom.N, midpoint)
        dconv_2 = jnp.einsum("ijl,j->il", rom.N, midpoint)

        J_rhs = (-dconv_1- dconv_2- rom.nu * rom.A)

        J_rhs = J_rhs * mask2

        J = (jnp.eye(a.shape[0])- 0.5 * dt * J_rhs)

        return J

    b0 = a + dt * rom_rhs(a, rom)

    initial = ( jnp.array(0, dtype=jnp.int32), b0, jnp.array(jnp.inf, dtype=a.dtype),)

    def condition(state):
        i, _, change = state
        return (i < max_iter) & (change > tol)

    def body(state):
        i, b, _ = state
        G = residual(b)
        J = jacobian(b)
        db = jnp.linalg.solve(J, -G)
        db = db * mask
        change = (jnp.linalg.norm(db)/ jnp.maximum(1.0, jnp.linalg.norm(b)))
        return i + 1, (b + db) * mask, change

    it, b, _ = jax.lax.while_loop(condition,body,initial,)
    return b * mask, it, jnp.linalg.norm(residual(b))


@jax.jit
def rom_step(a, dt, rom):
    return rom_step_info(a, dt, rom)[0]


@partial(jax.jit,static_argnames=("steps", "save_every"),)
def simulate_rom_fast(a0,dt,steps,rom,save_every=1,):
    
    if save_every == 1:
        def one_step(a, _):
            b = rom_step(a, dt, rom)
            return b, b

        _, saved = jax.lax.scan( one_step, a0, None, length=steps,)
    else:
        def one_step(a, _):
            b = rom_step(a, dt, rom)
            return b, b

        def block(a, _):
            a, history = jax.lax.scan(one_step,a,None,length=save_every,)
            return a, history[-1]

        _, saved = jax.lax.scan(block,a0,None,length=steps // save_every,)

    A = jnp.concatenate([a0[None, :], saved], axis=0)

    t = (dt* save_every* jnp.arange(saved.shape[0] + 1))

    return t, A


@partial(jax.jit,static_argnames=("steps", "save_every"),)
def simulate_rom(a0,dt,steps,rom,save_every):
    """Diagnostic integration path."""
    def one_step(carry, _):
        a, worst = carry

        b = rom_step(a, dt, rom)
        midpoint = 0.5 * (a + b)

        inj, dis, _ = rom_energy_terms(midpoint, rom)

        defect = (rom_energy(b)- rom_energy(a)- dt * (inj - dis))

        return (b,jnp.maximum(worst, jnp.abs(defect)),), None

    def block(carry, _):
        carry, _ = jax.lax.scan(one_step,carry,None,length=save_every,)
        return carry, carry[0]

    (_, worst), hist = jax.lax.scan(block,(a0, jnp.array(0.0, dtype=a0.dtype)),None,length=steps // save_every,)

    t = (dt* save_every* jnp.arange(steps // save_every + 1))

    A = jnp.concatenate([a0[None, :], hist], axis=0)

    return t, A, worst


@jax.jit
def reduce_state(u_flat, rom):
    return rom.MV.T @ u_flat


@jax.jit
def lift(a, rom):
    return rom.V @ a
