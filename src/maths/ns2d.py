from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["kx", "ky", "k2", "mask", "force"],
    meta_fields=["nx", "ny", "lx", "ly", "nu", "forcing_mode", "amplitude", "tol", "max_iter"],
)
@dataclass(frozen=True)
class NSModel:
    kx: Array
    ky: Array
    k2: Array
    mask: Array
    force: Array
    nx: int
    ny: int
    lx: float
    ly: float
    nu: float
    forcing_mode: int
    amplitude: float
    tol: float
    max_iter: int

    @property
    def dx(self):
        return self.lx / self.nx

    @property
    def dy(self):
        return self.ly / self.ny


def make_model(
    nx=64,
    ny=32,
    lx=4 * jnp.pi,
    ly=2 * jnp.pi,
    nu=0.02,
    forcing_mode=4,
    amplitude=1.0,
    tol=1e-12,
    max_iter=40,
):
    if nx < 12 or ny < 12 or nx % 2 or ny % 2:
        raise ValueError("nx and ny must be even and at least 12")
    if nu < 0 or forcing_mode < 1 or forcing_mode >= ny / 3:
        raise ValueError("Require nu >= 0 and 1 <= forcing_mode < ny/3")
    if tol <= 0 or max_iter < 1:
        raise ValueError("Require tol > 0 and max_iter >= 1")

    lx, ly = float(lx), float(ly)
    if lx <= 0 or ly <= 0:
        raise ValueError("Require positive domain lengths")

    dx, dy = lx / nx, ly / ny
    mx = jnp.fft.fftfreq(nx) * nx
    my = jnp.fft.fftfreq(ny) * ny
    kx = (2 * jnp.pi / lx) * mx[None, :]
    ky = (2 * jnp.pi / ly) * my[:, None]
    k2 = kx**2 + ky**2
    mask = (jnp.abs(mx[None, :]) < nx / 3) & (jnp.abs(my[:, None]) < ny / 3)

    y = jnp.arange(ny) * dy
    forcing_x = amplitude * jnp.sin(2 * jnp.pi * forcing_mode * y / ly)
    force = jnp.stack((jnp.broadcast_to(forcing_x[:, None], (ny, nx)),
                       jnp.zeros((ny, nx), dtype=jnp.float64)))
    return NSModel(
        kx=kx, ky=ky, k2=k2, mask=mask, force=force,
        nx=int(nx), ny=int(ny), lx=lx, ly=ly, nu=float(nu),
        forcing_mode=int(forcing_mode), amplitude=float(amplitude),
        tol=float(tol), max_iter=int(max_iter),
    )


def _fft(u):
    return jnp.fft.fft2(u, axes=(-2, -1))


def _ifft(uh):
    return jnp.fft.ifft2(uh, axes=(-2, -1)).real


@jax.jit
def filter_modes(u: Array, model: NSModel) -> Array:
    return _ifft(_fft(u) * model.mask)


@jax.jit
def project(u: Array, model: NSModel) -> Array:
    uh = _fft(u) * model.mask
    dot = model.kx * uh[0] + model.ky * uh[1]
    inv_k2 = jnp.where(model.k2 > 0, 1 / jnp.where(model.k2 > 0, model.k2, 1), 0)
    pu = uh[0] - model.kx * inv_k2 * dot
    pv = uh[1] - model.ky * inv_k2 * dot
    return _ifft(jnp.stack((pu, pv)))


@jax.jit
def divergence(u: Array, model: NSModel) -> Array:
    uh = _fft(u)
    return _ifft(1j * (model.kx * uh[0] + model.ky * uh[1]))


@jax.jit
def laplace(u: Array, model: NSModel) -> Array:
    return _ifft(-model.k2 * _fft(u) * model.mask)


@jax.jit
def apply_M(u: Array, model: NSModel) -> Array:
    return (model.dx * model.dy) * u


@jax.jit
def apply_K(u: Array, model: NSModel) -> Array:
    return -(model.dx * model.dy) * laplace(u, model)


@jax.jit
def mass_dot(u: Array, v: Array, model: NSModel) -> Array:
    return jnp.sum(u * apply_M(v, model))


@jax.jit
def conv(w: Array, u: Array, model: NSModel) -> Array:
    w = filter_modes(w, model)
    u = filter_modes(u, model)
    uh = _fft(u)
    ux = _ifft(1j * model.kx * uh)
    uy = _ifft(1j * model.ky * uh)
    advective = w[0] * ux + w[1] * uy

    wx_u = w[0] * u
    wy_u = w[1] * u
    conservative = _ifft(1j * model.kx * _fft(wx_u) +
                         1j * model.ky * _fft(wy_u))
    return filter_modes(0.5 * (advective + conservative), model)


@jax.jit
def F_h(u: Array, model: NSModel) -> Array:
    nonlinear = project(conv(u, u, model), model)
    return -nonlinear + model.nu * laplace(u, model) + model.force


@jax.jit
def energy(u: Array, model: NSModel) -> Array:
    return 0.5 * mass_dot(u, u, model)


@jax.jit
def energy_terms(u: Array, model: NSModel) -> tuple[Array, Array, Array]:
    injection = mass_dot(model.force, u, model)
    dissipation = model.nu * jnp.sum(u * apply_K(u, model))
    convective_work = mass_dot(u, conv(u, u, model), model)
    return injection, dissipation, convective_work


@jax.jit
def _helmholtz(rhs: Array, dt: float, model: NSModel) -> Array:
    uh = _fft(rhs) * model.mask
    return project(_ifft(uh / (1 + 0.5 * dt * model.nu * model.k2)), model)


@jax.jit
def fom_step_info(u: Array, dt: float, model: NSModel):
    linear = u + 0.5 * dt * model.nu * laplace(u, model) + dt * model.force

    def update(guess):
        midpoint = 0.5 * (u + guess)
        nonlinear = project(conv(midpoint, midpoint, model), model)
        return _helmholtz(linear - dt * nonlinear, dt, model)

    def cond(state):
        i, _, change = state
        return (i < model.max_iter) & (change > model.tol)

    def body(state):
        i, guess, _ = state
        next_guess = update(guess)
        change = jnp.linalg.norm(next_guess - guess) / jnp.maximum(1.0, jnp.linalg.norm(next_guess))
        return i + 1, next_guess, change

    iterations, next_u, update = jax.lax.while_loop(
        cond, body, (jnp.array(0), u, jnp.asarray(jnp.inf))
    )
    residual = midpoint_residual(u, next_u, dt, model)
    relative_residual = jnp.linalg.norm(residual) / jnp.maximum(1.0, jnp.linalg.norm(next_u))
    return next_u, iterations, update, relative_residual


@jax.jit
def fom_step(u: Array, dt: float, model: NSModel) -> Array:
    return fom_step_info(u, dt, model)[0]


@jax.jit
def midpoint_residual(u: Array, next_u: Array, dt: float, model: NSModel) -> Array:
    midpoint = 0.5 * (u + next_u)
    return next_u - u - dt * F_h(midpoint, model)


@partial(jax.jit, static_argnames=("steps", "save_every"))
def simulate(u0: Array, dt: float, steps: int, model: NSModel, save_every: int = 1):
    if steps < 0 or save_every < 1 or steps % save_every:
        raise ValueError("steps must be nonnegative and divisible by save_every")

    def block(state, _):
        state = jax.lax.fori_loop(0, save_every,
                                  lambda i, current: fom_step(current, dt, model), state)
        return state, state

    _, history = jax.lax.scan(block, u0, None, length=steps // save_every)
    frames = jnp.concatenate((u0[None, ...], history), axis=0)
    times = dt * save_every * jnp.arange(steps // save_every + 1)
    return times, frames


def initial_velocity(model: NSModel, amplitude: float = 0.1) -> Array:
    x = jnp.arange(model.nx) * model.dx
    y = jnp.arange(model.ny) * model.dy
    xx, yy = jnp.meshgrid(2 * jnp.pi * x / model.lx, 2 * jnp.pi * y / model.ly)
    stream = amplitude * (jnp.sin(xx) * jnp.sin(yy) + 0.35 * jnp.sin(2 * xx + yy))
    sh = _fft(stream)
    vel_x = _ifft(1j * model.ky * sh)
    vel_y = -_ifft(1j * model.kx * sh)
    return project(jnp.stack((vel_x, vel_y)), model)


def flatten_velocity(u: Array, model: NSModel) -> Array:
    if u.shape != (2, model.ny, model.nx):
        raise ValueError("Expected velocity with shape (2, ny, nx)")
    return jnp.reshape(u, (-1,))


def unflatten_velocity(a: Array, model: NSModel) -> Array:
    if a.shape != (2 * model.ny * model.nx,):
        raise ValueError("Expected flattened velocity with length 2*ny*nx")
    return jnp.reshape(a, (2, model.ny, model.nx))


def snapshot_matrix(velocity: Array, model: NSModel) -> Array:
    if velocity.ndim != 4 or velocity.shape[1:] != (2, model.ny, model.nx):
        raise ValueError("Expected snapshots with shape (frames, 2, ny, nx)")
    return jnp.reshape(velocity, (velocity.shape[0], -1)).T


@jax.jit
def F_h_flat(a: Array, model: NSModel) -> Array:
    return F_h(a.reshape((2, model.ny, model.nx)), model).reshape(-1)


@jax.jit
def conv_flat(a: Array, b: Array, model: NSModel) -> Array:
    return conv(a.reshape((2, model.ny, model.nx)),
                b.reshape((2, model.ny, model.nx)), model).reshape(-1)


@jax.jit
def project_flat(a: Array, model: NSModel) -> Array:
    return project(a.reshape((2, model.ny, model.nx)), model).reshape(-1)


@jax.jit
def apply_M_flat(a: Array, model: NSModel) -> Array:
    return apply_M(a.reshape((2, model.ny, model.nx)), model).reshape(-1)


@jax.jit
def apply_K_flat(a: Array, model: NSModel) -> Array:
    return apply_K(a.reshape((2, model.ny, model.nx)), model).reshape(-1)


@jax.jit
def fom_step_flat(a: Array, dt: float, model: NSModel) -> Array:
    return fom_step(a.reshape((2, model.ny, model.nx)), dt, model).reshape(-1)


@jax.jit
def relative_high_mode_energy(u: Array, model: NSModel) -> Array:
    uh = _fft(u)
    mx = jnp.abs(model.kx * model.lx / (2 * jnp.pi))
    my = jnp.abs(model.ky * model.ly / (2 * jnp.pi))
    edge = (mx >= model.nx / 4) | (my >= model.ny / 4)
    spectrum = jnp.sum(jnp.abs(uh) ** 2, axis=0)
    return jnp.sum(jnp.where(edge & model.mask, spectrum, 0)) / jnp.maximum(jnp.sum(spectrum), 1e-30)


@partial(jax.jit, static_argnames=("steps", "save_every"))
def simulate_audited(u0: Array, dt: float, steps: int, model: NSModel, save_every: int = 1):
    if steps < 0 or save_every < 1 or steps % save_every:
        raise ValueError("steps must be nonnegative and divisible by save_every")

    metrics0 = (u0, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    def one_step(carry, _):
        u, max_balance, max_residual, max_div, max_update, cumulative_input, cumulative_loss, max_iter, failures = carry
        next_u, iterations, update, residual = fom_step_info(u, dt, model)
        midpoint = 0.5 * (u + next_u)
        injection, dissipation, _ = energy_terms(midpoint, model)
        defect = energy(next_u, model) - energy(u, model) - dt * (injection - dissipation)
        div = jnp.linalg.norm(divergence(next_u, model))
        failed = (update > model.tol) | (residual > 1e-9) | ~jnp.isfinite(residual)
        return (next_u,
                jnp.maximum(max_balance, jnp.abs(defect)),
                jnp.maximum(max_residual, residual),
                jnp.maximum(max_div, div),
                jnp.maximum(max_update, update),
                cumulative_input + dt * injection,
                cumulative_loss + dt * dissipation,
                jnp.maximum(max_iter, iterations),
                failures + failed.astype(jnp.int32)), None

    def block(carry, _):
        carry, _ = jax.lax.scan(one_step, carry, None, length=save_every)
        return carry, (carry[0], carry[5], carry[6])

    carry, (history, inputs, losses) = jax.lax.scan(
        block, metrics0, None, length=steps // save_every)
    states = jnp.concatenate((u0[None, ...], history))
    times = dt * save_every * jnp.arange(steps // save_every + 1)
    audit = dict(
        maximum_energy_defect=carry[1], maximum_midpoint_residual=carry[2],
        maximum_divergence=carry[3], maximum_relative_update=carry[4],
        max_picard_iterations=carry[7], failed_steps=carry[8],
        cumulative_injection=jnp.concatenate((jnp.array([0.0]), inputs)),
        cumulative_dissipation=jnp.concatenate((jnp.array([0.0]), losses)),
    )
    return times, states, audit


@jax.jit
def weighted_dot_flat(a: Array, b: Array, alpha: float, model: NSModel) -> Array:
    return jnp.vdot(a, apply_M_flat(b, model) + alpha * apply_K_flat(b, model)).real


def grid_coordinates(model: NSModel):
    return jnp.arange(model.nx) * model.dx, jnp.arange(model.ny) * model.dy
