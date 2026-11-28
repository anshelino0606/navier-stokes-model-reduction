import numpy as np
import jax
import jax.numpy as jnp

from maths.ns2d import apply_M_flat, unflatten_velocity, flatten_velocity


def _lowpass_mask(nx, ny):
    mx = np.abs(np.fft.fftfreq(nx) * nx) < nx / 3
    my = np.abs(np.fft.fftfreq(ny) * ny) < ny / 3
    return jnp.asarray(np.outer(my, mx))


def _fit_axis(h, axis, n_src, n_dst):
    if n_dst == n_src:
        return h
    if n_dst < n_src:
        start = (n_src - n_dst) // 2
        index = [slice(None)] * h.ndim
        index[axis] = slice(start, start + n_dst)
        return h[tuple(index)]
    pad = (n_dst - n_src) // 2
    widths = [(0, 0)] * h.ndim
    widths[axis] = (pad, pad)
    return jnp.pad(h, widths)


def _resize(fields, src, dst):
    for n in (src.nx, src.ny, dst.nx, dst.ny):
        if n % 2:
            raise ValueError("Spectral transfer needs even grid sizes.")
    if (dst.nx - src.nx) % 2 or (dst.ny - src.ny) % 2:
        raise ValueError("Grid sizes must differ by an even number.")

    hat = jnp.fft.fftshift(jnp.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1))
    hat = _fit_axis(hat, -2, src.ny, dst.ny)
    hat = _fit_axis(hat, -1, src.nx, dst.nx)
    hat = jnp.fft.ifftshift(hat, axes=(-2, -1))
    hat = hat * _lowpass_mask(dst.nx, dst.ny)

    scale = (dst.nx * dst.ny) / (src.nx * src.ny)
    return jnp.fft.ifft2(hat * scale, axes=(-2, -1)).real


def _resize_columns(X, src, dst):
    """X: (N_src, m) flat fields as columns -> (N_dst, m)."""
    m = X.shape[1]
    fields = X.T.reshape(m, 2, src.ny, src.nx)
    out = _resize(fields, src, dst)
    return out.reshape(m, -1).T


def restrict_to_coarse(u_fine, fine_model, coarse_model):
    f = unflatten_velocity(u_fine, fine_model)
    return flatten_velocity(_resize(f, fine_model, coarse_model), coarse_model)


def prolong_to_fine(u_coarse, coarse_model, fine_model):
    f = unflatten_velocity(u_coarse, coarse_model)
    return flatten_velocity(_resize(f, coarse_model, fine_model), fine_model)


def restrict_basis_to_coarse(V, fine_model, coarse_model):
    return _resize_columns(V, fine_model, coarse_model)


def correction_projection_rows(V, fine_model, coarse_model):
    Vc = _resize_columns(V, fine_model, coarse_model)
    MVc = jax.vmap(
        lambda v: apply_M_flat(v, coarse_model), in_axes=1, out_axes=1
    )(Vc)
    return MVc.T


def prolong_columns_to_fine(U_coarse, coarse_model, fine_model):
    return _resize_columns(U_coarse, coarse_model, fine_model)


def least_squares_target(B_rows, V_coarse, u_coarse, r, ridge=1e-2):
    B = B_rows[:r]
    Vc = V_coarse[:, :r]
    G = B @ Vc
    reg = ridge * jnp.trace(G) / r
    return jnp.linalg.solve(G + reg * jnp.eye(r), B @ u_coarse)


def nudge(a, a_target, r, kappa):
    """a <- a + kappa (a_target - a) on the first r coefficients."""
    head = a[:r] + kappa * (a_target - a[:r])
    return jnp.concatenate([head, a[r:]])



"""
Replacements for the grid-transfer functions in your optimized
coarse-adaptive script.  Same names and signatures as the block-average /
np.repeat versions, plus two helpers.

Why: the piecewise-constant prolongation (np.repeat) of a coarse field is a
staircase.  Its Fourier images sit at multiples of the coarse grid size.  With
a 4x coarsening those images fall inside the fine solver's 2/3-dealiasing band,
so the enrichment modes and the state correction carry spurious high-wavenumber
content: gradient energy (viscous dissipation) is inflated and the added content
is damped at the wrong rate.  Zero-padding the coarse spectrum (and truncating
the fine spectrum in the other direction) adds nothing above the coarse band.

With these transfers, for fields inside the coarse band:
    * prolongation then restriction is the identity,
    * prolongation preserves the M-norm,
    * V^T M Pi P d  ==  (M_c R V)^T d     (so no fine-grid projection is needed).

Requires even grid sizes (true for 384x288, 96x72, 48x36, ...).
"""