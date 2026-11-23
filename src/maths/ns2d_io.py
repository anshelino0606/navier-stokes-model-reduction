from pathlib import Path

import jax.numpy as jnp
import numpy as np

from maths.ns2d import make_model


def load_velocity(path: str | Path, *, as_jax: bool = False):
    with np.load(path, allow_pickle=False) as data:
        model = make_model(
            nx=int(data["nx"]), ny=int(data["ny"]),
            lx=float(data["lx"]), ly=float(data["ly"]),
            nu=float(data["nu"]), forcing_mode=int(data["forcing_mode"]),
            amplitude=float(data["forcing_amplitude"]),
        )
        t = np.asarray(data["t"])
        velocity = np.asarray(data["velocity"])

    if velocity.shape != (len(t), 2, model.ny, model.nx):
        raise ValueError("Invalid velocity snapshot dimensions")
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Snapshot time points must be strictly increasing")
    if as_jax:
        return jnp.asarray(t), jnp.asarray(velocity), model
    return t, velocity, model


def load_snapshot_matrix(path: str | Path, *, as_jax: bool = False):
    t, velocity, model = load_velocity(path, as_jax=as_jax)
    states = velocity.reshape((len(t), -1)).T
    return t, states, model
