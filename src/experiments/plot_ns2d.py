import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def load_data(path):
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    t = arrays["t"]
    velocity = arrays["velocity"]
    if velocity.ndim != 4 or velocity.shape[0] != t.size or velocity.shape[1] != 2:
        raise ValueError("Expected velocity with shape (frames, 2, ny, nx)")
    ny, nx = velocity.shape[-2:]
    if nx != int(arrays["nx"]) or ny != int(arrays["ny"]):
        raise ValueError("Grid dimensions do not match saved metadata")
    return arrays


def vorticity(velocity, lx, ly):
    ny, nx = velocity.shape[-2:]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)[:, None]
    u_hat = np.fft.fft2(velocity[0])
    v_hat = np.fft.fft2(velocity[1])
    return np.fft.ifft2(1j * kx * v_hat - 1j * ky * u_hat).real


def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fields(data, out_dir, times):
    t = np.asarray(data["t"])
    frames = np.asarray(data["velocity"])
    lx, ly = float(data["lx"]), float(data["ly"])
    ii = np.array([np.argmin(abs(t - target)) for target in times])
    snaps = frames[ii]
    selected_t = t[ii]
    extent = (0, lx, 0, ly)
    comps = ("Horizontal velocity $u(x,y)$", "Vertical velocity $v(x,y)$")
    fig, axs = plt.subplots(2, len(ii), figsize=(4.35 * len(ii), 6.0),
                            sharex=True, sharey=True, constrained_layout=True)
    for row, label in enumerate(comps):
        lim = max(np.max(abs(snaps[:, row])), 1e-9)
        scale = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        for col, at in enumerate(ii):
            ax = axs[row, col]
            im = ax.imshow(snaps[col, row], origin="lower", extent=extent,
                           aspect="equal", cmap="RdBu_r", norm=scale,
                           interpolation="nearest")
            ax.set_title(f"$t={selected_t[col]:g}$")
            if col == 0:
                ax.set_ylabel(f"{label}\n$y$")
            if row == 1:
                ax.set_xlabel("$x$")
        fig.colorbar(im, ax=axs[row].tolist(), shrink=0.87, pad=0.012, label="Velocity")
    save(fig, out_dir, "velocity_components")

    speed = np.hypot(snaps[:, 0], snaps[:, 1])
    omega = np.asarray([vorticity(u, lx, ly) for u in snaps])
    om_lim = max(np.max(abs(omega)), 1e-9)
    fig, axs = plt.subplots(2, len(ii), figsize=(4.35 * len(ii), 6.0),
                            sharex=True, sharey=True, constrained_layout=True)
    for col in range(len(ii)):
        sp = axs[0, col].imshow(speed[col], origin="lower", extent=extent,
                 aspect="equal", cmap="viridis", vmin=0, vmax=speed.max())
        vort = axs[1, col].imshow(omega[col], origin="lower", extent=extent,
                 aspect="equal", cmap="RdBu_r", vmin=-om_lim, vmax=om_lim)
        axs[0, col].set_title(f"$t={selected_t[col]:g}$")
        axs[1, col].set_xlabel("$x$")
        if col == 0:
            axs[0, col].set_ylabel("Speed $|\\mathbf{u}|$\n$y$")
            axs[1, col].set_ylabel("Vorticity $\\omega$\n$y$")
    fig.colorbar(sp, ax=axs[0].tolist(), shrink=0.87, pad=0.012, label="Speed")
    fig.colorbar(vort, ax=axs[1].tolist(), shrink=0.87, pad=0.012, label="Vorticity")
    save(fig, out_dir, "speed_vorticity")

    ncols = min(2, len(ii))
    chosen = [0, len(ii) - 1] if len(ii) > 1 else [0]
    x = np.arange(frames.shape[-1]) * lx / frames.shape[-1]
    y = np.arange(frames.shape[-2]) * ly / frames.shape[-2]
    fig, axs = plt.subplots(1, ncols, figsize=(6.3 * ncols, 3.8),
                            sharex=True, sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs)
    for k, col in enumerate(chosen[:ncols]):
        u, v = snaps[col]
        ax = axs[k]
        img = ax.imshow(speed[col], origin="lower", extent=extent,
                        aspect="equal", cmap="viridis", vmin=0, vmax=speed.max())
        ax.streamplot(x, y, u, v, density=1.0, color="#161616",
                      linewidth=0.6, arrowsize=0.65, maxlength=3.5)
        ax.set_title(f"Instantaneous streamlines at $t={selected_t[col]:g}$")
        ax.set_xlabel("$x$")
        if k == 0:
            ax.set_ylabel("$y$")
    fig.colorbar(img, ax=axs.tolist(), shrink=0.8, pad=0.018,
                 label="Speed $|\\mathbf{u}|$")
    save(fig, out_dir, "speed_streamlines")


def diagnostics(data, out_dir):
    t = np.asarray(data["t"])
    E = np.asarray(data["energy"])
    Pin = np.asarray(data["injection"])
    D = np.asarray(data["dissipation"])
    E_budget = E[0] + np.asarray(data["cumulative_injection"]) - np.asarray(data["cumulative_dissipation"])
    defect = E - E_budget
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 6.3), sharex=True, constrained_layout=True)
    axs[0].plot(t, E, label="Kinetic energy $E(t)$", lw=2.0)
    axs[0].plot(t, E_budget, ls="--", label="Energy balance reconstruction", lw=1.5)
    axs[0].set_ylabel("Energy")
    axs[0].legend(frameon=False)
    axs[1].plot(t, Pin, label="Forcing power $\\langle f,u\\rangle_M$", lw=1.8)
    axs[1].plot(t, D, label="Viscous dissipation $\\nu u^T K u$", lw=1.8)
    axs[1].plot(t, Pin - D, label="Net power $dE/dt$", ls="--", lw=1.2)
    axs[1].axhline(0, color="0.4", lw=0.65)
    axs[1].set_xlabel("Time $t$")
    axs[1].set_ylabel("Power")
    axs[1].legend(frameon=False, ncol=2, loc="upper left")
    save(fig, out_dir, "energy_diagnostics")
    return float(np.max(abs(defect)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/ns2d/reference.npz")
    parser.add_argument("--output", default="figures/ns2d")
    parser.add_argument("--times", type=float, nargs="+", default=[2., 4., 6.])
    args = parser.parse_args()
    data = load_data(Path(args.input))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 9, "figure.dpi": 120})
    fields(data, out, args.times)
    defect = diagnostics(data, out)
    print(f"Plots saved to {out}")
    print(f"Max |E(t) - E(0) - cumulative injection + cumulative dissipation|: {defect:.3e}")


if __name__ == "__main__":
    main()
