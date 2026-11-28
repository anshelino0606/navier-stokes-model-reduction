# Residual-Guided Adaptive POD-ROM with FOM Bursts

Online adaptive reduced-order modeling for the 2D incompressible Navier–Stokes equations, using a residual error indicator to trigger short full-order model (FOM) bursts and enrich the POD basis during prediction.

This directory contains two experiment scripts:
- **`adaptive_rom.py`** — Runs the residual-guided adaptive POD-ROM and a fixed-rank baseline from saved training and reference `.npz` snapshot files.
- **`run_adaptive_rom_scaling.py`** — Self-contained grid-scaling benchmark that generates fine-grid training and reference trajectories on the fly across spatial resolutions and separates FOM generation runtimes from ROM prediction runtimes.

---

## Method Overview

### 1. Initial POD-ROM
The initial reduced basis is constructed from fine-grid training snapshots $S = [u(t_1), u(t_2), \ldots, u(t_N)]$. Weighted Proper Orthogonal Decomposition (`weighted_pod` with energy weight $\alpha$ from `get_alpha`) extracts the reduced basis $V \in \mathbb{R}^{n \times r}$, approximating the velocity field as

$$u_{\mathrm{ROM}}(t) = V a(t)$$

where $a(t) \in \mathbb{R}^r$ are the modal coefficients and the initial rank $r$ is selected from the singular value energy tail (`--pod-tolerance`). Distances, projections, and orthogonality are measured in the discrete mass-matrix inner product:

$$\langle u, v \rangle_M = u^T M v, \qquad \|u\|_M = \sqrt{u^T M u}$$

The Navier–Stokes operators are projected onto $V$ via `ops_from_ns2d` and `build_rom`, and `rom_step` advances $a(t)$ in time.

### 2. Residual Indicator & Offline Threshold Calibration
During online integration, model accuracy is monitored via a residual-based error indicator $\rho(t)$ (`residual_error_indicator`).

Before prediction begins, the trigger threshold $\tau$ is calibrated offline across the training interval $[t_1, T_0]$ (where $T_0 = t_N$):
1. Project the initial training snapshot onto the basis: $a(t_1) = V^T M u(t_1)$.
2. Integrate the initial ROM over the training window to obtain $\{a(t_k)\}$.
3. Evaluate $\rho_{\mathrm{train}}(t_k)$ at every training step.

Unless explicitly overridden with `--tau`, $\tau$ is scaled from the peak training residual using `--threshold-factor` ($\gamma$):

$$\tau = \gamma \max_{t \in \mathcal{T}_{\mathrm{train}}} \rho_{\mathrm{train}}(t)$$

### 3. Online Prediction with FOM Bursts
Starting from $u_0 = u_{\mathrm{ref}}(T_0)$ and $a_0 = V^T M u_0$, the adaptive solver advances in time:
1. **Reduced time-stepping:** Advance $a(t)$ via `rom_step` and lift to the fine grid via $u_{\mathrm{ROM}}(t) = V a(t)$.
2. **Periodic indicator check:** Every `--check-every` steps, evaluate $\rho(t)$.
3. **FOM burst trigger:** If $\rho(t) > \tau$ and the cooldown period has elapsed (`step >= cooldown_until`), pause ROM stepping and advance the fine-grid FOM (`fom_step_flat`) for `--window-steps` starting from $u_{\mathrm{ROM}}(t_{\mathrm{trig}})$:

$$S_{\mathrm{window}} = \bigl[u_{\mathrm{ROM}}(t_{\mathrm{trig}}),\, u_{\mathrm{FOM}}(t_{\mathrm{trig}} + \Delta t),\, \ldots,\, u_{\mathrm{FOM}}(t_{\mathrm{end}})\bigr]$$

Any reference output times falling inside the burst window record the full-order state $u_{\mathrm{FOM}}(t)$ directly.

### 4. Subspace Enrichment & State Re-Projection
At the end of each FOM burst (`enrich_rom`), the collected window snapshots $S_{\mathrm{window}}$ update the reduced subspace:
1. **Orthogonal projection:** Remove components already spanned by $V$:

$$S_{\perp} = (I - V V^T M) S_{\mathrm{window}}$$

2. **Secondary weighted POD:** If $\|S_{\perp}\|_2 > 10^{-14}$, compute a weighted POD of $S_{\perp} \approx W \Sigma Z^T$ and retain $p$ modes according to `--pod-tolerance`.
3. **Mass orthonormalization & operator extension:** Orthonormalize $W$ against $V$ via `m_orthonormalize` ($W^T M W = I$, $V^T M W = 0$) and rebuild the reduced operators (`extend_rom`) on the expanded basis $V_{\mathrm{new}} = [V \;\; W]$.
4. **State re-projection & cooldown:** Project the final FOM burst state onto $V_{\mathrm{new}}$ to resume ROM integration,

$$a_{\mathrm{new}} = V_{\mathrm{new}}^T M u_{\mathrm{FOM}}(t_{\mathrm{end}})$$

and enforce a cooldown of `--cooldown-steps` before the next burst can trigger.

### 5. Baseline Comparison & Diagnostics
The adaptive ROM is compared against a fixed-rank POD-ROM built from the same initial basis $V$ and integrated over the full prediction window without enrichment. Both models are evaluated against the fine-grid reference trajectory $u_{\mathrm{ref}}(t)$ using:

$$e(t) = \frac{\|u_{\mathrm{ROM}}(t) - u_{\mathrm{ref}}(t)\|_M}{\|u_{\mathrm{ref}}(t)\|_M + \varepsilon}, \qquad e_{\mathrm{proj}}(t) = \frac{\|V(t) V(t)^T M u_{\mathrm{ref}}(t) - u_{\mathrm{ref}}(t)\|_M}{\|u_{\mathrm{ref}}(t)\|_M + \varepsilon}$$

with $\varepsilon = 10^{-12}$, alongside kinetic energy $E(t) = \frac{1}{2} u^T M u$, forcing power injection $I(t) = u^T M f$, viscous dissipation $D(t) = \nu u^T K u$, and their cumulative trapezoidal integrals.

---

## Key Functions

- **`adaptive_rom.py`**
  - `pod_energy_basis`: Computes the initial weighted POD basis with rank fallback and energy-tail truncation.
  - `enrich_rom`: Projects out existing subspace components from FOM burst snapshots, extracts new $M$-orthonormal modes, and extends the ROM operators.
  - `run_adaptive`: Executes offline threshold calibration, the online ROM/FOM-burst loop, fixed-ROM baseline rollout, and metric/plot exports.
- **`run_adaptive_rom_scaling.py`**
  - `make_fom_simulator` / `generate_fom_trajectory`: Builds a JIT-compiled `jax.lax.scan` fine-grid FOM solver (on domain $[0, 4\pi] \times [0, 2\pi]$, $\nu = 0.02$, forcing mode $4$) with warm-up timing to generate training snapshots $S$ and reference data $U_{\mathrm{ref}}$.
  - `run_benchmark`: Wraps `run_adaptive` and records fine FOM generation runtimes alongside adaptive and fixed ROM prediction costs in `scaling_metrics.json`.

---

## Usage

### Run from saved `.npz` trajectories
```bash
python experiments/adaptive_rom.py \
  --train data/raw/ns2d/training.npz \
  --reference data/raw/ns2d/reference.npz \
  --dt 0.01 \
  --window-steps 20 \
  --check-every 5 \
  --cooldown-steps 20 \
  --threshold-factor 1.5 \
  --pod-tolerance 1e-4 \
  --out figures/adaptive_rom