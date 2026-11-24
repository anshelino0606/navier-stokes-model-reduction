# 2D periodic Navier–Stokes: full-model handoff (A)

Layer A for the structure-preserving, adaptive MOR project. This package contains the **full-order JAX model, tests, validated training/reference trajectories, and interfaces for B/C**. It does not implement or modify the POD–Galerkin model, reduced tensor, adaptive controller, or basis updates.

## Mathematical model

We solve on $\Omega=[0,L_x)\times[0,L_y)$ with periodic conditions,

$$
\partial_t \boldsymbol u = -\mathbb P\mathcal C(\boldsymbol u,\boldsymbol u)
+\nu\Delta\boldsymbol u+\boldsymbol f,\qquad
\nabla\cdot\boldsymbol u=0,
\qquad \boldsymbol f=(\sin(4y),0).
$$

Here $\mathbb P$ is the Fourier Helmholtz projection (pressure elimination), and $\mathcal C$ is the skew-symmetric pseudo-spectral advection

$$
\mathcal C(\boldsymbol w,\boldsymbol u)=\tfrac12\left[
(\boldsymbol w\cdot\nabla)\boldsymbol u
+\nabla\cdot(\boldsymbol w\otimes\boldsymbol u)\right].
$$

FFT differentiation uses the **2/3 dealiasing mask**. The discrete mass and stiffness operators are

$$
M=\Delta x\Delta y\,I,\qquad K=-M\Delta_h,
\qquad E=\tfrac12\boldsymbol u^TM\boldsymbol u.
$$

On the divergence-free, filtered grid, $\langle\boldsymbol u,\mathcal C(\boldsymbol w,\boldsymbol u)\rangle_M=0$. The implicit-midpoint step (solved by Picard iteration) satisfies

$$
E^{k+1}-E^k=\Delta t\left(\langle\boldsymbol f,\boldsymbol u^{k+1/2}\rangle_M
-\nu\,\boldsymbol u^{k+1/2,T}K\boldsymbol u^{k+1/2}\right)
$$

up to nonlinear-solver and floating-point tolerances.

## Install and run

Python 3.11+ and a double-precision JAX installation are required.

```bash
python -m pip install 'jax[cpu]' numpy pytest
PYTHONPATH=src python -m pytest -q src/tests/test_ns2d.py src/tests/test_ns2d_handoff.py

PYTHONPATH=src python src/experiments/ns2d_handoff.py \
    --nx 96 --ny 72 --dt 0.01 --train-end 2 --final-time 6 --save-every 5

PYTHONPATH=src python src/experiments/ns2d_interface_check.py
```

The generator runs one **continuous** trajectory from $t=0$ to $t=6$, then writes the prefix $[0,2]$ as a separate training file. Training and reference states are identical at their common sample times. The supplied NPZ files reproduce the specified parameters without needing to rerun the generator.

The default reproducible configuration is $L_x=4\pi$, $L_y=2\pi$, $(n_x,n_y)=(96,72)$, $\nu=0.02$, forcing mode 4, unit forcing amplitude, streamfunction perturbation amplitude $0.1$, $\Delta t=0.01$, and output every 5 steps. This is a **demonstrated unsteady test regime**, not a claim of statistically stationary turbulence.

## Interface contract for B and C

`src/maths/ns2d.py` exports the following operations. Array layout uses **component-first, y-major, x-minor** order.

| Interface | Purpose and convention |
|---|---|
| `make_model(...)` | Creates spectral grid, projection mask and forcing |
| `project(u, model)` | Projects an `(2,ny,nx)` field onto filtered divergence-free modes |
| `F_h(u, model)` | Projected, full-order velocity time derivative; **includes** $-\mathcal C$, diffusion and forcing |
| `conv(w, u, model)` | Skew-symmetric $\mathcal C(w,u)$; **no minus sign, no projection** |
| `apply_M(u, model)` | $Mu$ for field-shaped velocity |
| `apply_K(u, model)` | $Ku=-M\Delta_h u$ for field-shaped velocity |
| `fom_step(u, dt, model)` | One fully implicit-midpoint step (Picard iteration) |
| `fom_step_info(u, dt, model)` | Next velocity, Picard iterations, last relative update and midpoint residual |
| `simulate(u0, dt, steps, model, save_every)` | Simple time stepping, returns `(t, velocity)` |
| `simulate_audited(...)` | Time stepping with **per-step** energy, divergence and convergence auditing |
| `energy`, `energy_terms`, `divergence` | Diagnostics |
| `flatten_velocity`, `unflatten_velocity`, `snapshot_matrix` | Convert between field arrays and matrix snapshots |
| `F_h_flat`, `conv_flat`, `project_flat` | Same operations on 1D vectors of length $D=2n_xn_y$ |
| `apply_M_flat`, `apply_K_flat` | Weighted operations on 1D vectors of length $D$ |
| `weighted_dot_flat(a,b,alpha,model)` | $a^T(M+\alpha K)b$, with $\alpha\ge0$ |

**Flattening convention:** `u.reshape(-1)` is the full-state vector. Its first $n_xn_y$ entries are horizontal velocity $u_x$; its remaining entries are vertical velocity $u_y$. For a snapshot tensor `(frames,2,ny,nx)`, the state matrix is `velocity.reshape(frames,-1).T`, of shape `(D,frames)`. No pressure component is stored.

For B, the reduced basis $V\in\mathbb R^{D\times r}$ should satisfy $V^TMV=I$, and individual columns must remain projected and divergence-free. The reduced right-hand side is $V^TM F_h(Va)$, with tensor entries for convection based on $-V^TM\mathbb P\mathcal C(V_j,V_k)$. `conv_flat` returns **positive** $\mathcal C$; apply `-project_flat` when building the reduced nonlinear tensor. `apply_K_flat` does **not** include $\nu$.

For C, the recommended full-dimensional residual indicator uses `F_h_flat(V @ a, model)` and the same $M$-orthonormal basis; the weighted basis-selection metric uses `apply_M_flat + alpha * apply_K_flat`. The real full-order restart state is a flattened velocity vector, with projection coefficients $a=V^TMu_{\mathrm{FOM}}$.

### Reading the supplied data

```python
from maths.ns2d_io import load_velocity, load_snapshot_matrix

train_t, train_S, model = load_snapshot_matrix('data/raw/ns2d/training.npz')
ref_t, ref_velocity, _ = load_velocity('data/raw/ns2d/reference.npz')

# train_S.shape == (2*ny*nx, number_of_training_frames)
# ref_velocity.shape == (number_of_reference_frames, 2, ny, nx)
```

`load_velocity` and `load_snapshot_matrix` return NumPy arrays by default; pass `as_jax=True` to obtain JAX arrays. Both reconstruct the same model from saved metadata. The reference and training files each contain:

- `t`, `velocity` and the matching saved-frame diagnostics `energy`, `injection`, `dissipation`, `convection_work`, `divergence`, `high_mode_energy_fraction`, `cumulative_injection`, and `cumulative_dissipation`;
- `nx`, `ny`, `lx`, `ly`, `nu`, `forcing_mode`, `forcing_amplitude`, `initial_amplitude`, `dt`, `save_every`, `mass_weight`, `train_end_time`, `final_time`, `train_end_index`.

`data/raw/ns2d/diagnostics.csv` contains the reference trajectory diagnostics, and `validation.json` records worst-case **per-step** energy-balance defects, midpoint nonlinear residual, divergence, Picard iteration counts, convergence failures, and high-frequency energy fraction.

## Verification and regime selection

The automated tests cover Fourier projection/incompressibility, the symmetry and positivity of $K$, absence of convective work, discrete midpoint energy identity, analytic Kolmogorov shear, consistency between flat/grid interfaces, and the audited snapshot continuity and I/O interfaces.

On the supplied $96\times72$ run to $T=6$:

| Check | Result |
|---|---:|
| Failed implicit steps | 0 |
| Maximum per-step energy-balance defect | $3.49\times10^{-13}$ |
| Maximum divergence norm | $5.83\times10^{-13}$ |
| Maximum relative midpoint residual | $1.86\times10^{-13}$ |
| Maximum high-frequency energy fraction | $6.11\times10^{-5}$ |

Separate verification runs with the **same initial condition** found at $t=6$:

- time-step refinement $\Delta t=0.01\to0.005$ at $96\times72$: relative $L^2$ difference $9.90\times10^{-5}$;
- grid refinement $96\times72\to128\times96$ at $\Delta t=0.01$, with Fourier restriction to the coarse grid: relative $L^2$ difference $1.93\times10^{-3}$.

This is adequate for an initial handoff, but the team should recheck time and grid convergence if increasing final time, forcing amplitude, or decreasing viscosity.

## Scope and integration

All supplied files are Layer A only. Import this module into the shared repository without changing existing B/C code. The handoff fixes the **proposed** `(2,ny,nx)` / flattened-vector interface; confirm signatures at the joint kickoff before teammates start integration. The initial snapshots cover $[0,2]$, and the reference covers $[0,6]$, so B can train on the former and C can evaluate adaptation beyond the training interval. Reuse the entire continuous reference trajectory rather than restarting the FOM separately at $T_0$.

# 2D periodic Navier–Stokes: adaptive enrichment (C)

Layer C implements the residual-based monitoring and adaptive basis-enrichment framework for the 2D periodic Navier–Stokes model. It connects the full-order model provided by Layer A with the structure-preserving reduced-order modelling framework developed in Layer B.

The main objective is to improve reduced-order model accuracy by detecting when the current reduced basis no longer adequately represents the flow dynamics and enriching it with additional modes. Two enrichment strategies are considered: standard mass-weighted POD and energy-informed POD.

## Mathematical formulation

Let the reduced approximation be

$$
u_r=V_ra,
$$

where $V_r$ is an $M$-orthonormal, divergence-free reduced basis satisfying

$$
V_r^TMV_r=I.
$$

The unresolved part of the full-order tendency is measured using the residual indicator

$$
\eta_r(t)=\left\|(I-P_r)F_h(V_ra)\right\|_M,
\qquad P_r=V_rV_r^TM.
$$

A relative residual indicator is defined as

$$
\rho_r(t)=
\frac{\eta_r(t)}
{\|F_h(V_ra)\|_M+\varepsilon}.
$$

It measures how much of the full-order tendency is not represented by the current reduced space. A large residual indicates that additional basis directions may be needed. This is a residual-based monitoring indicator rather than a rigorous bound on the ROM solution error.

Using the orthonormality of the basis, the residual can be evaluated efficiently:

$$
\eta_r^2
=
F^TMF-\|V_r^TMF\|_2^2,
\qquad F=F_h(V_ra).
$$

## Snapshot-based enrichment

When enrichment is required, new full-order snapshots are collected in a matrix

$$
S=[u_1,u_2,\ldots,u_m].
$$

The existing reduced space is removed using the $M$-orthogonal projection:

$$
S_\perp=(I-P_r)S.
$$

New basis directions are extracted from the unresolved snapshot components using two approaches.

**Mass-weighted POD**

The correlation matrix is

$$
C_M=S_\perp^TMS_\perp.
$$

Its eigenvalue decomposition provides the mass-orthonormal enrichment modes. This approach prioritises the representation of the velocity field in the mass inner product.

**Energy-informed POD**

To incorporate spatial-gradient information, a weighted inner product is introduced:

$$
G_\alpha=M+\alpha K,
$$

where $K$ is the stiffness operator and $\alpha$ controls the contribution of gradients. The corresponding correlation matrix is

$$
C_G=S_\perp^TG_\alpha S_\perp.
$$

The eigenvalue decomposition of this matrix provides the energy-informed enrichment modes. These modes are initially orthonormal with respect to $G_\alpha$. Before being appended to the reduced basis, they are converted to an $M$-orthonormal set.

The weighting parameter is scaled using the forcing wavenumber:

$$
k_f=\frac{2\pi n_f}{L_y},
\qquad
\boxed{
\alpha=\frac{1}{k_f^2}
=\left(\frac{L_y}{2\pi n_f}\right)^2.
}
$$

This provides a physically motivated reference length scale. A dimensionless multiplier $\beta$ can also be introduced:

$$
\alpha=\frac{\beta}{k_f^2},
$$

allowing the influence of gradient weighting to be investigated.

## Residual-based enrichment controller

A rolling mean and standard deviation of the relative residual are used to calculate the standardised residual score:

$$
z_r(t)=
\frac{\rho_r(t)-\mu_\rho}
{\sigma_\rho+\varepsilon}.
$$

Enrichment is triggered when both conditions are satisfied:

$$
z_r(t)>z_{\mathrm{threshold}},
\qquad
\rho_r(t)>\rho_{\min}.
$$

This combines the absolute residual level with its deviation from recent behaviour.

The controller includes several safeguards:

- A cooldown period between enrichment events.
- A maximum permitted reduced-basis rank.
- A maximum number of modes added per event.
- A relative discarded-energy tolerance for selecting the number of enrichment modes.

These conditions help prevent unnecessary enrichment and excessive basis growth.

## Adaptive ROM workflow

The complete adaptive enrichment procedure consists of the following steps:

1. **Initialisation:** Construct an initial divergence-free, mass-orthonormal POD basis from training snapshots.
2. **Reduced-order simulation:** Evolve the reduced coefficients using the structure-preserving ROM time integrator.
3. **Residual monitoring:** Evaluate the relative residual indicator during the simulation.
4. **Trigger evaluation:** Check the residual thresholds, cooldown period and maximum-rank constraint.
5. **Snapshot collection:** When enrichment is triggered, obtain additional full-order snapshots over a selected time window.
6. **Projection:** Remove the components already represented by the current reduced basis.
7. **Mode extraction:** Compute new modes using mass-weighted POD or energy-informed POD.
8. **Orthogonalisation:** Ensure that the new modes are mass-orthogonal to the existing basis and to each other.
9. **Basis update:** Append the accepted modes to the reduced basis.
10. **Operator update:** Recompute or extend the reduced operators consistently with the updated basis.
11. **Continuation:** Resume reduced-order time integration using the enriched basis.

This procedure allows the reduced space to adapt to changes in the flow dynamics while maintaining the required mass-orthonormal basis structure.

## Implementation and files

The Layer C implementation is organised as follows:

| File | Description |
|---|---|
| `src/maths/ns2d_ae.py` | Residual monitoring, snapshot projection, mass-POD, energy-POD, orthogonalisation and enrichment-controller utilities |
| `src/tests/test_ns2d_ae.py` | Tests for residual evaluation, projection, POD orthogonality, enrichment, mode selection and triggering |
| `src/experiments/ns2d_adaptive.py` | Offline comparison of mass-POD and energy-POD enrichment |

The implementation uses the flattened interfaces provided by Layer A, including `F_h_flat`, `apply_M_flat` and `apply_K_flat`.

The flattened velocity vectors have dimension

$$
D=2n_xn_y,
$$

and snapshot matrices have the shape

$$
S\in\mathbb{R}^{D\times n_{\mathrm{snapshots}}}.
$$

## Numerical experiments and evaluation

The enrichment strategies are compared using the same initial reduced basis and the same enrichment snapshots.

The initial basis is constructed using mass-weighted POD. Later snapshots are then used to calculate additional modes using either mass-weighted POD or energy-informed POD.

The current offline experiment evaluates the enrichment procedures and saves the resulting bases, modes and eigenvalues for further analysis.

The broader evaluation of the adaptive ROM considers:

- **Approximation accuracy:** How well the reduced basis represents the full-order velocity snapshots.
- **Energy behaviour:** Whether the reduced model preserves the expected kinetic-energy behaviour.
- **Dissipation:** How accurately the reduced model represents viscous dissipation.
- **Gradient representation:** Whether energy-informed enrichment improves the representation of spatial gradients.
- **Basis dimension:** The number of modes required to achieve a given approximation quality.
- **Computational cost:** The computational expense of the reduced simulation, including enrichment and operator updates.

The main objective is to determine whether adaptive enrichment improves the accuracy and robustness of the reduced model without requiring an unnecessarily large basis.

## Testing and execution

Run the adaptive enrichment tests from the project root:

```bash
PYTHONPATH=src python -m pytest src/tests/test_ns2d_ae.py -v
```

Run the offline enrichment experiment:

```bash
PYTHONPATH=src python -m experiments.ns2d_adaptive
```

The default experiment uses the first half of the training snapshots to construct the initial basis and the remaining half for enrichment. This split can be adjusted using `--initial-fraction`.

The experiment saves its results to:

```text
data/results/ns2d/adaptive.npz
```

## Integration with the other project layers

Layer C forms the adaptive component of the complete structure-preserving model reduction framework:

- **Layer A:** Provides the full-order Navier–Stokes solver, discretisation, snapshots and mass and stiffness operators.
- **Layer B:** Provides the reduced-order formulation, reduced operators, time integration and basis-update functionality.
- **Layer C:** Provides residual monitoring, enrichment decisions, new basis construction and adaptive control.

Together, the three layers provide a framework for adaptive structure-preserving model order reduction of the 2D incompressible Navier–Stokes equations. The framework combines efficient reduced-order simulation with residual-based detection of unresolved dynamics and controlled enrichment of the reduced basis.