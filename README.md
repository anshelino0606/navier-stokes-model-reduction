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

# 2D periodic Navier–Stokes: structure-preserving ROM (B)

Layer B builds the **reduced-order model (ROM)**: an $M$-orthonormal, divergence-free POD–Galerkin model that inherits the kinetic-energy balance of the full model exactly, for any basis size, and whose reduced operators can be **extended** when Layer C adds modes. 

## Mathematical formulation

### Full model as a port-Hamiltonian system

With $E(u)=\tfrac12u^TMu$ and $\nabla E=Mu$, Layer A's semi-discrete model on the divergence-free, filtered space $\mathcal V_h=\{u:\mathbb Pu=u\}$ can be written as

$$
\dot u=\bigl(J(u)-R\bigr)Mu+f,\qquad
J(u)=-M^{-1}\mathbb P\,\widetilde C(u)\,\mathbb PM^{-1},\qquad
R=\nu M^{-1}KM^{-1}\succeq0,
$$

where $\widetilde C(w)v=M\,\mathcal C(w,v)$. Skew-symmetry of $\mathcal C$, $\;v^TM\mathcal C(w,u)=-u^TM\mathcal C(w,v)$ for divergence-free $w$, makes $J(u)$ skew, so

$$
\dot E=u^TMf-\nu\,u^TKu .
$$

### Structure-preserving projection

Following Chaturantabut, Beattie and Gugercin (2016), with trial basis $V$ and test basis $W=MV(V^TMV)^{-1}$, the reduced model is again port-Hamiltonian. Choosing $V^TMV=I$ and divergence-free columns ($\mathbb PV=V$) gives $W=MV$ and the $M$-Galerkin ROM of Sanderse (2020):

$$
u\approx Va,\qquad
\dot a=-N(a,a)-\nu A_ra+f_r,
$$

$$
N(a,a)_i=\sum_{j,k}N_{ijk}a_ja_k,\qquad
N_{ijk}=\varphi_i^TM\,\mathcal C(\varphi_j,\varphi_k),\qquad
A_r=V^TKV,\qquad
f_r=V^TMf .
$$

Because $V^TM\mathbb P=V^TM$, the projection $\mathbb P$ and the pressure drop out: the ROM is velocity-only. This is identical to $\dot a=V^TMF_h(Va)$.

### Inherited properties

- **Skew-symmetric tensor:** $N_{ijk}=-N_{kji}$, hence $a^TN(a,a)=0$.
- **Energy balance for any basis:** with $E_r=\tfrac12\|a\|^2=E(Va)$,
  $$\dot E_r=f_r^Ta-\nu\,a^TA_ra .$$
- **Uniform stability:** with $\lambda_*=\lambda_{\min}(A_r)>0$,
  $$\|a(t)\|\le\max\bigl(\|a(0)\|,\ \|f_r\|/(\nu\lambda_*)\bigr)\quad\text{for all }t,$$
  independent of $r$ and of the prediction horizon. Stability does **not** imply accuracy outside the training window.

### Time integration

The ROM uses implicit midpoint, as the full model does:

$$
a^{n+1}=a^n+\Delta t\,g(\bar a),\qquad \bar a=\tfrac12(a^n+a^{n+1}),
$$

solved by Newton on the $r\times r$ system (Jacobian from `jax.jacfwd`). It satisfies the discrete identity

$$
E_r^{n+1}-E_r^n=\Delta t\bigl(f_r^T\bar a-\nu\,\bar a^TA_r\bar a\bigr)
$$

up to the Newton tolerance, mirroring Layer A's full-model identity step by step.

### Weighted POD

For snapshots $S\in\mathbb R^{D\times m}$ and $W_\alpha=M+\alpha K$, the POD modes are computed without forming $S^TW_\alpha S$ (which squares the condition number):

$$
S=QR,\qquad Q^TW_\alpha Q=LL^T,\qquad L^TR=U\Sigma Y^T,\qquad \Phi=QL^{-T}U .
$$

The modes are then projected with $\mathbb P$ to remove round-off and $M$-orthonormalised (two Gram–Schmidt passes), keeping their span. POD is applied to the **raw** snapshots, without mean subtraction: centring introduces a linear term with no sign and breaks the energy identity.

### Basis extension

For new modes $\Phi_{\rm new}$ ($p$ columns), $M$-orthonormal to $V$ and to each other, `extend_rom` evaluates only the $(r+p)^2-r^2$ convection terms $\mathcal C(\varphi_j,\varphi_k)$ with $j$ or $k$ new. The block with $i$ new and $j,k$ old follows from skew-symmetry, $N_{ijk}=-N_{kji}$. The old block of $N$ is reused, and $A_r$, $f_r$ are bordered. The result equals a full rebuild to $10^{-11}$.

## Interface contract for C

`src/maths/sp_rom.py` exports:

| Interface | Purpose and convention |
|---|---|
| `ops_from_ns2d(model)` | Wraps Layer A's `conv_flat`, `apply_M_flat`, `apply_K_flat`, `project_flat`, forcing and $\nu$ into a `FOMOps` tuple |
| `weighted_pod(S, ops, r, alpha=0.0)` | $M$-orthonormal, divergence-free basis `V` `(D, r)` and the $W_\alpha$-singular values; raises if `r` exceeds the numerical rank |
| `m_orthonormalize(W, ops, against=None)` | Two-pass $M$-Gram–Schmidt, optionally against an existing basis (use before `extend_rom`) |
| `check_basis(V, ops)` | $\max\lvert V^TMV-I\rvert$ and $\|\mathbb PV-V\|/\|V\|$ |
| `build_rom(V, ops)` | Returns a `ROM` with fields `V`, `MV`, `A` ($A_r$, without $\nu$), `N`, `f`, `nu`; raises if the basis is invalid |
| `extend_rom(rom, Phi_new, ops)` | Appends `Phi_new` `(D, p)`; raises if the extended basis is not $M$-orthonormal and divergence-free |
| `rom_rhs(a, rom)` | $-N(a,a)-\nu A_ra+f_r$ |
| `rom_step(a, dt, rom)` | One implicit-midpoint step |
| `rom_step_info(a, dt, rom)` | Next state, Newton iterations and final residual |
| `simulate_rom(a0, dt, steps, rom, save_every)` | Returns `(t, coefficients, max per-step energy defect)`; coefficients have shape `(frames, r)` |
| `rom_energy(a)`, `rom_energy_terms(a, rom)` | $E_r$; $(f_r^Ta,\ \nu a^TA_ra,\ a^TN(a,a))$ |
| `reduce_state(u_flat, rom)` | Restart coefficients $a=V^TMu$ |
| `lift(a, rom)` | Full-grid velocity $u=Va$ |

The `ROM` object is a JAX pytree and can be passed to jitted functions. A change of $r$ triggers one recompilation.

**Guarantees from B:** $V^TMV=I$ to $10^{-10}$, $\mathbb PV=V$, and the exact discrete energy identity for **any** valid basis. An energy jump after enrichment therefore comes from the restart $a\leftarrow V^TMu_{\rm FOM}$, never from the ROM.

**Requirements on C:** new modes must be $M$-orthonormal to `rom.V` and to each other to $10^{-10}$. Computing POD via $\operatorname{eigh}(S^TMS)$ loses orthonormality for modes with small singular values, so apply `m_orthonormalize(W, ops, against=rom.V)` to every set of enrichment modes, for both mass-POD and energy-POD.

For C's residual indicator, `rom.V`, `rom.MV` and `lift` give $F=F_h(Va)$ and $V^TMF=$ `rom.MV.T @ F` directly.

## Implementation and files

| File | Description |
|---|---|
| `src/maths/sp_rom.py` | Weighted POD, $M$-orthonormalisation, reduced operators, basis extension, implicit-midpoint ROM, energy diagnostics |
| `src/tests/test_sp_rom.py` | Tests for basis properties, consistency with Layer A, energy conservation, discrete energy balance and extension |
| `src/experiments/sp_rom_baseline.py` | Fixed-POD baseline: train on $[0,T_0]$, predict to $T$, compare with the reference trajectory |
| `figures/sp_rom/` | Baseline figure and metrics (`baseline_alpha0.png`, `baseline_alpha0.json`) |

## Testing and execution

Run the ROM tests from the project root:

```bash
PYTHONPATH=src python -m pytest -q src/tests/test_sp_rom.py   # 7 tests
PYTHONPATH=src python -m pytest -q src/tests                  # all layers
```

The tests check:

- the POD basis is $M$-orthonormal and divergence-free, for $\alpha=0$ and $\alpha=0.1$;
- the tensor right-hand side equals $V^TMF_h(Va)$ (consistency with Layer A);
- $a^TN(a,a)=0$ and the energy identity hold for a **random** divergence-free basis, and $A_r\succeq0$;
- the discrete midpoint energy balance holds over 200 steps;
- `extend_rom` equals a full rebuild;
- `extend_rom` rejects modes that are not $M$-orthogonal.

Run the fixed-POD baseline (about 20 s on a laptop CPU):

```bash
python src/experiments/sp_rom_baseline.py --ranks 2 4 6 8 10 --alpha 0
```

The script locates the repository from its own path, so it runs from any working directory and without `PYTHONPATH` (including the VS Code *Run* button). It prints one line of metrics per rank and then the paths of its outputs, `figures/sp_rom/baseline_alpha0.png` and `figures/sp_rom/baseline_alpha0.json`. Options `--train`, `--reference`, `--ranks`, `--alpha` and `--out` select the snapshot files, basis sizes, POD weight and output folder; paths default to the repository's `data/raw/ns2d/` and `figures/sp_rom/`.

The figure has three panels: the ROM error $\|u_{\rm ref}-Va\|_M/\|u_{\rm ref}\|_M$, the best-approximation (projection) error in span $V$, and the kinetic energy of ROM and full model. A dotted line marks $T_0$. The JSON file stores, per rank, build and online times, the tensor's skew defect, the maximum per-step energy defect, errors at $T_0$ and $T$, and the leading normalised singular values.

Minimal usage:

```python
from maths.ns2d_io import load_snapshot_matrix
from maths.sp_rom import (ops_from_ns2d, weighted_pod, build_rom,
                          reduce_state, simulate_rom, lift)

t, S, model = load_snapshot_matrix("data/raw/ns2d/training.npz", as_jax=True)
ops = ops_from_ns2d(model)
V, sigma = weighted_pod(S, ops, r=8)
rom = build_rom(V, ops)
t_rom, A, worst_defect = simulate_rom(reduce_state(S[:, 0], rom), 0.01, 600, rom, save_every=5)
u_final = lift(A[-1], rom)
```

## Baseline results

Training on $[0,2]$ (41 snapshots), prediction to $T=6$, $\Delta t=0.01$, compared at the 121 reference frames:

| $r$ | Max step energy defect | Rel. error at $T_0=2$ | Rel. error at $T=6$ | Projection error at $T=6$ | Online time (600 steps) |
|---:|---:|---:|---:|---:|---:|
| 2 | $4.8\times10^{-14}$ | $7.5\times10^{-3}$ | 1.74 | 0.97 | 1.3 ms |
| 4 | $3.8\times10^{-14}$ | $2.3\times10^{-4}$ | 1.72 | 0.97 | 2.7 ms |
| 6 | $4.6\times10^{-14}$ | $3.3\times10^{-6}$ | 1.75 | 0.96 | 4.1 ms |
| 8 | $9.3\times10^{-14}$ | $6.0\times10^{-8}$ | 1.71 | 0.90 | 4.5 ms |
| 10 | $7.1\times10^{-14}$ | $7.5\times10^{-10}$ | 1.67 | 0.89 | 8.9 ms |

Errors are relative $M$-norm errors, $\|u_{\rm ref}-Va\|_M/\|u_{\rm ref}\|_M$. The projection error is the best approximation in span $V$. The full model takes about 6 s for the same 600 steps.

The ROM is started at $t=0$ from $a_0=V^TMu(0)$, so the error on $[0,T_0]$ measures reproduction of the training data and only $[T_0,T]$ is prediction. Because the ROM error at $T_0$ is already at projection level, restarting at $T_0$ from $a=V^TMu(T_0)$ gives practically the same prediction errors.

Observations:

- **Structure is preserved:** per-step energy defects are $\le10^{-13}$, the raw tensor is skew to $\sim10^{-15}$, and every run is stable.
- **The ROM is as good as its basis:** its error follows the projection error closely (Galerkin quasi-optimality).
- **The basis fails after $T_0$:** on $[0,2]$ the flow is essentially the laminar spin-up of the Kolmogorov shear, while the instability (energy peak near $t\approx5.3$) lies outside the training span. The ROM stays on the laminar branch. This is the gap Layer C's enrichment is designed to close.
- **The training set is nearly rank-deficient:** $\sigma_k/\sigma_1$ decays geometrically to about $10^{-12}$ (the full model's solver tolerance) at $k\approx14$, and to round-off by $k\approx17$. A fixed basis from $[0,2]$ therefore has about a dozen meaningful modes. `weighted_pod` rejects $\sigma_k/\sigma_1<10^{-10}$.
- **$\alpha=0.1$ changes nothing measurable** for a fixed basis on this snapshot set; its role is in Layer C's mode selection.

## Limitations

- The online cost of the reduced convection term is $O(r^3)$ per right-hand-side evaluation and $N$ needs $r^3$ storage. This is negligible for $r\lesssim50$. Much larger bases would need hyper-reduction, which in general breaks the skew-symmetry and hence the energy guarantee.
- Every basis extension costs $(r+p)^2-r^2$ full-grid convection evaluations; this belongs in the runtime budget of the adaptive method.
- Stability is guaranteed for any basis, accuracy is not: prediction quality outside the training window is limited by the basis, not by the ROM.
- ROM errors are measured against Layer A's discrete model, which is the correct reference for model reduction. As a physical reference, that model is converged to better than $10^{-4}$ up to $t\approx5.5$; during the instability (around $t\approx6$–$7$) the $96\times72$ grid differs from $128\times96$ by a few $10^{-3}$ (see Layer A's convergence checks).

## References

- B. Sanderse, *Non-linearly stable reduced-order models for incompressible flow with energy-conserving finite volume methods*, J. Comput. Phys. 421 (2020) 109736.
- S. Chaturantabut, C. Beattie, S. Gugercin, *Structure-preserving model reduction for nonlinear port-Hamiltonian systems*, SIAM J. Sci. Comput. 38(5) (2016).

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

Using the orthonormality of the basis, the residual indicator can also be evaluated efficiently using

$$
\eta_r^2 = F^T M F - \left\|V_r^T M F\right\|_2^2,
\qquad F = F_h(V_r a).
$$

This avoids explicitly constructing the projection matrix.

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