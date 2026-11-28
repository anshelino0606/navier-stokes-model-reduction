# Coarse-FOM-Guided Adaptive POD-ROM

Adaptive Proper Orthogonal Decomposition Reduced-Order Model (POD-ROM) for 2D incompressible Navier–Stokes flows, guided online by an independent coarse-grid Full-Order Model (FOM).

This directory contains two experiment scripts, These run coarse fom on coarser grid:

- **`coarse_rom_optimized.py`** — Evaluates the adaptive POD-ROM and a fixed-rank baseline using saved training and reference `.npz` snapshot files. 
- **`run_coarse_rom_scaling.py`** — Benchmarks grid scaling across spatial resolutions by generating fine-grid training and reference trajectories on the fly and separating offline vs. online runtimes.
---

## Core Methodology

### 1. Weighted POD & Fixed-Capacity Padding
Training snapshots $S$ are compressed using weighted POD under the discrete mass-matrix inner product:

$$\langle u, v \rangle_M = u^T M v, \qquad \|u\|_M = \sqrt{u^T M u}$$

The fine-grid velocity is approximated as $u_{\mathrm{ROM}}(t) = V a(t)$, where the initial rank $r$ is chosen from the singular value energy tail (`--pod-tolerance`). The ROM operators are pre-padded to `--max-rank` so matrix shapes stay static during online enrichment, avoiding JAX JIT recompilation.

### 2. Coarse-Grid Guidance & Convergence Auditing
Alongside the fine-grid ROM (grid $n_x \times n_y$, step $\Delta t$), an independent coarse FOM runs on grid $(n_x / f) \times (n_y / f)$ (`--coarse-factor` $f \ge 2$) with timestep $\Delta t_c = k \Delta t$ (`--dt-coarse`, where $k =$ `--check-every`).

Every coarse step is audited for Picard iteration count, relative Picard update (`1e-4`), and midpoint residual (`2e-4`); the run aborts if convergence fails.

### 3. Coarse Discrepancy & Offline Threshold Calibration
Restricting the fine ROM basis to the coarse grid via $V_c = R V$, the relative coarse-grid discrepancy is

$$d_c(t) = u_{\mathrm{FOM},c}(t) - V_c a(t), \qquad \delta_c(t) = \frac{\|d_c(t)\|_{M_c}}{\|u_{\mathrm{FOM},c}(t)\|_{M_c} + \varepsilon}$$

Before prediction, $\delta_c(t)$ is evaluated across the training interval to calibrate the online enrichment threshold with safety factor $\gamma$ (`--threshold-factor`):

$$\delta_{\mathrm{threshold}} = \max\left(\gamma \max_t \delta_c^{\mathrm{train}}(t),\, 10^{-12}\right)$$

### 4. Basis Enrichment & Reduced-Coordinate State Correction
During prediction, $\delta_c(t)$ is checked every $\Delta t_c$ and $d_c(t)$ is pushed to a rolling buffer (`--difference-window`). When $\delta_c(t) > \delta_{\mathrm{threshold}}$, the cooldown (`--cooldown-steps`) has elapsed, and $r < r_{\max}$:

1. **Subspace enrichment:** Buffered coarse discrepancies are prolonged to the fine grid ($D_f = [P d_c(t_1), \dots, P d_c(t_s)]$). Components already spanned by $V$ are projected out, weighted POD (`--enrichment-tolerance`) extracts new modes $W$, and $W$ is $M$-orthonormalized against $V$ into the padded slots ($V_{\mathrm{new}} = [V \;\; W]$).
2. **Reduced-space state correction:** Using precomputed projection rows $B = V^T M \Pi P$ (updated incrementally only for the newly added modes), the state is corrected directly in reduced coordinates without lifting to the fine grid:

$$a_{\mathrm{new}} = \begin{bmatrix} a \\ 0 \end{bmatrix} + B_{1:r_{\mathrm{new}}}\, d_c(t)$$

### 5. Validation & Diagnostics
Both the adaptive ROM and a fixed-rank POD baseline are evaluated against an unseen fine-grid reference FOM ($u_{\mathrm{ref}}$):

$$e(t) = \frac{\|u_{\mathrm{ROM}}(t) - u_{\mathrm{ref}}(t)\|_M}{\|u_{\mathrm{ref}}(t)\|_M + \varepsilon}, \qquad e_{\mathrm{proj}}(t) = \frac{\|(I - V V^T M) u_{\mathrm{ref}}(t)\|_M}{\|u_{\mathrm{ref}}(t)\|_M + \varepsilon}$$

> **Note:** `time_history.csv` reports $e_{\mathrm{proj}}(t)$ for the fixed ROM and stores `NaN` for the adaptive ROM because intermediate basis states are not retained across enrichment events.

---

## Key Functions

### `coarse_rom_optimized.py`
- **`simulate_coarse` / `run_coarse_fom`**: Runs the JIT-compiled `lax.scan` coarse FOM rollout and enforces Picard convergence checks.
- **`calibrate_coarse_threshold`**: Evolves the initial ROM and coarse FOM over the training window to set $\delta_{\mathrm{threshold}}$.
- **`enrich_from_coarse_differences`**: Prolongs buffered coarse discrepancies, projects out existing modes, extracts new $M$-orthonormal POD modes, and updates the padded ROM in-place.
- **`run_experiment`**: Coordinates offline calibration, precomputation of $B$, the online adaptive loop with state correction, fixed-ROM baseline rollout, and metric/plot exports.

### `run_coarse_rom_scaling.py`
- **`make_fom_simulator` / `generate_fom_trajectory`**: Builds a JIT-compiled `lax.scan` fine-grid FOM solver with warm-up timing to generate the training snapshots $S$ and continuation reference trajectory $U_{\mathrm{ref}}$.
- **`run_benchmark`**: Wraps `run_experiment` and aggregates full offline costs (fine FOM training + calibration + precomputation) separately from online prediction costs (coarse FOM + adaptive ROM) and validation-only reference FOM runtime.

---

## Usage

### Run from saved `.npz` trajectories
```bash
python experiments/coarse_rom_optimized.py \
  --train data/raw/ns2d/training.npz \
  --reference data/raw/ns2d/reference.npz \
  --dt 0.01 \
  --dt-coarse 0.5 \
  --check-every 50 \
  --coarse-factor 2 \
  --max-rank 25 \
  --out figures/coarse_rom_optimized