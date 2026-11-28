from pathlib import Path
import argparse
import json
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from maths.ns2d import make_model, initial_velocity, fom_step_info, flatten_velocity
from experiments.coarse_rom_optimized import run_experiment

jax.config.update("jax_enable_x64", True)

LX = 4.0 * np.pi
LY = 2.0 * np.pi
NU = 0.02
FORCING_MODE = 4
FORCING_AMPLITUDE = 1.0
INITIAL_AMPLITUDE = 0.1
FOM_TOL = 1e-12
FOM_MAX_ITER = 40


@jax.jit
def _fom_step_flat(u, dt, model):
    u_next, _, _, _ = fom_step_info(u.reshape(2, model.ny, model.nx), dt, model)
    return u_next.reshape(-1)


def make_fom_simulator(model, dt, steps):
    @jax.jit
    def simulate(u0):
        def one_step(u, _):
            u_next = _fom_step_flat(u, dt, model)
            return u_next, u_next

        _, trajectory = jax.lax.scan(one_step, u0, None, length=steps)
        return jnp.concatenate([u0[None, :], trajectory], axis=0)

    return simulate


def generate_fom_trajectory(u0, model, dt, steps, label):
    simulate = make_fom_simulator(model, dt, steps)

    print(f"\nPreparing {label} FOM trajectory...")
    print(f"  Grid:  {model.nx} x {model.ny}")
    print(f"  dt:    {dt}")
    print(f"  Steps: {steps}")
    print(f"  Time:  {steps * dt:.3f} seconds")

    warm = simulate(u0)
    jax.block_until_ready(warm)
    start = time.perf_counter()

    trajectory = simulate(u0)
    jax.block_until_ready(trajectory)

    runtime = time.perf_counter() - start
    print(f"  Runtime: {runtime:.6f} s")

    if not bool(jnp.all(jnp.isfinite(trajectory))):
        raise RuntimeError(f"{label} FOM produced non-finite values.")

    return trajectory, runtime


def build_fine_model(nx, ny):
    return make_model(
        nx=nx, ny=ny, lx=LX, ly=LY, nu=NU, forcing_mode=FORCING_MODE,
        amplitude=FORCING_AMPLITUDE, tol=FOM_TOL, max_iter=FOM_MAX_ITER,
    )


def build_initial_condition(model):
    u0_field = initial_velocity(model, amplitude=INITIAL_AMPLITUDE)
    return flatten_velocity(u0_field, model)



def run_benchmark(args):

    dt = args.dt
    dt_coarse = args.dt_coarse

    training_duration = args.training_duration
    prediction_duration = args.prediction_duration

    training_steps = int(round(training_duration / dt))
    prediction_steps = int(round(prediction_duration / dt))

    training_save_every = args.training_save_every

    if not np.isclose(training_steps * dt, training_duration):
        raise ValueError("Training duration must be an integer multiple of dt.")

    if not np.isclose(prediction_steps * dt, prediction_duration):
        raise ValueError("Prediction duration must be an integer multiple of dt.")

    if training_steps % training_save_every != 0:
        raise ValueError("Training steps must be divisible by training_save_every.")

    ratio_float = dt_coarse / dt
    ratio = int(round(ratio_float))

    if ratio < 1 or not np.isclose(ratio_float, ratio, atol=1e-8):
        raise ValueError("dt_coarse must be an integer multiple of dt.")

    if args.check_every != ratio:
        raise ValueError(f"check_every must equal dt_coarse / dt = {ratio}.")

    if prediction_steps % ratio != 0:
        raise ValueError("Prediction steps must be divisible by check_every.")

    if training_steps % ratio != 0:
        raise ValueError("Training steps must be divisible by dt_coarse / dt.")

    fine_model = build_fine_model(args.nx, args.ny)

    print("\n================================================")
    print("COARSE-FOM-GUIDED ADAPTIVE POD-ROM SCALING")
    print("================================================")

    print("\nPHYSICAL AND GRID CONFIGURATION")
    print("--------------------------------")
    print(f"Fine grid:             {args.nx} x {args.ny}")
    print(f"Fine velocity DOFs:    {2 * args.nx * args.ny}")
    print(f"Training duration:     {training_duration:.3f} s")
    print(f"Prediction duration:   {prediction_duration:.3f} s")
    print(f"Fine timestep:         {dt}")
    print(f"Coarse timestep:       {dt_coarse}")
    print(f"Coarse factor:         {args.coarse_factor}")
    print(f"Check interval:        {args.check_every} fine steps")
    print(f"Training FOM steps:    {training_steps}")
    print(f"Prediction FOM steps:  {prediction_steps}")

    u0 = build_initial_condition(fine_model)
    jax.block_until_ready(u0)

    print("\n================================================")
    print("1. FINE FOM TRAINING TRAJECTORY")
    print("================================================")

    training_trajectory, training_fom_runtime = generate_fom_trajectory(
        u0=u0, model=fine_model, dt=dt, steps=training_steps, label="Training"
    )
    training_indices = np.arange(0, training_steps + 1, training_save_every)
    S = training_trajectory[jnp.asarray(training_indices)].T
    t_train = training_indices.astype(float) * dt

    print("\nTraining snapshot information")
    print("-----------------------------")
    print(f"Training duration:     {t_train[-1]:.3f} s")
    print(f"Training snapshots:    {S.shape[1]}")
    print(f"Snapshot dimension:    {S.shape[0]}")
    print(f"Snapshot interval:     {training_save_every * dt:.3f} s")
    print(f"Training FOM runtime:   {training_fom_runtime:.6f} s")

    u_train_end = training_trajectory[-1]

    print("\n================================================")
    print("2. FINE FOM PREDICTION REFERENCE")
    print("================================================")

    reference_trajectory, reference_fom_runtime = generate_fom_trajectory(
        u0=u_train_end, model=fine_model, dt=dt, steps=prediction_steps, label="Reference"
    )
    t_ref = training_duration + np.arange(prediction_steps + 1, dtype=float) * dt
    U_ref = reference_trajectory.T

    print("\nReference information")
    print("---------------------")
    print(f"Reference start:       {t_ref[0]:.3f} s")
    print(f"Reference end:         {t_ref[-1]:.3f} s")
    print(f"Reference snapshots:   {U_ref.shape[1]}")
    print(f"Reference FOM runtime: {reference_fom_runtime:.6f} s")

    # -------------------------------------------------------------------------
    # 3. CONFIGURE THE ADAPTIVE ROM EXPERIMENT
    # -------------------------------------------------------------------------

    out = args.out / f"{args.nx}x{args.ny}"

    experiment_args = SimpleNamespace(
        dt=dt, dt_coarse=dt_coarse, check_every=args.check_every,
        coarse_factor=args.coarse_factor, threshold_factor=args.threshold_factor,
        difference_window=args.difference_window, cooldown_steps=args.cooldown_steps,
        pod_tolerance=args.pod_tolerance, enrichment_tolerance=args.enrichment_tolerance,
        max_rank=args.max_rank, out=out,
    )

    print("\n================================================")
    print("3. ADAPTIVE ROM EXPERIMENT")
    print("================================================")

    result = run_experiment(t_train=t_train, S=S, t_ref=t_ref, U_ref=U_ref, fine_model=fine_model, args=experiment_args)

    calibration_rom = result["calibration_rom_runtime_seconds"]
    calibration_coarse = result["calibration_coarse_runtime_seconds"]
    precomputation = result["offline_precomputation_runtime_seconds"]
    coarse_prediction = result["coarse_fom_runtime_seconds"]
    adaptive_prediction = result["adaptive_rom_runtime_seconds"]
    fixed_prediction = result["fixed_runtime_seconds"]

    offline_total = training_fom_runtime + calibration_rom + calibration_coarse + precomputation
    online_adaptive_total = coarse_prediction + adaptive_prediction
    complete_adaptive_total = offline_total + online_adaptive_total
    result.update({
        "training_duration_seconds": float(training_duration),
        "prediction_duration_seconds": float(prediction_duration),
        "training_fom_steps": int(training_steps),
        "prediction_fom_steps": int(prediction_steps),
        "training_snapshot_count": int(S.shape[1]),
        "training_snapshot_interval_seconds": float(training_save_every * dt),
        "training_fom_runtime_seconds": float(training_fom_runtime),
        "reference_fom_runtime_seconds": float(reference_fom_runtime),
        "offline_total_including_training_seconds": float(offline_total),
        "online_adaptive_total_seconds": float(online_adaptive_total),
        "complete_adaptive_workflow_seconds": float(complete_adaptive_total),
        "fixed_rom_prediction_seconds": float(fixed_prediction),
        "reference_runtime_excluded_from_method_comparison": True,
    })

    out.mkdir(parents=True, exist_ok=True)

    with open(out / "scaling_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n================================================")
    print("GRID-SCALING BENCHMARK SUMMARY")
    print("================================================")

    print(f"\nGrid: {args.nx} x {args.ny}")
    print(f"Fine DOFs: {2 * args.nx * args.ny}")
    print(f"Training interval: {training_duration:.2f} s")
    print(f"Prediction interval: {prediction_duration:.2f} s")

    print("\nOFFLINE COST")
    print("------------")
    print(f"Fine FOM training:          {training_fom_runtime:.6f} s")
    print(f"ROM calibration:            {calibration_rom:.6f} s")
    print(f"Coarse FOM calibration:     {calibration_coarse:.6f} s")
    print(f"Correction precomputation:  {precomputation:.6f} s")
    print(f"Total offline cost:         {offline_total:.6f} s")

    print("\nONLINE PREDICTION COST")
    print("----------------------")
    print(f"Coarse FOM prediction:      {coarse_prediction:.6f} s")
    print(f"Adaptive ROM prediction:    {adaptive_prediction:.6f} s")
    print(f"Adaptive online total:      {online_adaptive_total:.6f} s")
    print(f"Fixed ROM prediction:       {fixed_prediction:.6f} s")

    print("\nCOMPLETE WORKFLOW")
    print("-----------------")
    print(f"Adaptive workflow:          {complete_adaptive_total:.6f} s")
    print(f"Reference FOM (separate):   {reference_fom_runtime:.6f} s")

    print("\nACCURACY")
    print("--------")
    print(f"Initial rank:               {result['initial_rank']}")
    print(f"Final rank:                 {result['final_rank']}")
    print(f"Enrichment events:          {len(result['enrichment_events'])}")
    print(f"Adaptive final error:       {result['final_adaptive_error']:.6e}")
    print(f"Fixed ROM final error:      {result['final_fixed_error']:.6e}")

    print("\nSaved results to:")
    print(out)

    return result

def main():
    parser = argparse.ArgumentParser(description="Grid-scaling benchmark for the coarse-FOM-guided adaptive POD-ROM.")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=72)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dt-coarse", type=float, default=0.1)
    parser.add_argument("--training-duration", type=float, default=2.0)
    parser.add_argument("--prediction-duration", type=float, default=6.0)
    parser.add_argument("--training-save-every", type=int, default=5)
    parser.add_argument("--coarse-factor", type=int, default=2)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--threshold-factor", type=float, default=1.5)
    parser.add_argument("--difference-window", type=int, default=10)
    parser.add_argument("--cooldown-steps", type=int, default=50)
    parser.add_argument("--pod-tolerance", type=float, default=1e-4)
    parser.add_argument("--enrichment-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-rank", type=int, default=25)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/coarse_rom_384_288")

    args = parser.parse_args()

    if args.nx < 24 or args.ny < 24:
        parser.error("Grid dimensions must be at least 24.")

    if args.nx % args.coarse_factor:
        parser.error("nx must be divisible by coarse-factor.")

    if args.ny % args.coarse_factor:
        parser.error("ny must be divisible by coarse-factor.")

    if args.dt <= 0 or args.dt_coarse <= 0:
        parser.error("Timesteps must be positive.")

    if args.training_duration <= 0:
        parser.error("Training duration must be positive.")

    if args.prediction_duration <= 0:
        parser.error("Prediction duration must be positive.")

    if args.training_save_every < 1:
        parser.error("training-save-every must be positive.")

    if args.coarse_factor < 2:
        parser.error("coarse-factor must be at least 2.")

    if args.max_rank < 1:
        parser.error("max-rank must be positive.")

    run_benchmark(args)


if __name__ == "__main__":
    main()