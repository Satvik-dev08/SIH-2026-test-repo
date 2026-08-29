"""
Benchmark: FCFS vs Simulated Annealing vs Genetic Algorithm
============================================================

Runs all three scheduling strategies on the SAME SchedulerSnapshot
and produces a comprehensive comparative report.

Strategies compared:
    1. FCFS       — First Come First Serve (arrival-time baseline)
    2. SA         — Simulated Annealing    (sa_scheduler.py)
    3. GA         — Genetic Algorithm      (ga_scheduler.py)

All strategies use the SAME:
    - Input data  (SchedulerSnapshot)
    - Fitness function (evaluate_schedule_numba)
    - Constraints (headway, maintenance, priorities)
    - Output format (OptimizationResult)

The ONLY difference is the optimisation strategy.

Usage
-----
    python benchmark.py                # standard benchmark
    python benchmark.py --scalability  # include scalability test
"""

import multiprocessing as mp
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Import shared components from existing GA implementation
# ================================================================
from ga_scheduler import (
    RANDOM_SEED,
    MIN_HEADWAY,
    PRIORITY_WEIGHTS,
    TRAIN_COUNT,
    OPTIMIZATION_RUNS,
    Train,
    SchedulerSnapshot,
    ScheduleEntry,
    OptimizationResult,
    validate_snapshot,
    snapshot_train_data,
    prepare_numba_data,
    evaluate_schedule_numba,
    build_schedule_entries,
    build_combined_scenario,
    print_schedule,
    plot_schedule,
    SchedulerService,
)

from sa_scheduler import (
    SASchedulerService,
    SA_RUNS,
    create_arrival_order_solution,
)


# ================================================================
# FCFS Baseline Scheduler
# ================================================================

def run_fcfs(snapshot: SchedulerSnapshot) -> OptimizationResult:
    """First Come First Serve baseline scheduler.

    Schedules trains strictly in arrival-time order with no
    optimisation.  Serves as the lower-bound baseline against
    which SA and GA improvements are measured.

    This is equivalent to what a dispatcher would do manually
    without any decision-support system.

    Args:
        snapshot: SchedulerSnapshot containing trains and constraints.

    Returns:
        OptimizationResult with the FCFS schedule.
    """
    validate_snapshot(snapshot)

    trains = snapshot_train_data(snapshot)
    numba_data = prepare_numba_data(trains)
    train_ids, arrivals, crosses, platforms, weights = numba_data

    # Sort by arrival time (FCFS)
    order = np.argsort(arrivals).astype(np.int32)

    score = float(evaluate_schedule_numba(
        order, arrivals, crosses, platforms, weights, MIN_HEADWAY
    ))

    final_schedule = [train_ids[x] for x in order]
    entries = build_schedule_entries(final_schedule, snapshot)

    total_delay = sum(e.delay_minutes for e in entries)
    weighted_delay = sum(e.weighted_delay for e in entries)

    return OptimizationResult(
        optimization_id="fcfs_" + snapshot.snapshot_id,
        snapshot_id=snapshot.snapshot_id,
        status="completed",
        objective_score=score,
        schedule=entries,
        metrics={
            "total_delay": total_delay,
            "weighted_delay": weighted_delay,
            "maximum_delay": max(
                [e.delay_minutes for e in entries], default=0
            ),
            "trains_scheduled": len(entries),
        },
        generations_completed=0,
        random_seed=RANDOM_SEED,
        algorithm_version="fcfs-baseline-v1",
    )


# ================================================================
# Benchmark Runner
# ================================================================

def run_benchmark(snapshot: SchedulerSnapshot) -> Dict:
    """Run all three algorithms on the same snapshot.

    Returns a dictionary keyed by algorithm name, each containing
    the OptimizationResult, runtime, and convergence history.
    """
    results: Dict = {}

    # --- FCFS ---
    print("\n" + "=" * 60)
    print("  Running FCFS Baseline...")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.perf_counter()
    fcfs_result = run_fcfs(snapshot)
    fcfs_time = time.perf_counter() - t0

    results["FCFS"] = {
        "result": fcfs_result,
        "runtime": fcfs_time,
        "history": [],
    }

    # --- SA ---
    print("\n" + "=" * 60)
    print("  Running Simulated Annealing...")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sa_service = SASchedulerService({}, snapshot.maintenance_windows)
    t0 = time.perf_counter()
    sa_result, sa_history = sa_service.optimize(snapshot)
    sa_time = time.perf_counter() - t0

    results["SA"] = {
        "result": sa_result,
        "runtime": sa_time,
        "history": sa_history,
    }

    # --- GA ---
    print("\n" + "=" * 60)
    print("  Running Genetic Algorithm...")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ga_service = SchedulerService({}, snapshot.maintenance_windows)
    t0 = time.perf_counter()
    ga_result = ga_service.optimize(snapshot)
    ga_time = time.perf_counter() - t0

    results["GA"] = {
        "result": ga_result,
        "runtime": ga_time,
        "history": [],
    }

    return results


# ================================================================
# Comparison Table
# ================================================================

def print_comparison_table(results: Dict) -> None:
    """Print a formatted side-by-side comparison of all algorithms."""

    print("\n" + "=" * 94)
    print("  BENCHMARK COMPARISON")
    print("=" * 94)

    header = (
        f"{'Metric':<30} | "
        f"{'FCFS':>15} | "
        f"{'SA':>15} | "
        f"{'GA':>15}"
    )
    print(header)
    print("-" * 94)

    # Rows
    rows = [
        ("Final Fitness Score", "objective_score", True),
        ("Total Delay (min)", "total_delay", False),
        ("Weighted Delay", "weighted_delay", False),
        ("Maximum Delay (min)", "maximum_delay", False),
        ("Trains Scheduled", "trains_scheduled", False),
    ]

    for label, key, is_top_level in rows:
        values = []
        for algo in ["FCFS", "SA", "GA"]:
            r = results[algo]["result"]
            if is_top_level:
                values.append(f"{r.objective_score:.0f}")
            else:
                values.append(f"{r.metrics[key]:.0f}")
        print(
            f"{label:<30} | "
            f"{values[0]:>15} | "
            f"{values[1]:>15} | "
            f"{values[2]:>15}"
        )

    # Runtime
    runtimes = [
        f"{results[algo]['runtime']:.3f}s"
        for algo in ["FCFS", "SA", "GA"]
    ]
    print(
        f"{'Runtime':<30} | "
        f"{runtimes[0]:>15} | "
        f"{runtimes[1]:>15} | "
        f"{runtimes[2]:>15}"
    )

    # Iterations / Generations
    iters = [
        str(results[algo]["result"].generations_completed)
        for algo in ["FCFS", "SA", "GA"]
    ]
    print(
        f"{'Iterations / Generations':<30} | "
        f"{iters[0]:>15} | "
        f"{iters[1]:>15} | "
        f"{iters[2]:>15}"
    )

    # Algorithm version
    versions = [
        results[algo]["result"].algorithm_version
        for algo in ["FCFS", "SA", "GA"]
    ]
    print(
        f"{'Algorithm Version':<30} | "
        f"{versions[0]:>15} | "
        f"{versions[1]:>15} | "
        f"{versions[2]:>15}"
    )

    print("=" * 94)

    # --- Percentage improvement over FCFS ---
    fcfs_score = results["FCFS"]["result"].objective_score
    fcfs_delay = results["FCFS"]["result"].metrics["weighted_delay"]

    if fcfs_score > 0:
        print("\n  Improvement over FCFS (fitness score):")
        for algo in ["SA", "GA"]:
            score = results[algo]["result"].objective_score
            pct = ((fcfs_score - score) / fcfs_score) * 100
            print(f"    {algo}: {pct:+.2f}% {'reduction' if pct > 0 else 'increase'}")

    if fcfs_delay > 0:
        print("\n  Improvement over FCFS (weighted delay):")
        for algo in ["SA", "GA"]:
            delay = results[algo]["result"].metrics["weighted_delay"]
            pct = ((fcfs_delay - delay) / fcfs_delay) * 100
            print(f"    {algo}: {pct:+.2f}% {'reduction' if pct > 0 else 'increase'}")

    print()


# ================================================================
# Convergence Plot
# ================================================================

def plot_convergence(
    sa_history: List[Tuple[int, float]],
    output: str = "convergence_comparison.png",
) -> None:
    """Plot the SA convergence curve.

    GA convergence can be overlaid here in the future by passing
    its history data as an additional parameter.
    """
    if not sa_history:
        print("[Plot] No SA convergence data available.")
        return

    iterations = [h[0] for h in sa_history]
    scores = [h[1] for h in sa_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        iterations, scores,
        color="#2196F3", linewidth=2, label="SA Best Score",
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness Score (lower is better)", fontsize=12)
    ax.set_title(
        "Simulated Annealing — Convergence Curve", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"[Plot] Saved {output}")


# ================================================================
# Scalability Benchmark
# ================================================================

def build_scenario_with_count(train_count: int) -> SchedulerSnapshot:
    """Build a scenario with a specific number of trains.

    Uses the same random seed and generation logic as
    build_combined_scenario() to ensure consistency.
    """
    rng = random.Random(RANDOM_SEED)
    train_types = ("express", "passenger", "passenger", "freight")

    trains = []
    for i in range(1, train_count + 1):
        train_type = rng.choice(train_types)
        trains.append(
            Train(
                train_id=f"T{i}",
                train_type=train_type,
                arrival=rng.randint(600, 660),
                cross_time=rng.randint(2, 6),
                platform=rng.randint(1, 24),
            )
        )

    maintenance_start = rng.randint(620, 640)

    return SchedulerSnapshot(
        snapshot_id=f"scale_{train_count}",
        trains=tuple(trains),
        minimum_headway_minutes=MIN_HEADWAY,
        maintenance_windows=((maintenance_start, maintenance_start + 5),),
    )


def run_scalability_benchmark(
    train_counts: Optional[List[int]] = None,
) -> Dict:
    """Run all three algorithms across increasing train counts.

    For scalability tests, each algorithm uses a single run (no
    multi-restart) to isolate the scaling behaviour of the core
    optimisation loop.

    Args:
        train_counts: List of train counts to test.
                      Default: [50, 100, 250, 500, 1000]

    Returns:
        Dictionary with scalability data for each algorithm.
    """
    if train_counts is None:
        train_counts = [50, 100, 250, 500, 1000]

    print("\n" + "=" * 60)
    print("  SCALABILITY BENCHMARK")
    print("=" * 60)

    data: Dict = {
        "train_counts": train_counts,
        "FCFS": {"runtime": [], "fitness": [], "avg_delay": []},
        "SA":   {"runtime": [], "fitness": [], "avg_delay": []},
        "GA":   {"runtime": [], "fitness": [], "avg_delay": []},
    }

    for count in train_counts:
        print(f"\n--- {count} trains ---")
        snapshot = build_scenario_with_count(count)

        # FCFS
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        t0 = time.perf_counter()
        fcfs_r = run_fcfs(snapshot)
        fcfs_t = time.perf_counter() - t0
        fcfs_avg = (
            fcfs_r.metrics["total_delay"]
            / max(1, fcfs_r.metrics["trains_scheduled"])
        )
        data["FCFS"]["runtime"].append(fcfs_t)
        data["FCFS"]["fitness"].append(fcfs_r.objective_score)
        data["FCFS"]["avg_delay"].append(fcfs_avg)
        print(f"  FCFS : fitness={fcfs_r.objective_score:>8.0f}   time={fcfs_t:.3f}s")

        # SA (single run)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        sa_svc = SASchedulerService({}, snapshot.maintenance_windows)
        t0 = time.perf_counter()
        sa_r, _ = sa_svc.optimize(snapshot, runs=1)
        sa_t = time.perf_counter() - t0
        sa_avg = (
            sa_r.metrics["total_delay"]
            / max(1, sa_r.metrics["trains_scheduled"])
        )
        data["SA"]["runtime"].append(sa_t)
        data["SA"]["fitness"].append(sa_r.objective_score)
        data["SA"]["avg_delay"].append(sa_avg)
        print(f"  SA   : fitness={sa_r.objective_score:>8.0f}   time={sa_t:.3f}s")

        # GA (single run)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        ga_svc = SchedulerService({}, snapshot.maintenance_windows)
        t0 = time.perf_counter()
        ga_r = ga_svc.optimize(snapshot, runs=1)
        ga_t = time.perf_counter() - t0
        ga_avg = (
            ga_r.metrics["total_delay"]
            / max(1, ga_r.metrics["trains_scheduled"])
        )
        data["GA"]["runtime"].append(ga_t)
        data["GA"]["fitness"].append(ga_r.objective_score)
        data["GA"]["avg_delay"].append(ga_avg)
        print(f"  GA   : fitness={ga_r.objective_score:>8.0f}   time={ga_t:.3f}s")

    return data


def plot_scalability(
    data: Dict,
    output_prefix: str = "scalability",
) -> None:
    """Generate scalability comparison plots.

    Produces a three-panel figure:
        1. Runtime vs train count
        2. Fitness vs train count
        3. Average delay vs train count
    """
    counts = data["train_counts"]

    colors = {"FCFS": "#888888", "SA": "#2196F3", "GA": "#E91E63"}
    markers = {"FCFS": "s", "SA": "^", "GA": "o"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1 — Runtime
    ax = axes[0]
    for algo in ["FCFS", "SA", "GA"]:
        ax.plot(
            counts, data[algo]["runtime"],
            marker=markers[algo], color=colors[algo],
            label=algo, linewidth=2, markersize=8,
        )
    ax.set_xlabel("Number of Trains", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime Scalability", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2 — Fitness
    ax = axes[1]
    for algo in ["FCFS", "SA", "GA"]:
        ax.plot(
            counts, data[algo]["fitness"],
            marker=markers[algo], color=colors[algo],
            label=algo, linewidth=2, markersize=8,
        )
    ax.set_xlabel("Number of Trains", fontsize=12)
    ax.set_ylabel("Fitness Score (lower is better)", fontsize=12)
    ax.set_title("Solution Quality Scalability", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Average Delay
    ax = axes[2]
    for algo in ["FCFS", "SA", "GA"]:
        ax.plot(
            counts, data[algo]["avg_delay"],
            marker=markers[algo], color=colors[algo],
            label=algo, linewidth=2, markersize=8,
        )
    ax.set_xlabel("Number of Trains", fontsize=12)
    ax.set_ylabel("Average Delay (minutes)", fontsize=12)
    ax.set_title("Average Delay Scalability", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output = f"{output_prefix}_comparison.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"[Plot] Saved {output}")


# ================================================================
# Main
# ================================================================

def main() -> None:
    """Run the full benchmark suite."""

    print("=" * 60)
    print("  Railway Scheduler Benchmark")
    print("  FCFS vs Simulated Annealing vs Genetic Algorithm")
    print("=" * 60)

    run_scalability = "--scalability" in sys.argv

    # ==============================================================
    # Standard benchmark (default scenario)
    # ==============================================================
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    snapshot = build_combined_scenario()

    results = run_benchmark(snapshot)
    print_comparison_table(results)

    # Convergence plot (SA)
    plot_convergence(results["SA"]["history"])

    # Gantt charts for each algorithm
    for algo in ["FCFS", "SA", "GA"]:
        plot_schedule(
            snapshot,
            results[algo]["result"],
            output=f"{algo.lower()}_schedule.png",
        )

    # ==============================================================
    # Scalability benchmark (optional)
    # ==============================================================
    if run_scalability:
        scalability_data = run_scalability_benchmark()
        plot_scalability(scalability_data)
    else:
        print(
            "\nTip: Run with --scalability to include the "
            "scalability benchmark (50–1000 trains)."
        )

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    mp.freeze_support()
    main()
