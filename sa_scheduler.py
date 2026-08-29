"""
Simulated Annealing Railway Train Scheduler
============================================

A production-quality Simulated Annealing (SA) optimizer for the railway
train scheduling and platform optimization problem.

This module works alongside the existing Genetic Algorithm (GA) in
ga_scheduler.py. Both optimizers share the same data structures, fitness
function, and output format to enable fair, scientifically defensible
comparison.

Algorithm Overview
------------------
- Initial solution : arrival-time ordering (deterministic heuristic)
- Neighbor generation: swap (40%), insert (40%), adjacent swap (20%)
- Acceptance : Metropolis criterion  P = exp(-delta / T)
- Cooling : exponential  T *= alpha  (pluggable strategy)
- Temperature : adaptive calibration via Ben-Ameur (2004)
- Stopping : max_iterations OR temperature < min_temperature

Usage
-----
    python sa_scheduler.py           # standalone run
    python benchmark.py              # comparative benchmark
"""

import math
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ================================================================
# Import shared components from the existing GA implementation.
# NO duplication — guarantees identical fitness evaluation.
# ================================================================
from ga_scheduler import (
    RANDOM_SEED,
    MIN_HEADWAY,
    PRIORITY_WEIGHTS,
    TRAIN_COUNT,
    Train,
    SchedulerSnapshot,
    ScheduleEntry,
    OptimizationResult,
    validate_snapshot,
    snapshot_train_data,
    prepare_numba_data,
    evaluate_schedule_numba,
    build_schedule_entries,
    print_schedule,
    build_combined_scenario,
    plot_schedule,
)

# ================================================================
# SA Configuration Constants
# ================================================================

# --- Cooling schedule ---
COOLING_RATE: float = 0.995
"""Exponential cooling multiplier. Each iteration: T_new = T * COOLING_RATE.
A value of 0.995 means ~1,000 iterations to reach ~0.7 % of T0."""

MIN_TEMPERATURE: float = 0.01
"""Temperature below which SA stops. Effectively forces convergence."""

# --- Iteration scaling ---
ITERATIONS_PER_TRAIN: int = 500
"""MAX_ITERATIONS = max(MIN_ITERATIONS, train_count * ITERATIONS_PER_TRAIN).
Scales the search budget with problem size so that larger instances
receive proportionally more exploration."""

MIN_ITERATIONS: int = 25_000
"""Floor for small datasets — ensures adequate search even for < 50 trains."""

# --- Adaptive temperature calibration ---
CALIBRATION_SAMPLES: int = 200
"""Number of random neighbors sampled to estimate T0."""

INITIAL_ACCEPTANCE_PROB: float = 0.8
"""Target acceptance probability for uphill moves at T0.
Higher values → more exploration at the start."""

# --- Multi-restart ---
SA_RUNS: int = 10
"""Number of independent SA restarts. Matches GA's OPTIMIZATION_RUNS."""

# --- Neighbor operator probabilities ---
SWAP_PROBABILITY: float = 0.40
INSERT_PROBABILITY: float = 0.40
# Adjacent swap = 1.0 - SWAP - INSERT = 0.20

# --- Convergence logging ---
LOG_INTERVAL: int = 100
"""Record the best score every LOG_INTERVAL iterations."""

# --- Algorithm identifier ---
SA_ALGORITHM_VERSION: str = "numba-sa-scheduler-v1"


# ================================================================
# Cooling Strategies
# ================================================================

def exponential_cooling(temperature: float, cooling_rate: float) -> float:
    """Exponential (geometric) cooling: T_new = T * alpha.

    This is the most widely used cooling schedule in practice.
    It provides a smooth, gradual temperature decrease that
    balances exploration and exploitation effectively.

    The structure of this function (and its signature) makes it
    trivial to add alternative strategies later — simply define
    a new function with the same signature and pass it to
    run_simulated_annealing().

    Future alternatives (not yet implemented):
        - linear_cooling:      T_new = T - constant
        - logarithmic_cooling: T_new = T0 / (1 + alpha * ln(1 + k))

    Args:
        temperature: Current temperature.
        cooling_rate: Multiplicative factor (0 < alpha < 1).

    Returns:
        New temperature after one cooling step.
    """
    return temperature * cooling_rate


# ================================================================
# Initial Solution
# ================================================================

def create_arrival_order_solution(arrivals: np.ndarray) -> np.ndarray:
    """Generate an initial schedule by sorting trains by arrival time.

    Arrival-time ordering is a natural, deterministic heuristic:
    trains that arrive first are processed first, which tends to
    minimise platform conflicts and produce low-delay schedules.

    This gives SA a reasonable starting point to improve upon
    (as opposed to a completely random permutation).

    Args:
        arrivals: Array of arrival times indexed by train index.

    Returns:
        Permutation of train indices sorted by ascending arrival time.
    """
    return np.argsort(arrivals).astype(np.int32)


# ================================================================
# Neighbor Generation
# ================================================================

def neighbor_swap(solution: np.ndarray) -> np.ndarray:
    """Swap two random (non-adjacent) positions in the schedule.

    A large perturbation — the two trains can be far apart in the
    ordering.  Good for exploration: allows the algorithm to escape
    local minima by making significant structural changes.

    Time complexity : O(N)  (array copy)
    Space complexity: O(N)  (one new array)

    Args:
        solution: Current schedule (permutation of train indices).

    Returns:
        New schedule with two positions swapped.
    """
    neighbor = solution.copy()
    n = len(neighbor)
    i, j = random.sample(range(n), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def neighbor_insert(solution: np.ndarray) -> np.ndarray:
    """Remove a train and re-insert it at a random position.

    Preserves the relative order of all other trains while
    repositioning one train.  This is a medium-scale perturbation
    that can discover orderings unreachable by simple swaps.

    Time complexity : O(N)  (list pop + insert)
    Space complexity: O(N)  (one temporary list + one new array)

    Args:
        solution: Current schedule (permutation of train indices).

    Returns:
        New schedule with one train repositioned.
    """
    # Work on a Python list for efficient pop/insert
    lst = solution.tolist()
    n = len(lst)
    i = random.randint(0, n - 1)
    element = lst.pop(i)
    j = random.randint(0, n - 2)
    lst.insert(j, element)
    return np.array(lst, dtype=np.int32)


def neighbor_adjacent_swap(solution: np.ndarray) -> np.ndarray:
    """Swap two adjacent trains in the schedule.

    The smallest possible perturbation.  Ideal for fine-tuning:
    when the temperature is low and the algorithm is converging,
    adjacent swaps explore the immediate neighbourhood without
    disrupting the overall schedule structure.

    Time complexity : O(N)  (array copy)
    Space complexity: O(N)  (one new array)

    Args:
        solution: Current schedule (permutation of train indices).

    Returns:
        New schedule with two adjacent positions swapped.
    """
    neighbor = solution.copy()
    n = len(neighbor)
    i = random.randint(0, n - 2)
    neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
    return neighbor


def generate_neighbor(solution: np.ndarray) -> np.ndarray:
    """Generate a neighbouring solution using a probabilistic operator mix.

    Operator selection probabilities:
        Swap           40 %  — large moves for exploration
        Insert         40 %  — medium moves preserving partial order
        Adjacent Swap  20 %  — small moves for local refinement

    All operators produce valid permutations (no duplicates, no
    missing trains).

    Args:
        solution: Current schedule (permutation of train indices).

    Returns:
        A new neighbouring schedule.
    """
    r = random.random()
    if r < SWAP_PROBABILITY:
        return neighbor_swap(solution)
    elif r < SWAP_PROBABILITY + INSERT_PROBABILITY:
        return neighbor_insert(solution)
    else:
        return neighbor_adjacent_swap(solution)


# ================================================================
# Adaptive Temperature Calibration
# ================================================================

def calibrate_initial_temperature(
    initial_solution: np.ndarray,
    initial_score: float,
    arrivals: np.ndarray,
    crosses: np.ndarray,
    platforms: np.ndarray,
    weights: np.ndarray,
    minimum_headway: int,
    num_samples: int = CALIBRATION_SAMPLES,
    target_acceptance: float = INITIAL_ACCEPTANCE_PROB,
) -> float:
    """Compute initial temperature adaptively from the fitness landscape.

    Method (Ben-Ameur, 2004 — *Computing the Initial Temperature of
    Simulated Annealing*):

        1. Generate ``num_samples`` random neighbours of the initial
           solution.
        2. Evaluate their fitness and compute deltas (new − current).
        3. Collect all *positive* deltas (uphill / worsening moves).
        4. Set  T0 = −avg_positive_delta / ln(target_acceptance)

    This ensures that at T0, approximately ``target_acceptance``
    fraction of uphill moves will be accepted — regardless of the
    problem scale, number of trains, or penalty magnitudes.

    Args:
        initial_solution: The starting schedule.
        initial_score   : Fitness of the starting schedule.
        arrivals, crosses, platforms, weights: Numba data arrays.
        minimum_headway : Minimum headway between trains (minutes).
        num_samples     : Neighbours to sample for calibration.
        target_acceptance: Desired acceptance probability at T0.

    Returns:
        Calibrated initial temperature (float > 0).
    """
    positive_deltas: List[float] = []

    for _ in range(num_samples):
        nbr = generate_neighbor(initial_solution)
        nbr_score = float(evaluate_schedule_numba(
            nbr, arrivals, crosses, platforms, weights, minimum_headway
        ))
        delta = nbr_score - initial_score
        if delta > 0:
            positive_deltas.append(delta)

    if not positive_deltas:
        # Initial solution is already very strong — use a conservative
        # fallback so the algorithm still explores somewhat.
        return max(1.0, initial_score * 0.1)

    avg_delta = sum(positive_deltas) / len(positive_deltas)

    # T0 = -avg_delta / ln(acceptance_prob)
    # For target_acceptance = 0.8:  ln(0.8) ≈ -0.2231
    t0 = -avg_delta / math.log(target_acceptance)

    return max(t0, 0.01)  # safety floor


# ================================================================
# Compute Maximum Iterations
# ================================================================

def compute_max_iterations(train_count: int) -> int:
    """Derive the iteration budget from the number of trains.

    Scaling rationale
    -----------------
    The solution space is N! permutations.  More trains require more
    iterations to adequately explore the landscape.  We use:

        MAX_ITERATIONS = max(MIN_ITERATIONS, N × ITERATIONS_PER_TRAIN)

    With the defaults (MIN_ITERATIONS=25 000, ITERATIONS_PER_TRAIN=500):

        50 trains  → 25 000 iterations
       100 trains  → 50 000 iterations
       250 trains  → 125 000 iterations
       500 trains  → 250 000 iterations
      1000 trains  → 500 000 iterations

    Each iteration performs a single O(N) fitness evaluation via the
    numba-compiled function, so total work is O(N² × 500) — comfortably
    within practical bounds for N ≤ 1 000.

    Args:
        train_count: Number of trains in the schedule.

    Returns:
        Maximum number of SA iterations.
    """
    return max(MIN_ITERATIONS, train_count * ITERATIONS_PER_TRAIN)


# ================================================================
# Core Simulated Annealing Engine
# ================================================================

def run_simulated_annealing(
    numba_data: Tuple,
    cooling_rate: float = COOLING_RATE,
    min_temperature: float = MIN_TEMPERATURE,
    cooling_fn: Callable[[float, float], float] = exponential_cooling,
) -> Tuple[np.ndarray, float, int, List[Tuple[int, float]]]:
    """Execute one run of the Simulated Annealing optimiser.

    Performs a single SA trajectory starting from an arrival-time-
    ordered initial solution, using adaptive temperature calibration
    and exponential cooling to converge on a near-optimal schedule.

    The algorithm maintains:
        - **current** solution  (may accept worse solutions to escape
          local minima)
        - **best** solution  (global best — never lost)

    Args:
        numba_data     : Tuple of (train_ids, arrivals, crosses,
                         platforms, weights) from prepare_numba_data().
        cooling_rate   : Multiplicative cooling factor (default 0.995).
        min_temperature: Stopping threshold (default 0.01).
        cooling_fn     : Temperature update function. Signature:
                         (temperature, cooling_rate) → new_temperature.
                         Default: exponential_cooling.

    Returns:
        (best_solution, best_score, iterations_completed, history)

        best_solution       : np.ndarray — best permutation found.
        best_score          : float — fitness of that permutation.
        iterations_completed: int — total iterations executed.
        history             : list of (iteration, best_score) tuples
                              for convergence plotting.
    """
    train_ids, arrivals, crosses, platforms, weights = numba_data
    train_count = len(train_ids)
    minimum_headway = MIN_HEADWAY

    # ----------------------------------------------------------
    # 1. Initial solution — arrival-time ordering
    # ----------------------------------------------------------
    current_solution = create_arrival_order_solution(arrivals)
    current_score = float(evaluate_schedule_numba(
        current_solution, arrivals, crosses, platforms, weights,
        minimum_headway
    ))

    best_solution = current_solution.copy()
    best_score = current_score

    # ----------------------------------------------------------
    # 2. Adaptive temperature calibration
    # ----------------------------------------------------------
    temperature = calibrate_initial_temperature(
        current_solution, current_score,
        arrivals, crosses, platforms, weights, minimum_headway,
    )

    # ----------------------------------------------------------
    # 3. Compute iteration budget
    # ----------------------------------------------------------
    max_iterations = compute_max_iterations(train_count)
    progress_step = max(1, max_iterations // 10)

    # ----------------------------------------------------------
    # 4. Convergence history
    # ----------------------------------------------------------
    history: List[Tuple[int, float]] = [(0, best_score)]

    # ----------------------------------------------------------
    # 5. Main SA loop
    # ----------------------------------------------------------
    iteration = 0
    for iteration in range(1, max_iterations + 1):

        # Stopping criterion — temperature floor
        if temperature < min_temperature:
            break

        # Generate a neighbour
        neighbor = generate_neighbor(current_solution)
        neighbor_score = float(evaluate_schedule_numba(
            neighbor, arrivals, crosses, platforms, weights,
            minimum_headway
        ))

        # Acceptance decision (Metropolis criterion)
        delta = neighbor_score - current_score

        if delta < 0:
            # Improvement — always accept
            current_solution = neighbor
            current_score = neighbor_score
        else:
            # Worsening — accept with Boltzmann probability
            acceptance_prob = math.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_score = neighbor_score

        # Track global best
        if current_score < best_score:
            best_solution = current_solution.copy()
            best_score = current_score

        # Cool down
        temperature = cooling_fn(temperature, cooling_rate)

        # Log convergence
        if iteration % LOG_INTERVAL == 0:
            history.append((iteration, best_score))

        # Progress reporting (10 messages per run)
        if iteration % progress_step == 0:
            print(
                f"[SA] Iteration {iteration}/{max_iterations}  "
                f"T={temperature:.4f}  "
                f"Current={current_score:.0f}  "
                f"Best={best_score:.0f}"
            )

    # Final history entry
    if not history or history[-1][0] != iteration:
        history.append((iteration, best_score))

    return best_solution, best_score, iteration, history


# ================================================================
# SA Scheduler Service
# ================================================================

class SASchedulerService:
    """Simulated Annealing based train scheduling service.

    Mirrors the interface of ``SchedulerService`` from ga_scheduler.py
    to enable drop-in usage and fair benchmarking.

    The service runs multiple independent SA restarts — each with the
    same deterministic initial solution but different random neighbour
    sequences — and selects the best result across all runs.

    Attributes:
        trains             : dict — train data (interface compatibility).
        maintenance_windows: list — (start, end) maintenance periods.
    """

    def __init__(
        self,
        trains: Dict,
        maintenance_windows: Optional[List] = None,
    ):
        self.trains = trains
        self.maintenance_windows = maintenance_windows or []

    def optimize(
        self,
        snapshot: SchedulerSnapshot,
        runs: int = SA_RUNS,
    ) -> Tuple[OptimizationResult, List[Tuple[int, float]]]:
        """Run SA optimisation and return the best result.

        Executes ``runs`` independent SA restarts on the given snapshot.
        Each run uses a different random-seed offset to produce diverse
        neighbour sequences while maintaining reproducibility.

        Args:
            snapshot: SchedulerSnapshot with trains and constraints.
            runs    : Number of independent SA restarts (default SA_RUNS).

        Returns:
            (OptimizationResult, convergence_history)

            OptimizationResult   : best result across all runs.
            convergence_history  : list of (iteration, best_score) from
                                   the best run.
        """
        validate_snapshot(snapshot)

        print(f"[SA Service] Starting {runs} SA runs...")

        # Prepare shared data
        trains = snapshot_train_data(snapshot)
        numba_data = prepare_numba_data(trains)

        # Track best across all runs
        best_solution: Optional[np.ndarray] = None
        best_score = float("inf")
        best_iterations = 0
        best_history: List[Tuple[int, float]] = []

        for run_idx in range(runs):
            # Unique seed per run for diverse neighbour sequences
            run_seed = RANDOM_SEED + run_idx
            random.seed(run_seed)
            np.random.seed(run_seed)

            solution, score, iterations, history = run_simulated_annealing(
                numba_data,
            )

            print(
                f"[SA Service] Run {run_idx + 1}/{runs}  "
                f"Score={score:.0f}  "
                f"Iterations={iterations}"
            )

            if score < best_score:
                best_score = score
                best_solution = solution
                best_iterations = iterations
                best_history = history

        # Restore deterministic seeds
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # Convert integer indices back to train IDs
        train_ids = numba_data[0]
        final_schedule = [train_ids[x] for x in best_solution]

        # Build schedule entries (same function as GA)
        entries = build_schedule_entries(final_schedule, snapshot)

        total_delay = sum(e.delay_minutes for e in entries)
        weighted_delay = sum(e.weighted_delay for e in entries)

        result = OptimizationResult(
            optimization_id="sa_opt_" + snapshot.snapshot_id,
            snapshot_id=snapshot.snapshot_id,
            status="completed",
            objective_score=float(best_score),
            schedule=entries,
            metrics={
                "total_delay": total_delay,
                "weighted_delay": weighted_delay,
                "maximum_delay": max(
                    [e.delay_minutes for e in entries], default=0
                ),
                "trains_scheduled": len(entries),
            },
            generations_completed=best_iterations,
            random_seed=RANDOM_SEED,
            algorithm_version=SA_ALGORITHM_VERSION,
        )

        return result, best_history


# ================================================================
# Main Entry Point
# ================================================================

def main() -> None:
    """Run SA optimiser standalone on the demo scenario."""

    print("=" * 60)
    print("  Simulated Annealing Railway Scheduler")
    print("=" * 60)
    print()

    # Deterministic seeding
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Creating scenario...")
    snapshot = build_combined_scenario()

    service = SASchedulerService({}, snapshot.maintenance_windows)

    start_time = time.perf_counter()
    result, history = service.optimize(snapshot, SA_RUNS)
    elapsed = time.perf_counter() - start_time

    print()
    print(f"BEST SCORE    : {result.objective_score}")
    print(f"Runtime       : {elapsed:.3f}s")
    print(f"Iterations    : {result.generations_completed}")
    print(f"Algorithm     : {result.algorithm_version}")
    print()

    trains = snapshot_train_data(snapshot)
    print_schedule([x.train_id for x in result.schedule], trains)
    plot_schedule(snapshot, result, output="sa_train_schedule.png")

    print("Finished.")


if __name__ == "__main__":
    main()
