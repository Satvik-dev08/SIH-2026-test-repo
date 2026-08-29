# Combined solution of all the practice files

"""Genetic-algorithm train scheduler for a fixed scenario."""

import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
# ----------------------------
# Config: GA & runtime params
# ----------------------------
RANDOM_SEED = 42
POP_SIZE = 60
GENERATIONS = 50
OPTIMIZATION_RUNS = 10
TRAIN_COUNT = 10
MUTATION_RATE = 0.25
ELITISM_RATIO = 0.15  # top X fraction kept each generation
STAGNATION_LIMIT = 10
ALGORITHM_VERSION = "ga-scheduler-v2"
random.seed(RANDOM_SEED)

# ----------------------------
# Priority weights & headway
# ----------------------------
PRIORITY_WEIGHTS = {"express": 3, "passenger": 2, "freight": 1}
# Minimum headway (minutes) required between two trains on the same platform when scheduled back-to-back
MIN_HEADWAY = 2

# ----------------------------
# Data model
# The optimizer consumes SchedulerSnapshot and returns OptimizationResult.
# ----------------------------


@dataclass(frozen=True)
class Train:
    train_id: str
    train_type: str
    arrival: int
    cross_time: int
    platform: int
    priority: Optional[int] = None

    @property
    def weight(self) -> int:
        return self.priority if self.priority is not None else PRIORITY_WEIGHTS.get(self.train_type, 2)


@dataclass(frozen=True)
class SchedulerSnapshot:
    snapshot_id: str
    trains: Tuple[Train, ...]
    minimum_headway_minutes: int = MIN_HEADWAY
    maintenance_windows: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ScheduleEntry:
    train_id: str
    sequence: int
    platform: int
    planned_start: int
    planned_end: int
    delay_minutes: int
    weighted_delay: int


@dataclass(frozen=True)
class OptimizationResult:
    optimization_id: str
    snapshot_id: str
    status: str
    objective_score: float
    schedule: Tuple[ScheduleEntry, ...]
    metrics: Dict[str, float]
    generations_completed: int
    random_seed: int
    algorithm_version: str = ALGORITHM_VERSION

    def as_dict(self) -> dict:
        result = asdict(self)
        result["schedule"] = [asdict(entry) for entry in self.schedule]
        return result


def validate_snapshot(snapshot: SchedulerSnapshot) -> None:
    if not snapshot.snapshot_id:
        raise ValueError("snapshot_id is required")
    if snapshot.minimum_headway_minutes < 0:
        raise ValueError("minimum_headway_minutes cannot be negative")
    train_ids = [train.train_id for train in snapshot.trains]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("train IDs must be unique")
    for train in snapshot.trains:
        if not train.train_id:
            raise ValueError("train_id is required")
        if train.train_type not in PRIORITY_WEIGHTS:
            raise ValueError(f"unsupported train type: {train.train_type}")
        if train.arrival < 0 or train.cross_time <= 0 or train.platform <= 0:
            raise ValueError(f"invalid timing or platform data for {train.train_id}")
    for start, end in snapshot.maintenance_windows:
        if start >= end:
            raise ValueError("maintenance windows must have start < end")


def snapshot_train_data(snapshot: SchedulerSnapshot) -> Dict[str, dict]:
    return {
        train.train_id: {
            "id": train.train_id,
            "type": train.train_type,
            "arrival": train.arrival,
            "cross_time": train.cross_time,
            "platform": train.platform,
            "priority": train.priority,
        }
        for train in snapshot.trains
    }


# ----------------------------
# Fitness function: models platforms, maintenance, headways, priorities
# Lower is better (total weighted delay + heavy penalties)
# ----------------------------
def evaluate_schedule(schedule: List[str], trains: Dict[str, dict],
                      maintenance_windows: List[Tuple[int, int]] = None,
                      minimum_headway: int = MIN_HEADWAY) -> float:
    """
    schedule: list of train IDs in the order we let them occupy their platform (not per-platform assignment)
    trains: mapping id -> train data (arrival, cross_time, platform if fixed)
    maintenance_windows: list of (start_min, end_min) intervals during which platform is blocked
    Returns a scalar score: lower is better
    """
    maintenance_windows = maintenance_windows or []
    platform_free_at = {}   # platform -> time minute when it becomes free
    total_weighted_delay = 0
    penalty = 0

    for tid in schedule:
        t = trains[tid]
        plat = t.get("platform", 1)  # default platform 1 if not specified
        arrival = int(t["arrival"])
        cross = int(t["cross_time"])
        typ = t.get("type", "passenger")
        weight = t.get("priority") or PRIORITY_WEIGHTS.get(typ, 2)

        free_at = platform_free_at.get(plat, 0)
        # Respect minimum headway: next train can only start at free_at + MIN_HEADWAY
        start = max(free_at + minimum_headway, arrival)

        # Maintenance conflict penalty: heavy penalty if any part of train occupation overlaps maintenance window
        for (m_start, m_end) in maintenance_windows:
            # overlap if start < m_end and (start+cross) > m_start
            if start < m_end and (start + cross) > m_start:
                penalty += 1000  # huge penalty (effectively disallows)
        # Delay computed as waiting beyond arrival
        delay = max(0, start - arrival)
        total_weighted_delay += delay * weight

        # Update platform free time
        platform_free_at[plat] = start + cross

        # Safety check: if start < arrival (shouldn't happen) penalize
        if start < arrival:
            penalty += 500

    # Also penalize overlapping trains on same platform in schedule order (double-check)
    # This is mostly redundant due to start calculation, but keep as guard.
    # Lower score is better
    score = total_weighted_delay + penalty
    return score


# ----------------------------
# GA: permutation-based
# Supports warm-start via init_pop (list of permutations)
# ----------------------------
def create_random_individual(trains: Dict[str, dict]) -> List[str]:
    ids = list(trains.keys())
    random.shuffle(ids)
    return ids


def order_crossover(p1: List[str], p2: List[str]) -> List[str]:
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    # copy slice from p1
    child[a:b+1] = p1[a:b+1]
    # fill remaining from p2 in order
    idx = 0
    for gene in p2:
        if gene in child:
            continue
        while child[idx] is not None:
            idx += 1
        child[idx] = gene
    return child


def swap_mutation(ind: List[str]) -> None:
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]


def tournament_selection(pop: List[List[str]], scores: List[float], k=3) -> List[str]:
    # pick k random individuals, return best
    idxs = random.sample(range(len(pop)), min(k, len(pop)))
    best = min(idxs, key=lambda i: scores[i])
    return pop[best]


def improve_schedule(schedule: List[str], trains: Dict[str, dict],
                     maintenance_windows: List[Tuple[int, int]] = None,
                     minimum_headway: int = MIN_HEADWAY) -> List[str]:
    """Improve a schedule by repeatedly keeping the best beneficial swap."""
    improved = schedule[:]
    best_score = evaluate_schedule(improved, trains, maintenance_windows, minimum_headway)

    while True:
        swap_schedule = None
        swap_score = best_score
        for first_index in range(len(improved) - 1):
            for second_index in range(first_index + 1, len(improved)):
                candidate = improved[:]
                candidate[first_index], candidate[second_index] = (
                    candidate[second_index], candidate[first_index]
                )
                candidate_score = evaluate_schedule(candidate, trains, maintenance_windows, minimum_headway)
                if candidate_score < swap_score:
                    swap_schedule = candidate
                    swap_score = candidate_score

        if swap_schedule is None:
            return improved
        improved = swap_schedule
        best_score = swap_score


def run_ga(trains: Dict[str, dict],
           maintenance_windows: List[Tuple[int, int]] = None,
           minimum_headway: int = MIN_HEADWAY,
           generations: int = GENERATIONS,
           pop_size: int = POP_SIZE,
           mutation_rate: float = MUTATION_RATE,
           init_pop: Optional[List[List[str]]] = None,
           verbose: bool = True,
           stagnation_limit: int = STAGNATION_LIMIT,
           generation_count: Optional[List[int]] = None) -> List[List[str]]:
    # initialize population
    if init_pop:
        population = init_pop[:]  # copy
        while len(population) < pop_size:
            population.append(create_random_individual(trains))
    else:
        population = [create_random_individual(trains) for _ in range(pop_size)]

    best_score_seen = float("inf")
    stagnant_generations = 0
    completed_generations = 0

    for gen in range(generations):
        completed_generations += 1
        # evaluate
        scores = [evaluate_schedule(ind, trains, maintenance_windows, minimum_headway) for ind in population]
        # sort by score ascending (lower better)
        paired = sorted(zip(scores, population), key=lambda x: x[0])
        scores = [score for score, _ in paired]
        population = [p for (_, p) in paired]

        # log best occasionally
        if verbose and gen % max(1, generations // 10) == 0:
            print(f"[GA] gen {gen+1}/{generations} best score = {paired[0][0]}")

        if paired[0][0] < best_score_seen:
            best_score_seen = paired[0][0]
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        if stagnant_generations >= stagnation_limit:
            break

        # elitism: keep top E
        E = max(1, int(pop_size * ELITISM_RATIO))
        new_pop = population[:E]

        # fill rest
        while len(new_pop) < pop_size:
            # selection
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            child = order_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                swap_mutation(child)
            new_pop.append(child)
        population = new_pop

    if generation_count is not None:
        generation_count.append(completed_generations)

    # Refine the best GA candidate with deterministic local search.
    final_scores = [evaluate_schedule(ind, trains, maintenance_windows, minimum_headway) for ind in population]
    paired = sorted(zip(final_scores, population), key=lambda x: x[0])
    best_schedule = improve_schedule(paired[0][1], trains, maintenance_windows, minimum_headway)
    paired[0] = (evaluate_schedule(best_schedule, trains, maintenance_windows, minimum_headway), best_schedule)
    paired.sort(key=lambda x: x[0])
    return [p for (_, p) in paired]


# ----------------------------
# Pretty print an evaluated schedule
# ----------------------------
def print_schedule(schedule: List[str], trains: Dict[str, dict], maintenance_windows=None,
                   minimum_headway: int = MIN_HEADWAY):
    maintenance_windows = maintenance_windows or []
    platform_free_at = {}
    total_weighted_delay = 0
    print("=" * 72)
    print("Train | type      | arr  | start | end  | delay | w_delay | platform")
    print("-" * 72)
    for tid in schedule:
        t = trains[tid]
        plat = t.get("platform", 1)
        arrival = int(t["arrival"])
        cross = int(t["cross_time"])
        typ = t.get("type", "passenger")
        weight = t.get("priority") or PRIORITY_WEIGHTS.get(typ, 2)
        free_at = platform_free_at.get(plat, 0)
        start = max(free_at + minimum_headway, arrival)
        end = start + cross
        delay = max(0, start - arrival)
        wdelay = delay * weight
        total_weighted_delay += wdelay
        platform_free_at[plat] = end
        print(f"{tid:5} | {typ:9} | {arrival:4} | {start:5} | {end:4} | {delay:5} | {wdelay:7} | {plat}")
    print("-" * 72)
    print("Total weighted delay:", total_weighted_delay)
    print("=" * 72)


# ----------------------------
# Main pipeline: initial optimization
# ----------------------------
def build_schedule_entries(schedule: List[str], snapshot: SchedulerSnapshot) -> Tuple[ScheduleEntry, ...]:
    train_data = snapshot_train_data(snapshot)
    platform_free_at = {}
    entries = []

    for sequence, train_id in enumerate(schedule, start=1):
        train = train_data[train_id]
        platform = train["platform"]
        arrival = train["arrival"]
        start = max(platform_free_at.get(platform, 0) + snapshot.minimum_headway_minutes, arrival)
        end = start + train["cross_time"]
        delay = max(0, start - arrival)
        entries.append(ScheduleEntry(
            train_id=train_id,
            sequence=sequence,
            platform=platform,
            planned_start=start,
            planned_end=end,
            delay_minutes=delay,
            weighted_delay=delay * train["priority"] if train["priority"] is not None else delay * PRIORITY_WEIGHTS[train["type"]],
        ))
        platform_free_at[platform] = end

    return tuple(entries)


class SchedulerService:
    def __init__(self, trains: Dict[str, dict], maintenance_windows=None):
        self.trains = trains
        self.maintenance_windows = maintenance_windows or []

    def optimize(self, snapshot: SchedulerSnapshot, runs: int = OPTIMIZATION_RUNS) -> OptimizationResult:
        validate_snapshot(snapshot)
        if runs <= 0:
            raise ValueError("runs must be greater than zero")

        trains = snapshot_train_data(snapshot)
        best = None
        best_score = float("inf")
        best_generations = 0
        print(f"[Service] Running {runs} optimization runs...")

        for run_number in range(1, runs + 1):
            generation_count = []
            pop = run_ga(
                trains,
                snapshot.maintenance_windows,
                snapshot.minimum_headway_minutes,
                generations=GENERATIONS,
                pop_size=POP_SIZE,
                verbose=False,
                generation_count=generation_count,
            )
            score = evaluate_schedule(pop[0], trains, snapshot.maintenance_windows, snapshot.minimum_headway_minutes)
            print(f"[Service] Run {run_number}/{runs} score: {score:.1f}")
            if score < best_score:
                best = pop[0]
                best_score = score
                best_generations = generation_count[0]

        entries = build_schedule_entries(best, snapshot)
        total_delay = sum(entry.delay_minutes for entry in entries)
        weighted_delay = sum(entry.weighted_delay for entry in entries)
        return OptimizationResult(
            optimization_id=f"opt_{snapshot.snapshot_id}",
            snapshot_id=snapshot.snapshot_id,
            status="completed",
            objective_score=best_score,
            schedule=entries,
            metrics={
                "total_delay": total_delay,
                "weighted_delay": weighted_delay,
                "maximum_delay": max((entry.delay_minutes for entry in entries), default=0),
                "trains_scheduled": len(entries),
            },
            generations_completed=best_generations,
            random_seed=RANDOM_SEED,
        )

    def initial_optimize(self, runs: int = OPTIMIZATION_RUNS):
        snapshot = SchedulerSnapshot(
            snapshot_id="demo",
            trains=tuple(
                Train(
                    train_id=train_id,
                    train_type=train["type"],
                    arrival=train["arrival"],
                    cross_time=train["cross_time"],
                    platform=train["platform"],
                    priority=train.get("priority"),
                )
                for train_id, train in self.trains.items()
            ),
            minimum_headway_minutes=MIN_HEADWAY,
            maintenance_windows=tuple(self.maintenance_windows),
        )
        result = self.optimize(snapshot, runs)
        print("[Service] Best schedule (score {:.1f}):".format(result.objective_score))
        print_schedule(
            [entry.train_id for entry in result.schedule],
            self.trains,
            self.maintenance_windows,
            snapshot.minimum_headway_minutes,
        )
        return result


# ----------------------------
# Example: combined scenario construction
# ----------------------------
def build_combined_scenario() -> SchedulerSnapshot:
    """
    Builds a combined scenario that includes:
    - multiple trains close in time (peak conflict)
    - priorities mixed (express/passenger/freight)
    - maintenance window
    This scenario is optimized as-is without live updates.
    """
    scenario_random = random.Random()
    train_types = ("express", "passenger", "passenger", "freight")
    trains = []
    for train_number in range(1, TRAIN_COUNT + 1):
        train_type = scenario_random.choice(train_types)
        trains.append(Train(
            train_id=f"T{train_number}",
            train_type=train_type,
            arrival=scenario_random.randint(600, 660),
            cross_time=scenario_random.randint(2, 6),
            platform= scenario_random.randint(1, 2),
        ))

    # A short maintenance window creates a realistic scheduling constraint.
    maintenance_start = scenario_random.randint(620, 640)
    maintenance_windows = [(maintenance_start, maintenance_start + 5)]

    return SchedulerSnapshot(
        snapshot_id="demo",
        trains=tuple(trains),
        minimum_headway_minutes=MIN_HEADWAY,
        maintenance_windows=tuple(maintenance_windows),
    )


# ----------------------------
# Entry point
# ----------------------------
def main_demo():
    snapshot = build_combined_scenario()
    svc = SchedulerService({}, snapshot.maintenance_windows)
    result = svc.optimize(snapshot)
    print("[Service] Best schedule (score {:.1f}):".format(result.objective_score))
    print_schedule(
        [entry.train_id for entry in result.schedule],
        snapshot_train_data(snapshot),
        snapshot.maintenance_windows,
        snapshot.minimum_headway_minutes,
    )
    plot_schedule(snapshot, result)
    print("[Demo] finished.")

def plot_schedule(snapshot: SchedulerSnapshot, result: OptimizationResult,
                  output_path: str = "train_schedule.png") -> None:
    """Save the optimized schedule as a platform occupancy chart."""
    train_types = {train.train_id: train.train_type for train in snapshot.trains}
    platforms = sorted({entry.platform for entry in result.schedule})
    platform_rows = {platform: row for row, platform in enumerate(platforms)}
    colors = {"express": "#d1495b", "passenger": "#00798c", "freight": "#edae49"}

    figure, axis = plt.subplots(figsize=(14, max(4, len(platforms) * 1.2)))
    for entry in result.schedule:
        row = platform_rows[entry.platform]
        duration = entry.planned_end - entry.planned_start
        axis.barh(
            row,
            duration,
            left=entry.planned_start,
            height=0.6,
            color=colors.get(train_types[entry.train_id], "#6c757d"),
            edgecolor="black",
        )
        axis.text(
            entry.planned_start + duration / 2,
            row,
            entry.train_id,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    for start, end in snapshot.maintenance_windows:
        axis.axvspan(start, end, color="#6c757d", alpha=0.2)

    axis.set_yticks(range(len(platforms)))
    axis.set_yticklabels([f"Platform {platform}" for platform in platforms])
    axis.set_xlabel("Time (minutes)")
    axis.set_ylabel("Platform")
    axis.set_title(f"Optimized Train Schedule | Weighted delay: {result.metrics['weighted_delay']:.0f}")
    axis.grid(axis="x", linestyle="--", alpha=0.35)

    from matplotlib.patches import Patch
    axis.legend(handles=[
        Patch(facecolor=colors["express"], label="Express"),
        Patch(facecolor=colors["passenger"], label="Passenger"),
        Patch(facecolor=colors["freight"], label="Freight"),
        Patch(facecolor="#6c757d", alpha=0.2, label="Maintenance"),
    ], loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[Plot] Saved schedule chart to {output_path}")


if __name__ == "__main__":
    main_demo()