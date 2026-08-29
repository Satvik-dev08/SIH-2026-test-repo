"""Backtracking solver for the SYNAPSE train-scheduling problem."""

import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

PRIORITY_WEIGHTS = {"express": 3, "passenger": 2, "freight": 1}
MIN_HEADWAY = 2
TRAIN_COUNT = 50
RANDOM_SEED = 42
MIN_CURRENT_DELAY = 20
MAX_CURRENT_DELAY = 200


@dataclass(frozen=True)
class Train:
    train_id: str
    train_type: str
    arrival: int
    cross_time: int
    platform: int
    priority: Optional[int] = None
    delay_minutes: int = 0


@dataclass(frozen=True)
class SchedulerSnapshot:
    snapshot_id: str
    trains: Tuple[Train, ...]
    minimum_headway_minutes: int = MIN_HEADWAY
    maintenance_windows: Tuple[Tuple[int, int], ...] = ()


def build_combined_scenario():
    rng = random.Random(RANDOM_SEED)
    train_types = ("express", "passenger", "passenger", "freight")
    trains = []

    for train_number in range(1, TRAIN_COUNT + 1):
        trains.append(
            Train(
                train_id=f"T{train_number}",
                train_type=rng.choice(train_types),
                arrival=rng.randint(600, 660),
                cross_time=rng.randint(2, 6),
                platform=rng.randint(1, 24),
                delay_minutes=rng.randint(MIN_CURRENT_DELAY, MAX_CURRENT_DELAY),
            )
        )

    maintenance_start = rng.randint(620, 640)
    return SchedulerSnapshot(
        snapshot_id="backtracking-demo",
        trains=tuple(trains),
        minimum_headway_minutes=MIN_HEADWAY,
        maintenance_windows=((maintenance_start, maintenance_start + 5),),
    )


class BacktrackingScheduler:
    def __init__(self, snapshot, max_nodes=250_000, time_limit_seconds=15):
        self.snapshot = snapshot
        self.trains = {train.train_id: train for train in snapshot.trains}
        self.train_ids = tuple(self.trains)
        self.max_nodes = max_nodes
        self.time_limit_seconds = time_limit_seconds
        self.nodes = 0
        self.started_at = 0.0
        self.best_schedule = []
        self.best_score = float("inf")
        self.complete = True

    def score_schedule(self, schedule):
        platform_free = {}
        score = 0
        for train_id in schedule:
            train = self.trains[train_id]
            arrival = train.arrival + train.delay_minutes
            start = max(
                platform_free.get(train.platform, 0)
                + self.snapshot.minimum_headway_minutes,
                arrival,
            )
            score += max(0, start - arrival) * (
                train.priority
                or {"express": 3, "passenger": 2, "freight": 1}[train.train_type]
            )
            platform_free[train.platform] = start + train.cross_time
        return score

    def schedule_details(self, schedule):
        platform_free = {}
        details = []

        for sequence, train_id in enumerate(schedule, start=1):
            train = self.trains[train_id]
            effective_arrival = train.arrival + train.delay_minutes
            start = max(
                platform_free.get(train.platform, 0)
                + self.snapshot.minimum_headway_minutes,
                effective_arrival,
            )
            exit_time = start + train.cross_time
            new_delay = max(0, start - effective_arrival)
            weight = train.priority or PRIORITY_WEIGHTS[train.train_type]
            details.append(
                {
                    "sequence": sequence,
                    "train_id": train_id,
                    "platform": train.platform,
                    "planned_arrival": train.arrival,
                    "existing_delay": train.delay_minutes,
                    "effective_arrival": effective_arrival,
                    "start": start,
                    "exit": exit_time,
                    "new_delay": new_delay,
                    "weighted_delay": new_delay * weight,
                    "total_delay": train.delay_minutes + new_delay,
                }
            )
            platform_free[train.platform] = exit_time

        return details

    def lower_bound(self, remaining, platform_free):
        bound = 0
        headway = self.snapshot.minimum_headway_minutes
        for train_id in remaining:
            train = self.trains[train_id]
            arrival = train.arrival + train.delay_minutes
            earliest_start = max(
                platform_free.get(train.platform, 0) + headway, arrival
            )
            weight = (
                train.priority
                or {"express": 3, "passenger": 2, "freight": 1}[train.train_type]
            )
            bound += max(0, earliest_start - arrival) * weight
        return bound

    def search(self, remaining, schedule, platform_free, score):
        self.nodes += 1
        if (
            self.nodes >= self.max_nodes
            or time.perf_counter() - self.started_at >= self.time_limit_seconds
        ):
            self.complete = False
            return
        if not remaining:
            if score < self.best_score:
                self.best_score = score
                self.best_schedule = schedule[:]
            return
        if score + self.lower_bound(remaining, platform_free) >= self.best_score:
            return

        candidates = sorted(
            remaining,
            key=lambda train_id: (
                self.trains[train_id].arrival + self.trains[train_id].delay_minutes,
                (
                    -self.trains[train_id].priority
                    if self.trains[train_id].priority is not None
                    else -{"express": 3, "passenger": 2, "freight": 1}[
                        self.trains[train_id].train_type
                    ]
                ),
            ),
        )
        for train_id in candidates:
            train = self.trains[train_id]
            arrival = train.arrival + train.delay_minutes
            start = max(
                platform_free.get(train.platform, 0)
                + self.snapshot.minimum_headway_minutes,
                arrival,
            )
            train_score = max(0, start - arrival) * (
                train.priority
                or {"express": 3, "passenger": 2, "freight": 1}[train.train_type]
            )
            next_free = platform_free.copy()
            next_free[train.platform] = start + train.cross_time
            self.search(
                tuple(item for item in remaining if item != train_id),
                schedule + [train_id],
                next_free,
                score + train_score,
            )
            if not self.complete:
                return

    def optimize(self):
        self.started_at = time.perf_counter()
        greedy = sorted(
            self.train_ids,
            key=lambda train_id: self.trains[train_id].arrival
            + self.trains[train_id].delay_minutes,
        )
        self.best_schedule = greedy
        self.best_score = self.score_schedule(greedy)
        self.search(self.train_ids, [], {}, 0)
        return (
            self.best_schedule,
            self.best_score,
            self.nodes,
            time.perf_counter() - self.started_at,
            self.complete,
        )


def main():
    snapshot = build_combined_scenario()
    backtracking = BacktrackingScheduler(snapshot)
    schedule, score, nodes, elapsed, exact = backtracking.optimize()
    print("[Backtracking]")
    print(f"Weighted delay: {score}")
    print(f"Nodes explored: {nodes}")
    print(f"Runtime: {elapsed:.3f}s")
    print(f"Exact result: {exact}")
    print()
    print(
        "Train | Platform | Planned | Existing | Effective | Start | Exit | "
        "New delay | Weighted | Total delay"
    )
    print("-" * 112)
    for row in backtracking.schedule_details(schedule):
        print(
            f"{row['train_id']:5} | "
            f"{row['platform']:8} | "
            f"{row['planned_arrival']:7} | "
            f"{row['existing_delay']:8} | "
            f"{row['effective_arrival']:9} | "
            f"{row['start']:5} | "
            f"{row['exit']:4} | "
            f"{row['new_delay']:9} | "
            f"{row['weighted_delay']:8} | "
            f"{row['total_delay']:11}"
        )
    details = backtracking.schedule_details(schedule)
    print("-" * 112)
    print(f"New delay: {sum(row['new_delay'] for row in details)} minutes")
    print(f"Weighted delay: {sum(row['weighted_delay'] for row in details)}")
    print(f"Total delay: {sum(row['total_delay'] for row in details)} minutes")


if __name__ == "__main__":
    main()
