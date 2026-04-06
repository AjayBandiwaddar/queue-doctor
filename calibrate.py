"""
calibrate.py — Find true optimal reward for each Queue Doctor task.

Run this locally BEFORE inference.py to determine correct optimal_reward values.
The optimal policy is: always serve the highest-priority servable patient
(lowest severity number, break ties by longest wait time).

Usage:
    python calibrate.py

Output: optimal_reward values to paste into server/tasks.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.models import Patient, SEVERITY_NAMES
from server.queue_engine import QueueEngine
from server.tasks import TASKS


def optimal_policy(engine) -> float:
    """
    Run the theoretically optimal greedy policy:
    1. Serve severity-1 patients immediately
    2. Among equal severity, serve longest-waiting first
    3. Respect resource constraints (skip if can't serve)
    4. Never wait if any servable patient exists

    Returns total cumulative reward achieved.
    """
    total_reward = 0.0
    steps_taken  = 0

    while engine.step < engine.max_steps:
        state    = engine.get_state()
        queue    = state["queue"]
        servable = [p for p in queue if p.get("can_serve_now", True)]

        if servable:
            # Sort: severity ascending (1=most critical), wait_time descending
            best = sorted(
                servable,
                key=lambda p: (p["severity"], -p["wait_time"])
            )[0]
            reward, _, _ = engine.serve_patient(best["patient_id"])
            total_reward += reward
        else:
            # Nothing servable — must wait
            reward, _, _ = engine.wait()
            total_reward += reward

        steps_taken += 1

    return round(total_reward, 4)


def run_calibration():
    print("=" * 60)
    print("Queue Doctor — Optimal Policy Calibration")
    print("=" * 60)
    print()
    print("Running optimal greedy policy on each task (no seed = deterministic)...")
    print()

    results = {}

    for task_id, task_config in TASKS.items():
        engine       = QueueEngine(task_config, seed=None)
        optimal      = optimal_policy(engine)
        served       = len(engine.served)
        total        = len(task_config["arrivals"])
        current_est  = task_config.get("optimal_reward", "N/A")

        results[task_id] = optimal

        print(f"Task: {task_config['task_name']} ({task_config['difficulty']})")
        print(f"  Patients served:        {served}/{total}")
        print(f"  True optimal reward:    {optimal}")
        print(f"  Current estimate:       {current_est}")
        print(f"  Difference:             {round(optimal - current_est, 4) if isinstance(current_est, (int, float)) else 'N/A'}")
        print()

        # Show what the normalized scores would be with correct optimal
        print(f"  Score normalization preview:")
        print(f"    If agent gets 90% of optimal: {round(0.9 * optimal / optimal, 4)}")
        print(f"    If agent gets 75% of optimal: {round(0.75, 4)}")
        print(f"    If agent gets 60% of optimal: {round(0.60, 4)}")
        print()

    print("=" * 60)
    print("PASTE THESE VALUES INTO server/tasks.py")
    print("=" * 60)
    for task_id, optimal in results.items():
        print(f'  "{task_id}": optimal_reward = {optimal}')
    print()

    # Also run with a few seeds to show variance
    print("=" * 60)
    print("Seed variance check (optimal policy on randomized episodes)")
    print("=" * 60)
    print()

    for seed in [42, 7, 99]:
        print(f"Seed {seed}:")
        for task_id, task_config in TASKS.items():
            engine  = QueueEngine(task_config, seed=seed)
            optimal = optimal_policy(engine)
            # Score relative to deterministic optimal
            det_optimal = results[task_id]
            relative = round(optimal / det_optimal, 4) if det_optimal > 0 else 0
            print(f"  {task_config['task_name']}: optimal={optimal}, relative={relative}")
        print()


if __name__ == "__main__":
    run_calibration()