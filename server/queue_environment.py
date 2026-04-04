# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
Queue Doctor — OpenEnv MCPEnvironment.

A genuine multi-step reinforcement learning environment for hospital
emergency department triage. The agent makes sequential decisions
each step — which patient to serve — and the environment state changes
meaningfully in response (new arrivals, wait time increases, deterioration).

This is a true Markov Decision Process: the agent's action at step N
changes the state available at step N+1. A better policy produces
measurably better outcomes across all three tasks.

Stochasticity:
    start_task() accepts an optional seed parameter. When provided, small
    random perturbations are applied to patient attributes (severity ±1
    with 15% probability, arrival step ±1 with 10% probability). This
    ensures each episode is distinct, prevents solution memorization, and
    produces non-zero score variance across runs (required by Phase 2).

Episode workflow:
    list_tasks()
    → start_task(task_id, seed=<int>)   # seed optional
    → get_queue_state()
    → [serve_patient(patient_id) | wait()] × N steps
    → finalize_episode()
"""

import json
from typing import Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

from .tasks import TASKS
from .queue_engine import QueueEngine
from .graders import GRADERS


class QueueDoctorEnvironment(MCPEnvironment):
    """
    Queue Doctor — Hospital Triage RL Environment.

    Three tasks of increasing difficulty:
        task_1_easy   — Static queue, 1 doctor, 10 steps
        task_2_medium — Dynamic arrivals, 2 doctors, 20 steps
        task_3_hard   — Mass casualty, deterioration, ICU, 3 doctors, 30 steps

    MCP tools:
        list_tasks()                    → task catalogue with metadata
        start_task(task_id, seed)       → init episode (seed optional for stochasticity)
        get_queue_state()               → observe current state (no time advance)
        serve_patient(patient_id)       → treat patient, advance 1 step
        wait()                          → skip step (penalized), advance 1 step
        finalize_episode()              → compute final normalized score
        get_current_state()             → environment-level metadata
    """

    def __init__(self):
        mcp = FastMCP("queue_doctor")

        @mcp.tool
        def list_tasks() -> str:
            """
            List all available triage tasks with metadata.
            Returns task IDs, names, difficulty, resources, and descriptions.
            """
            return json.dumps([
                {
                    "task_id":         tid,
                    "task_name":       t["task_name"],
                    "difficulty":      t["difficulty"],
                    "max_steps":       t["max_steps"],
                    "num_doctors":     t["num_doctors"],
                    "icu_beds":        t.get("icu_beds", 0),
                    "total_patients":  len(t["arrivals"]),
                    "description":     t["description"],
                }
                for tid, t in TASKS.items()
            ], indent=2)

        @mcp.tool
        def start_task(task_id: str, seed: int = None) -> str:
            """
            Initialize a task episode. Must be called before any actions.

            Args:
                task_id: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
                seed:    Optional integer seed for episode randomization.
                         When provided, small stochastic perturbations are
                         applied to patient attributes (severity ±1 with 15%
                         probability, arrival step ±1 with 10% probability).
                         Use different seeds across runs to get score variance.
                         Omit for the deterministic baseline episode.

            Returns task description, rules, initial queue state, and workflow.
            """
            if task_id not in TASKS:
                return json.dumps({
                    "error": f"Unknown task_id '{task_id}'. "
                             f"Valid: {list(TASKS.keys())}"
                })

            self._active_task_id = task_id
            self._engine         = QueueEngine(TASKS[task_id], seed=seed)
            self._state.step_count += 1

            task          = TASKS[task_id]
            initial_state = self._engine.get_state()

            return json.dumps({
                "task_id":         task_id,
                "task_name":       task["task_name"],
                "difficulty":      task["difficulty"],
                "description":     task["description"],
                "max_steps":       task["max_steps"],
                "num_doctors":     task["num_doctors"],
                "icu_beds":        task.get("icu_beds", 0),
                "seed":            seed,
                "initial_queue":   initial_state["queue"],
                "queue_length":    initial_state["queue_length"],
                "triage_advisory": initial_state["triage_advisory"],
                "workflow": (
                    "1. Call get_queue_state() to observe current patients.\n"
                    "2. Call serve_patient(patient_id) to treat a patient "
                    "   — this advances time by 1 step.\n"
                    "3. OR call wait() to skip a step "
                    "   (penalized if patients are waiting).\n"
                    "4. Repeat until done=true.\n"
                    "5. Call finalize_episode() to get your final score."
                ),
            }, indent=2)

        @mcp.tool
        def get_queue_state() -> str:
            """
            Observe the current emergency department state. Does NOT advance time.

            Returns:
                - Current step and steps remaining
                - All patients sorted by priority (severity, then wait time)
                - can_serve_now flag per patient (resource availability check)
                - Available doctors and ICU beds
                - Patients served and missed emergencies
                - Cumulative reward
                - Triage advisory (for inspection — not used by the inference agent)
                - done flag
            """
            if self._engine is None:
                return json.dumps({
                    "error": "No active task. Call start_task(task_id) first."
                })
            self._state.step_count += 1
            return json.dumps(self._engine.get_state(), indent=2)

        @mcp.tool
        def serve_patient(patient_id: str) -> str:
            """
            Assign a doctor to treat a patient. ADVANCES SIMULATION BY 1 STEP.

            After this action:
            - Patient removed from queue
            - Wait times increase for all remaining patients
            - New patients may arrive (deterministic or seeded schedule)
            - Deteriorating patients' countdowns decrease (Task 3)
            - Step counter increments

            Resource errors (no ICU bed, insufficient doctors) do NOT advance
            time — the agent receives an error message and must choose again.

            Args:
                patient_id: Patient ID (e.g. 'P001', 'P007')

            Returns step reward, updated queue state, and events log.
            """
            if self._engine is None:
                return json.dumps({
                    "error": "No active task. Call start_task(task_id) first."
                })
            if self._engine.step >= self._engine.max_steps:
                return json.dumps({
                    "error": "Episode complete. Call finalize_episode().",
                    "done":  True,
                })

            reward, new_state, events = self._engine.serve_patient(patient_id)
            self._cumulative_reward  += reward
            self._state.step_count   += 1

            return json.dumps({
                "action":      f"serve_patient({patient_id})",
                "step_reward": round(reward, 4),
                "events":      events,
                "state":       new_state,
                "done":        new_state["done"],
                "hint": (
                    "Call finalize_episode() to get your final score."
                    if new_state["done"] else
                    "Continue serving patients or call finalize_episode() anytime."
                ),
            }, indent=2)

        @mcp.tool
        def wait() -> str:
            """
            Skip this step without serving any patient. ADVANCES SIMULATION BY 1 STEP.

            Penalties:
              Emergency (severity 1) in queue: -0.30 per patient
              Urgent (severity 2-3) in queue:  -0.10
              Any patient in queue:             -0.05
              Empty queue:                       0.00

            Returns step penalty, updated queue state, and events log.
            """
            if self._engine is None:
                return json.dumps({
                    "error": "No active task. Call start_task(task_id) first."
                })
            if self._engine.step >= self._engine.max_steps:
                return json.dumps({
                    "error": "Episode complete. Call finalize_episode().",
                    "done":  True,
                })

            penalty, new_state, events = self._engine.wait()
            self._cumulative_reward   += penalty
            self._state.step_count    += 1

            return json.dumps({
                "action":      "wait()",
                "step_reward": round(penalty, 4),
                "events":      events,
                "state":       new_state,
                "done":        new_state["done"],
            }, indent=2)

        @mcp.tool
        def finalize_episode() -> str:
            """
            Finalize the current task and compute the final normalized score.

            Applies the principled grader to produce a score in [0, 1].
            Grader weights are derived from published clinical literature —
            not tuned to hit target scores.

            Returns final score, component scores, and full episode statistics.
            """
            if self._engine is None:
                return json.dumps({
                    "error": "No active task. Call start_task(task_id) first."
                })

            task_id = self._active_task_id
            task    = TASKS[task_id]
            result  = GRADERS[task["grader"]](self._engine)

            self._finalized_tasks[task_id] = result["score"]
            done       = len(self._finalized_tasks) >= len(TASKS)
            self._done = done
            self._state.step_count += 1

            return json.dumps({
                "task_id":         task_id,
                "task_name":       task["task_name"],
                "difficulty":      task["difficulty"],
                **result,
                "episode_steps":   self._engine.step,
                "patients_served": len(self._engine.served),
                "served_detail":   self._engine.served,
                "tasks_completed": len(self._finalized_tasks),
                "tasks_total":     len(TASKS),
                "all_done":        done,
            }, indent=2)

        @mcp.tool
        def get_current_state() -> str:
            """Get environment-level metadata (episode state, not queue state)."""
            return json.dumps({
                "episode_id":        self._state.episode_id,
                "step_count":        self._state.step_count,
                "active_task":       self._active_task_id,
                "finalized_scores":  self._finalized_tasks,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "done":              self._done,
                "tasks_available":   list(TASKS.keys()),
            }, indent=2)

        super().__init__(mcp)
        self._state                         = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_reward: float      = 0.0
        self._done: bool                    = False
        self._active_task_id: Optional[str] = None
        self._engine: Optional[QueueEngine] = None
        self._finalized_tasks: dict         = {}

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state               = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._cumulative_reward   = 0.0
        self._done                = False
        self._active_task_id      = None
        self._engine              = None
        self._finalized_tasks     = {}
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status":  "ready",
                "message": (
                    "Queue Doctor ready. "
                    "Workflow: list_tasks() → start_task(task_id, seed=<int>) → "
                    "get_queue_state() → "
                    "[serve_patient(patient_id) or wait()] × N → "
                    "finalize_episode()"
                ),
                "tasks_available": list(TASKS.keys()),
            },
        )

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False, reward=0.0,
            metadata={
                "error": f"Unknown action: {type(action).__name__}. Use MCP tools."
            },
        )

    def step(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state