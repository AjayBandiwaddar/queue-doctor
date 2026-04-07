# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
Queue Engine -- Core RL State Machine for Queue Doctor.

Every serve_patient() or wait() call advances the simulation by 1 step.
Resource errors (no ICU bed, insufficient doctors) do NOT advance the step.

Key design decisions:
    1. Queue sorted by REPORTED severity -- misreporting genuinely deceives
       the agent (realistic: triage nurses rely on self-report initially).
    2. wait() penalty only fires for SERVABLE patients -- patients that
       cannot be served due to resource constraints (ICU full, insufficient
       doctors) do not contribute to the wait penalty. This prevents
       double-counting: unservable patients already penalize via the hard
       grader's missed_emergency counter at finalize_episode().
"""

import random
from typing import List, Dict, Tuple
from .models import Patient, SEVERITY_NAMES


class QueueEngine:
    """Deterministic (or seeded-stochastic) hospital queue simulator."""

    def __init__(self, task_config: dict, seed: int = None):
        self.task_config      = task_config
        self.max_steps        = task_config["max_steps"]
        self.num_doctors      = task_config["num_doctors"]
        self.icu_capacity     = task_config.get("icu_beds", 0)
        self.seed             = seed

        self.step             = 0
        self.queue: List[Patient] = []
        self.served: List[Dict]   = []
        self.available_doctors    = self.num_doctors
        self.available_icu        = self.icu_capacity
        self.missed_emergencies   = 0
        self.cumulative_reward    = 0.0
        self.step_rewards: List[float] = []
        self.events_log: List[str]     = []

        rng = random.Random(seed) if seed is not None else None
        self._arrival_schedule: Dict[int, List[Patient]] = {}

        for a in task_config["arrivals"]:
            step_offset = 0
            sev_offset  = 0

            if rng is not None:
                if rng.random() < 0.10:
                    step_offset = rng.choice([-1, 1])
                if rng.random() < 0.15:
                    sev_offset = rng.choice([-1, 1])
                    if a["severity"] != 1 and a["severity"] + sev_offset < 1:
                        sev_offset = 0
                    if a["severity"] == 1 and sev_offset == 1:
                        sev_offset = 0

            arrival_step = max(0, a["step"] + step_offset)
            true_sev     = max(1, min(5, a["severity"] + sev_offset))
            reported_sev = a.get("reported_severity", a["severity"])
            if reported_sev == a["severity"] and sev_offset != 0:
                reported_sev = max(1, min(5, reported_sev + sev_offset))

            if arrival_step not in self._arrival_schedule:
                self._arrival_schedule[arrival_step] = []

            self._arrival_schedule[arrival_step].append(Patient(
                patient_id=a["patient_id"],
                severity=true_sev,
                reported_severity=reported_sev,
                arrival_step=arrival_step,
                deterioration_countdown=a.get("deterioration_countdown", -1),
                requires_icu=a.get("requires_icu", False),
                requires_specialist=a.get("requires_specialist", False),
            ))

        self._process_arrivals(0)

    def _can_serve(self, p: Patient) -> bool:
        """Check if patient can be served given current resources."""
        doctors_needed = 2 if p.requires_specialist else 1
        return (
            self.available_doctors >= doctors_needed and
            (not p.requires_icu or self.available_icu >= 1)
        )

    def get_state(self) -> dict:
        # Sort by REPORTED severity -- misreporting genuinely deceives agent
        sorted_queue = sorted(
            self.queue,
            key=lambda p: (p.reported_severity, -p.wait_time)
        )
        state = {
            "step":               self.step,
            "max_steps":          self.max_steps,
            "steps_remaining":    self.max_steps - self.step,
            "queue":              [self._patient_dict(p) for p in sorted_queue],
            "queue_length":       len(self.queue),
            "available_doctors":  self.available_doctors,
            "patients_served":    len(self.served),
            "missed_emergencies": self.missed_emergencies,
            "cumulative_reward":  round(self.cumulative_reward, 4),
            "done":               self.step >= self.max_steps,
        }
        if self.icu_capacity > 0:
            state["available_icu_beds"] = self.available_icu
            state["total_icu_beds"]     = self.icu_capacity
        state["triage_advisory"] = self._build_advisory(sorted_queue)
        return state

    def _patient_dict(self, p: Patient) -> dict:
        d              = p.to_dict()
        can_serve      = self._can_serve(p)
        d["can_serve_now"] = can_serve
        if not can_serve:
            doctors_needed = 2 if p.requires_specialist else 1
            if p.requires_icu and self.available_icu == 0:
                d["cannot_serve_reason"] = "No ICU beds available"
            elif self.available_doctors < doctors_needed:
                d["cannot_serve_reason"] = (
                    f"Needs {doctors_needed} doctors, "
                    f"only {self.available_doctors} available"
                )
        return d

    def serve_patient(self, patient_id: str) -> Tuple[float, dict, List[str]]:
        """
        Serve a patient. ADVANCES TIME BY 1 STEP only if action is valid.
        Resource errors do NOT advance time.
        """
        patient = next((p for p in self.queue if p.patient_id == patient_id), None)

        if patient is None:
            self.step += 1
            self._advance_step()
            self.step_rewards.append(-0.1)
            self.cumulative_reward -= 0.1
            return -0.1, self.get_state(), [
                f"Error: Patient {patient_id} not found in queue."
            ]

        doctors_needed = 2 if patient.requires_specialist else 1

        if self.available_doctors < doctors_needed:
            return 0.0, self.get_state(), [
                f"Cannot serve {patient_id}: needs {doctors_needed} doctors, "
                f"{self.available_doctors} available. Choose another patient."
            ]
        if patient.requires_icu and self.available_icu < 1:
            return 0.0, self.get_state(), [
                f"Cannot admit {patient_id}: no ICU beds available. "
                f"Choose a non-ICU patient."
            ]

        self.queue.remove(patient)
        self.available_doctors -= doctors_needed
        if patient.requires_icu:
            self.available_icu -= 1

        reward = self._compute_reward(patient)
        self.cumulative_reward += reward

        resource_note = ""
        if patient.requires_specialist:
            resource_note = " [2 doctors used]"
        if patient.requires_icu:
            resource_note += (
                f" [ICU bed consumed, "
                f"{self.available_icu}/{self.icu_capacity} remaining]"
            )

        self.served.append({
            "patient_id":        patient.patient_id,
            "true_severity":     patient.severity,
            "reported_severity": patient.reported_severity,
            "wait_time":         patient.wait_time,
            "reward":            round(reward, 4),
            "served_step":       self.step,
        })

        events = [
            f"Step {self.step}: Served {patient_id} "
            f"(reported={patient.reported_severity}, true={patient.severity}, "
            f"waited {patient.wait_time} steps) -> reward {reward:.3f}{resource_note}"
        ]

        self.step += 1
        self.available_doctors = self.num_doctors
        events.extend(self._advance_step())
        self.step_rewards.append(reward)

        return reward, self.get_state(), events

    def wait(self) -> Tuple[float, dict, List[str]]:
        """
        Skip step. Penalized only for SERVABLE patients waiting.

        Key fix: patients that cannot be served due to resource constraints
        (ICU full, insufficient doctors) do NOT contribute to the wait penalty.
        They already penalize via missed_emergencies at finalize_episode().
        Counting them twice would make waiting with ICU-blocked patients
        catastrophically punishing even when the agent has no alternative.

        Penalty tiers (based on highest reported severity among SERVABLE patients):
            Severity-1 servable in queue: -0.30 per patient
            Severity 2-3 servable:        -0.10
            Any servable patient:         -0.05
            No servable patients:          0.00 (agent has no valid action)
        """
        penalty = 0.0
        events  = []

        # Only count SERVABLE patients for the penalty
        servable_patients = [p for p in self.queue if self._can_serve(p)]

        if servable_patients:
            emergencies = sum(
                1 for p in servable_patients if p.reported_severity == 1
            )
            urgent = sum(
                1 for p in servable_patients if 2 <= p.reported_severity <= 3
            )
            if emergencies > 0:
                penalty = -0.3 * emergencies
                events.append(
                    f"Waited with {emergencies} IMMEDIATE servable patient(s)! "
                    f"Penalty: {penalty:.2f}"
                )
            elif urgent > 0:
                penalty = -0.1
                events.append(
                    f"Waited with {urgent} urgent servable patient(s). Penalty: -0.10"
                )
            else:
                penalty = -0.05
                events.append("Waited with non-urgent servable patients. Penalty: -0.05")
        elif self.queue:
            # Patients present but ALL resource-blocked -- no penalty
            # (agent physically cannot serve anyone; missed_emergencies tracks this)
            blocked_count = len(self.queue)
            events.append(
                f"Step {self.step}: {blocked_count} patient(s) present but all "
                f"resource-blocked -- no penalty. Missed emergencies tracked at finalize."
            )
        else:
            events.append(f"Step {self.step}: Queue empty -- no penalty.")

        self.cumulative_reward += penalty
        self.step += 1
        events.extend(self._advance_step())
        self.step_rewards.append(penalty)

        return penalty, self.get_state(), events

    def _advance_step(self) -> List[str]:
        events       = []
        new_arrivals = self._arrival_schedule.get(self.step, [])

        if new_arrivals:
            for p in new_arrivals:
                self.queue.append(p)
            names = ", ".join(
                f"{p.patient_id}(reported={p.reported_severity})"
                for p in new_arrivals
            )
            events.append(f"Step {self.step}: New arrivals -- {names}")
            if len(new_arrivals) >= 4:
                events.append(
                    f"MASS CASUALTY EVENT: "
                    f"{len(new_arrivals)} patients arrived simultaneously!"
                )

        for patient in list(self.queue):
            if patient.deterioration_countdown > 0:
                patient.deterioration_countdown -= 1
                if patient.deterioration_countdown == 0:
                    old_sev          = patient.severity
                    patient.severity = max(1, patient.severity - 1)
                    patient.reported_severity = patient.severity
                    patient.condition         = (
                        "critical" if patient.severity == 1 else "at_risk"
                    )
                    patient.deterioration_countdown = -1
                    events.append(
                        f"DETERIORATION: {patient.patient_id} worsened "
                        f"severity {old_sev}->{patient.severity}"
                    )

        for patient in self.queue:
            patient.wait_time += 1
            if patient.severity == 1:
                self.missed_emergencies += 1

        self.available_doctors = self.num_doctors
        return events

    def _process_arrivals(self, step: int) -> None:
        for p in self._arrival_schedule.get(step, []):
            self.queue.append(p)

    def _compute_reward(self, patient: Patient) -> float:
        """
        Manchester Triage System reward using TRUE severity.
        1 step = approx 10 minutes of real time.
        """
        w = patient.wait_time
        s = patient.severity

        if s == 1:
            if w == 0: return 1.00
            if w == 1: return 0.60
            if w == 2: return 0.20
            return 0.00
        elif s == 2:
            return max(0.0, 1.00 - w * 0.125)
        elif s == 3:
            return max(0.0, 0.85 - w * 0.071)
        elif s == 4:
            return max(0.0, 0.60 - w * 0.040)
        else:
            return max(0.0, 0.40 - w * 0.020)

    def _build_advisory(self, sorted_queue: list) -> str:
        """Advisory based on REPORTED severity. Excluded from LLM prompt."""
        if not sorted_queue:
            return "Queue is empty."
        parts        = []
        emergencies  = [p for p in self.queue if p.reported_severity == 1]
        deteriorating = [
            p for p in self.queue
            if p.deterioration_countdown > 0 and p.deterioration_countdown <= 2
        ]

        if emergencies:
            ids = ", ".join(p.patient_id for p in emergencies)
            parts.append(f"IMMEDIATE (reported): {ids}")
        if deteriorating:
            ids = ", ".join(
                f"{p.patient_id}(in {p.deterioration_countdown} steps)"
                for p in deteriorating
            )
            parts.append(f"DETERIORATING SOON: {ids}")
        if self.icu_capacity > 0 and self.available_icu == 0:
            icu_waiting = [p for p in self.queue if p.requires_icu]
            if icu_waiting:
                parts.append(
                    f"ICU FULL: {len(icu_waiting)} patient(s) cannot be admitted."
                )
        if not parts:
            top = sorted_queue[0]
            parts.append(
                f"Highest priority: {top.patient_id} "
                f"(reported sev={top.reported_severity}, "
                f"waited {top.wait_time} steps)"
            )
        return " | ".join(parts)