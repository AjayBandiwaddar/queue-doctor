# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
Queue Engine — Core RL State Machine for Queue Doctor.

Every serve_patient() or wait() call advances the simulation by 1 step.
Resource errors (no ICU bed, insufficient doctors) do NOT advance the step —
the agent must choose a different patient.
"""

from typing import List, Dict, Tuple
from .models import Patient, SEVERITY_NAMES


class QueueEngine:
    """Deterministic hospital queue simulator."""

    def __init__(self, task_config: dict):
        self.task_config = task_config
        self.max_steps = task_config["max_steps"]
        self.num_doctors = task_config["num_doctors"]
        self.icu_capacity = task_config.get("icu_beds", 0)

        self.step = 0
        self.queue: List[Patient] = []
        self.served: List[Dict] = []
        self.available_doctors = self.num_doctors
        self.available_icu = self.icu_capacity
        self.missed_emergencies = 0
        self.cumulative_reward = 0.0
        self.step_rewards: List[float] = []
        self.events_log: List[str] = []

        self._arrival_schedule: Dict[int, List[Patient]] = {}
        for a in task_config["arrivals"]:
            s = a["step"]
            if s not in self._arrival_schedule:
                self._arrival_schedule[s] = []
            self._arrival_schedule[s].append(Patient(
                patient_id=a["patient_id"],
                severity=a["severity"],
                reported_severity=a.get("reported_severity", a["severity"]),
                arrival_step=s,
                deterioration_countdown=a.get("deterioration_countdown", -1),
                requires_icu=a.get("requires_icu", False),
                requires_specialist=a.get("requires_specialist", False),
            ))

        self._process_arrivals(0)

    def get_state(self) -> dict:
        sorted_queue = sorted(self.queue, key=lambda p: (p.severity, -p.wait_time))
        state = {
            "step": self.step,
            "max_steps": self.max_steps,
            "steps_remaining": self.max_steps - self.step,
            "queue": [self._patient_dict(p) for p in sorted_queue],
            "queue_length": len(self.queue),
            "available_doctors": self.available_doctors,
            "patients_served": len(self.served),
            "missed_emergencies": self.missed_emergencies,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "done": self.step >= self.max_steps,
        }
        if self.icu_capacity > 0:
            state["available_icu_beds"] = self.available_icu
            state["total_icu_beds"] = self.icu_capacity
        state["triage_advisory"] = self._build_advisory(sorted_queue)
        return state

    def _patient_dict(self, p: Patient) -> dict:
        """Patient dict with servability flag."""
        d = p.to_dict()
        doctors_needed = 2 if p.requires_specialist else 1
        can_serve = (
            self.available_doctors >= doctors_needed and
            (not p.requires_icu or self.available_icu >= 1)
        )
        d["can_serve_now"] = can_serve
        if not can_serve:
            if p.requires_icu and self.available_icu == 0:
                d["cannot_serve_reason"] = "No ICU beds available"
            elif self.available_doctors < doctors_needed:
                d["cannot_serve_reason"] = f"Needs {doctors_needed} doctors, only {self.available_doctors} available"
        return d

    def serve_patient(self, patient_id: str) -> Tuple[float, dict, List[str]]:
        """
        Serve a patient. ADVANCES TIME BY 1 STEP only if action is valid.
        Resource errors (no ICU/doctors) do NOT advance time — agent must choose differently.
        """
        patient = next((p for p in self.queue if p.patient_id == patient_id), None)
        if patient is None:
            # Patient not in queue — still advance (agent made wrong call)
            self.step += 1
            self._advance_step()
            self.step_rewards.append(-0.1)
            self.cumulative_reward -= 0.1
            return -0.1, self.get_state(), [f"Error: Patient {patient_id} not found in queue."]

        doctors_needed = 2 if patient.requires_specialist else 1

        # Resource checks — do NOT advance time, let agent choose again
        if self.available_doctors < doctors_needed:
            return -0.0, self.get_state(), [
                f"Cannot serve {patient_id}: needs {doctors_needed} doctors, "
                f"{self.available_doctors} available. Choose another patient."
            ]
        if patient.requires_icu and self.available_icu < 1:
            return -0.0, self.get_state(), [
                f"Cannot admit {patient_id}: no ICU beds available. "
                f"Choose a non-ICU patient or wait."
            ]

        # Valid serve — remove from queue, consume resources
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
            resource_note += f" [ICU bed consumed, {self.available_icu}/{self.icu_capacity} remaining]"

        self.served.append({
            "patient_id": patient.patient_id,
            "true_severity": patient.severity,
            "reported_severity": patient.reported_severity,
            "wait_time": patient.wait_time,
            "reward": round(reward, 4),
            "served_step": self.step,
        })

        events = [
            f"Step {self.step}: Served {patient_id} "
            f"(severity {patient.severity}, waited {patient.wait_time} steps)"
            f" → reward {reward:.3f}{resource_note}"
        ]

        self.step += 1
        self.available_doctors = self.num_doctors
        events.extend(self._advance_step())
        self.step_rewards.append(reward)

        return reward, self.get_state(), events

    def wait(self) -> Tuple[float, dict, List[str]]:
        """Skip step. Penalized if patients are waiting. Always advances time."""
        penalty = 0.0
        events = []

        if self.queue:
            emergencies = sum(1 for p in self.queue if p.severity == 1)
            urgent = sum(1 for p in self.queue if 2 <= p.severity <= 3)
            if emergencies > 0:
                penalty = -0.3 * emergencies
                events.append(f"CRITICAL: Waited with {emergencies} IMMEDIATE patient(s)! Penalty: {penalty:.2f}")
            elif urgent > 0:
                penalty = -0.1
                events.append(f"Waited with {urgent} urgent patient(s). Penalty: -0.10")
            else:
                penalty = -0.05
                events.append("Waited with non-urgent patients. Minor penalty: -0.05")
        else:
            events.append(f"Step {self.step}: Queue empty — no penalty.")

        self.cumulative_reward += penalty
        self.step += 1
        events.extend(self._advance_step())
        self.step_rewards.append(penalty)

        return penalty, self.get_state(), events

    def _advance_step(self) -> List[str]:
        events = []
        new_arrivals = self._arrival_schedule.get(self.step, [])
        if new_arrivals:
            for p in new_arrivals:
                self.queue.append(p)
            names = ", ".join(f"{p.patient_id}(sev={p.severity})" for p in new_arrivals)
            events.append(f"Step {self.step}: New arrivals — {names}")
            # Warn about mass casualty
            if len(new_arrivals) >= 4:
                events.append(f"⚠️  MASS CASUALTY EVENT: {len(new_arrivals)} patients arrived simultaneously!")

        for patient in list(self.queue):
            if patient.deterioration_countdown > 0:
                patient.deterioration_countdown -= 1
                if patient.deterioration_countdown == 0:
                    old_sev = patient.severity
                    patient.severity = max(1, patient.severity - 1)
                    patient.reported_severity = patient.severity
                    patient.condition = "critical" if patient.severity == 1 else "at_risk"
                    patient.deterioration_countdown = -1
                    events.append(
                        f"⚠️  DETERIORATION: {patient.patient_id} worsened "
                        f"severity {old_sev}→{patient.severity}"
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
        """Manchester Triage System-based reward. 1 step ≈ 10 minutes."""
        w = patient.wait_time
        s = patient.severity   # True severity

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
        if not sorted_queue:
            return "Queue is empty."
        parts = []
        emergencies = [p for p in self.queue if p.severity == 1]
        deteriorating = [p for p in self.queue
                        if p.deterioration_countdown > 0 and p.deterioration_countdown <= 2]
        upcoming = self._arrival_schedule.get(self.step, [])

        if emergencies:
            ids = ", ".join(p.patient_id for p in emergencies)
            parts.append(f"IMMEDIATE: {ids} need urgent treatment now!")
        if deteriorating:
            ids = ", ".join(
                f"{p.patient_id}(⚠️{p.deterioration_countdown}s)" for p in deteriorating)
            parts.append(f"DETERIORATING: {ids}")
        if self.icu_capacity > 0 and self.available_icu == 0:
            icu_waiting = [p for p in self.queue if p.requires_icu]
            if icu_waiting:
                parts.append(f"ICU FULL: {len(icu_waiting)} ICU patient(s) cannot be admitted.")
        if upcoming and any(a.severity <= 2 for a in upcoming):
            parts.append("INCOMING: Emergency/urgent patient arrives this step!")
        if not parts:
            top = sorted_queue[0]
            parts.append(
                f"Highest priority: {top.patient_id} "
                f"(severity {top.severity}, waited {top.wait_time} steps)"
            )
        return " | ".join(parts)