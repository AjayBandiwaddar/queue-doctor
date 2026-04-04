# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
Patient data model for Queue Doctor.

Severity levels follow the Manchester Triage System (MTS):
  1 = IMMEDIATE    — Immediately life-threatening. Target: 0 min.
  2 = VERY_URGENT  — Serious condition.            Target: 10 min.
  3 = URGENT       — Significant problem.          Target: 60 min.
  4 = LESS_URGENT  — Less serious.                 Target: 120 min.
  5 = NON_URGENT   — Minor problem.                Target: 240 min.

Reference: Manchester Triage Group (2014). Emergency Triage, 3rd Ed.
"""

from dataclasses import dataclass, field
from typing import Optional

SEVERITY_NAMES = {
    1: "IMMEDIATE",
    2: "VERY_URGENT",
    3: "URGENT",
    4: "LESS_URGENT",
    5: "NON_URGENT",
}

SEVERITY_COLORS = {
    1: "RED",
    2: "ORANGE",
    3: "YELLOW",
    4: "GREEN",
    5: "BLUE",
}


@dataclass
class Patient:
    """
    Represents a patient in the emergency department queue.

    True severity vs reported severity may differ in Task 3
    (patient self-report can be inaccurate — simulates real triage uncertainty).
    """
    patient_id: str
    severity: int            # True clinical severity (1-5)
    reported_severity: int   # What the patient reports (may differ in Task 3)
    arrival_step: int
    wait_time: int = 0

    # Task 3 complexity attributes
    deterioration_countdown: int = -1   # Steps until condition worsens. -1 = won't deteriorate.
    requires_icu: bool = False           # Needs 1 ICU bed when served
    requires_specialist: bool = False    # Needs 2 doctors when served
    condition: str = "stable"            # stable | at_risk | critical

    def to_dict(self) -> dict:
        d = {
            "patient_id": self.patient_id,
            "severity": self.reported_severity,     # Agent sees reported severity
            "true_severity": self.severity,          # Same unless misreported
            "severity_name": SEVERITY_NAMES.get(self.reported_severity, "UNKNOWN"),
            "triage_color": SEVERITY_COLORS.get(self.reported_severity, "UNKNOWN"),
            "wait_time": self.wait_time,
            "condition": self.condition,
            "requires_icu": self.requires_icu,
            "requires_specialist": self.requires_specialist,
        }
        if self.deterioration_countdown > 0:
            d["deterioration_countdown"] = self.deterioration_countdown
            d["deterioration_warning"] = (
                f"URGENT: Patient deteriorates in {self.deterioration_countdown} step(s)!"
                if self.deterioration_countdown <= 2
                else f"Patient will worsen in {self.deterioration_countdown} steps if untreated."
            )
        return d