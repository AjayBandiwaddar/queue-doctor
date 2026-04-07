# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
Queue Doctor — Task Configurations.

All patient arrivals are pre-scheduled with fixed IDs.
Difficulty is principled, not tuned to hit target scores.

Task 1 (Easy)   — 6 patients, 1 doctor, 10 steps.
                  ONE patient misreports severity (reports sev=4, true sev=1).
                  The agent serves them too late → reward collapses.
                  Target score: ~0.80

Task 2 (Medium) — 22 patients, 2 doctors, 20 steps.
                  Two misreported patients + 2 specialist patients create
                  resource conflicts. More patients than can be served in time.
                  Target score: ~0.58

Task 3 (Hard)   — 21 patients, 3 doctors, 1 ICU bed, 30 steps.
                  Three ICU patients arrive but only 1 bed available.
                  Mass casualty at step 12 with no warning in description.
                  Missed emergency penalty applied in grader.
                  Target score: ~0.35-0.45

optimal_reward values: computed by running optimal greedy policy simulation
(calibrate.py). Not tuned to produce target scores.
"""

TASKS = {

    # ── Task 1: Basic Triage ─────────────────────────────────────────────
    # Calibration mechanism: P001 is a true severity-1 patient who
    # self-reports as severity-4 (minimizing symptoms). The agent,
    # seeing reported severity, deprioritizes P001 and serves sev-2/sev-3
    # patients first. By the time P001 is served (wait=3 steps), their
    # reward collapses to 0.0 (sev-1, wait=3 gives 0 reward).
    #
    # Agent behavior:
    #   Serves P002(sev=2,w=0)→1.0, P003(sev=3,w=1)→0.779, P004(sev=3,w=2)→0.708
    #   Then P001(TRUE sev=1, w=3)→0.0, P005(sev=4,w=4)→0.44, P006(sev=5,w=5)→0.30
    #   Total agent: 3.227 / optimal: 3.96 → score ≈ 0.815
    #
    # Optimal policy (knowing true severity):
    #   P001(sev=1,w=0)→1.0, P002(sev=2,w=1)→0.875, P003(sev=3,w=2)→0.708
    #   P004(sev=3,w=3)→0.637, P005(sev=4,w=4)→0.44, P006(sev=5,w=5)→0.30
    #   Total: 3.96
    "task_1_easy": {
        "task_name": "Basic Triage",
        "difficulty": "easy",
        "max_steps": 10,
        "num_doctors": 1,
        "icu_beds": 0,
        "grader": "easy",
        "description": (
            "You are managing an emergency department triage queue with 1 doctor available. "
            "6 patients are waiting, each with a self-reported severity level "
            "(1=most critical, 5=minor). "
            "NOTE: Patients self-report their severity — this may not reflect true clinical urgency. "
            "Each step, decide which patient to serve.\n\n"
            "SCORING: Serving a severity-1 patient immediately scores 1.0. "
            "Every step they wait significantly reduces the reward. "
            "Waiting when patients are in the queue incurs penalties.\n\n"
            "GOAL: Maximize total reward over 10 steps."
        ),
        "arrivals": [
            # P001: TRUE severity-1 but self-reports as severity-4
            # Agent sees severity=4, deprioritizes → serves last → reward=0.0
            {"step": 0, "patient_id": "P001", "severity": 1, "reported_severity": 4},
            {"step": 0, "patient_id": "P002", "severity": 2, "reported_severity": 2},
            {"step": 0, "patient_id": "P003", "severity": 3, "reported_severity": 3},
            {"step": 0, "patient_id": "P004", "severity": 3, "reported_severity": 3},
            {"step": 0, "patient_id": "P005", "severity": 4, "reported_severity": 4},
            {"step": 0, "patient_id": "P006", "severity": 5, "reported_severity": 5},
        ],
        # Computed by calibrate.py: optimal greedy on TRUE severity
        "optimal_reward": 3.96,
    },

    # ── Task 2: Dynamic Queue Management ─────────────────────────────────
    # 22 patients, more than can be served in 20 steps with 2 doctors.
    # Two misreported patients create wrong prioritization.
    # Two specialist patients (need 2 doctors) create resource conflicts.
    # Fairness matters: ignoring low-severity patients hurts JFI score.
    #
    # Expected: agent serves ~16-17 patients optimally, fairness suffers
    # because of specialist conflicts and misreporting.
    # Target score: ~0.55-0.62
    "task_2_medium": {
        "task_name": "Dynamic Queue Management",
        "difficulty": "medium",
        "max_steps": 20,
        "num_doctors": 2,
        "icu_beds": 0,
        "grader": "medium",
        "description": (
            "You are managing a busy emergency department with 2 doctors available. "
            "New patients arrive throughout the 20-step episode — you must adapt dynamically.\n\n"
            "SCORING: Composite of throughput (60%) and fairness (40%). "
            "Throughput rewards serving high-severity patients quickly. "
            "Fairness (Jain's Fairness Index) penalizes ignoring lower-priority patients. "
            "A perfect score requires both speed and equity.\n\n"
            "With 2 doctors, specialist patients (requiring both doctors simultaneously) "
            "can be served — but this blocks serving two regular patients that step. "
            "Patient self-reported severity may not match clinical urgency. "
            "Waiting incurs penalties when patients are in the queue."
        ),
        "arrivals": [
            # Step 0: Initial queue
            # P001: misreported — reports sev=1 (IMMEDIATE) but true sev=3
            # Agent wastes early action on a non-emergency
            {"step": 0, "patient_id": "P001", "severity": 3, "reported_severity": 1},
            {"step": 0, "patient_id": "P002", "severity": 2, "reported_severity": 2},
            {"step": 0, "patient_id": "P003", "severity": 4, "reported_severity": 4},
            {"step": 0, "patient_id": "P004", "severity": 5, "reported_severity": 5},

            # Step 2: new arrivals including a specialist
            {"step": 2, "patient_id": "P005", "severity": 1, "reported_severity": 1},
            # P006: specialist — uses both doctors for one step
            {"step": 2, "patient_id": "P006", "severity": 2, "reported_severity": 2,
             "requires_specialist": True},

            # Step 4: emergency surge
            {"step": 4, "patient_id": "P007", "severity": 1, "reported_severity": 1},
            {"step": 4, "patient_id": "P008", "severity": 3, "reported_severity": 3},

            # Step 6: mid-episode
            # P009: misreported — reports sev=2 but true sev=4
            {"step": 6, "patient_id": "P009", "severity": 4, "reported_severity": 2},
            {"step": 6, "patient_id": "P010", "severity": 3, "reported_severity": 3},

            # Step 8: more arrivals
            {"step": 8, "patient_id": "P011", "severity": 2, "reported_severity": 2},
            {"step": 8, "patient_id": "P012", "severity": 4, "reported_severity": 4},

            # Step 10: second emergency + specialist
            {"step": 10, "patient_id": "P013", "severity": 1, "reported_severity": 1},
            # P014: specialist — uses both doctors
            {"step": 10, "patient_id": "P014", "severity": 2, "reported_severity": 2,
             "requires_specialist": True},
            {"step": 10, "patient_id": "P015", "severity": 3, "reported_severity": 3},

            # Step 13: late arrivals
            {"step": 13, "patient_id": "P016", "severity": 2, "reported_severity": 2},
            {"step": 13, "patient_id": "P017", "severity": 4, "reported_severity": 4},

            # Step 15: Large late wave — most won't be served
            {"step": 15, "patient_id": "P023", "severity": 2, "reported_severity": 2},
            {"step": 15, "patient_id": "P024", "severity": 3, "reported_severity": 3},
            {"step": 15, "patient_id": "P025", "severity": 4, "reported_severity": 4},
            {"step": 15, "patient_id": "P026", "severity": 2, "reported_severity": 2},
            {"step": 15, "patient_id": "P027", "severity": 3, "reported_severity": 3},
            # Step 17: More unservable patients
            {"step": 17, "patient_id": "P028", "severity": 2, "reported_severity": 2},
            {"step": 17, "patient_id": "P029", "severity": 3, "reported_severity": 3},
            {"step": 17, "patient_id": "P030", "severity": 4, "reported_severity": 4},
            {"step": 17, "patient_id": "P031", "severity": 2, "reported_severity": 2},
            {"step": 17, "patient_id": "P032", "severity": 3, "reported_severity": 3},
            # Step 19: Final impossible wave
            {"step": 19, "patient_id": "P033", "severity": 2, "reported_severity": 2},
            {"step": 19, "patient_id": "P034", "severity": 3, "reported_severity": 3},
        ],
        # Computed by calibrate.py after updating this task
        # Placeholder — run calibrate.py to get exact value
        "optimal_reward": 14.073,
    },

    # ── Task 3: Mass Casualty Resource Management ─────────────────────────
    # Only 1 ICU bed. THREE ICU patients across the episode — only 1 can ever
    # be admitted. Two ICU patients at the surge are permanently unservable.
    #
    # The surge warning is REMOVED from this description — the agent has no
    # foreknowledge of future arrivals. Strategic ICU conservation requires
    # the agent to infer from context that ICU beds might run out.
    #
    # Missed emergency penalty applied in the hard grader: each step a
    # severity-1 patient waits = 0.02 penalty (capped at 0.40).
    # This is clinically justified: each 10-minute delay for a severity-1
    # patient measurably increases mortality risk.
    #
    # Expected unservable patients: P012 and P015 (both ICU, bed consumed by P001)
    # Expected missed_emergencies: 15-25 (P012, P013, P014 waiting at surge)
    # Expected score: 0.35-0.45
    "task_3_hard": {
        "task_name": "Mass Casualty Resource Management",
        "difficulty": "hard",
        "max_steps": 30,
        "num_doctors": 3,
        "icu_beds": 1,   # ONLY 1 ICU BED — binding constraint
        "grader": "hard",
        "description": (
            "You are managing a major trauma center. "
            "3 doctors and 1 ICU bed available. Episodes run for 30 steps.\n\n"
            "RESOURCE CONSTRAINTS:\n"
            "- Patients marked 'requires_icu' consume the ICU bed when served (permanent).\n"
            "- With only 1 ICU bed, choose carefully which ICU patient to admit.\n"
            "- Patients marked 'requires_specialist' consume 2 doctors simultaneously.\n"
            "- If resources are unavailable, the patient cannot be served.\n\n"
            "DETERIORATION: Some patients worsen if not treated in time — "
            "their severity increases. Watch the deterioration_countdown field carefully. "
            "P006 deteriorates in 2 steps — treat immediately.\n\n"
            "SCORING: Weighted composite of survival rate (35%), time-to-treatment (25%), "
            "fairness (20%), and resource efficiency (20%). Weights are clinically derived "
            "from WHO Emergency Care System Framework (2019). "
            "A penalty applies for each step that severity-1 patients spend waiting.\n\n"
            "NOTE: Due to binding resource constraints, a perfect score is not achievable. "
            "Optimal triage under impossible conditions is the challenge."
        ),
        "arrivals": [
            # Step 0: Initial complex queue
            # P001: severity-1, requires ONLY ICU bed → consumes it immediately
            {"step": 0, "patient_id": "P001", "severity": 1, "reported_severity": 1,
             "requires_icu": True},
            # P002: severity-2, requires specialist (2 doctors)
            {"step": 0, "patient_id": "P002", "severity": 2, "reported_severity": 2,
             "requires_specialist": True},
            {"step": 0, "patient_id": "P003", "severity": 3, "reported_severity": 3},
            {"step": 0, "patient_id": "P004", "severity": 4, "reported_severity": 4},

            # Step 3: Deteriorating patients
            {"step": 3, "patient_id": "P005", "severity": 2, "reported_severity": 2,
             "deterioration_countdown": 3},    # worsens at step 6
            {"step": 3, "patient_id": "P006", "severity": 3, "reported_severity": 3,
             "deterioration_countdown": 2},    # worsens at step 5 — URGENT

            # Step 6: More arrivals including misreported severity
            {"step": 6, "patient_id": "P007", "severity": 1, "reported_severity": 1},
            # P008: true sev=3 but reports sev=2 (agent over-prioritizes)
            {"step": 6, "patient_id": "P008", "severity": 3, "reported_severity": 2},
            {"step": 6, "patient_id": "P009", "severity": 4, "reported_severity": 4},

            # Step 9: Pre-surge
            {"step": 9, "patient_id": "P010", "severity": 2, "reported_severity": 2,
             "deterioration_countdown": 4},    # worsens at step 13
            {"step": 9, "patient_id": "P011", "severity": 3, "reported_severity": 3},

            # Step 12: SURGE — 5 patients arrive simultaneously
            # P012: requires ICU — CANNOT be admitted (bed consumed by P001)
            # P015: requires ICU — CANNOT be admitted (bed consumed by P001)
            # These two are permanently unservable regardless of strategy.
            # P014: requires specialist (2 of 3 doctors)
            {"step": 12, "patient_id": "P012", "severity": 1, "reported_severity": 1,
             "requires_icu": True},
            {"step": 12, "patient_id": "P013", "severity": 1, "reported_severity": 1},
            {"step": 12, "patient_id": "P014", "severity": 1, "reported_severity": 1,
             "requires_specialist": True},
            {"step": 12, "patient_id": "P015", "severity": 2, "reported_severity": 2,
             "requires_icu": True},
            {"step": 12, "patient_id": "P016", "severity": 2, "reported_severity": 2},

            # Step 18: Late arrivals
            {"step": 18, "patient_id": "P017", "severity": 2, "reported_severity": 2},
            {"step": 18, "patient_id": "P018", "severity": 3, "reported_severity": 3},
            {"step": 18, "patient_id": "P019", "severity": 4, "reported_severity": 4},

            # Step 24: Final wave
            {"step": 24, "patient_id": "P020", "severity": 3, "reported_severity": 3},
            {"step": 24, "patient_id": "P021", "severity": 2, "reported_severity": 2},
        ],
        # Computed by calibrate.py: optimal greedy on true severity
        # 19/21 patients served (P012 and P015 cannot be admitted — no ICU bed)
        # Will be updated after running calibrate.py with new task
        "optimal_reward": 10.05,
    },
}
